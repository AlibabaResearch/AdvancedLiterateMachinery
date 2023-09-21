# Copyright (2023) Alibaba Group and its affiliates

import os
import re
import cv2
import lmdb
import six
import string
from fastai.vision import *
from torchvision import transforms
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import math
from itertools import chain
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import unicodedata
from typing import List

from .img_transforms import CVColorJitter, CVDeterioration, CVGeometry


class MyConcatDataset(ConcatDataset):
    def __getattr__(self, k): 
        return getattr(self.datasets[0], k)


class ImageDataset(Dataset):
    "`ImageDataset` read data from LMDB database."
    def __init__(self,
                 path:PathOrStr,
                 is_training:bool=True,
                 img_h:int=32,
                 img_w_max:int=384,
                 do_resize=False,
                 max_len:int=25,
                 use_ctc=False,
                 char94:bool=False,
                 convert_mode:str='RGB',
                 data_aug:bool=True,
                 deteriorate_ratio:float=0.,
                 return_idx:bool=False,
                 **kwargs):
        self.path, self.name = Path(path), Path(path).name
        assert self.path.is_dir() and self.path.exists(), f"{path} is not a valid directory."
        self.convert_mode = convert_mode
        self.img_h = img_h
        self.img_w_max = img_w_max
        self.do_resize = do_resize
        self.return_idx = return_idx
        self.char94, self.is_training = char94, is_training
        self.data_aug = data_aug and is_training
        if char94:
            charlist = list(string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation)
        else:
            charlist = list(string.ascii_lowercase + string.digits)
        self.unsupported = f"[^{re.escape(''.join(charlist))}]"
        if use_ctc:
            self.blank_token = '[BLK]'
            self.charlist = [self.blank_token] + charlist
        else:
            self.eos_token = '[EOS]'
            self.charlist = [self.eos_token] + charlist
        self.use_ctc = use_ctc
        self.c = len(self.charlist)
        self.char2id = dict(zip(self.charlist, range(len(self.charlist))))

        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {path}.'
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode()))
        print(f'training-{str(is_training)}\t{path}:', len(self))
        
        self.max_len = max_len
        
        if self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15),
                    distortion=0.5, antiphase=0, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
    
    def __len__(self):
        return self.length

    def _next_image(self, index):
        next_index = random.randint(0, len(self) - 1)
        return self.get(next_index)

    def _check_image(self, x, pixels=6):
        if x.size[0] <= pixels or x.size[1] <= pixels: return False
        else: return True

    def resize(self, img):
        def _resize_ratio(img, ratio):
            trg_h, trg_w = self.img_h, int(self.img_h * ratio)
            if trg_w < 128:
                trg_w = int(trg_w * 0.33 + 85)
                # trg_w = int(trg_w * 0.21 + 121)
                # trg_w = int(trg_w * 0.26 + 95)

            # trg_w = ((trg_w - 1) // 4 + 1) * 4
            trg_w = min(trg_w, self.img_w_max)
            img = cv2.resize(img, (trg_w, trg_h))
            return img
        
        # rotate vertical ones
        h, w = img.shape[:2]
        if h / w > 4.5:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h
        if self.do_resize:
            return cv2.resize(img, (self.img_w_max, self.img_h))
        if self.is_training and random.random() < 0.6:
            w_norm = w / h * self.img_h
            base_h, maxh = self.img_h, self.img_h
            base_w = max(int(self.img_h * 0.8), int(w_norm * 0.33))
            r_max = min(4, max(1.1, self.img_w_max / w_norm))
            maxw = max(self.img_h, int(w_norm * r_max))
            h, w = random.randint(base_h, maxh), random.randint(base_w, maxw)
        ratio = w / h
        return _resize_ratio(img, ratio)  # keep aspect ratio

    def preprocess_label(self, label:str):
        if self.char94:
            label = ''.join(label.split())
            label = label.replace('，', ',')
            label = label.replace('；', ';')
            label = label.replace('：', ':')
            label = label.replace('？', '?')
            label = label.replace('！', '!')
            label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
        else:
            label = label.lower()
        label = re.sub(self.unsupported, '', label)
        return label

    def get(self, idx):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx+1:09d}', f'label-{idx+1:09d}'
            try:
                label = str(txn.get(label_key.encode()), 'utf-8')  # label
                label = self.preprocess_label(label)
                if self.is_training and len(label) == 0:
                    return self._next_image(idx)
                if self.max_len is not None and len(label) > self.max_len:
                    return self._next_image(idx)

                imgbuf = txn.get(image_key.encode())  # image
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
                    image = PIL.Image.open(buf).convert(self.convert_mode)
                if self.is_training and not self._check_image(image):
                    return self._next_image(idx)
            except:
                import traceback
                traceback.print_exc()
                return self._next_image(idx)
            return image, label, idx

    def _process_image(self, image):
        if self.data_aug:
            image = self.augment_tfs(image)
        image = self.resize(np.array(image))
        return image

    def word2label_att(self, word):
        label_list = [self.char2id[ch] for ch in word] + [self.char2id[self.eos_token]]
        label = torch.LongTensor(label_list)
        label_length = len(label_list)
        return label, label_length
    
    def word2label_ctc(self, word):
        label_list = [self.char2id[ch] for ch in word]
        label = torch.IntTensor(label_list)
        label_length = len(word)
        return label, label_length

    def __getitem__(self, idx):
        image, word, idx_new = self.get(idx)
        if not self.is_training: assert idx == idx_new, \
            f'idx {idx} != idx_new {idx_new} during testing.'

        image = self._process_image(image)
        label, label_len = self.word2label_ctc(word) if self.use_ctc else self.word2label_att(word)

        if self.return_idx: return image, label, label_len, word, idx_new
        else: return image, label, label_len, word


class AlignCollate(object):
    def __init__(self, dataset):
        self.use_ctc = dataset.use_ctc
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    
    def pad_image(self, img, max_w, borderType=cv2.BORDER_CONSTANT):
        mask = np.ones_like(img[:, :, 0])
        if self.use_ctc:
            left, right = 0, max_w - img.shape[1]
        else:
            pad_w = (max_w - img.shape[1]) / 2
            left, right = math.ceil(pad_w), math.floor(pad_w)
        img = cv2.copyMakeBorder(img, 0, 0, left, right, borderType)
        mask = cv2.copyMakeBorder(mask, 0, 0, left, right, borderType)
        mask = mask.astype(np.float32)
        return img, mask

    def __call__(self, batch):
        n_items = len(batch[0])
        if n_items == 4:
            images, labels, lengths, words = zip(*batch)
        elif n_items == 5:
            images, labels, lengths, words, idxes = zip(*batch)
        
        # to know the max length of words and the max width of images
        max_width = 0
        for img, label in zip(images, labels):
            # max_length = max(max_length, label.shape[0])
            max_width = max(max_width, img.shape[1])
        max_width = ((max_width - 1) // 32 + 1) * 32 # multiple of 32
        # pad
        masks = []
        images = list(images)
        for i in range(len(images)):
            images[i], mask = self.pad_image(images[i], max_width)
            images[i] = self.totensor(images[i])
            masks.append(torch.from_numpy(mask))
        b_images = torch.stack(images, dim=0)
        b_masks = torch.stack(masks, dim=0)

        if self.use_ctc:
            b_labels = torch.cat(labels, dim=0)
            b_lengths = torch.IntTensor(lengths)
        else:
            b_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
            b_lengths = torch.LongTensor(lengths)
        if n_items == 4:
            return b_images, b_masks, b_labels, b_lengths, words
        else:
            return b_images, b_masks, b_labels, b_lengths, words, idxes


def find_all_lmdb_dir(root:str):
    lmdb_dirs = []
    if not os.path.isdir(root):
        return lmdb_dirs
    sub_dirs = os.listdir(root)
    if "data.mdb" in sub_dirs and "lock.mdb" in sub_dirs:
        return [root]
    for sd in sub_dirs:
        sub_root = os.path.join(root, sd)
        lmdb_dirs.extend(find_all_lmdb_dir(sub_root))
    return lmdb_dirs


def merge_all_lmdb_dir(roots:List[str]):
    all_dirs = []
    for root in roots:
        all_dirs.extend(find_all_lmdb_dir(root))
    all_dirs = list(set(all_dirs)) # to avoid repeated dirs
    return all_dirs


def get_data(data_dir, img_h, img_w_max, max_len, batch_size, do_resize=False, data_aug=True, use_ctc=False,
             char94=False, workers=4, shuffle=False, is_train=False, distributed=False):
    if isinstance(data_dir, str):
        data_dir = [data_dir]
    data_dir = merge_all_lmdb_dir(data_dir)
    dataset_list = []
    for data_dir_ in data_dir:
        dataset_list.append(
            ImageDataset(data_dir_, is_train, img_h, img_w_max, do_resize,
                max_len, use_ctc, char94, convert_mode='RGB', data_aug=data_aug)
        )
    if len(data_dir) > 1:
        dataset = MyConcatDataset(dataset_list)
    else:
        dataset = dataset_list[0]
    # print('total image: ', len(dataset))
    if distributed:
        data_sampler = DistributedSampler(dataset=dataset)
        shuffle = None
    else:
        data_sampler = None

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
        shuffle=shuffle, pin_memory=True, drop_last=is_train, collate_fn=AlignCollate(dataset), sampler=data_sampler)

    return dataset, data_loader


def test():
    import torchvision
    lmdb_path = [
        # "/home/admin/workspace/dataset/STR_abi_lmdb/training/MJ/MJ_train/",
        # "/home/admin/workspace/dataset/STR_abi_lmdb/training/MJ/MJ_test/",
        # "/home/admin/workspace/dataset/STR_abi_lmdb/training/MJ/MJ_valid/",
        # "/home/admin/workspace/dataset/STR_abi_lmdb/training/ST",
        "/mnt/workspace/workgroup/ccx/dataset/STR_PARSeq/train/real/MLT19/train"
    ]
    train_dataset, train_dataloader = get_data(
        lmdb_path, 32, 256, None, 64, do_resize=False, use_ctc=False, char94=True, shuffle=True, is_train=True)
    print(len(train_dataset))
    for images, masks, labels, label_lens, words in iter(train_dataloader):
        print(images.size())
        # import ipdb; ipdb.set_trace()
        images = torchvision.utils.make_grid(images, nrow=2)
        mu = torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1)
        images = (images * std + mu) * 255
        cv2.imwrite("./data/data_view.jpg", images.numpy().transpose(1, 2, 0))
        print(masks[0, 16])
        print(words)
        print(labels)
        print(label_lens)
        input()


if __name__ == "__main__":
    test()
