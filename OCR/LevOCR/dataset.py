import os
import sys
import re
import six
import math
import lmdb
import torch
import random
import cv2

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
from transforms import CVColorJitter, CVDeterioration, CVGeometry


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []
        balanced_batch_texts_noise = []
        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text, text_noise = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
                balanced_batch_texts_noise += text_noise
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text, text_noise = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
                balanced_batch_texts_noise += text_noise
            except Exception as e:
                print(e)
                pass
        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        return balanced_batch_images, balanced_batch_texts, balanced_batch_texts_noise

def hierarchical_dataset(root, opt, select_data='/', test_flag=False):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            print('dirpath:', dirpath)
            for selected_d in select_data:
                if selected_d in dirpath.split('/') + ['/']:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt, test_flag)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset, dataset_log

class LmdbDataset(Dataset):
    def __init__(self, root, opt, test_flag):

        self.root = root
        self.opt = opt
        if not opt.eval:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])        
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        self.disturb = SpellingMutation(pn0=0.1, pn1=0.7, pn2=0.95, pt0=0.6, pt1=0.8)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')
                    if len(label) > self.opt.batch_max_length:
                        continue
                    
                    self.filtered_index_list.append(index)
                self.nSamples = len(self.filtered_index_list)
        self.test_flag = test_flag

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')
            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            label = re.sub('[^0-9a-zA-Z]', '', label)
            if not self.opt.sensitive:
                label = label.lower()
            
            if self.opt.eval: 
                return (img, label, label)

            img = self.augment_tfs(img)
            label_noise = self.disturb(label)      
        return (img, label, label_noise)

class RawDataset(Dataset):
    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])

class AlignCollateTest(object):

    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, imgs_path = zip(*batch)

        resized_max_w = self.imgW
        input_channel = 3 if images[0].mode == 'RGB' else 1

        resized_images = []
        for image in images:
            image = preprocess(image, 128, 32)
            resized_images.append(image)

        image_tensors = torch.cat(resized_images, 0)

        return image_tensors, imgs_path

class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad

        return Pad_img


def preprocess(img, width, height):
    img = cv2.resize(np.array(img), (width, height))
    img = transforms.ToTensor()(img).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    return (img-mean[...,None,None]) / std[...,None,None]

class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels, label_noise = zip(*batch)

        resized_max_w = self.imgW
        input_channel = 3 if images[0].mode == 'RGB' else 1

        resized_images = []
        for image in images:
            image = preprocess(image, 128, 32)
            resized_images.append(image)

        image_tensors = torch.cat(resized_images, 0)

        return image_tensors, labels, label_noise


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

class TextDataset(Dataset):
    def __init__(self, path, opt):
        self.path = path
        with open(self.path) as f:
            content = f.readlines()
        self.data_list = [x.strip() for x in content]
        self.opt = opt
        self.sm = SpellingMutation_TEXT(pn0=0.2, pn1=0.6, pn2=0.95, pt0=0.5, pt1=0.75)

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        gt = sample_ = self.data_list[idx]
        gt = re.sub('[^0-9a-zA-Z]+', '', gt)
        if not self.opt.sensitive:
            gt = gt.lower()
        gt_noise = self.sm(gt)
        return gt, gt_noise
    
class SpellingMutation(object):
    def __init__(self, pn0=0.2, pn1=0.6, pn2=0.95, pt0=0.5, pt1=0.75):
        """ 
        Args:
            pn0: the prob of not modifying characters is (pn0)
            pn1: the prob of modifying one characters is (pn1 - pn0)
            pn2: the prob of modifying two characters is (pn2 - pn1), 
                 and three (1 - pn2)
            pt0: the prob of replacing operation is pt0.
            pt1: the prob of inserting operation is (pt1 - pt0),
                 and deleting operation is (1 - pt1)
        """
        super().__init__()
        self.pn0, self.pn1, self.pn2 = pn0, pn1, pn2
        self.pt0, self.pt1 = pt0, pt1
        self.digits = '0123456789'
        self.alphabets = 'abcdefghijklmnopqrstuvwxyz'
        self.symbols = self.digits + self.alphabets
        self.max_length = 25

    def is_digit(self, text, ratio=0.5):
        length = max(len(text), 1)
        digit_num = sum([t in self.digits for t in text])
        if digit_num / length < ratio: return False
        return True

    def is_unk_char(self, char):
        # return char == self.charset.unk_char
        return (char not in self.digits) and (char not in self.alphabets)

    def get_num_to_modify(self, length):
        prob = random.random()
        if prob < self.pn0: num_to_modify = 0
        elif prob < self.pn1: num_to_modify = 1
        elif prob < self.pn2: num_to_modify = 2
        else: num_to_modify = 3
        
        if length <= 1: num_to_modify = min(num_to_modify, 1)
        elif length >= 2 and length <= 4: num_to_modify = min(num_to_modify, 1) # min(num_to_modify, 1) 
        else: num_to_modify = min(num_to_modify, length // 2)  # smaller than length // 2
        return num_to_modify

    def __call__(self, text, debug=False):
        if self.is_digit(text): return text
        length = len(text)
        num_to_modify = self.get_num_to_modify(length)
        if num_to_modify <= 0: return text

        chars = []
        index = np.arange(0, length)
        random.shuffle(index)
        index = index[: num_to_modify]
        if debug: self.index = index
        for i, t in enumerate(text):
            if i not in index: chars.append(t)
            elif self.is_unk_char(t): chars.append(t)
            else:
                prob = random.random()
                if prob < self.pt0: # replace
                    chars.append(random.choice(self.alphabets))
                elif prob < self.pt1: # insert
                    chars.append(random.choice(self.alphabets))
                    chars.append(t)
                else: 
                    chars.append(t)
                    chars.append(t)
        new_text = ''.join(chars[: self.max_length-1])
        return new_text if len(new_text) >= 1 else text

class SpellingMutation_TEXT(object):
    def __init__(self, pn0=0.1, pn1=0.6, pn2=0.95, pt0=0.25, pt1=0.5, pt2=0.75 ):
        """ 
        Args:
            pn0: the prob of not modifying characters is (pn0)
            pn1: the prob of modifying one characters is (pn1 - pn0)
            pn2: the prob of modifying two characters is (pn2 - pn1), 
                 and three (1 - pn2)
            pt0: the prob of replacing operation is pt0.
            pt1: the prob of inserting operation is (pt1 - pt0),
                 and deleting operation is (1 - pt1)
        """
        super().__init__()
        self.pn0, self.pn1, self.pn2 = pn0, pn1, pn2
        self.pt0, self.pt1, self.pt2 = pt0, pt1, pt2
        self.digits = '0123456789'
        self.alphabets = 'abcdefghijklmnopqrstuvwxyz'
        self.symbols = self.digits + self.alphabets
        self.max_length = 25

    def is_digit(self, text, ratio=0.5):
        length = max(len(text), 1)
        digit_num = sum([t in self.digits for t in text])
        if digit_num / length < ratio: return False
        return True

    def is_unk_char(self, char):
        # return char == self.charset.unk_char
        return (char not in self.digits) and (char not in self.alphabets)

    def get_num_to_modify(self, length):
        prob = random.random()
        if prob < self.pn0: num_to_modify = 0
        elif prob < self.pn1: num_to_modify = 1
        elif prob < self.pn2: num_to_modify = 2
        else: num_to_modify = 3
        
        if length <= 1: num_to_modify = min(num_to_modify, 1)
        elif length >= 2 and length <= 4: num_to_modify = min(num_to_modify, 1) # min(num_to_modify, 1) 
        else: num_to_modify = min(num_to_modify, length // 2)  # smaller than length // 2
        return num_to_modify

    def __call__(self, text, debug=False):
        if self.is_digit(text): return text
        length = len(text)
        num_to_modify = self.get_num_to_modify(length)
        if num_to_modify <= 0: return text

        chars = []
        index = np.arange(0, length)
        random.shuffle(index)
        index = index[: num_to_modify]
        if debug: self.index = index
        for i, t in enumerate(text):
            if i not in index: chars.append(t)
            elif self.is_unk_char(t): chars.append(t)
            else:
                prob = random.random()
                if prob < self.pt0: # replace
                    chars.append(random.choice(self.alphabets))
                elif prob < self.pt1: # insert
                    chars.append(random.choice(self.alphabets))
                    chars.append(t)
                elif prob < self.pt2: # insert
                    chars.append(t)
                    chars.append(t)
                else: # delete
                    continue

        new_text = ''.join(chars[: self.max_length-1])
        return new_text if len(new_text) >= 1 else text
