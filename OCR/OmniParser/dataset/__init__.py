import torch 

from pathlib import Path
from torch.utils.data import ConcatDataset

from . import transforms as T
from .text_spotting import TextSpottingDataset
from .cord import CordDataset
from .open_image_v5 import OpenImageV5Dataset
from .coco_text import COCOTextDataset
from .sroie import SroieDataset

from .collate_fn import CollateFN

def build_dataloader(dataset, image_set, args):
    if args.distributed:
        shuffle = True if image_set == 'train' else False
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
    else:
        if image_set == 'train':
            sampler = torch.utils.data.RandomSampler(dataset)
        elif image_set == 'val':
            sampler = torch.utils.data.SequentialSampler(dataset)
    
    collate_fn = CollateFN(image_set, args)
    if image_set == 'train':
        batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=args.num_workers
        )
    elif image_set == 'val':
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=args.num_workers
        )

    return dataloader, sampler


def build_dataset(image_set, args):

    transforms = build_transform(image_set, args)

    root = Path(args.data_root)
    if image_set == 'train':
        dataset_names = args.train_dataset 
    elif image_set == 'val':
        dataset_names = args.val_dataset

    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'totaltext_train':
            img_folder = root / "totaltext" / "train_images"; ann_file = root / "totaltext" / "train.json"
        elif dataset_name == 'totaltext_val':
            img_folder = root / "totaltext" / "test_images"; ann_file = root / "totaltext" / "test.json"
        elif dataset_name == 'mlt_train':
            img_folder = root / "mlt2017" / "MLT_train_images"; ann_file = root / "mlt2017" / "train.json"
        elif dataset_name == 'ctw1500_train':
            img_folder = root / "CTW1500" / "ctwtrain_text_image"; ann_file = root / "CTW1500" / "annotations" / "train_ctw1500_maxlen100_v2.json"
        elif dataset_name == 'ctw1500_val':
            img_folder = root / "CTW1500" / "ctwtest_text_image"; ann_file = root / "CTW1500" / "annotations" / "test_ctw1500_maxlen100.json"
        elif dataset_name == 'syntext1_train':
            img_folder = root / "syntext1" / "syntext_word_eng"; ann_file = root / "syntext1" / "train.json"
        elif dataset_name == 'syntext2_train':
            img_folder = root / "syntext2" / "emcs_imgs"; ann_file = root / "syntext2" / "train.json"
        elif dataset_name == 'ic13_train':
            img_folder = root / "icdar2013" / "train_images"; ann_file = root / "icdar2013" / "ic13_train_dc.json"
        elif dataset_name == 'ic13_val':
            img_folder = root / "icdar2013" / "test_images"; ann_file = root / "icdar2013" / "ic13_test.json"
        elif dataset_name == 'ic15_train':
            img_folder = root / "icdar2015" / "train_images"; ann_file = root / "icdar2015" / "ic15_train_dc.json"
        elif dataset_name == 'ic15_val':
            img_folder = root / "icdar2015" / "test_images"; ann_file = root / "icdar2015" / "ic15_test.json"
        elif dataset_name == 'hiertext_trainval':
            img_folder = root / "hiertext" / "train"; ann_file = root / "hiertext" / "hiertext_trainval.json"
        elif dataset_name == 'textocr_trainval':
            img_folder = root / "textocr" / "train_images"; ann_file = root / "textocr" / "textocr_trainval.json"
        elif dataset_name in ['cord_train', 'cord_val', 'open_image_v5_trainval', 'cocotext_trainval', 'sroie_train', 'sroie_val']:
            pass
        else:
            raise ValueError

        if 'cord' in dataset_name:
            dataset = CordDataset(dataset_name, transforms, args)
        elif 'open_image_v5' in dataset_name:
            dataset = OpenImageV5Dataset(dataset_name, transforms, args)
        elif 'cocotext' in dataset_name:
            dataset = COCOTextDataset(dataset_name, transforms, args)
        elif 'sroie' in dataset_name:
            dataset = SroieDataset(dataset_name, transforms, args)
        else:
            dataset = TextSpottingDataset(img_folder, ann_file, dataset_name, transforms, args)
        datasets.append(dataset)

    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    
    return dataset 

def build_transform(image_set, args):
    transforms = []
    if image_set == 'train':
        if args.train_vie:
            transforms.append(T.RandomCropInst(args.crop_min_size_ratio, args.crop_max_size_ratio, args.crop_prob))
        else:
            transforms.append(T.RandomCrop(args.crop_min_size_ratio, args.crop_max_size_ratio, args.crop_prob))
        transforms.append(T.RandomRotate(args.rotate_max_angle, args.rotate_prob))
        transforms.append(T.RandomResize(args.train_min_size, args.train_max_size))
        transforms.append(T.RandomDistortion(args.dist_brightness, args.dist_contrast, args.dist_saturation, args.dist_hue, args.distortion_prob))
    if image_set == 'val':
        transforms.append(T.RandomResize([args.test_min_size], args.test_max_size))

    transforms.append(T.ToTensor())
    transforms.append(T.Normalize())

    transforms = T.Compose(transforms) if len(transforms) > 0 else None 

    return transforms