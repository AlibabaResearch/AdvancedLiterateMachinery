import time

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.utils.data.dataloader import DataLoader

import cv2
import numpy as np

from lightning_modules.data_modules.vie_dataset import VIEDataset


class VIEDataModule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None
        self.tokenizer = tokenizer
        self.collate_fn = None

        if self.cfg.model.backbone in [
            "alibaba-damo/geolayoutlm-base-uncased",
            "alibaba-damo/geolayoutlm-large-uncased",
        ]:
            self.backbone_type = "geolayoutlm"
        else:
            raise ValueError(
                f"Not supported model: self.cfg.model.backbone={self.cfg.model.backbone}"
            )

    @overrides
    def setup(self, stage=None):
        self.train_loader = self._get_train_loader()
        self.val_loader = self._get_val_test_loaders(mode="val")

    @overrides
    def train_dataloader(self):
        return self.train_loader

    @overrides
    def val_dataloader(self):
        return self.val_loader

    def _get_train_loader(self):
        start_time = time.time()

        dataset = VIEDataset(
            self.cfg.dataset,
            self.cfg.task,
            self.backbone_type,
            self.cfg.model.head,
            self.cfg.dataset_root_path,
            self.tokenizer,
            self.cfg.train.max_seq_length,
            self.cfg.train.max_block_num,
            self.cfg.img_h,
            self.cfg.img_w,
            mode="train",
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for loading training data: {elapsed_time}", flush=True)

        return data_loader

    def _get_val_test_loaders(self, mode):
        dataset = VIEDataset(
            self.cfg.dataset,
            self.cfg.task,
            self.backbone_type,
            self.cfg.model.head,
            self.cfg.dataset_root_path,
            self.tokenizer,
            self.cfg.train.max_seq_length,
            self.cfg.train.max_block_num,
            self.cfg.img_h,
            self.cfg.img_w,
            mode=mode,
        )
        # debug_by_visualization(0, dataset)

        data_loader = DataLoader(
            dataset,
            batch_size=self.cfg[mode].batch_size,
            shuffle=False,
            num_workers=self.cfg[mode].num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return data_loader

    @overrides
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        return batch


def debug_by_visualization(idx, dataset):
    obj_dict = dataset[idx]
    bio_class_names = dataset.bio_class_names

    image = obj_dict['image'].permute(1, 2, 0).numpy().astype(np.uint8)
    h, w = image.shape[:2]
    blk_box = obj_dict['bbox']
    blk_box[:, 0::2] = blk_box[:, 0::2] / 1000.0 * w
    blk_box[:, 1::2] = blk_box[:, 1::2] / 1000.0 * h
    blk_box = blk_box.numpy().astype(np.int32)
    first_token_idxes = obj_dict['first_token_idxes'].tolist()
    bio_labels = obj_dict['bio_labels'].numpy()
    
    for blk_id, tok_idx in enumerate(first_token_idxes):
        if tok_idx == 0:
            break
        bbox = blk_box[tok_idx]
        category = bio_class_names[bio_labels[tok_idx]]
        cv2.rectangle(image, bbox[:2], bbox[2:], (205, 116, 24), 2)
        cv2.putText(image, category, tuple(bbox[:2] + np.array([1, 1])), \
            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.4, color=(0, 0, 255))
    cv2.imwrite('vis.jpg', image)
    input('Continue')
