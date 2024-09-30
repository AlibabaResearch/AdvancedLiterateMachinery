# coding:utf-8
import os

import torch
import torch.optim as optim
from torchvision import transforms

from dataset.lvtr_dataset import DatasetLVTR
global_cfgs = {
    "testmode": False,
    "out_dir": "outputs/train_lvtr/",
    "logging_dir": "log",
    "total_epoch": 100,
    "seed": 123,
    "max_train_steps": 5000000,
    "ckpt_interval": 2000,
    "log_interval": 100,
    "experience_name": "basecode",
    "load_from_recognizer": False,
    "max_grad_norm": 1.0,
    "backloss": 0.,
    "foreloss": 0.,
    "recognizer_init_state_dict": "./ckpts/recognizer.pth",  # change to your path
    "decoder_init_state_dict": "./ckpts/lvtr.pth",  # change to your path
}

accelerator_cfgs = {
    "gradient_accumulation_steps": 1,
    "mixed_precision": "no",  # choices=["no", "fp16", "bf16"],
    "log_with": "tensorboard",
}

dataset_cfgs = {
    "dict_file": "dataset/chardict.txt",
    "dataset_train": DatasetLVTR,
    "dataset_train_args": {
        "roots": [
            "./SceneVTG-Erase/LVTR_data/scripts/LVTR_trainlist.txt",  # change to your path
        ],
        "img_height": 64,
        "img_width": 512,
        "render_height": 64,
        "global_state": "Train",
        "dataset_root": './SceneVTG-Erase/LVTR_data/data',  # change to your path
    },
    "dataloader_train": {
        "batch_size": 12,
        "shuffle": True,
        "num_workers": 8,
    },
    "dataset_test": DatasetLVTR,
    "dataset_test_args": {
        "roots": [
            "./SceneVTG-Erase/LVTR_data/scripts/LVTR_testlist.txt",  # change to your path
        ],
        "img_height": 64,
        "img_width": 512,
        "render_height": 64,
        "global_state": "Test",
        "dataset_root": './SceneVTG-Erase/LVTR_data/data',  # change to your path
    },
    "dataloader_test": {
        "batch_size": 4,
        "shuffle": False,
        "num_workers": 4,
    },
}

models_cfgs = {
    "unet_cfgs": {
        "dim": 128,
        "image_embed_dim": 512,
        "text_embed_dim": 512,
        "cond_dim": 128,
        "channels": 3,
        "dim_mults": (1, 2, 2, 4, 4, 8),
        "cond_on_text_encodings": True,
        "max_text_len": 128,
        "init_with_removal": True,
        "init_with_linepolymask": True,
        "init_with_wordpolymask": True,
    },
    "decoder_cfgs": {
        "image_size": (64, 512),
        "image_pos_len": 128,
        "text_pos_len": 128,
        "channels": 3,
        "timesteps": 100,
        "drop_prob": {
            "renderimage": 0.5,
            "textstring": 0.1,
        },
        "learned_variance_constrain_frac": True,
    },
}

optimizer_cfgs = {
    "optimizer": optim.AdamW,
    "optimizer_args": {
        "lr": 0.0001,
        "weight_decay": 0.01,
        "betas": (0.9, 0.99),
    },
    "optimizer_scheduler": optim.lr_scheduler.CosineAnnealingLR,
}
