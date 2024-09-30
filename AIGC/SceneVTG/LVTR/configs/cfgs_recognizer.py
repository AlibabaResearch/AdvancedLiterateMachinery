# coding:utf-8
import os

import torch
import torch.optim as optim
from torchvision import transforms

from dataset.lvtr_dataset import DatasetLVTR

global_cfgs = {
    "testmode": False,
    "out_dir": "./outputs/train_recognizer/",
    "logging_dir": "log",
    "total_epoch": 50,
    "seed": 123,
    "max_train_steps": 2000000,
    "ckpt_interval": 2000,
    "log_interval": 100,
    "max_grad_norm": 1.0,
    "init_state_dict": "./ckpts/recognizer.pth",  # change to your path
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
        "batch_size": 24,
        "shuffle": True,
        "num_workers": 16,
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

models_cfgs = {}

optimizer_cfgs = {
    "optimizer": optim.AdamW,
    "optimizer_args": {
        "lr": 0.001,
    },
    "optimizer_scheduler": optim.lr_scheduler.MultiStepLR,
    "optimizer_scheduler_args": {
        "milestones": [30, 40],
        "gamma": 0.1,
    },
}
