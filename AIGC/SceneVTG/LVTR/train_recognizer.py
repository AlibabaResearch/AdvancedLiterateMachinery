import datetime
import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import configs.cfgs_recognizer as cfgs
from modules.recognizer.recognizer import ExCTC
from utils import (CTC_AR_counter, Loss_counter, cha_encdec, load_model,
                   save_args_to_yaml)

torch.multiprocessing.set_sharing_strategy("file_system")

out_dir = cfgs.global_cfgs["out_dir"]
os.makedirs(out_dir, exist_ok=True)

logging_dir = f"{out_dir}/{cfgs.global_cfgs['logging_dir']}"

accelerator = Accelerator(
    kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    project_dir=logging_dir,
    **cfgs.accelerator_cfgs,
)

if cfgs.global_cfgs["testmode"]:
    log_file = f"{out_dir}/testing.log"
else:
    log_file = f"{out_dir}/training.log"

logging.basicConfig(filename=log_file, datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

# model preparing
model = ExCTC(cfgs.dataset_cfgs["dict_file"])

init_state_dict = cfgs.global_cfgs["init_state_dict"]
if init_state_dict is not None:
    load_model(model, init_state_dict)

# dataset preparing
dataset = cfgs.dataset_cfgs["dataset_train"](**cfgs.dataset_cfgs["dataset_train_args"])
train_loader = DataLoader(dataset, **cfgs.dataset_cfgs["dataloader_train"])
test_data_set = cfgs.dataset_cfgs["dataset_test"](
    **cfgs.dataset_cfgs["dataset_test_args"]
)
test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs["dataloader_test"])
char_encoder = cha_encdec(cfgs.dataset_cfgs["dict_file"])

# optimizer preparing
total_epoch = cfgs.global_cfgs["total_epoch"]
optimizer = cfgs.optimizer_cfgs["optimizer"](
    model.parameters(), **cfgs.optimizer_cfgs["optimizer_args"]
)

lr_scheduler = cfgs.optimizer_cfgs["optimizer_scheduler"](
    optimizer, **cfgs.optimizer_cfgs["optimizer_scheduler_args"]
)

model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

if accelerator.is_main_process:
    accelerator.init_trackers("trackers")
    save_args_to_yaml(
        args=cfgs,
        output_file=f"{out_dir}/config.yaml",
    )

loss_counter_CTC = Loss_counter()
loss_counter_WHACE = Loss_counter()
loss_counter_Concentrate = Loss_counter()
acc_counter = CTC_AR_counter(cfgs.dataset_cfgs["dict_file"])

epoch_iters = len(train_loader)
total_iters = epoch_iters * total_epoch

if accelerator.is_main_process:
    print("{}, total {} iters, begin training".format(out_dir, total_iters))

progress_bar = tqdm(
    range(total_iters),
    disable=not accelerator.is_local_main_process,
)
progress_bar.set_description("Steps")


def test(test_loader, model, ep=0, global_step=0):
    device = model.device
    model = model.module
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        val_parser = CTC_AR_counter(cfgs.dataset_cfgs["dict_file"])
        for it, (sample_batched) in enumerate(test_loader):
            inputs = sample_batched["crop_image_wobk"]
            labels = char_encoder.encode(sample_batched["crop_ocrstring"])
            inputs = inputs.to(device); labels = labels.to(device)
            output = model(inputs, labels)
            val_parser.add_iter(output["output"], labels, images=inputs)
        res_str = val_parser.show_eval(ep, it, True, start_time)
        logging.info(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Epoch {ep} global_step {global_step} => Accs: {res_str}"
        )


if cfgs.global_cfgs["testmode"] and accelerator.is_main_process:
    test(test_loader, model, 0)
    exit()

global_step = 0
for epoch_idx in range(1, total_epoch):
    train_loss = 0.0
    for batch_idx, sample_batch in enumerate(train_loader):
        model.train()
        images = sample_batch["crop_image_wobk"]
        labels = char_encoder.encode(sample_batch["crop_ocrstring"])
        images_render = sample_batch["crop_render"]

        # augments
        B, C, H, W = images_render.shape
        probmask = (
            torch.zeros((images.shape[0],), device=images.device).float().uniform_(0, 1)
            < 0.05
        )
        probmask = probmask.view(B, 1, 1, 1).expand(B, C, H, W).int()
        inputs = probmask * images_render + (1 - probmask) * images

        with accelerator.accumulate(model):
            output = model(inputs, labels)
            loss = output["loss"]

            avg_loss = accelerator.gather(loss).mean()
            train_loss += avg_loss.item() / cfgs.accelerator_cfgs["gradient_accumulation_steps"]
            
            # Backpropagate
            accelerator.backward(loss)

            acc_counter.add_iter(output["output"], labels, inputs)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), cfgs.global_cfgs["max_grad_norm"], norm_type=2
                )

            optimizer.step()
            optimizer.zero_grad()                

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"train_loss": train_loss}, step=global_step)
            train_loss = 0.0
            
            if accelerator.is_main_process:
                if global_step % cfgs.global_cfgs["ckpt_interval"] == 0:
                    test(test_loader, model, epoch_idx, global_step)
                    save_dir = f"{out_dir}/global_step_{global_step}"
                    os.makedirs(save_dir, exist_ok=True)
                    if accelerator.num_processes > 1:
                        model4save = model.module
                    else:
                        model4save = model
                    torch.save(model4save.state_dict(), f"{save_dir}/total_model.pth")
                    logging.info(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Save the checkpoint on global step {global_step}"
                    )
                    print("Save the checkpoint on global step {}".format(global_step))            

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        if accelerator.is_main_process and global_step % cfgs.global_cfgs["log_interval"] == 0:
            logging.info(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Global Step {global_step} => train_loss = {loss}, Accs: {acc_counter.show()}, lr: {lr_scheduler.get_last_lr()[0]}"
            )
        progress_bar.set_postfix(**logs)

        if global_step >= cfgs.global_cfgs["max_train_steps"]:
            break

    lr_scheduler.step()

accelerator.end_training()
