import datetime
import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import configs.cfgs_lvtr as cfgs
from dalle2_pytorch import Decoder, Unet
from modules.recognizer.recognizer import ExCTC
from utils import Loss_counter, cha_encdec, load_model, save_args_to_yaml

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
recognizer = ExCTC(cfgs.dataset_cfgs["dict_file"])
recognizer_frame = ExCTC(cfgs.dataset_cfgs["dict_file"])
recognizer_ocr = ExCTC(cfgs.dataset_cfgs["dict_file"])

unet = Unet(**cfgs.models_cfgs["unet_cfgs"])

decoder = Decoder(
    unet=unet, recognizer=recognizer_frame, **cfgs.models_cfgs["decoder_cfgs"]
)

recognizer_init_state_dict = cfgs.global_cfgs["recognizer_init_state_dict"]
if recognizer_init_state_dict is not None:
    load_model(recognizer, recognizer_init_state_dict)
    load_model(recognizer_ocr, recognizer_init_state_dict)

decoder_init_state_dict = cfgs.global_cfgs["decoder_init_state_dict"]
if decoder_init_state_dict is not None:
    load_model(decoder, decoder_init_state_dict)

if cfgs.global_cfgs["load_from_recognizer"]:
    decoder.reloadrecognizer(recognizer=recognizer)

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
    decoder.parameters(), **cfgs.optimizer_cfgs["optimizer_args"]
)

lr_scheduler = cfgs.optimizer_cfgs["optimizer_scheduler"](optimizer, T_max=total_epoch)

decoder, optimizer, train_loader, recognizer_ocr = accelerator.prepare(
    decoder, optimizer, train_loader, recognizer_ocr
)

if accelerator.is_main_process:
    accelerator.init_trackers("trackers")
    save_args_to_yaml(
        args=cfgs,
        output_file=f"{out_dir}/config.yaml",
    )

loss_counter = Loss_counter()
loss_counter_ocr = Loss_counter()
loss_counter_mse = Loss_counter()

epoch_iters = len(train_loader)
total_iters = epoch_iters * total_epoch

if accelerator.is_main_process:
    print("{}, total {} iters, begin training".format(out_dir, total_iters))

progress_bar = tqdm(
    # range(cfgs.global_cfgs["max_train_steps"]),
    range(total_iters),
    disable=not accelerator.is_local_main_process,
)
progress_bar.set_description("Steps")


def test(test_loader, model, ep=0, batch_idx=0, fasttest=True, testcounts=2):
    device = model.device
    model = model.module
    model.eval()
    with torch.no_grad():
        for it, (sample_batch) in enumerate(test_loader):
            images = sample_batch["crop_image"]
            labels = sample_batch["crop_ocrstring"]
            images_removal = sample_batch["crop_image_removal"]
            images_render = sample_batch["crop_render"]
            mask_line = sample_batch["mask_line"]
            image_removal_fix = sample_batch["image_removal_fix"]
            crop_box = sample_batch["crop_box"]
            line_poly_mask = sample_batch["line_poly_mask"]
            word_poly_mask = sample_batch["word_poly_mask"]
            width_resize = sample_batch["width_resize"]
            image_full = sample_batch["image_full"]

            all_conditions = {}
            all_conditions["images_style"] = images.to(device)
            all_conditions["text"] = char_encoder.encode(labels).to(device)
            all_conditions["images_removal"] = images_removal.to(device)
            all_conditions["images_render"] = images_render.to(device)
            all_conditions["line_poly_mask"] = line_poly_mask.to(device)
            all_conditions["word_poly_mask"] = word_poly_mask.to(device)

            rec_images = model.sample(
                drop_prob={
                    "textstring": 0.0,
                    "renderimage": 0.0,
                },
                cond_scale=7.0,
                all_conditions=all_conditions,
            )

            images_render = F.interpolate(
                images_render, size=(64, 512), mode="bilinear"
            )
            line_poly_mask = line_poly_mask.repeat(1, 3, 1, 1)
            word_poly_mask = word_poly_mask.repeat(1, 3, 1, 1)

            col_items = 1
            draw_h = 64
            draw_w = 512
            row_items = images.size(0) // col_items
            canvas = np.zeros(((row_items) * draw_h * 6, col_items * draw_w, 3)).astype(
                np.uint8
            )
            images_np = (images.permute(0, 2, 3, 1).cpu().data.numpy() * 255).astype(
                np.uint8
            )
            images_removal_np = (
                images_removal.permute(0, 2, 3, 1).cpu().data.numpy() * 255
            ).astype(np.uint8)
            images_render_np = (
                images_render.permute(0, 2, 3, 1).cpu().data.numpy() * 255
            ).astype(np.uint8)
            rec_images_np = (
                rec_images.permute(0, 2, 3, 1).cpu().data.numpy() * 255
            ).astype(np.uint8)
            line_poly_mask_np = (
                line_poly_mask.permute(0, 2, 3, 1).cpu().data.numpy() * 255
            ).astype(np.uint8)
            word_poly_mask_np = (
                word_poly_mask.permute(0, 2, 3, 1).cpu().data.numpy() * 255
            ).astype(np.uint8)

            for i in range(0, rec_images.size(0)):
                row_idx = i * 6
                col_idx = 0
                canvas[
                    row_idx * draw_h : (row_idx + 1) * draw_h,
                    col_idx * draw_w : (col_idx + 1) * draw_w,
                    :,
                ] = rec_images_np[i]
                canvas[
                    (row_idx + 1) * draw_h : (row_idx + 2) * draw_h,
                    col_idx * draw_w : (col_idx + 1) * draw_w,
                    :,
                ] = images_np[i]
                canvas[
                    (row_idx + 2) * draw_h : (row_idx + 3) * draw_h,
                    col_idx * draw_w : (col_idx + 1) * draw_w,
                    :,
                ] = images_render_np[i]
                canvas[
                    (row_idx + 3) * draw_h : (row_idx + 4) * draw_h,
                    col_idx * draw_w : (col_idx + 1) * draw_w,
                    :,
                ] = images_removal_np[i]
                canvas[
                    (row_idx + 4) * draw_h : (row_idx + 5) * draw_h,
                    col_idx * draw_w : (col_idx + 1) * draw_w,
                    :,
                ] = line_poly_mask_np[i]
                canvas[
                    (row_idx + 5) * draw_h : (row_idx + 6) * draw_h,
                    col_idx * draw_w : (col_idx + 1) * draw_w,
                    :,
                ] = word_poly_mask_np[i]

            vis_path = out_dir + "/vis_reg"
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)

            savepath = vis_path + "/image_%d_%d_%d_%d.jpg" % (
                ep,
                batch_idx,
                it,
                device.index,
            )
            cv2.imwrite(savepath, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

            row_items = 1
            draw_h = 512
            draw_w = 512
            col_items = images.size(0) // row_items
            canvas = np.zeros(((row_items) * draw_h, col_items * draw_w, 3)).astype(
                np.uint8
            )
            image_removal_fix_np = (
                image_removal_fix.permute(0, 2, 3, 1).cpu().data.numpy() * 255
            ).astype(np.uint8)
            rec_images_np = (
                rec_images.permute(0, 2, 3, 1).cpu().data.numpy() * 255
            ).astype(np.uint8)
            for i in range(0, image_removal_fix.size(0)):
                row_idx = i // col_items
                col_idx = i % col_items
                image_cat = image_removal_fix_np[i]
                width_resize_i = int(width_resize[i])
                rec_images_i = rec_images[i : i + 1, :, :, :]
                image_cat[
                    crop_box[i][1] : crop_box[i][3], crop_box[i][0] : crop_box[i][2], :
                ] = (
                    F.interpolate(
                        rec_images_i,
                        size=(
                            crop_box[i][3] - crop_box[i][1],
                            crop_box[i][2] - crop_box[i][0],
                        ),
                        mode="bilinear",
                    )
                    .permute(0, 2, 3, 1)
                    .cpu()[0]
                    .data.numpy()
                    * 255
                ).astype(
                    np.uint8
                )
                canvas[
                    row_idx * draw_h : (row_idx + 1) * draw_h,
                    col_idx * draw_w : (col_idx + 1) * draw_w,
                    :,
                ] = image_cat

            vis_path = out_dir + "/vis_reg"
            savepath = vis_path + "/image_full_%d_%d_%d_%d.jpg" % (
                ep,
                batch_idx,
                it,
                device.index,
            )
            cv2.imwrite(savepath, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

            if fasttest and it == testcounts:
                break


if cfgs.global_cfgs["testmode"]:
    test(test_loader, decoder, 0, 0, fasttest=False)
    exit()


global_step = 0
for epoch_idx in range(1, total_epoch):
    train_loss = 0.0
    for batch_idx, sample_batch in enumerate(train_loader):
        decoder.train()
        images = sample_batch["crop_image"]
        labels = sample_batch["crop_ocrstring"]
        images_removal = sample_batch["crop_image_removal"]
        images_render = sample_batch["crop_render"]
        line_poly_mask = sample_batch["line_poly_mask"]
        word_poly_mask = sample_batch["word_poly_mask"]

        with accelerator.accumulate(decoder):
            all_conditions = {}
            all_conditions["images_style"] = images
            all_conditions["text"] = char_encoder.encode(labels)
            all_conditions["images_removal"] = images_removal
            all_conditions["images_render"] = images_render
            all_conditions["line_poly_mask"] = line_poly_mask
            all_conditions["word_poly_mask"] = word_poly_mask

            outputs = decoder(images, all_conditions=all_conditions)
            loss_dm = outputs["loss"]

            if cfgs.global_cfgs["backloss"] > 0:
                x0_denoise = torch.clamp(outputs['x0_denoise'], min=0, max=1)
                bk_mask = (1-line_poly_mask).repeat(1, images.shape[1], 1, 1)
                loss_background = cfgs.global_cfgs["backloss"] * torch.sum(bk_mask * F.mse_loss(x0_denoise, images, reduction="none")) / torch.sum(bk_mask)
            else:
                loss_background = torch.Tensor([0])
                loss_background = loss_background.to(loss_dm.device)

            if cfgs.global_cfgs["foreloss"] > 0:
                x0_denoise = torch.clamp(outputs['x0_denoise'], min=0, max=1)
                line_poly_mask = line_poly_mask.repeat(1, images.shape[1], 1, 1)
                x0_features = recognizer_ocr.module.get_image_features(line_poly_mask * x0_denoise)
                images_features = recognizer_ocr.module.get_image_features(line_poly_mask * images)
                loss_foreground = cfgs.global_cfgs["foreloss"] * F.mse_loss(x0_features, images_features, reduction="none").mean()
            else:
                loss_foreground = torch.Tensor([0])
                loss_foreground = loss_foreground.to(loss_dm.device)

            loss = loss_dm + loss_background + loss_foreground

            avg_loss = accelerator.gather(loss).mean()
            train_loss += (
                avg_loss.item() / cfgs.accelerator_cfgs["gradient_accumulation_steps"]
            )
            # Backpropagate
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    decoder.parameters(), cfgs.global_cfgs["max_grad_norm"], norm_type=2
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
                    test(test_loader, decoder, epoch_idx, batch_idx)
                    save_dir = f"{out_dir}/global_step_{global_step}"
                    os.makedirs(save_dir, exist_ok=True)
                    if accelerator.num_processes > 1:
                        model4save = decoder.module
                    else:
                        model4save = decoder
                    torch.save(model4save.state_dict(), f"{save_dir}/total_model.pth")
                    logging.info(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Save the checkpoint on global step {global_step}"
                    )
                    print("Save the checkpoint on global step {}".format(global_step))

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "dm_loss": loss_dm.detach().item(), "bk_loss": loss_background.detach().item(), "fore_loss": loss_foreground.detach().item()}
        if global_step % cfgs.global_cfgs["log_interval"] == 0:
            logging.info(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Global Step {global_step} => train_loss = {loss}; dm_loss = {loss_dm}; bk_loss = {loss_background}, fore_loss = {loss_foreground}, lr: {lr_scheduler.get_last_lr()[0]}"
            )
        progress_bar.set_postfix(**logs)
        # Quit
        if global_step >= cfgs.global_cfgs["max_train_steps"]:
            break
    lr_scheduler.step()

accelerator.end_training()
