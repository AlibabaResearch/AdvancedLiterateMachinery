# Copyright (2023) Alibaba Group and its affiliates

import os
import sys
import time
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

from utils import *
from test import evaluate

device = torch.device('cuda')


def main(config_dict):
    # basic preparation
    writer = SummaryWriter(config_dict['tfboard_path'])
    # data preparation
    train_dataset, train_dataloader = default_data(config_dict, config_dict['train_data_path'], training=True, shuffle=True)
    _, test_dataloader = default_data(config_dict, config_dict['test_data_path'], training=False, shuffle=True)
    # string-id encoder & decoder
    if config_dict['use_ctc']:
        blank_id=train_dataset.char2id[train_dataset.blank_token]
        converter = CTCLabelConverter(train_dataset.charlist,
            blank_id=blank_id)
    else:
        blank_id = None
        converter = AttnSequenceDecoder(train_dataset.charlist, eos_token=train_dataset.eos_token)
    num_classes = len(converter.character)
    config_dict['num_classes'] = num_classes
    # model preparation
    net = get_model(config_dict, device, training=True, blank_id=blank_id)
    save_load_tool = SaveAndLoad(config_dict, net)
    # loss averager
    loss_avg = Averager()
    # setup optimizer
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config_dict['lr'],
        weight_decay=config_dict['weight_decay'],
        eps=config_dict['opt_eps'],
    )
    print(optimizer)
    print("-" * 80)
    # lr_scheduler
    num_training_steps_per_epoch = len(train_dataloader)
    lr_schedule_values = cosine_scheduler(
        config_dict['lr'], config_dict['min_lr'], config_dict['num_epochs'], num_training_steps_per_epoch,
        warmup_epochs=config_dict['warmup_epochs'], warmup_steps=config_dict['warmup_steps'],
    )
    # Initial setting
    start_epoch = config_dict['start_epoch']
    start_iter = config_dict['start_iter']
    # Load checkpoint
    if config_dict['pretrained_path'] is not None:
        start_epoch, start_iter = save_load_tool.load(config_dict['pretrained_path'])
    """ start training """
    start_time = time.time()
    # start_iter = start_epoch * num_training_steps_per_epoch
    i = start_iter
    # pdb.set_trace()
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, config_dict['num_epochs']):
        for it, (images, img_masks, labels, lengths, words_raw) in enumerate(train_dataloader):
            # update lr
            lr = lr_schedule_values[min(i, len(lr_schedule_values) - 1)]
            for param_group in iter(optimizer.param_groups):
                param_group['lr'] = lr

            images, img_masks, labels, lengths = \
                images.to(device), img_masks.to(device), labels.to(device), lengths.to(device)
            # forward pass
            if config_dict['model_type'] in ['ctc']:
                ret_dict = net(images, img_masks, labels=labels, label_lens=lengths)
                cost = ret_dict['loss']
            elif config_dict['model_type'] in ['lister', 'pat', 'rnn']:
                ret_dict = net(images, img_masks, max_char=labels.size(1), labels=labels, label_lens=lengths)
                cost = ret_dict['loss']
            else:
                raise TypeError
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config_dict['grad_clip'])
            optimizer.step()
            # pdb.set_trace()

            loss_avg.add(cost)

            # validation part
            if i % config_dict['show_interval'] == 0:
                elapsed_time = (time.time() - start_time) / 3600
                writer.add_scalar('lr/train', lr, i)
                # loss_log = '-' * 80 + '\n'
                loss_log = ''
                loss_log += f'[i{i} | e{epoch}-{it}/{len(train_dataloader)}] '
                loss_log += f'Total avg loss: {loss_avg.val():0.4f}, lr: {lr:.1e}'
                if len(ret_dict) > 1:
                    for loss_k, loss_v in ret_dict.items():
                        if loss_k == 'loss' or loss_k.startswith('l_'):
                            loss_log += f', {loss_k}: {loss_v.item():.3f}'
                loss_log += f', Elapsed_time: {elapsed_time:0.1f}h'
                print(loss_log, flush=True)
                writer.add_scalar('Loss/train', loss_avg.val(), i)
                loss_avg.reset()
            if i % config_dict['test_interval'] == 0:
                net.eval()
                acc, pred_str, gt_str, test_imgs, *_ = evaluate(
                    test_dataloader, net, converter, model_type=config_dict['model_type'])
                net.train()
                writer.add_scalar('Accuracy/val', acc, i)
                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s}{"Prediction":25s}T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}'
                for gt, pred in zip(gt_str[-config_dict['num_to_show']:], pred_str[-config_dict['num_to_show']:]):
                    gt, pred = gt.lower(), pred.lower()
                    predicted_result_log += f'\n{gt:25s}{pred:25s}{str(pred == gt)}\n{dashed_line}'
                print(predicted_result_log, flush=True)
                # show related imgs
                test_imgs = torchvision.utils.make_grid(
                    test_imgs[-config_dict['num_to_show']:].cpu(), nrow=1)
                test_imgs = (test_imgs * 0.5 + 0.5) * 255
                cv2.imwrite("./data/val_view.jpg", test_imgs.numpy().transpose(1, 2, 0))
                # save
                save_load_tool.save(epoch, i, acc)
                # 
                if (i + 1) >= config_dict['num_iters']:
                    print(f'Finished at iter {i} - epoch {epoch}')
                    sys.exit()
            i += 1
    writer.close()


if __name__ == "__main__":
    config_dict = get_configs(is_training=True)
    """ Seed and GPU setting """
    random.seed(config_dict['seed'])
    np.random.seed(config_dict['seed'])
    torch.manual_seed(config_dict['seed'])
    torch.cuda.manual_seed(config_dict['seed'])
    torch.cuda.manual_seed_all(config_dict['seed'])

    cudnn.benchmark = False
    cudnn.deterministic = True

    os.makedirs('data', exist_ok=True)
    main(config_dict)
    print('End the training!')
