# Copyright (2023) Alibaba Group and its affiliates

import os
import yaml
import time
import shutil
import torch
import torchvision
import numpy as np
import cv2
from timm.models import create_model
from scipy import interpolate
import Levenshtein
import math
from omegaconf import OmegaConf
import re
from typing import List
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import matplotlib.pyplot as plt

from dataset.dataset import get_data
from model.model import LISTER, PATModel, CTCModel, RNNAttnSTR


def get_configs(is_training:bool):
    # cmd first
    config_dict = OmegaConf.from_cli()
    cli_keys = list(config_dict.keys())
    for cli_k in cli_keys:
        if cli_k[0] == '-':
            config_dict[re.sub(r'^-+', '', cli_k)] = config_dict[cli_k]
            del config_dict[cli_k]
    if 'c' in config_dict:
        cfg_fp = config_dict['c']
        del config_dict['c']
    elif 'config' in config_dict:
        cfg_fp = config_dict['config']
        del config_dict['config']
    config_dict_from_yml = OmegaConf.load(cfg_fp)
    config_dict = OmegaConf.merge(config_dict_from_yml, config_dict)

    # workers
    # num_gpus = torch.cuda.device_count()
    # config_dict['workers'] = config_dict['workers'] * num_gpus
    # ctc usage
    config_dict['use_ctc'] = (config_dict['model_type'].lower() == 'ctc')
    # ckpt_path
    config_dict['ckpt_path'] = os.path.join(config_dict['ckpt_path'], config_dict['model_name'])
    if is_training:
        os.makedirs(config_dict['ckpt_path'], exist_ok=True)
        # tfboard_path
        tfboard_path = os.path.join(config_dict['tb_dir'], config_dict['model_name'])
        # if os.path.exists(tfboard_path):
        #     # decision = input(f'{tfboard_path} has existed. Please choose the next operation (d to delete | c for another creation | other to reuse)\n')
        #     decision = 'd'
        #     if decision == 'd':
        #         shutil.rmtree(tfboard_path)
        #     elif decision == 'c':
        #         tfboard_path += '_' + time.asctime().replace(' ', '_')
        config_dict['tfboard_path'] = tfboard_path
    else:
        res_dir = os.path.dirname(config_dict['obj_fn'])
        if len(res_dir) > 0:
            os.makedirs(res_dir, exist_ok=True)
        config_dict['saved_model'] = os.path.join(config_dict['ckpt_path'], config_dict['ckpt_name'][1])
        if config_dict['test_speed']:
            config_dict['batch_size'] = 1

    # continue training
    if config_dict['continue']:
        config_dict['pretrained_path'] = os.path.join(config_dict['ckpt_path'], config_dict['ckpt_name'][0])
        if os.path.exists(config_dict['pretrained_path']):
            config_dict['finetune'] = False
        else:
            config_dict['continue'] = False
            config_dict['pretrained_path'] = None

    # dump fn
    config_dict['obj_fn'] = '_'.join([config_dict['obj_fn'], config_dict['model_name']]) + '.txt'

    # used for exchange root path from DLC to DSW
    if not os.path.exists('/ccx_dataset'):
        for ds_split in ['train_data_path', 'test_data_path']:
            for i in range(len(config_dict[ds_split])):
                config_dict[ds_split][i] = config_dict[ds_split][i].replace('ccx_dataset', 'mnt/workspace/dataset')
                config_dict[ds_split][i] = config_dict[ds_split][i].replace('nlp_ocr_v100', 'workspace/workgroup')

    print('-' * 80)
    for k, v in config_dict.items():
        print(f"{k}: {v}", end=", ")
    print()
    print('-' * 80)
    return config_dict


def default_data(config_dict:dict, data_path, training=False, shuffle=False, distributed=False):
    dataset, dataloader = get_data(
        data_path,
        config_dict['img_h'],
        config_dict['img_w_max'],
        config_dict['max_len'],
        config_dict['batch_size'],
        config_dict['do_resize'],
        data_aug=training,
        use_ctc=config_dict['use_ctc'],
        char94=config_dict.get('char94', False),
        workers=config_dict['workers'],
        shuffle=shuffle,
        is_train=training,
        distributed=distributed,
    )
    return dataset, dataloader

                 
def get_model(config_dict, device, training=False, **kwargs):
    if config_dict['model_type'] == 'ctc':
        net = CTCModel(
            num_classes=config_dict['num_classes'],
            blank_id=kwargs['blank_id'],
            drop_path_rate=config_dict['drop_path_rate'],
            layer_scale_init_value=config_dict['layer_scale_init_value'],
            h_fm=config_dict['h_fm'],
            enc=config_dict['enc'],
            timer=config_dict['test_speed'],
        )
    elif config_dict['model_type'] == 'lister':
        net = LISTER(
            num_classes=config_dict['num_classes'],
            max_ch=config_dict['max_len'],
            drop_path_rate=config_dict['drop_path_rate'],
            layer_scale_init_value=config_dict['layer_scale_init_value'],
            iters=config_dict['iters'],
            nhead=config_dict['nhead'],
            window_size=config_dict['window_size'],
            num_sa_layers=config_dict['num_sa_layers'],
            num_mg_layers=config_dict['num_mg_layers'],
            detach_grad=config_dict['detach_grad'],
            h_fm=config_dict['h_fm'],
            enc=config_dict['enc'],
            sa4enc=config_dict['sa4enc'],
            enc_version=config_dict['enc_version'],
            mlm=config_dict.get('mlm', False),
            timer=config_dict['test_speed'],
            attn_scaling=(config_dict['enc_version']=='base'),
        )
    elif config_dict['model_type'] == 'pat':
        net = PATModel(
            num_classes=config_dict['num_classes'],
            max_ch=config_dict['max_len'],
            drop_path_rate=config_dict['drop_path_rate'],
            layer_scale_init_value=config_dict['layer_scale_init_value'],
            h_fm=config_dict['h_fm'],
            enc=config_dict['enc'],
            timer=config_dict['test_speed'],
        )
    elif config_dict['model_type'] == 'rnn':
        net = RNNAttnSTR(
            num_classes=config_dict['num_classes'],
            max_ch=config_dict['max_len'],
            drop_path_rate=config_dict['drop_path_rate'],
            layer_scale_init_value=config_dict['layer_scale_init_value'],
            h_fm=config_dict['h_fm'],
            enc=config_dict['enc'],
            timer=config_dict['test_speed'],
        )
    else:
        raise TypeError
    net = net.to(device)
    # net = torch.nn.DataParallel(net)
    if training:
        net.train()
    else:
        net.eval()
    return net


class SaveAndLoad(object):
    def __init__(self, config_dict, model):
        self.configs = config_dict
        self.model = model
        self.best_acc = 0.0

    def save(self, epoch, iteration, acc):
        save_obj = {
                'state_dict': self.model.state_dict(),
                'epoch': epoch,
                'iteration': iteration,
                'best_acc': max(acc, self.best_acc)
        }
        torch.save(save_obj, 
            os.path.join(self.configs['ckpt_path'], self.configs['ckpt_name'][0]))
        if acc >= self.best_acc:
            self.best_acc = acc
            torch.save(save_obj,
                os.path.join(self.configs['ckpt_path'], self.configs['ckpt_name'][1]))
        print("Best Acc: {:.2%} in {}".format(self.best_acc, self.configs['ckpt_path']))
    
    def load(self, ckpt_fp, local_rank=0):
        print('loading pretrained model from ' + ckpt_fp)
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ckpt = torch.load(ckpt_fp, map_location=map_location)
        state_dict = ckpt['state_dict']
        if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.rename_state_keys(state_dict)
        print(
            f"Preloaded info\t{ckpt['epoch']}-{ckpt['iteration']}: best_acc: {ckpt['best_acc']:.2%}"
        )
        if self.configs['finetune']:
            self.model.load_state_dict(state_dict, strict=False)
            start_epoch, start_iter = 0, 0
        else:
            self.model.load_state_dict(state_dict)
            start_epoch = ckpt['epoch']
            start_iter = ckpt['iteration']
            self.best_acc = ckpt['best_acc']
        return start_epoch, start_iter
    
    @staticmethod
    def rename_state_keys(state_dict):
        new_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = re.sub(r'^module\.', '', k)
            new_dict[k] = v
        return new_dict


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """
    def __init__(self, character, blank_id=0):
        self.character = character
        self.blank_id = blank_id
        self.dict = dict(zip(character, range(len(character))))

    def decode(self, text_index, remove=True):
        """ convert text-index into text-label. """
        texts = []
        batch_size, T = text_index.size()
        for k in range(batch_size):
            t = text_index[k].tolist()
            char_list = []
            for i in range(T):
                if remove:
                    if t[i] != self.blank_id and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                        char_list.append(self.character[t[i]])
                else:
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)
            texts.append(text)
        return texts


class AttnSequenceDecoder(object):
    """ Convert between text-label and text-index.
    text is organised as like: "hello[EOS]"
    """
    def __init__(self, character, eos_token='[EOS]'):
        self.character = character
        self.char2id = dict(zip(character, range(len(character))))
        self.eos_id = self.char2id[eos_token]

    def decode(self, text_index) -> List[str]:
        """ convert text-index into text-label. 
        text_index: [b, T]
        """
        texts = []
        batch_size, T = text_index.size()
        for k in range(batch_size):
            t = text_index[k].tolist()
            char_list = []
            for i in range(T):
                if t[i] == self.eos_id:
                    break
                char_list.append(self.character[t[i]])
            text = ''.join(char_list)
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class AR_counter(object):
    def __init__(self, unsupported:str=None):
        self.total_samples = 0
        self.correct = 0
        self.norm_ed = 0.
        self.acc = 0.
        self.cr = 0.
        self.total_accumulated = 0
        self.correct_accumulated = 0

        self.unsupported = unsupported 

    def clear(self):
        self.total_samples = 0.
        self.correct = 0
        self.norm_ed = 0.
        self.acc = 0.
        self.cr = 0.
    
    def equal(self, s1, s2):
        if self.unsupported is None:
            s1 = s1.lower()
            s1 = re.sub(r'[^0-9a-z]', '', s1)
            s2 = s2.lower()
            s2 = re.sub(r'[^0-9a-z]', '', s2)
        else:
            s1 = re.sub(self.unsupported, '', s1)
            s2 = re.sub(self.unsupported, '', s2)
        return s1 == s2
        
    def add(self, prdt_texts, labels):
        self.total_accumulated += len(labels)
        for gt, pred in zip(labels, prdt_texts):
            # if we only care the words of lengths <= 25
            # to be fair when comparing with others
            if len(gt) > 25:
                self.total_accumulated -= 1
                continue
            #####
            self.total_samples += 1
            gt, pred = gt.lower(), pred.lower()
            if self.equal(gt, pred):
                self.correct += 1
                self.correct_accumulated += 1
            if len(gt) == 0 or len(pred) == 0:
                self.norm_ed += 0
            elif len(gt) > len(pred):
                self.norm_ed += 1 - Levenshtein.distance(pred, gt) / len(gt)
            else:
                self.norm_ed += 1 - Levenshtein.distance(pred, gt) / len(pred)
    
    def val(self):
        self.acc = self.correct / (self.total_samples + 1e-5)
        self.cr = self.norm_ed / (self.total_samples + 1e-5)
    
    def accuracy_by_now(self):
        return self.correct_accumulated / (self.total_accumulated + 1e-5)
    
    def show(self):
        self.val()
        split_line = '-' * 50
        print(split_line)
        print('Acc: {:.2%}, Character_Rate: {:.2%}'.format(self.acc, self.cr))
        # print(split_line)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=1e-7, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_steps if warmup_steps > 0 else warmup_epochs * niter_per_ep
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    cos_schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
    
    schedule = np.concatenate((warmup_schedule, cos_schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


class DynamicAlpha(object):
    def __init__(self, low, high, T1, T2):
        self.low = low
        self.step = high - low
        self.T1 = T1
        self.span = T2 - T1
    
    def __call__(self, t):
        alpha = self.step * ((t - self.T1) / self.span) ** 2 + self.low
        return alpha


def tensor2img(ts):
    img = torchvision.utils.make_grid(ts.cpu(), nrow=1)
    mu = torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1)
    img = (img * std + mu) * 255
    img = img.numpy().transpose(1, 2, 0)
    return img


def draw_attn_in_image(attn_map, img):
    attn_list = [img]
    for att in attn_map:
        att = att.cpu().numpy() * 255
        att = att.astype(np.uint8)
        att = cv2.resize(att, (img.shape[1], img.shape[0]))
        att = cv2.applyColorMap(att, 2)
        att = img * 0.5 + att * 0.5
        attn_list.append(att.astype(np.uint8))
    out_img = np.concatenate(attn_list, axis=0)
    return out_img


def draw_matrix(matrix:np.ndarray, save_fn:str):
    # Normalize by row
    matrix = matrix.astype(np.float)
    # plot
    # plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(1))
    # for i in range(matrix.shape[0]):
    #     ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    # ax.set_xticklabels([''] + classes, rotation=90)
    # ax.set_yticklabels([''] + classes)
    #save
    plt.savefig(save_fn)


def add_patch_box(img, patch_size=(16, 16), thickness=2):
    h, w = img.shape[:2]
    # horizontal line
    line_hor = np.ones((thickness, w), dtype=np.uint8) * 222
    for i in range(h // patch_size[0]):
        start = i * patch_size[0]
        end = start + thickness
        img[start:end, :, 0] = line_hor
        img[start:end, :, 1:] = 0
    # vertical line
    line_ver = np.ones((h, thickness), dtype=np.uint8) * 222
    for i in range(w // patch_size[1]):
        start = i * patch_size[1]
        end = start + thickness
        img[:, start:end, 0] = line_ver
        img[:, start:end, 1:] = 0
    return img


def count_params(model):
    cnt = 0
    for k, v in model.named_parameters():
        cnt += v.numel()
    print(f'Params: {cnt/1e6:.3f} M')
    return cnt
