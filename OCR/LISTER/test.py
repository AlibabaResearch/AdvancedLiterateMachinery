# Copyright (2023) Alibaba Group and its affiliates

import os
import cv2
import numpy as np
import yaml
import string
import time
from tabulate import tabulate
from tqdm import tqdm
from collections.abc import Iterable
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn

from utils import *


device = torch.device('cuda')


def evaluate(dataloader, net, label_cvter, model_type='vit', accountant=None, need_debug=False, test_speed=False, ret_probs=False):
    if accountant is None:
        accountant = AR_counter()
    # cnt_dict = {}
    preds_str_all, words_all, probs_all = [], [], []
    if test_speed:
        tt_len = defaultdict(float)
        te_len = defaultdict(float)
        td_len = defaultdict(float)
        ti_len = defaultdict(float)
        cnt_len = defaultdict(int)
    time0 = time.time()
    for images, img_masks, labels, lengths, words in tqdm(dataloader):
        images = images.to(device)
        img_masks = img_masks.to(device)
        with torch.no_grad():
            if model_type == 'ctc':
                ret_dict = net(images, img_masks)
            else:
                # To avoid that: stop steps earlier or later
                ret_dict = net(images, img_masks, max_char=labels.size(1))
            if test_speed:
                torch.cuda.synchronize()
                ret_dict, t_dict = ret_dict
                tt_len[len(words[0])] += t_dict['t_total']
                te_len[len(words[0])] += t_dict['t_enc']
                td_len[len(words[0])] += t_dict['t_dec']
                cnt_len[len(words[0])] += 1
        if 'logit_best' in ret_dict.keys():
            output = ret_dict['logit_best']
        else:
            output = ret_dict['logits']
        if isinstance(output, list):
            output = output[-1]
        if 'tgt_img' in ret_dict.keys():
            tgt_imgs = ret_dict['tgt_img']
        else:
            tgt_imgs = None
        ti0 = time.time()
        if ret_probs:
            # calc prob
            probs = output.softmax(-1).max(-1)[0] # [b, L]
            char_masks = ret_dict['char_masks']
            if isinstance(char_masks, list): char_masks = char_masks[-1]
            probs.masked_fill_((1 - char_masks).round().bool(), 1.0)
            probs = (probs + 1e-10).log().sum(-1).exp() # [b]
            probs = probs.cpu().tolist()
            probs_all.extend(probs)

        preds_prob, preds_index = output.max(2)
        preds_str = label_cvter.decode(preds_index.data)
        if test_speed:
            ti_len[len(words[0])] += (time.time() - ti0)
            continue
        accountant.add(preds_str, words)

        preds_str_all.extend(preds_str)
        words_all.extend(words)

        if need_debug:
            attn_map = ret_dict['char_maps'] if 'char_maps' in ret_dict.keys() else None
            if isinstance(attn_map, list):
                attn_map = attn_map[-1]
            h = ret_dict['h'] if 'h' in ret_dict.keys() else None
            if isinstance(h, list):
                h = h[-1]
            nb_map = ret_dict['nb_map'] if 'nb_map' in ret_dict.keys() else None
            if isinstance(nb_map, list):
                nb_map = nb_map[-1]
            debug(preds_str, words, images, tgt_imgs, attn_map, h, nb_map)
    if test_speed:
        time1 = time.time()
        speed = (time1 - time0) / len(dataloader.dataset)
        print(f"Inference Speed: {speed*1000:.2f} ms")
        t_head = ['t_type'] + [i for i in range(2, 26)]
        t_vals = [
            ['t_enc'] + [te_len[i] / (cnt_len[i] + 1e-5) * 1000 for i in range(2, 26)],
            ['t_dec'] + [td_len[i] / (cnt_len[i] + 1e-5) * 1000 for i in range(2, 26)],
            ['t_model'] + [tt_len[i] / (cnt_len[i] + 1e-5) * 1000 for i in range(2, 26)],
            ['t_inf'] + [ti_len[i] / (cnt_len[i] + 1e-5) * 1000 for i in range(2, 26)],
        ]
        md_info = tabulate(t_vals, headers=t_head, tablefmt='pipe', floatfmt='.2f')
        print(md_info)
        
        acc = 0.0
    else:
        accountant.val()
        acc = accountant.acc
        accountant.show()
        accountant.clear()

    return acc, preds_str_all, words_all, images, probs_all


def debug(preds_str, words, images, tgt_imgs=None, attn_map=None, h=1, transfer_mat=None):
    batch_size = len(words)
    if attn_map is not None:
        b, L, N = attn_map.size()
        attn_map = attn_map[:, :-1, :-1].view(b, L-1, h, -1)
        # attn_map = attn_map[:, :-1, :].view(b, L-1, h, -1)
    for i in range(batch_size):
        if preds_str[i] != words[i]:
        # if words[i] in [
        #     "wwwtopstockresearchcom",
        #     "wwwloudbillboardscom",
        # ]:
        # if len(words[i]) > 25:
            src_img = images[i:i+1]
            src_img = tensor2img(src_img)
            if tgt_imgs is None:
                img = src_img
            else:
                tgt_img = tgt_imgs[i:i+1]
                tgt_img = tensor2img(tgt_img)
                tgt_img = add_patch_box(tgt_img)
                src_img = cv2.resize(src_img, (tgt_img.shape[1], src_img.shape[0]))
                img = np.concatenate((src_img, tgt_img), axis=0)
            cv2.imwrite("./data/test_view.jpg", img)
            print(f"gt: {words[i]}\tpred: {preds_str[i]}\t{str(words[i]==preds_str[i])}")
            print("length:", len(words[i]))
            if attn_map is not None:
                img_att = draw_attn_in_image(attn_map[i], src_img)
                cv2.imwrite("./data/test_attn_view.jpg", img_att)
            if transfer_mat is not None:
                transfer_mat_i = transfer_mat[i].cpu().numpy()
                draw_matrix(transfer_mat_i, "./data/test_mat_view.jpg")

            input()


def main(config_dict):
    # data 
    dataloaders = []
    if config_dict['test_speed']:
        config_dict['test_data_path'] = config_dict['test_data_path'][:1]
    for data_path in config_dict['test_data_path']:
        test_dataset, test_dataloader = default_data(config_dict, data_path, training=False, shuffle=False)
        dataloaders.append(test_dataloader)
    if config_dict['use_ctc']:
        blank_id = test_dataset.char2id[test_dataset.blank_token]
        converter = CTCLabelConverter(test_dataset.charlist,
            blank_id=blank_id)
    else:
        converter = AttnSequenceDecoder(test_dataset.charlist, eos_token=test_dataset.eos_token)
        blank_id = None
    num_classes = len(converter.character)
    config_dict['num_classes'] = num_classes
    accountant = AR_counter()
    # model preparation
    net = get_model(config_dict, device, training=False, blank_id=blank_id)
    save_load_tool = SaveAndLoad(config_dict, net)
    if not config_dict['test_speed']:
        # load trained parameters
        save_load_tool.load(config_dict['saved_model'])
    count_params(net)

    if not config_dict['test_speed']:
        f_obj = open(config_dict['obj_fn'], 'w')
    table_headers = ['Test set']
    acc_list = ['Acc.']
    for j, dataloader in enumerate(dataloaders):
        print('-' * 80)
        print(config_dict['test_data_path'][j])
        acc, preds_str_all, words_all, images, probs = evaluate(
            dataloader, net, converter, model_type=config_dict['model_type'], accountant=accountant,
            need_debug=config_dict['debug'], test_speed=config_dict['test_speed'], ret_probs=config_dict['ret_probs'])
        if not config_dict['test_speed']:
            for i in range(len(words_all)):
                if len(probs) == 0:
                    line_new = f"{i}\t{words_all[i]}\t{preds_str_all[i]}\n"
                else:
                    line_new = f"{i}\t{words_all[i]}\t{preds_str_all[i]}\t{probs[i]:.3f}\n"
                f_obj.writelines(line_new)
            # record info
            table_headers.append(os.path.basename(config_dict['test_data_path'][j].rstrip('/')))
            acc_list.append(f'{acc * 100:.2f}')
    if not config_dict['test_speed']:
        f_obj.close()
        if isinstance(config_dict['test_data_path'], Iterable) and len(config_dict['test_data_path']) > 0:
            # calculate the average accuracy
            acc_avg = accountant.accuracy_by_now()
            acc_list.append(f'{acc_avg * 100:.2f}')
            table_headers.append('avg.')

        md_info = tabulate([acc_list + ['-']*3], headers=table_headers+['Speed', 'Params', 'FLOPs'], tablefmt='pipe', floatfmt='.1f')
        print()
        print(md_info)
        with open(config_dict['obj_fn'].replace('.txt', '.res.txt'), 'w') as f:
            f.writelines(md_info)


if __name__ == "__main__":
    config_dict = get_configs(is_training=False)
    """GPU setting """
    main(config_dict)
