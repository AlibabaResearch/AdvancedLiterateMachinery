import codecs
import json
import math
import os
import pdb
import time

import cv2
import edit_distance
import editdistance as ed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.autograd import Variable


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return "%s" % (asMinutes(s))


def cal_distance(label_list, pre_list):
    y = edit_distance.SequenceMatcher(a=label_list, b=pre_list)
    yy = y.get_opcodes()
    insert = 0
    delete = 0
    replace = 0
    for item in yy:
        if item[0] == "insert":
            insert += item[-1] - item[-2]
        if item[0] == "delete":
            delete += item[2] - item[1]
        if item[0] == "replace":
            replace += item[-1] - item[-2]
    distance = insert + delete + replace
    return distance, (delete, replace, insert)


def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    print("loaded {}".format(model_path))
    state_dict_ = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        # if recognizerdict is not
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        k, model_state_dict[k].shape, state_dict[k].shape
                    )
                )
                state_dict[k] = model_state_dict[k]
        else:
            print("Drop parameter {}.".format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr
            print("Resumed optimizer with start lr", start_lr)
        else:
            print("No optimizer parameters in checkpoint.")
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


class cha_encdec:
    def __init__(self, dict_file, case_sensitive=True):
        self.case_sensitive = case_sensitive
        self.text_seq_len = 128
        self.dict = {}
        self.dict_len = 0
        for i, item in enumerate(open(dict_file, "r").readlines()):
            item = item.strip("\n").strip("\r")
            self.dict[item] = i
            self.dict_len += 1

    def encode(self, label_batch):
        max_len = max([len(s) for s in label_batch])
        out = torch.zeros(len(label_batch), max_len + 1).long()
        for i in range(0, len(label_batch)):
            if not self.case_sensitive:
                cur_encoded = (
                    torch.tensor(
                        [
                            self.dict[char.lower()]
                            if char.lower() in self.dict
                            else self.dict_len
                            for char in label_batch[i]
                        ]
                    )
                    + 1
                )
            else:
                cur_encoded = (
                    torch.tensor(
                        [
                            self.dict[char] if char in self.dict else self.dict_len
                            for char in label_batch[i]
                        ]
                    )
                    + 1
                )
            out[i][0 : len(cur_encoded)] = cur_encoded
        out = torch.cat(
            (
                out,
                torch.zeros((out.size(0), self.text_seq_len - out.size(1))).type_as(
                    out
                ),
            ),
            dim=1,
        )
        return out


class CTC_AR_counter:
    def __init__(self, dict_file):
        self.delete_error = 0.0
        self.replace_error = 0.0
        self.insert_error = 0.0
        self.character_total = 0
        self.word_error = 0
        self.total_word = 0
        self.alphabet = ""

        self.cnt = 0

        self.dt = {}
        for i, item in enumerate(open(dict_file, "r").readlines()):
            item = item.strip("\n").strip("\r")
            self.dt[i] = item
            self.alphabet += item
        self.dt[len(self.alphabet)] = "\\o"

    def clear(self):
        self.delete_error = 0.0
        self.replace_error = 0.0
        self.insert_error = 0.0
        self.character_total = 0
        self.word_error = 0
        self.total_word = 0

    def add_iter(self, output, labels, images=None):
        # B = output.size(1)
        B = output.size(0)
        for i in range(0, B):
            raw_pred = output[i, :, :-1].topk(1)[1].squeeze().tolist()
            if raw_pred[0] != 0:
                pred = [raw_pred[0]]
                pred.extend(
                    [
                        raw_pred[j]
                        for j in range(1, len(raw_pred))
                        if raw_pred[j] != raw_pred[j - 1]
                    ]
                )
            else:
                pred = [
                    raw_pred[j]
                    for j in range(1, len(raw_pred))
                    if raw_pred[j] != raw_pred[j - 1]
                ]
            pred = [_ - 1 for _ in pred if _ > 0]
            label_i = labels[i].tolist()
            label_i = [_ - 1 for _ in label_i if _ > 0]
            distance, (delete, replace, insert) = cal_distance(label_i, pred)
            label_chn = "".join(
                [self.dt[cha] for cha in label_i if cha <= len(self.alphabet)]
            )
            pred_chn = "".join(
                [self.dt[cha] for cha in pred if cha <= len(self.alphabet)]
            )

            if label_chn != pred_chn:
                self.word_error += 1
            self.total_word += 1

            self.character_total += len(label_i)
            self.delete_error += delete
            self.insert_error += insert
            self.replace_error += replace

    def check(self, output, labels):
        B = output.size(0)
        use_index = []
        nouse_index = []
        for i in range(0, B):
            raw_pred = output[i, :, :-1].topk(1)[1].squeeze().tolist()
            if raw_pred[0] != 0:
                pred = [raw_pred[0]]
                pred.extend(
                    [
                        raw_pred[j]
                        for j in range(1, len(raw_pred))
                        if raw_pred[j] != raw_pred[j - 1]
                    ]
                )
            else:
                pred = [
                    raw_pred[j]
                    for j in range(1, len(raw_pred))
                    if raw_pred[j] != raw_pred[j - 1]
                ]
            pred = [_ - 1 for _ in pred if _ > 0]
            label_i = labels[i].tolist()
            label_i = [_ - 1 for _ in label_i if _ > 0]
            distance, (delete, replace, insert) = cal_distance(label_i, pred)
            label_chn = "".join(
                [self.dt[cha] for cha in label_i if cha <= len(self.alphabet)]
            )
            pred_chn = "".join(
                [self.dt[cha] for cha in pred if cha <= len(self.alphabet)]
            )
            if label_chn == pred_chn:
                use_index.append(i)
            else:
                nouse_index.append(i)
        return use_index, nouse_index

    def show(self):
        CR = 1 - (self.delete_error + self.replace_error) / self.character_total
        AR = (
            1
            - (self.delete_error + self.replace_error + self.insert_error)
            / self.character_total
        )
        LA = 1 - float(self.word_error) / self.total_word
        res_str = (
            "CR: %4f, AR: %4f, LA: %4f, delete: %4d, replace: %4d, insert: %4d, gt_len: %4d, word_err: %4d, total_word: %4d"
            % (
                CR,
                AR,
                LA,
                self.delete_error,
                self.replace_error,
                self.insert_error,
                self.character_total,
                self.word_error,
                self.total_word,
            )
        )
        self.clear()
        return res_str

    def show_eval(self, ep=0, iter_id=0, clear=False, start_time=None):
        CR = 1 - (self.delete_error + self.replace_error) / self.character_total
        AR = (
            1
            - (self.delete_error + self.replace_error + self.insert_error)
            / self.character_total
        )
        LA = 1 - float(self.word_error) / self.total_word
        if start_time:
            print(
                "Test : %10s ep: %4d, iter_id: %6d, CR: %4f  AR: %4f  LA: %4f"
                % (timeSince(start_time), ep, iter_id, CR, AR, LA)
            )
        else:
            print(
                "ep: %4d, iter_id: %6d, CR: %4f  AR: %4f   LA: %4f"
                % (ep, iter_id, CR, AR, LA)
            )
        res_str = (
            "ep: %04d, iter_id: %6d, CR: %4f, AR: %4f, LA: %4f, delete: %4d, replace: %4d, insert: %4d, gt_len: %4d, word_err: %4d, total_word: %4d\n"
            % (
                ep,
                iter_id,
                CR,
                AR,
                LA,
                self.delete_error,
                self.replace_error,
                self.insert_error,
                self.character_total,
                self.word_error,
                self.total_word,
            )
        )
        return res_str


class Loss_counter:
    def __init__(self, display_interval=20):
        self.display_interval = display_interval
        self.total_iters = 0.0
        self.loss_sum = 0

    def add_iter(self, loss):
        self.total_iters += 1
        self.loss_sum += float(loss)

    def clear(self):
        self.total_iters = 0
        self.loss_sum = 0

    def get_loss(self):
        loss = self.loss_sum / self.total_iters if self.total_iters > 0 else 0
        self.total_iters = 0
        self.loss_sum = 0
        return loss


def save_args_to_yaml(args, output_file):
    # Convert args namespace to a dictionary

    # Write the dictionary to a YAML file
    with open(output_file, "w") as yaml_file:
        yaml.dump(args.global_cfgs, yaml_file, sort_keys=False)
        yaml.dump(args.accelerator_cfgs, yaml_file, sort_keys=False)
        yaml.dump(args.dataset_cfgs, yaml_file, sort_keys=False)
        yaml.dump(args.models_cfgs, yaml_file, sort_keys=False)
        yaml.dump(args.optimizer_cfgs, yaml_file, sort_keys=False)
