# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import numpy as np
import torch
import torch.utils.data
from overrides import overrides
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import BertTokenizer

from lightning_modules.bros_module import BROSModule
from utils import get_class_names, cfg_to_hparams, get_specific_pl_logger

TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

class GeoLayoutLMVIEModule(BROSModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        class_names = get_class_names(self.cfg.dataset_root_path)
        bio_class_names = ["O"]
        for class_name in class_names:
            if not class_name.startswith('O'):
                bio_class_names.extend([f"B-{class_name}", f"I-{class_name}"])
        self.eval_kwargs = {
            "bio_class_names": bio_class_names,
        }

        self.f1_res = {
            "f1_labeling": 0.0,
            "f1_linking": 0.0,
            "f1_all": 0.0,
        }
        self.max_f1 = {
            "f1_labeling": (-1, 0.0),
            "f1_linking": (-1, 0.0),
        }

    @overrides
    def training_step(self, batch, batch_idx, *args):
        _, loss_dict = self.net(batch)
        loss = loss_dict["total_loss"]

        log_dict_input = {
            "train_loss": loss, 
            "train_loss_labeling": loss_dict["labeling_loss"], 
            "train_loss_linking": loss_dict["linking_loss"]
        }
        self.log_dict(log_dict_input, sync_dist=False)
        ret_loss = {
            "loss": loss,
            "loss_labeling": loss_dict["labeling_loss"].detach(), 
            "loss_linking": loss_dict["linking_loss"].detach(),
        }
        return ret_loss

    @overrides
    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.tensor(0.0).to(self.device)
        avg_labeling_loss = torch.tensor(0.0).to(self.device)
        avg_linking_loss = torch.tensor(0.0).to(self.device)
        n_outputs = max(1, len(training_step_outputs))
        for step_out in training_step_outputs:
            avg_loss += step_out["loss"]
            avg_labeling_loss += step_out["loss_labeling"]
            avg_linking_loss += step_out["loss_linking"]

        log_dict = {
            "total_loss": avg_loss / n_outputs, 
            "labeling_loss": avg_labeling_loss / n_outputs,
            "linking_loss": avg_linking_loss / n_outputs
        }
        self._log_shell(log_dict, prefix="train ")

    @rank_zero_only
    @overrides
    def on_fit_end(self):
        self.print('-' * 20 + ' Best F1 scores in ' + self.cfg.save_weight_dir + '-' * 20)
        for k, v in self.max_f1.items():
            self.print(f'{k}: {v[1]:.2%} at epoch {v[0]}')
        
        hparam_dict = cfg_to_hparams(self.cfg, {})

        tb_logger = get_specific_pl_logger(self.logger, TensorBoardLogger)

        if tb_logger:
            tb_logger.log_hyperparams(hparam_dict, self.f1_res)

    @torch.no_grad()
    @overrides
    def validation_step(self, batch, batch_idx, *args):
        head_outputs, loss_dict = self.net(batch)
        step_out = do_eval_step(batch, head_outputs, loss_dict, self.eval_kwargs)
        return step_out

    @torch.no_grad()
    @overrides
    def validation_epoch_end(self, validation_step_outputs):
        scores = do_eval_epoch_end(validation_step_outputs)
        for task_name, score_task in scores.items():
            self.f1_res[f'f1_{task_name}'] = score_task['f1']
            suffix = ''
            if self.current_epoch > 0 and self.max_f1[f'f1_{task_name}'][1] < self.f1_res[f'f1_{task_name}']:
                self.max_f1[f'f1_{task_name}'] = (self.current_epoch, self.f1_res[f'f1_{task_name}'])
                suffix = ' (BEST BY NOW) '
            self.print(
                f"{task_name} --> precision: {score_task['precision']:.4f}, recall: {score_task['recall']:.4f}, f1: {score_task['f1']:.4f}{suffix}"
            )
        self.f1_res['f1_all'] = self.f1_res['f1_labeling'] + self.f1_res['f1_linking']
        self.log_dict(self.f1_res)


def do_eval_step(batch, head_outputs, loss, eval_kwargs, dump_dir=''):
    loss = loss["total_loss"]
    step_out_labeling = do_eval_step_ee(batch, head_outputs, loss, eval_kwargs, dump_dir)
    step_out_linking = do_eval_step_el(batch, head_outputs, loss, eval_kwargs, dump_dir)
    step_out = {"labeling": step_out_labeling, "linking": step_out_linking}
    return step_out


# for labeling
def do_eval_step_ee(batch, head_outputs, loss, eval_kwargs, dump_dir=''):
    bio_class_names = eval_kwargs["bio_class_names"]

    pr_labels = torch.argmax(head_outputs["logits4labeling"], -1)

    gt_str_list, pr_str_list = eval_ee_bio_batch(
        pr_labels,
        batch["bio_labels"],
        batch["are_box_first_tokens"],
        bio_class_names,
        dump_dir, batch,
        tokenizer=TOKENIZER,
    )

    step_out = {
        "loss": loss,
        "gt_str_list": gt_str_list,
        "pr_str_list": pr_str_list,
    }

    return step_out


def eval_ee_bio_batch(pr_labels, gt_labels, are_box_first_tokens, bio_class_names, 
        dump_dir='', batch=None, tokenizer=None):
    gt_str_list = []
    pr_str_list = []

    bsz = pr_labels.shape[0]
    for example_idx in range(bsz):
        gt_str_i = parse_str_from_seq(
            gt_labels[example_idx],
            are_box_first_tokens[example_idx],
            bio_class_names,
        )
        pr_str_i = parse_str_from_seq(
            pr_labels[example_idx],
            are_box_first_tokens[example_idx],
            bio_class_names,
        )

        gt_str_list.append(gt_str_i)
        pr_str_list.append(pr_str_i)

        # dump details
        if len(dump_dir) > 0:
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)
            assert batch is not None

            img_name = os.path.splitext(os.path.basename(batch["image_path"][example_idx]))[0]
            txt_fn = f'{img_name}_tagging.txt'
            f = open(os.path.join(dump_dir, txt_fn), 'w')
            f.writelines(batch["image_path"][example_idx] + '\n\n')

            box_first_token_mask = are_box_first_tokens[example_idx].cpu().tolist()
            num_valid_tokens = batch["attention_mask"][example_idx].sum().item()

            input_ids = batch["input_ids"][example_idx].cpu().tolist()

            width, height = batch["size_raw"][example_idx].cpu().tolist()
            block_boxes = batch["bbox"][example_idx].float()
            block_boxes[:, [0, 2]] = block_boxes[:, [0, 2]] / 1000 * width
            block_boxes[:, [1, 3]] = block_boxes[:, [1, 3]] / 1000 * height
            block_boxes = block_boxes.to(torch.long).cpu().tolist()

            for token_idx in range(num_valid_tokens):
                if box_first_token_mask[token_idx]:
                    valid_idx = sum(box_first_token_mask[:token_idx+1]) - 1
                    line = f"{token_idx}\t{gt_str_i[valid_idx]}\t{pr_str_i[valid_idx]}"
                    # add word info
                    ids = [input_ids[token_idx]]
                    tok_tmp_idx = token_idx + 1
                    while tok_tmp_idx < num_valid_tokens and not box_first_token_mask[tok_tmp_idx]:
                        ids.append(input_ids[tok_tmp_idx])
                        tok_tmp_idx += 1
                    word = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids))
                    line += f"\t{word}"
                    # add coord info
                    block_box = block_boxes[token_idx]
                    line += f"\t{','.join([str(coord) for coord in block_box])}\n"
                    f.writelines(line)
            f.close()

    return gt_str_list, pr_str_list


def parse_str_from_seq(seq, box_first_token_mask, bio_class_names):
    seq = seq[box_first_token_mask]
    res_str_list = []
    for i, label_id_tensor in enumerate(seq):
        label_id = label_id_tensor.item()
        if label_id < 0:
            raise ValueError("The label of words must not be negative!")
        res_str_list.append(bio_class_names[label_id])

    return res_str_list


# for linking, block level
def do_eval_step_el(batch, head_outputs, loss, eval_kwargs, dump_dir=''):
    bio_class_names = eval_kwargs["bio_class_names"]

    if head_outputs["max_prob_as_father"]:
        prob_linking = torch.sigmoid(head_outputs["logits4linking_list"][-1])
        head_outputs["pred4linking2"] = torch.where(
                prob_linking >= 0.5, \
                prob_linking,
                torch.zeros_like(head_outputs["logits4linking_list"][-1]))

        pr_el_labels = head_outputs["pred4linking2"]
    else:
        pr_el_labels = head_outputs["pred4linking"]

    n_batch_gt_rel, n_batch_pr_rel, n_batch_correct_rel = eval_el_geo_batch(
        pr_el_labels,
        batch["el_labels_blk"],
        batch["first_token_idxes"],
        batch["el_label_blk_mask"],
        batch["bio_labels"],
        bio_class_names,
        dump_dir, batch,
        max_prob_as_father=head_outputs["max_prob_as_father"],
        max_prob_as_father_upperbound=head_outputs["max_prob_as_father_upperbound"]
    )

    step_out = {
        "loss": loss,
        "n_batch_gt_rel": n_batch_gt_rel,
        "n_batch_pr_rel": n_batch_pr_rel,
        "n_batch_correct_rel": n_batch_correct_rel,
    }

    return step_out


def eval_el_geo_batch(
    pr_el_labels,
    gt_el_labels,
    first_token_idxes,
    el_label_blk_mask,
    bio_labels,
    bio_class_names,
    dump_dir='',
    batch=None,
    max_prob_as_father=False,
    max_prob_as_father_upperbound=False,
):
    n_batch_gt_rel, n_batch_pr_rel, n_batch_correct_rel = 0, 0, 0

    bsz = pr_el_labels.shape[0]
    for example_idx in range(bsz):
        n_gt_rel, n_pr_rel, n_correct_rel = eval_el_geo_example(
            pr_el_labels[example_idx],
            gt_el_labels[example_idx],
            first_token_idxes[example_idx],
            el_label_blk_mask[example_idx],
            bio_labels[example_idx],
            bio_class_names,
            dump_dir, batch, example_idx,
            max_prob_as_father=max_prob_as_father,
            max_prob_as_father_upperbound=max_prob_as_father_upperbound,
        )

        n_batch_gt_rel += n_gt_rel
        n_batch_pr_rel += n_pr_rel
        n_batch_correct_rel += n_correct_rel

    return n_batch_gt_rel, n_batch_pr_rel, n_batch_correct_rel


def eval_el_geo_example(
    pr_el_label,
    gt_el_label,
    first_token_idxes,
    el_label_blk_mask,
    bio_labels,
    bio_class_names,
    dump_dir='',
    batch=None,
    example_idx=None,
    max_prob_as_father=False,
    max_prob_as_father_upperbound=False,
):
    gt_relations, gt_s_memo, _ = parse_relations(
        gt_el_label, first_token_idxes, el_label_blk_mask, bio_labels, bio_class_names,
        max_prob_as_father=max_prob_as_father,
        max_prob_as_father_upperbound=max_prob_as_father_upperbound
    )
    pr_relations, pr_s_memo, flag = parse_relations2(
        pr_el_label, first_token_idxes, el_label_blk_mask, bio_labels, bio_class_names,
        max_prob_as_father=max_prob_as_father,
        max_prob_as_father_upperbound=max_prob_as_father_upperbound
    )

    if max_prob_as_father_upperbound and flag:

        for son, link_list in pr_s_memo.items():
            if len(link_list) == 1:
                continue

            for item in link_list:
                if item not in gt_relations:
                    pr_relations.remove(item)

    n_gt_rel = len(gt_relations)
    n_pr_rel = len(pr_relations)
    n_correct_rel = len(gt_relations & pr_relations)

    # dump details
    if len(dump_dir) > 0:
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir, exist_ok=True)
        assert batch is not None
        assert example_idx is not None
        img_name = os.path.splitext(os.path.basename(batch["image_path"][example_idx]))[0]
        txt_fn = f'{img_name}_linking.txt'
        gt_relations = sorted(list(gt_relations))
        pr_relations = sorted(list(pr_relations))
        with open(os.path.join(dump_dir, txt_fn), 'w') as f:
            f.writelines(batch["image_path"][example_idx] + '\n')
            f.writelines('\n')
            # record coordinates for each block (id)
            first_token_idxes = batch["first_token_idxes"][example_idx].cpu().tolist()
            block_mask = batch["block_mask"][example_idx].cpu().tolist()
            width, height = batch["size_raw"][example_idx].cpu().tolist()
            block_boxes = batch["bbox"][example_idx].float()
            block_boxes[:, [0, 2]] = block_boxes[:, [0, 2]] / 1000 * width
            block_boxes[:, [1, 3]] = block_boxes[:, [1, 3]] / 1000 * height
            block_boxes = block_boxes.to(torch.long).cpu().tolist()
            for blk_id, first_token_id in enumerate(first_token_idxes):
                if block_mask[blk_id] == 0:
                    break
                block_box = block_boxes[first_token_id]
                line = f"{blk_id}\t{','.join([str(coord) for coord in block_box])}\n"
                f.writelines(line)

            f.writelines('\n')
            # record relations (father,son)
            for rel in pr_relations:
                line = f"{rel[0]},{rel[1]}"
                if rel in gt_relations:
                    line += "\tRIGHT"
                else:
                    line += "\tERROR"
                f.writelines(line + '\n')
            for rel in gt_relations:
                if rel not in pr_relations:
                    line = f"{rel[0]},{rel[1]}\tMISS"
                    f.writelines(line + '\n')

    return n_gt_rel, n_pr_rel, n_correct_rel


def parse_relations(
    el_label,
    first_token_idxes,
    el_label_blk_mask,
    bio_labels,
    bio_class_names,
    max_prob_as_father=False,
    max_prob_as_father_upperbound=False,
):
    el_label = el_label * el_label_blk_mask
    blk_num = el_label.size(0)
    
    s_memo = {}
    prob_dict = {}
    flag = False
    link_map_tuples = []
    for son_id in range(blk_num):
        if el_label_blk_mask[son_id, 0] == 0:
            break
        for fthr_id in range(blk_num):
            if el_label_blk_mask[son_id, fthr_id] == 0:
                break
            if el_label[son_id, fthr_id] == 0:
                continue
            if bio_class_names[bio_labels[first_token_idxes[son_id]].item()] != "O" and \
                bio_class_names[bio_labels[first_token_idxes[fthr_id]].item()] != "O":

                link_map_tuples.append((fthr_id, son_id))
                
                if max_prob_as_father or max_prob_as_father_upperbound:
                    if son_id not in s_memo:
                        s_memo[son_id] = [(fthr_id, son_id)]
                        prob_dict[son_id] = {"prob":el_label[son_id, fthr_id], "item": (fthr_id, son_id)}
                    else:
                        flag = True
                        s_memo[son_id].append((fthr_id, son_id))

                        if not max_prob_as_father_upperbound:

                            if el_label[son_id, fthr_id] > prob_dict[son_id]["prob"]:
                                link_map_tuples.remove(prob_dict[son_id]["item"])
                                prob_dict[son_id]["prob"] = el_label[son_id, fthr_id]
                                prob_dict[son_id]["item"] = (fthr_id, son_id)
                            
                            elif el_label[son_id, fthr_id] == prob_dict[son_id]["prob"]:
                                pass

                            else:
                                link_map_tuples.remove((fthr_id, son_id))

    return set(link_map_tuples), s_memo, flag


def parse_relations2(
    el_label,
    first_token_idxes,
    el_label_blk_mask,
    bio_labels,
    bio_class_names,
    max_prob_as_father=False,
    max_prob_as_father_upperbound=False,
):
    el_label = el_label * el_label_blk_mask
    blk_num = el_label.size(0)

    link_map_tuples = []
    if max_prob_as_father:
        threshold = torch.topk(el_label, k=1)[0][:, -1]
        threshold = (threshold - 1e-3) * (threshold > 0.5).float() # 1e-3
    for son_id in range(blk_num):
        if el_label_blk_mask[son_id, 0] == 0:
            break
        for fthr_id in range(blk_num):
            if el_label_blk_mask[son_id, fthr_id] == 0:
                break
            if el_label[son_id, fthr_id] == 0:
                continue
            if bio_class_names[bio_labels[first_token_idxes[son_id]].item()] != "O" and \
                bio_class_names[bio_labels[first_token_idxes[fthr_id]].item()] != "O":

                if not max_prob_as_father or el_label[son_id, fthr_id] > threshold[son_id]:
                    link_map_tuples.append((fthr_id, son_id))

    return set(link_map_tuples), None, None


def do_eval_epoch_end(step_outputs):
    gt_str_list, pr_str_list = [], []
    n_total_gt_rel, n_total_pred_rel, n_total_correct_rel = 0, 0, 0

    for step_out in step_outputs:
        # labeling
        gt_str_list.extend(step_out["labeling"]["gt_str_list"])
        pr_str_list.extend(step_out["labeling"]["pr_str_list"])
        # linking
        n_total_gt_rel += step_out["linking"]["n_batch_gt_rel"]
        n_total_pred_rel += step_out["linking"]["n_batch_pr_rel"]
        n_total_correct_rel += step_out["linking"]["n_batch_correct_rel"]
    
    # labeling
    prec_lb = precision_score(gt_str_list, pr_str_list)
    reca_lb = recall_score(gt_str_list, pr_str_list)
    f1_lb = f1_score(gt_str_list, pr_str_list)

    # linking
    prec_lk = 0.0 if n_total_pred_rel == 0 else n_total_correct_rel / n_total_pred_rel
    reca_lk = 0.0 if n_total_gt_rel == 0 else n_total_correct_rel / n_total_gt_rel
    f1_lk = (
        0.0
        if reca_lk * prec_lk == 0
        else 2.0 * reca_lk * prec_lk / (reca_lk + prec_lk)
    )

    scores = {
        "labeling": {
            "precision": prec_lb,
            "recall": reca_lb,
            "f1": f1_lb,
        },
        "linking": {
            "precision": prec_lk,
            "recall": reca_lk,
            "f1": f1_lk,
        }
    }

    return scores
