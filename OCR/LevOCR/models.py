import torch.nn as nn

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.utils.data
import numpy as np

from levt import utils
from levt.levenshtein_transformer import LevenshteinTransformerModel
from levt.utils import new_arange

from abinet.model_vision import BaseVision


# for levocr loss
def _compute_levt_loss(
    outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
):
    """
    outputs: batch x len x d_model
    targets: batch x len
    masks:   batch x len

    policy_logprob: if there is some policy
        depends on the likelihood score as rewards.
    """

    def mean_ds(x: Tensor, dim=None) -> Tensor:
        return (
            x.float().mean().type_as(x)
            if dim is None
            else x.float().mean(dim).type_as(x)
        )

    if masks is not None:
        outputs, targets = outputs[masks], targets[masks]

    if masks is not None and not masks.any():
        nll_loss = torch.tensor(0)
        loss = nll_loss
    else:  
        logits = F.log_softmax(outputs, dim=-1)
        if targets.dim() == 1:  # true
            losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

        else:  # soft-labels
            losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
            losses = losses.sum(-1)

        nll_loss = mean_ds(losses)
        if label_smoothing > 0:   # true
            loss = (
                nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
            )
        else:
            loss = nll_loss

    loss = loss * factor
    return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

def inject_noise(opt, target_tokens, tgt_dict):
    def _random_delete(target_tokens):
        pad = tgt_dict.pad()
        bos = tgt_dict.bos()
        eos = tgt_dict.eos()

        max_len = target_tokens.size(1)
        target_mask = target_tokens.eq(pad)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(
            target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
        )
        target_score.masked_fill_(target_mask, 1)
        target_score, target_rank = target_score.sort(1)
        target_length = target_mask.size(1) - target_mask.float().sum(
            1, keepdim=True
        )

        # do not delete <bos> and <eos> (we assign 0 score for them)
        target_cutoff = (
            2
            + (
                (target_length - 2)
                * target_score.new_zeros(target_score.size(0), 1).uniform_()
            ).long()
        )
        target_cutoff = target_score.sort(1)[1] >= target_cutoff

        prev_target_tokens = (
            target_tokens.gather(1, target_rank)
            .masked_fill_(target_cutoff, pad)
            .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
        )
#         prev_target_tokens = prev_target_tokens[
#             :, : prev_target_tokens.ne(pad).sum(1).max()
#         ]

        return prev_target_tokens

    def _random_mask(target_tokens):
        pad = tgt_dict.pad()
        bos = tgt_dict.bos()
        eos = tgt_dict.eos()
        unk = tgt_dict.unk()

        target_masks = (
            target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
        )
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token.

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        prev_target_tokens = target_tokens.masked_fill(
            target_cutoff.scatter(1, target_rank, target_cutoff), unk
        )
        return prev_target_tokens

    def _full_mask(target_tokens):
        pad = tgt_dict.pad()
        bos = tgt_dict.bos()
        eos = tgt_dict.eos()
        unk = tgt_dict.unk()

        target_mask = (
            target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
        )
        return target_tokens.masked_fill(~target_mask, unk)

    if opt.noise == "random_delete":
        return _random_delete(target_tokens)
    elif opt.noise == "random_mask":
        return _random_mask(target_tokens)
    elif opt.noise == "full_mask":
        return _full_mask(target_tokens)
    elif opt.noise == "no_noise":
        return target_tokens
    else:
        raise NotImplementedError

class LevOCRModel(nn.Module):

    def __init__(self, opt, src_dict):
        super(LevOCRModel, self).__init__()
        self.opt = opt
        self.vision = BaseVision(opt)
        """ Levt """
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(512)  

        self.levt = LevenshteinTransformerModel.build_model(opt, src_dict)
    
    def extract_img_feature(self, img_feature):
        x = self.conv4_1(img_feature)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        return x    

    def forward(self, image, labels_add_del, tgt_vision, tgt_tokens, criterion):
        if image != None:
            vision_out = self.vision(image)
            pred_logit = vision_out['logits']
            features = vision_out['features']
            preds_vision = pred_logit.log_softmax(2)
            loss_vison = criterion(preds_vision.view(-1, preds_vision.shape[-1]), tgt_vision.contiguous().view(-1))
            img_feature = self.extract_img_feature(features)
        else:
            img_feature = None
            preds_vision = None
            loss_vison = 0.0

        preds = self.levt(labels_add_del, img_feature, tgt_tokens)

        losses, nll_loss = [], []
        outputs = preds
        for obj in outputs:
            _losses = _compute_levt_loss(
                outputs[obj].get("out"),
                outputs[obj].get("tgt"),
                outputs[obj].get("mask", None),
                outputs[obj].get("ls", 0.0),
                name=obj + "-loss",
                factor=outputs[obj].get("factor", 1.0),
            )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss_levt = sum(l["loss"] for l in losses) 
        nll_loss = sum(l for l in nll_loss)

        logging_output = {
            "loss": loss_levt.data,
            "nll_loss": nll_loss.data,
        }

        reduce=True
        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return [loss_levt, loss_vison, preds_vision, preds, logging_output]

