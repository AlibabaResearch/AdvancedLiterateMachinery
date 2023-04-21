# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys
import math
import logging
import re
import copy
import torch
import json

import torch.nn as nn
import torch.distributed as dist
from torchvision.ops import roi_align

from modeling_bros import BrosModel, BrosPreTrainedModel
from vision_model import VisionModel

from transformer_local import TransformerDecoderLayer, TransformerDecoder

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GeoLayoutLMModel(BrosPreTrainedModel):
    def __init__(self, config, hard_negtive_sampling=False):
        super().__init__(config)
        self.config = config
        self.hard_negtive_sampling = hard_negtive_sampling

        self.text_encoder = BrosModel(config)
        self.visual_encoder = VisionModel(img_size=self.config.image_size)
        self.pool = nn.AdaptiveAvgPool2d([1,1])
        self.vis_linear = nn.Linear(256, self.config.hidden_size)

        # !!!!!!!!!! Note !!!!!!!!!!
        # In PyTorch Transformer, the input size is [seq_len, batch_size, hidden_size]
        # In the attention_mask, 1 denotes masked
        cross_modal_text_layer = TransformerDecoderLayer(
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.intermediate_size, self_attn=True)
        self.cross_modal_text = TransformerDecoder(cross_modal_text_layer, 1)

        cross_modal_visual_layer = TransformerDecoderLayer(
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.intermediate_size, self_attn=True)
        self.cross_modal_visual = TransformerDecoder(cross_modal_visual_layer, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        image=None,
        bbox=None,
        bbox_4p_normalized=None,
        attention_mask=None,
        first_token_idxes=None,
        first_token_idxes_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):

        batch_size, seq_len = input_ids.shape

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        kwargs["line_bbox"] = bbox
        # ################ get text representation ################
        outputs = self.text_encoder(
            input_ids,
            bbox=bbox_4p_normalized,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        # sequence_output: [batch_size, seq_len, hidden_size]
        # pooled_output: [batch_size, hidden_size]
        sequence_output, pooled_output = outputs[:2]

        # ################ get visual representation ################
        _, num_first = first_token_idxes.shape
        B_batch_dim = torch.arange(0, batch_size,
            device=input_ids.device).reshape(
            batch_size,1).expand(batch_size, num_first)

        feature_bbox = bbox[B_batch_dim, first_token_idxes]
        _, block_num, _ = feature_bbox.shape

        visual_out = self.visual_encoder(image)
        batch_idxs = torch.arange(0, batch_size,
            device=sequence_output.device).reshape(batch_size,1).expand(batch_size, block_num).unsqueeze(-1)

        # [batch_size*block_num, 5]
        batch_idx_with_bbox = torch.cat((batch_idxs, feature_bbox), 2).reshape(
            batch_size*block_num, 5
        ).to(dtype=visual_out["feat_ms"].dtype)

        if visual_out["feat_ms"].dtype == torch.float16:
            # [batch_size*block_num, 256, 1, 1]
            blk_vis_features = roi_align(
                visual_out["feat_ms"].to(torch.float32), batch_idx_with_bbox.to(torch.float32), 1,
                spatial_scale=visual_out["feat_ms"].size(-1)/1000.0)
            blk_vis_features = blk_vis_features.to(dtype=visual_out["feat_ms"].dtype)
        else:
            blk_vis_features = roi_align(
                visual_out["feat_ms"], batch_idx_with_bbox.to(torch.float32), 1,
                spatial_scale=visual_out["feat_ms"].size(-1)/1000.0)

        # [batch_size*block_num, 256]
        blk_vis_features = blk_vis_features.squeeze(2).squeeze(2).reshape(batch_size, block_num, 256)

        # visual representation of text blocks - blk_vis_features: [batch_size, block_num, hidden_size]
        blk_vis_features = self.vis_linear(blk_vis_features)
        blk_vis_features = blk_vis_features * first_token_idxes_mask.unsqueeze(2)
        # Representation of full images [batch_size, 256]
        full_img_features = self.pool(visual_out["feat_ms"]).squeeze(2).squeeze(2)
        # [batch_size, hidden_size]
        full_img_features = self.vis_linear(full_img_features).unsqueeze(1)

        # ################ Vision-Text Fusion ################
        # cross attention inputs
        vis_inps = torch.cat((full_img_features, blk_vis_features), 1)

        glb_feat_attn = torch.ones((batch_size, 1)).to(input_ids.device)

        vis_mask = torch.cat((glb_feat_attn, first_token_idxes_mask), 1)

        new_attention_mask = (1 - attention_mask) > 0
        new_vis_mask = (1 - vis_mask) > 0

        text_mm_feat = self.cross_modal_text(
            tgt=sequence_output.transpose(0,1),
            memory=vis_inps.transpose(0,1),
            tgt_key_padding_mask=new_attention_mask,
            memory_key_padding_mask=new_vis_mask)

        vis_mm_feat = self.cross_modal_visual(
            tgt=vis_inps.transpose(0,1),
            memory=sequence_output.transpose(0,1),
            tgt_key_padding_mask=new_vis_mask,
            memory_key_padding_mask=new_attention_mask,
            
        )

        # [batch_size, seq_len, hidden_size]
        text_mm_feat = text_mm_feat.transpose(0,1)
        # [batch_size, 1+block_num, hidden_size]
        vis_mm_feat = vis_mm_feat.transpose(0,1)

        return sequence_output, blk_vis_features, text_mm_feat, vis_mm_feat



class MyBilinear(nn.Module):
    def __init__(self, in1, in2, out, bias=True):
        super(MyBilinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out, in1, in2))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out))
        else:
            self.bias = None
        torch.nn.init.xavier_normal_(self.weight, 0.1)
    
    def forward(self, x1, x2):
        '''
        input:
        x1: [b, T1, in1]
        x2: [b, T2, in2]
        output:
        y: [b, T1, T2, out]
        '''
        y = torch.einsum('bim,omn->bino', x1, self.weight) # [b, T1, in2, out]
        y = torch.einsum('bino,bjn->bijo', y, x2) # [b, T1, T2, out]
        if self.bias is not None:
            y = y + self.bias.contiguous().view(1, 1, 1, -1)
        return y


class PairGeometricHead(nn.Module):
    def __init__(self, config):
        super(PairGeometricHead, self).__init__()
        self.config = config
        self.bilinear = MyBilinear(self.config.hidden_size, self.config.hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states：[batch size, block_num, hidden]
        rel_blk = self.bilinear(hidden_states, hidden_states).squeeze(-1) # [batch_size, block_num, block_num]
        return rel_blk


class TripleGeometricHead(nn.Module):
    def __init__(self, config):
        super(TripleGeometricHead, self).__init__()
        self.config = config
        self.triple_cls = nn.Linear(self.config.hidden_size, self.config.triple_cls_num)

    @staticmethod
    def pretraining_loss(features, **kwargs):
        loss = 0.0
        if "geo_triplet_collinear_labels" in kwargs:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            triple_cls_num = features.shape[-1]
            triple_loss = loss_fct(
                features.view(-1, triple_cls_num),
                kwargs["geo_triplet_collinear_labels"].view(-1)
            )

            loss += triple_loss

        return loss

    def forward(self, hidden_states, triple_anchor_ids):
        # hidden_states：[batch size, block_num, hidden]
        # anchor ids [[1,2,3],[4,5,6],[9,10,11]]
        # label1 [0,2,3]
        batch_size, sample_triple_num, _ = triple_anchor_ids.shape
        B_batch_dim = torch.arange(0, batch_size,
            device=hidden_states.device).reshape(
            batch_size,1).expand(batch_size, sample_triple_num*3).reshape(
            batch_size, sample_triple_num, 3)
        
        # [batch_size, sample_triple_num, 3, hidden_size]
        triple_feat = hidden_states[B_batch_dim, triple_anchor_ids]
        # [batch_size, sample_triple_num, hidden_size]
        triple_feat = torch.sum(triple_feat, dim=2)

        # [batch_size, sample_triple_num, triple_cls_num]
        triple_cls_feat = self.triple_cls(triple_feat)

        return triple_cls_feat


class MultiPairsGeometricHead(nn.Module):
    def __init__(self, config):
        super(MultiPairsGeometricHead, self).__init__()
        self.config = config
        self.rel_layer = nn.Linear(self.config.hidden_size*2, self.config.hidden_size)

        enc_layer = nn.TransformerEncoderLayer(
            self.config.hidden_size, 2, self.config.hidden_size)  # 96
        self.trenc = nn.TransformerEncoder(enc_layer, 1)  # 2
        dec_layer = TransformerDecoderLayer(
            self.config.hidden_size, 2, self.config.hidden_size, self_attn=False)  # 96
        self.trdec = TransformerDecoder(dec_layer, 1) # 1
        self.fc = nn.Linear(self.config.hidden_size, 1)

        self.max_rel_enc = 96

    def forward(self, hidden_states, logits_prev, block_mask):
        """
        Args:
            hidden_states (TYPE): [batch size, block_num, hidden_size]
            logits_prev: [batch_size, block_num, block_num]
            block_mask: [batch_size, block_num]
        """
        batch_size, block_num, hidden_size = hidden_states.size()
        feat_pair = self.rel_layer(
            torch.cat([
                hidden_states.unsqueeze(2).expand(batch_size, block_num, block_num, hidden_size),
                hidden_states.unsqueeze(1).expand(batch_size, block_num, block_num, hidden_size),
            ], dim=3)
        ).contiguous().view(batch_size, block_num*block_num, -1)
        block_pair_mask = block_mask.unsqueeze(2).expand(batch_size, block_num, block_num) * \
            block_mask.unsqueeze(1).expand(batch_size, block_num, block_num)
        block_pair_mask = block_pair_mask.contiguous().view(batch_size, block_num*block_num)

        score_prev = torch.sigmoid(logits_prev)
        score_prev.detach_()

        negative_idx_pred = (score_prev < 0.5).float().view(batch_size, -1) # [b, block_num * block_num]
        negative_idx_pred = 1 - (1 - negative_idx_pred) * block_pair_mask # the padded blocks could not have links
        max_valid = min((1 - negative_idx_pred).sum(1).max().long().item(), self.max_rel_enc) # Restrict the number of relations to save memory usage
        max_valid = max(max_valid, 1)
        feat_pair_now = feat_pair * (1 - negative_idx_pred.unsqueeze(-1))
        idx_tmp = negative_idx_pred.argsort(1)
        idx_tmp = idx_tmp[:, :max_valid]
        enc_padding_mask = negative_idx_pred.gather(1, idx_tmp).bool()
        enc_padding_mask[:, 0] = False
        idx_tmp = idx_tmp.unsqueeze(-1).expand(batch_size, max_valid, feat_pair.size(2))
        positive_pairs = feat_pair_now.gather(1, idx_tmp)

        patterns = self.trenc(positive_pairs.transpose(0, 1), src_key_padding_mask=enc_padding_mask) # linking patterns
        # cmp out
        refined_pair = self.trdec(feat_pair.transpose(0, 1), patterns, memory_key_padding_mask=enc_padding_mask)
        refined_pair = refined_pair.transpose(0, 1).contiguous().view(batch_size, block_num, block_num, -1)
        ## fc
        logits_ref = self.fc(refined_pair).squeeze(-1)
        return logits_ref
