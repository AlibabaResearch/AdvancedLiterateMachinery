import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from .encoders import Image_Encoder, Text_Encoder, Co_Encoder
from .losses import compute_mlm, compute_global_local_contrast, compute_image_text_contrast

import time
import math
import numpy as np
import random


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class VLPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        embed_dim = config["embed_dim"]
        
        self.image_encoder = Image_Encoder(config)
        self.text_encoder = Text_Encoder(config)
        self.co_encoder = Co_Encoder(config)

        self.token_type_embeddings = nn.Embedding(2, embed_dim)
        self.mlm_score = MLMHead(self.text_encoder.bert_config)

        self.image_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.image_proj_wip = nn.Linear(embed_dim, embed_dim)
        self.text_proj_wip = nn.Linear(embed_dim, embed_dim)
        self.logit_scale_wip = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.max_logit_scale = math.log(100)
        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, batch):

        ret = {}
        text_ids = batch[f"text_ids"].cuda()
        text_ids_mlm = batch[f"text_ids_mlm"].cuda()
        text_labels_mlm = batch[f"text_labels_mlm"].cuda()
        text_masks = batch[f"text_masks"].cuda()
        
        text_embeds = self.text_encoder(text_ids_mlm, text_masks)

        images = batch['image'].cuda()
        image_embeds = self.image_encoder(images)
        image_masks = torch.ones(image_embeds.size()[:2], device=image_embeds.device).long()

        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks, device=image_embeds.device))
        image_embeds = image_embeds + self.token_type_embeddings(image_masks)

        # online find hard subword examples for wip
        trainable_dict = self.text_encoder.text_embeddings.word_embeddings.weight
        subword_embedding_dict = trainable_dict.detach() # N_dict, C
        subword_embedding_dict /= subword_embedding_dict.norm(dim=-1, keepdim=True)
        no_mlm_text_embeds = subword_embedding_dict[text_ids] # B, N1, C
        no_mlm_text_similarity = torch.matmul(no_mlm_text_embeds, subword_embedding_dict.t()) # B N1 N_dict
        similarity_indices = no_mlm_text_similarity.sort(descending=True).indices

        false_text_length = 64
        candidate_false_text_ids = similarity_indices[:, :, :false_text_length]

        valid_text_masks = text_masks.bool()
        for i, _len in enumerate(valid_text_masks.sum(dim=1)):
            valid_text_masks[i, _len - 1] = False
        valid_text_masks[:, 0] = False
        valid_text_masks = valid_text_masks & (text_labels_mlm == -100)

        candidate_false_text_embeds = trainable_dict[candidate_false_text_ids]

        candidate_false_text_feats_wip = self.text_proj_wip(candidate_false_text_embeds)
        image_feats_wip = self.image_proj_wip(image_embeds[:,1:2,:])
        self.logit_scale_wip.data = torch.clamp(self.logit_scale_wip.data, 0, self.max_logit_scale)
        
        image_feats_wip = image_feats_wip / image_feats_wip.norm(dim=-1, keepdim=True)
        candidate_false_text_feats_wip = candidate_false_text_feats_wip / candidate_false_text_feats_wip.norm(dim=-1, keepdim=True)
        logit_scale_wip = self.logit_scale_wip.exp()

        ret.update(compute_global_local_contrast(image_feats_wip, candidate_false_text_feats_wip, valid_text_masks, logit_scale_wip))

        # global contrast 
        text_feats = self.text_proj(text_embeds[:,0,:])
        image_feats = self.image_proj(image_embeds[:,0,:])
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, self.max_logit_scale)
        
        # normalized features
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        ret.update(compute_image_text_contrast(image_feats, text_feats, logit_scale))

        # mlm loss
        cross_text_feats = self.co_encoder(text_embeds, image_embeds, text_masks)
        mlm_logits = self.mlm_score(cross_text_feats)
        mlm_labels = text_labels_mlm
        ret.update(compute_mlm(mlm_logits, mlm_labels, self.config["vocab_size"]))

        return ret





