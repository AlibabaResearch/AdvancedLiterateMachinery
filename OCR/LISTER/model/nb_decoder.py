# Copyright (2023) Alibaba Group and its affiliates
# --------------------------------------------------------
# To decode arbitrary-length text images.
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from typing import List, Dict
import math
# import pysnooper

from .la_utils import LocalAttentionModule
from .feature_extractor import FocalNetBlock


def softmax_m1(x:torch.Tensor, dim:int):
    # for x >= 0
    fx = x.exp() - 1
    fx = fx / fx.sum(dim, keepdim=True)
    return fx


class BilinearLayer(nn.Module):
    def __init__(self, in1, in2, out, bias=True):
        super(BilinearLayer, self).__init__()
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


class FeatureMapEnhancer(nn.Module):
    """ Merge the global and local features
    """
    def __init__(self, feat_dim, num_layers=1, focal_level=3, max_kh=1, layerscale_value=1e-6, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.merge_layer = nn.ModuleList([
            FocalNetBlock(
                dim=feat_dim,
                mlp_ratio=4,
                drop=drop_rate,
                focal_level=focal_level,
                max_kh=max_kh,
                focal_window=3,
                use_layerscale=True,
                layerscale_value=layerscale_value,
            ) for i in range(num_layers)])
        # self.scale = 1. / (feat_dim ** 0.5)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, feat_map, feat_char, char_attn_map, vis_mask, h=1):
        """
        feat_map: [b, N, C]
        feat_char: [b, T, C], T include the EOS token
        char_attn_map: [b, T, N], N exclude the EOS token
        vis_mask: [b, N]
        h: height of the feature map

        return: [b, C, h, w]
        """
        b, _, C = feat_map.size()
        # 1. restore the char feats into the visual map
        # char_feat_map = char_attn_map.transpose(1, 2).matmul(feat_char * self.scale) # [b, N, C]
        char_feat_map = char_attn_map.transpose(1, 2).matmul(feat_char) # [b, N, C]
        char_feat_map = self.norm1(char_feat_map)
        feat_map = feat_map + char_feat_map

        # 2. merge
        vis_mask = vis_mask.contiguous().view(b, h, -1) # [b, h, w]
        for blk in self.merge_layer:
            blk.H, blk.W = h, vis_mask.size(-1)
            feat_map = blk(feat_map, vis_mask)
        feat_map = self.dropout(self.norm2(feat_map))
        feat_map = feat_map.transpose(1, 2).contiguous().view(b, C, h, -1) # [b, C, h, w]
        feat_map = feat_map * vis_mask.unsqueeze(1)
        return feat_map


class NeighborDecoder(nn.Module):
    """ Find neighbors for each character
    In this version, each iteration shares the same decoder with the local vision decoder.
    """
    def __init__(self, num_classes, feat_dim, max_ch=1000,
                iters=0, nhead=8, window_size=11, num_sa_layers=2, num_mg_layers=2, **kwargs):
        super().__init__()
        self.eos_emb = nn.Parameter(torch.ones(feat_dim))
        # self.eos_gen = nn.Linear(feat_dim, feat_dim)
        self.q_fc = nn.Linear(feat_dim, feat_dim, bias=True)
        self.k_fc = nn.Linear(feat_dim, feat_dim)
        
        self.neighbor_navigator = BilinearLayer(feat_dim, feat_dim, 1)
        # self.tau = nn.Parameter(torch.tensor(feat_dim ** (-0.5))) # weaker

        self.vis_cls = nn.Linear(feat_dim, num_classes)

        self.p_threshold = 0.6
        self.max_ch = max_ch or 1000 # to avoid endless loop
        self.iters = iters
        if iters > 0:
            self.cntx_module = LocalAttentionModule(
                feat_dim, nhead, window_size, num_sa_layers, drop_rate=0.1, proj_drop_rate=0.1,
                detach_grad=kwargs['detach_grad'],
                mlm=kwargs.get('mlm', False))
            self.merge_layer = FeatureMapEnhancer(feat_dim, num_layers=num_mg_layers)

        self.detach_grad = kwargs['detach_grad']
        self.attn_scaling = kwargs['attn_scaling']
        self._init()
    
    def _init(self):
        trunc_normal_(self.eos_emb, std=.02)
    
    def align_chars(self, start_map, nb_map, max_ch=None):
        if self.training:
            assert max_ch is not None
        max_ch = max_ch or self.max_ch # required during training to be efficient
        b, N = nb_map.shape[:2]

        char_map = start_map # [b, N]
        all_finished = torch.zeros(b, dtype=torch.long, device=nb_map.device)
        char_maps = []
        char_masks = []
        for i in range(max_ch):
            char_maps.append(char_map)
            char_mask = (all_finished == 0).float()
            char_masks.append(char_mask)
            if i == max_ch - 1:
                break

            # loc_ch = char_map.argmax(1) # [b,]
            # all_finished = all_finished + (loc_ch == N - 1).long()
            all_finished = all_finished + (char_map[:, -1] > self.p_threshold).long()
            if not self.training:
                # check if end
                if (all_finished > 0).sum().item() == b:
                    break
            
            # loc_ch_expd = loc_ch.contiguous().view(-1, 1, 1).expand(-1, 1, N)
            # char_map = torch.gather(nb_map, 1, loc_ch_expd).squeeze(1)

            if self.training:
            # if True:
                char_map = char_map.unsqueeze(1).matmul(nb_map).squeeze(1)
            else:
                # char_map_dt = (char_map.detach() * 50).softmax(-1)
                k = min(1 + i * 2, 16)
                char_map_dt = softmax_m1(char_map.detach() * k, dim=-1)
                char_map = char_map_dt.unsqueeze(1).matmul(nb_map).squeeze(1)

        char_maps = torch.stack(char_maps, dim=1) # [b, L, N], L = n_char + 1
        char_masks = torch.stack(char_masks, dim=1) # [b, L], 0 denotes masked
        return char_maps, char_masks
    
    # @pysnooper.snoop()
    def decode_once(self, x:torch.FloatTensor, mask, mask_pad, max_char:int=None, h:int=1):
        b, c, h, w = x.size()
        x = x.flatten(2).transpose(1, 2) # [b, N, c], N = h x w
        mask = mask.unsqueeze(2) # [b, N, 1]
        g = (x * mask).sum(1) / (mask.sum(1) + 1e-10) # global representation, [b, c]

        # append eos emb to x
        # eos_emb = self.eos_gen(g)
        # x_ext = torch.cat([x, eos_emb.unsqueeze(1)], dim=1) # [b, N+1, c]
        x_ext = torch.cat([x, self.eos_emb.unsqueeze(0).expand(b, -1).unsqueeze(1)], dim=1) # [b, N+1, c]
        
        # locate the first character feature
        q_start = self.q_fc(g) # [b, c]
        k_feat = self.k_fc(x_ext) # [b, N+1, c]
        start_map = k_feat.matmul(q_start.unsqueeze(-1)).squeeze(-1) # [b, N+1]
        # scaling, referring to: https://kexue.fm/archives/8823
        # start_map = start_map / c * torch.log(1 + mask.sum(1))
        if self.attn_scaling:
            start_map = start_map / (c ** 0.5)
        # start_map = start_map * self.tau
        start_map.masked_fill_(mask_pad, float('-inf'))
        start_map = start_map.softmax(1)

        # Neighbor discovering
        q_feat = self.q_fc(x)
        nb_map = self.neighbor_navigator(q_feat, k_feat).squeeze(-1) # [b, N, N+1]
        # nb_map = nb_map / c * torch.log(1 + mask.sum(1, keepdim=True))
        if self.attn_scaling:
            nb_map = nb_map / (c ** 0.5)
        # nb_map = nb_map * self.tau
        nb_map.masked_fill_(mask_pad.unsqueeze(1).expand(-1, h*w, -1), float('-inf'))
        nb_map = nb_map.softmax(2)
        # nb_map= (nb_map * 3).softmax(2)
        last_neighbor = torch.zeros(h*w+1, device=x.device)
        last_neighbor[-1] = 1.0
        nb_map = torch.cat([
            nb_map,
            last_neighbor.contiguous().view(1, 1, -1).expand(b, -1, -1)
        ], dim=1) # to complete the neighbor matrix, (N+1) x (N+1)
        
        # string (feature) decoding
        char_maps, char_masks = self.align_chars(start_map, nb_map, max_char)
        char_feats = char_maps.matmul(x_ext) # [b, L, c]
        char_feats = char_feats * char_masks.unsqueeze(-1)
        logits = self.vis_cls(char_feats) # [b, L, nC]
        # import ipdb;ipdb.set_trace()

        results = dict(
            logits=logits,
            char_feats=char_feats,
            char_maps=char_maps,
            char_masks=char_masks,
            h=h,
            nb_map=nb_map,
        )
        return results

    def forward(self, x:torch.FloatTensor, mask:torch.FloatTensor=None, max_char:int=None, use_fem=True):
        """
        Args:
            x (torch.FloatTensor): of size (b, c, h, w)
            mask (torch.FloatTensor, optional): of size (b, h, w). Defaults to None.
        """
        b, c, h, w = x.size()
        if mask is None:
            mask = torch.ones_like(x[:, :, 0]) # [b, N]
        else:
            mask = mask.flatten(1) # [b, N]
        mask_pad = (1 - mask).round().bool() # inverted mask, [b, N]
        mask_pad = torch.cat([mask_pad, torch.zeros(b, 1, device=x.device).bool()], dim=1) # [b, N+1]

        # 0. decode the visual feature map
        res_vis = self.decode_once(x, mask, mask_pad, max_char, h)
        if not use_fem:
            return res_vis
        res_list = [res_vis]
        # 1. iterate
        if self.detach_grad:
            x = x.detach()
        for it in range(self.iters):
            char_feat_cntx = self.cntx_module(res_list[-1]['char_feats'], res_list[-1]['char_masks'])
            # import ipdb;ipdb.set_trace()
            char_maps = res_list[-1]['char_maps']
            if self.detach_grad:
                char_maps = char_maps.detach()
            feat_map = self.merge_layer(
                x.flatten(2).transpose(1, 2),
                char_feat_cntx,
                char_maps[:, :, :-1],
                mask,
                h=h,
            )
            res_i = self.decode_once(feat_map, mask, mask_pad, max_char, h)
            res_list.append(res_i)

        res_list = merge_dict_list(res_list)
        # if not self.training:
        #     if len(res_list['logits']) > 1:
        #         logits = torch.stack(res_list['logits'], dim=1) # [b, I, L, nC]
        #         char_masks = torch.stack(res_list['char_masks'], dim=1) # [b, I, L]
        #         probs = logits.softmax(-1).max(-1)[0] # [b, I, L]
        #         probs.masked_fill_((1 - char_masks).round().bool(), 1.0)
        #         probs = (probs + 1e-10).log().sum(-1).exp() # [b, I]
        #         _, idx = probs.max(1) # [b,]
        #         logit_best = logits[torch.arange(b, device=x.device), idx]
        #     else:
        #         logit_best = res_list['logits'][0]
        #     res_list['logit_best'] = logit_best
        return res_list


def merge_dict_list(x:List[Dict]):
    d_new = {}
    for d in x:
        for k, v in d.items():
            if k not in d_new.keys():
                d_new[k] = [v]
            else:
                d_new[k].append(v)
    return d_new


if __name__ == "__main__":
    device = torch.device("cuda:0")
    x = torch.rand(2, 768, 8, 32, device=device)
    mask = torch.ones(2, 8, 32, device=device)
    mask[0, :, 24:] = 0
    mask[1, :, 18:] = 0

    model = NeighborDecoder(37, 768, max_ch=25)
    model = model.to(device)

    y = model(x, mask, 17)
    for k, v in y.items():
        print(k, v.size())
