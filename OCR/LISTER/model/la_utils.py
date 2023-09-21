# Copyright (2023) Alibaba Group and its affiliates

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math


class LocalSelfAttention(nn.Module):
    def __init__(self, feat_dim, nhead, window_size:int, add_pos_bias=False, qkv_drop=0.0, proj_drop=0.0, mlm=False):
        super().__init__()
        assert feat_dim % nhead == 0
        self.q_fc = nn.Linear(feat_dim, feat_dim)
        self.kv_fc = nn.Linear(feat_dim, feat_dim * 2)

        self.nhead = nhead
        self.head_dim = feat_dim // nhead
        self.window_size = window_size
        if add_pos_bias:
            self.kv_pos_bias = nn.Parameter(torch.zeros(window_size, feat_dim))
            trunc_normal_(self.kv_pos_bias, std=.02)
        else:
            self.kv_pos_bias = None
        self.qkv_dropout = nn.Dropout(qkv_drop)

        self.proj = nn.Linear(feat_dim, feat_dim)
        self.proj_dropout = nn.Dropout(proj_drop)
        self.mlm = mlm
        if mlm:
            print("Use mlm.")
    
    def _gen_t_index(self, real_len, device):
        idx = torch.stack([
            torch.arange(real_len, dtype=torch.long, device=device) + st for st in range(self.window_size)]).t() # [T, w]
        return idx
    
    def _apply_attn_mask(self, attn_score):
        attn_score[:, :, :, :, self.window_size // 2] = float('-inf')
        return attn_score
    
    def forward(self, x, mask):
        """
        Args:
            x: [b, T, C]
            mask: [b, T]
        """
        b, T, C = x.size()
        # mask with 0
        x = x * mask.unsqueeze(-1)

        q = self.q_fc(self.qkv_dropout(x)) # [b, T, C]
        pad_l = pad_r = self.window_size // 2
        x_pad = F.pad(x, (0, 0, pad_l, pad_r)) # [b, T+w, C]
        # organize the window-based kv
        b_idx = torch.arange(b, dtype=torch.long, device=x.device).contiguous().view(b, 1, 1)
        t_idx = self._gen_t_index(T, x.device).unsqueeze(0)
        x_pad = x_pad[b_idx, t_idx] # [b, T, w, C]
        if self.kv_pos_bias is not None:
            x_pad = self.qkv_dropout(x_pad + self.kv_pos_bias.unsqueeze(0).unsqueeze(1))
        else:
            x_pad = self.qkv_dropout(x_pad)
        kv = self.kv_fc(x_pad) # [b, T, w, 2*C]
        k, v = kv.chunk(2, -1) # both are [b, T, w, C]
        # multi-head splitting
        q = q.contiguous().view(b, T, self.nhead, -1) # [b, T, h, C/h]
        k = k.contiguous().view(b, T, self.window_size, self.nhead, -1).transpose(2, 3) # [b, T, h, w, C/h]
        v = v.contiguous().view(b, T, self.window_size, self.nhead, -1).transpose(2, 3)
        # calculate attention scores
        # the scaling of qk refers to: https://kexue.fm/archives/8823
        alpha = q.unsqueeze(3).matmul(k.transpose(-1, -2) / self.head_dim * math.log(self.window_size)) # [b, T, h, 1, w]
        if self.mlm:
            alpha = self._apply_attn_mask(alpha)
        alpha = alpha.softmax(-1)
        output = alpha.matmul(v).squeeze(-2).contiguous().view(b, T, -1) # [b, T, C]
        output = self.proj_dropout(self.proj(output))
        output = output * mask.unsqueeze(-1)
        return output


class LocalAttentionBlock(nn.Module):
    def __init__(self, feat_dim, nhead, window_size, add_pos_bias:bool, drop=0.0, proj_drop=0.0, init_values=1e-6, mlm=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.sa = LocalSelfAttention(feat_dim, nhead, window_size, add_pos_bias, drop, proj_drop, mlm=mlm)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(feat_dim*4, feat_dim),
            nn.Dropout(drop),
        )
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(feat_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(feat_dim),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = 1.0, 1.0
    
    def forward(self, x, mask):
        x = x + self.gamma_1 * self.sa(self.norm1(x), mask)
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        x = x * mask.unsqueeze(-1)
        return x


class LocalAttentionModule(nn.Module):
    def __init__(self, feat_dim, nhead, window_size, num_layers, drop_rate=0.0, proj_drop_rate=0.0, detach_grad=False, mlm=False):
        super().__init__()
        self.attn_blocks = nn.ModuleList([
            LocalAttentionBlock(
                feat_dim, nhead, window_size,
                add_pos_bias=(i==0),
                drop=drop_rate,
                proj_drop=proj_drop_rate,
                mlm=mlm,
            ) for i in range(num_layers)])
        
        self.detach_grad = detach_grad
    
    def forward(self, x, mask):
        if self.detach_grad:
            x = x.detach()
        for blk in self.attn_blocks:
            x = blk(x, mask)
        return x


if __name__ == "__main__":
    x = torch.rand(2, 6, 512).cuda()
    mask = torch.zeros(2, 6).cuda()
    mask[:, :5] = 1.0
    blk = LocalAttentionModule(512, 8, 11, 2, 0.1, 0.1).cuda()
    y = blk(x, mask)
    print(y.size())
