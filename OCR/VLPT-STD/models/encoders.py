import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

from .resnet import resnet50


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


class FPN(nn.Module):
    def __init__(self,):
        super().__init__()

        nin = [256, 512, 1024, 2048]
        ndim = 256
        self.fpn_in = nn.ModuleList([conv1x1(nin[-1], ndim), conv1x1(nin[-2], ndim),
                    conv1x1(nin[-3], ndim), conv1x1(nin[-4], ndim)])

    def forward(self, feature_list):
        _, c2, c3, c4, c5 = feature_list
        
        p5 = self.fpn_in[0](c5)
        p5_up = F.interpolate(p5, scale_factor=2, mode='nearest')

        c4_ = self.fpn_in[1](c4)
        p4 = c4_ + p5_up
        p4_up =  F.interpolate(p4, scale_factor=2, mode='nearest')

        c3_ = self.fpn_in[2](c3)
        p3 = c3_ + p4_up
        p3_up =  F.interpolate(p3, scale_factor=2, mode='nearest')

        c2_ = self.fpn_in[3](c2)
        p2 = c2_ + p3_up

        tmp_size = c3.size()[2:]

        p2 = F.interpolate(p2, size=tmp_size, mode='bilinear')
        p4 = F.interpolate(p4, size=tmp_size, mode='bilinear')
        p5 = F.interpolate(p5, size=tmp_size, mode='bilinear')

        concat_feature = torch.cat((p2, p3, p4, p5), dim=1)

        return concat_feature
    

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, memory, mask=None):
        B, N1, C = query.shape
        B, N2, C = memory.shape

        q = self.q_proj(query).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(memory).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(memory).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, memory, mask=None):
        _x, attn = self.attn(self.norm1(x), self.norm1(memory), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Cross_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, memory, mask=None):
        _x, attn = self.self_attn(self.norm1(x), self.norm1(x), mask=mask)
        x = x + self.drop_path1(_x)

        _x, attn = self.cross_attn(self.norm2(x), self.norm2(memory))
        x = x + self.drop_path2(_x)
        x = x + self.drop_path3(self.mlp(self.norm3(x)))
        return x


class Image_Encoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        embed_dim = config["embed_dim"]
        img_size = config["image_size"]
        depth = config["image_transformer_depth"]
        drop_rate = config["drop_rate"]

        self.backbone = resnet50(pretrained=True)
        self.fpn = FPN()

        self.proj = nn.Sequential(nn.Conv2d(1024, embed_dim, kernel_size=1, stride=2))

        num_patches = (img_size // 16)**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=config["num_heads"],
                    mlp_ratio=config["mlp_ratio"],
                    qkv_bias=config["qkv_bias"],
                    drop=drop_rate,
                    attn_drop=config["attn_drop_rate"],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = self.backbone(x)
        x = self.fpn(x)
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        image_masks = torch.ones(x.size()[:2], device=x.device).long()
 
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0], inputs[1], inputs[2])
            return custom_forward

        for _, blk in enumerate(self.blocks):
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(blk), x, x, image_masks)
        
        x = self.norm(x)

        return x


class Text_Encoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["embed_dim"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["embed_dim"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        embed_dim = config["embed_dim"]
        drop_rate = config["drop_rate"]

        self.text_embeddings = BertEmbeddings(self.bert_config)

        depth = config["text_transformer_depth"]

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=config["num_heads"],
                    mlp_ratio=config["mlp_ratio"],
                    qkv_bias=config["qkv_bias"],
                    drop=drop_rate,
                    attn_drop=config["attn_drop_rate"],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, text_masks):

        x = self.text_embeddings(x)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0], inputs[1], inputs[2])
            return custom_forward

        for _, blk in enumerate(self.blocks):
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(blk), x, x, text_masks)
        
        x = self.norm(x)

        return x


class Co_Encoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        depth = config["co_transformer_depth"]
        embed_dim = config["embed_dim"]
        drop_rate = config["drop_rate"]

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.blocks = nn.ModuleList(
            [
                Cross_Block(
                    dim=embed_dim,
                    num_heads=config["num_heads"],
                    mlp_ratio=config["mlp_ratio"],
                    qkv_bias=config["qkv_bias"],
                    drop=drop_rate,
                    attn_drop=config["attn_drop_rate"],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y, masks=None):

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0], inputs[1], inputs[2])
            return custom_forward

        for _, blk in enumerate(self.blocks):
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(blk), x, y, masks)
        
        x = self.norm(x)

        return x