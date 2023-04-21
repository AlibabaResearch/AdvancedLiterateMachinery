# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import trunc_normal_,DropPath
import random

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from convnext import convnext_tiny


class VisionModel(nn.Module):
    def __init__(self, trans_layers=2, inner_channels=256, img_size=(896, 896)):
        super(VisionModel, self).__init__()
        self.cnn = convnext_tiny(pretrained=False, in_22k=True)
        self.dims = self.cnn.dims
        self.img_size = img_size

        img_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).contiguous().view(1, 3, 1, 1)
        img_std = torch.tensor(IMAGENET_DEFAULT_STD).contiguous().view(1, 3, 1, 1)
        self.register_buffer('img_mean', img_mean)
        self.register_buffer('img_std', img_std)

        # FPN refers to DB [Liao et al.]
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(self.dims[-1], inner_channels, 1, bias=False)
        self.in4 = nn.Conv2d(self.dims[-2], inner_channels, 1, bias=False)
        self.in3 = nn.Conv2d(self.dims[-3], inner_channels, 1, bias=False)
        self.in2 = nn.Conv2d(self.dims[-4], inner_channels, 1, bias=False)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=False)
        
        self.pos_emb1 = nn.Parameter(
            torch.randn(inner_channels, self.img_size[0]//32, self.img_size[1]//32)
        )
        trunc_normal_(self.pos_emb1, std=.02)
        self.aggr = nn.Conv2d(inner_channels, inner_channels, 3, padding=1)
        self.dropout_pos = nn.Dropout(0.1)

        self.drop_path = DropPath(0.1)
    
    def forward(self, x):
        x = x / 255.0
        x = (x - self.img_mean) / self.img_std
        ms_features = self.cnn(x)
        # import pdb;pdb.set_trace()
        c2, c3, c4, c5 = ms_features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)
        N, D5, H5, W5 = in5.size()
        
        in5_pos = self.dropout_pos(in5 + self.pos_emb1.unsqueeze(0))
        in5 = self.dropout_pos(self.aggr(in5_pos))

        # FPN for fused multi-scale visual feature
        out4 = self.up5(in5) + self.drop_path(in4)
        out3 = self.up4(out4) + self.drop_path(in3)
        out2 = self.up3(out3) + self.drop_path(in2)
        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        feat_ms = torch.cat((p5, p4, p3, p2), 1)
        ret_dict = dict(
            feat_ms = feat_ms,
        )
        return ret_dict
