import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class ParallelAttention(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.k_map = nn.Linear(feat_dim, feat_dim)
        self.order_att = nn.Linear(feat_dim, 26)
        self.fc = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x, mask):
        h = x.size(2)
        mask = mask.flatten(1)
        mask_pad = (1 - mask).round().bool()

        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        key = self.k_map(x)
        att = self.order_att(key).transpose(1, 2)
        att.masked_fill_(mask_pad.unsqueeze(1).expand_as(att), float('-inf'))
        att = att.softmax(2)
        char_feat = att.matmul(x)
        y_pred = self.fc(char_feat)
        ret = dict(
            logits=y_pred,
            char_maps=att,
            h=h,
        )
        return ret
