'''
Implementation of A3 module based on TokenLearner.

Copyright 2022 Alibaba
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearner(nn.Module):
    
    def __init__(self, input_embed_dim, out_token=30):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner = nn.Sequential(nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size = (1,1), stride=1, groups=8, bias=False),
                                          nn.Conv2d(input_embed_dim, out_token, kernel_size = (1,1), stride=1, bias=False))
        self.feat = nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size = (1,1), stride=1, groups=8, bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.token_norm(x) # [bs, 257, 768]
        x = x.transpose(1, 2).unsqueeze(-1) # [bs, 768, 257, 1]
        selected = self.tokenLearner(x) # [bs, 27, 257, 1].
        selected = selected.flatten(2)  # [bs, 27, 257].
        selected = F.softmax(selected, dim=-1) 
        feat = self.feat(x) #  [bs, 768, 257, 1].
        feat = feat.flatten(2).transpose(1,2)  # [bs, 257, 768]
        x = torch.einsum('...si,...id->...sd', selected, feat) # [bs, 27, 768]
        
        x = self.norm(x)
        return selected, x