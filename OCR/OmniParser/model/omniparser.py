import torch.nn as nn 

from .block import MLP
from utils.nested_tensor import NestedTensor
from .fpn import FPN

class OmniParser(nn.Module):
    def __init__(self, backbone, transformer, num_classes, use_fpn=False):
        super(OmniParser, self).__init__()
        self.backbone = backbone 
        self.transformer = transformer
        self.use_fpn = use_fpn
        if self.use_fpn: 
            self.fpn = FPN()
            self.input_proj = nn.Conv2d(1024, transformer.d_model, kernel_size=1, stride=2)
        else:
            self.input_proj = nn.Conv2d(1024, transformer.d_model, kernel_size=1)

    def forward(self, samples: NestedTensor, sequence):
        features, pos = self.backbone(samples)

        if self.use_fpn:
            src = self.fpn([x.tensors for x in features])
            _, mask = features[-2].decompose()
            pos = pos[-2]
        else:
            src = features[-1].tensors
            _, mask = features[-1].decompose()
            pos = pos[-1]

        out = self.transformer(self.input_proj(src), mask, pos, sequence)
        return out
