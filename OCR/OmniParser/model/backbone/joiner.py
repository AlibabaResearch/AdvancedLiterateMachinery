import torch.nn as nn 

from typing import List
from utils.nested_tensor import NestedTensor

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super(Joiner, self).__init__(backbone, position_embedding)
    
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name in ['0', '1', '2', '3']:        
            out.append(xs[name])
            pos.append(self[1](xs[name]).to(xs[name].tensors.dtype))

        return out, pos
