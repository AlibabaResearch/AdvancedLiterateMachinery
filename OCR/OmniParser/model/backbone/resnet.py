import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from torchvision.models._utils import IntermediateLayerGetter

from utils.dist import is_main_process
from utils.nested_tensor import NestedTensor
from model.block import FrozenBatchNorm2d


class ResNet(nn.Module):
    def __init__(self, name: str,
                train_backbone: bool,
                return_interm_layers: bool,
                dilation: bool,
                freeze_bn: bool):
        super(ResNet, self).__init__()
        
        norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=norm_layer)
        
        # backbone.load_state_dict(torch.load('./resnet50-19c8e357.pth', map_location=torch.device('cpu')))
        # for name, parameter in backbone.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        else:
            return_layers = {'layer4': '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}

        for name in ['0', '1', '2', '3']:        
            x = xs[name]
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out