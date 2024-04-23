import torch 
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


class FPN(nn.Module):
    def __init__(self,):
        super().__init__()

        nin = [128, 256, 512, 1024]
        ndim = 256
        self.fpn_in = nn.ModuleList([conv1x1(nin[-1], ndim), conv1x1(nin[-2], ndim),
                    conv1x1(nin[-3], ndim), conv1x1(nin[-4], ndim)])

    def forward(self, feature_list):

        c2, c3, c4, c5 = feature_list
        p5 = self.fpn_in[0](c5)
        p5_up = F.interpolate(p5, size=c4.size()[2:], mode='nearest')

        c4_ = self.fpn_in[1](c4)
        p4 = c4_ + p5_up
        p4_up =  F.interpolate(p4, size=c3.size()[2:], mode='nearest')

        c3_ = self.fpn_in[2](c3)
        p3 = c3_ + p4_up
        p3_up =  F.interpolate(p3, size=c2.size()[2:], mode='nearest')

        c2_ = self.fpn_in[3](c2)
        p2 = c2_ + p3_up

        tmp_size = c3.size()[2:]

        p2 = F.interpolate(p2, size=tmp_size, mode='bilinear')
        p4 = F.interpolate(p4, size=tmp_size, mode='bilinear')
        p5 = F.interpolate(p5, size=tmp_size, mode='bilinear')

        concat_feature = torch.cat((p2, p3, p4, p5), dim=1)
        return concat_feature
    