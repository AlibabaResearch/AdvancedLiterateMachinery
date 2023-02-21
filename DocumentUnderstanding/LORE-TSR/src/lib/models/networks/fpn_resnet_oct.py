from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
__all__ = ['OctResNet', 'oct_resnet26', 'oct_resnet50', 'oct_resnet101', 'oct_resnet152', 'oct_resnet200']

def pad_same(x,ksize,stride=1,Pool=False):
      shape = x.shape
      n_in,w,h = shape[1],shape[2],shape[3]
      if h % stride == 0:
          pad_along_height = max(ksize - stride, 0)
      else:
          pad_along_height = max(ksize - (h % stride), 0)
      if w % stride == 0:
          pad_along_width = max(ksize - stride, 0)
      else:
          pad_along_width = max(ksize - (w % stride), 0)
      pad_bottom = pad_along_height // 2
      pad_top = pad_along_height - pad_bottom
      pad_right = pad_along_width // 2
      pad_left = pad_along_width - pad_right
      dim = (pad_left,pad_right,pad_top,pad_bottom)
      if Pool:
          dim = (pad_right,pad_left,pad_bottom,pad_top)
      x = F.pad(x,dim,"constant",value=0)
      return x

class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, bias=False):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.kernel_size = kernel_size
        self.inchannel = in_channels - int(alpha_in * in_channels)
        self.l = int(alpha_in * in_channels)
        self.h = int((1-alpha_in) * in_channels)
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, stride=1, padding=0, bias=bias)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 else \
                        nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, stride=1, padding=0, bias=bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, stride=1, padding=0, bias=bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        if x_h is not None:
            x_h = self.downsample(x_h) if self.stride == 2 else x_h
            x_h_ = pad_same(x_h,self.kernel_size,1)
            x_h2h = self.conv_h2h(x_h_)
            x_h2l = self.conv_h2l(pad_same(self.downsample(x_h),self.kernel_size,1)) if self.alpha_out > 0 else None
        if x_l is not None:
            x_l2h = pad_same(x_l,self.kernel_size,1)
            x_l2h = self.conv_l2h(x_l2h)
            x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = pad_same(x_l2l,self.kernel_size,1)
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None 
            x_h = x_l2h + x_h2h
            x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
            return x_h, x_l
        else:
            return x_h2h, x_h2l

class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, 
                 bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l

class Conv_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0,
                 bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l

class BasicBlock(nn.Module):
    expansion = 1
  
    def __init__(self, inplanes, planes, stride=1, downsample=None,groups=1,
                base_width=64, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes
        self.conv1 = Conv_BN_ACT(inplanes, width, kernel_size=3, stride=stride, alpha_in=alpha_in, alpha_out=alpha_out, norm_layer=norm_layer)
        self.conv2 = Conv_BN(width, width, kernel_size=3, alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, norm_layer=norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
  
    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None
        h = identity_h.shape
        l = identity_l.shape if identity_l is not None else 0
        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))

        if self.downsample is not None:
            identity_h, identity_l = self.downsample(x)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv_BN_ACT(inplanes, width, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, norm_layer=norm_layer)
        self.conv2 = Conv_BN_ACT(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, norm_layer=norm_layer,
                                 alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.conv3 = Conv_BN(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer,
                             alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        x_h, x_l = self.conv3((x_h, x_l))

        if self.downsample is not None:
            identity_h, identity_l = self.downsample(x)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l

class OctResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(OctResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.deconv_with_bias = False
        self.heads = heads

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, norm_layer=norm_layer, alpha_in=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, norm_layer=norm_layer, alpha_out=0, output=True)

        self.adaption3 = nn.Conv2d(128,256, kernel_size=1, stride=1, padding=0,bias=False)
        self.adaption2 = nn.Conv2d(64,256, kernel_size=1, stride=1, padding=0,bias=False)
        self.adaption1 = nn.Conv2d(32 ,256, kernel_size=1, stride=1, padding=0,bias=False)
        self.adaption0 = nn.Conv2d(64 ,256, kernel_size=1, stride=1, padding=0,bias=False)
        self.adaptionU1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0,bias=False)

        self.deconv_layers1 = self._make_deconv_layer(1, [256], [4],)
        self.deconv_layers2 = self._make_deconv_layer(1, [256], [4],)
        self.deconv_layers3 = self._make_deconv_layer(1, [256], [4],)
        self.deconv_layers4 = self._make_deconv_layer(1, [256], [4],)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                inchannel = 256 
                fc = nn.Sequential(
                    nn.Conv2d(inchannel, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))
            else:
                inchannel = 256 
                fc = nn.Conv2d(in_channels=inchannel,out_channels=num_output,kernel_size=1,stride=1,padding=0)   
            self.__setattr__(head, fc) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv_BN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, alpha_in=alpha_in, alpha_out=alpha_out)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1 
            output_padding = 0 
        elif deconv_kernel == 3:
            padding = 1 
            output_padding = 1 
        elif deconv_kernel == 2:
            padding = 0 
            output_padding = 0 
        elif deconv_kernel == 7:
            padding = 3 
            output_padding = 0 

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


    def forward(self, x):
        x = pad_same(x,7,2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = pad_same(x,3,2,True)
        x0 = self.maxpool(x)

        x_h1, x_l1 = self.layer1(x0)
        x_h2, x_l2 = self.layer2((x_h1,x_l1))
        x_h3, x_l3 = self.layer3((x_h2,x_l2))
        x_h4, x_l4 = self.layer4((x_h3,x_l3))

        x3_ = self.deconv_layers1(x_h4)
        x3_ = self.adaption3(x_h3) + x3_

        x2_ = self.deconv_layers2(x3_)
        x2_ = self.adaption2(x_h2) + x2_

        x1_ = self.deconv_layers3(x2_)
        x1_ = self.adaption1(x_h1) + x1_

        x0_ = self.deconv_layers4(x1_) + self.adaption0(x0)
        x0_ = self.adaptionU1(x0_)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x0_)

        return [ret]

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_oct_resnet_fpn_mask(num_layers, heads, head_conv):
  block_class, layers = resnet_spec[num_layers]

  model = OctResNet(block_class, layers, heads, head_conv=head_conv)
  return model
