# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------
# --------------------------------------------------------
# To encode arbitrary-length text images with masks.
# Based on FocalNet.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import math


class MaskedConv2d(nn.Conv2d):
    """ Mask the convolved features
    """
    def forward(self, x: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, C, H, W)
            mask (torch.Tensor): (B, H, W)
        """
        x = x * mask.unsqueeze(1)
        x = self._conv_forward(x, self.weight, self.bias)

        if x.shape[2] != mask.shape[1] or x.shape[3] != mask.shape[2]:
            mask = F.adaptive_max_pool2d(mask.unsqueeze(1), (x.shape[2], x.shape[3])).squeeze(1)
        return x, mask


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, max_kh, focal_factor=2, bias=True, proj_drop=0., use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2*dim + (self.focal_level+1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            k_h, k_w = min(kernel_size, max_kh), kernel_size
            p_h, p_w = k_h // 2, k_w // 2
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=(k_h, k_w), stride=1, 
                        groups=dim, padding=(p_h, p_w), bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x, mask):
        """
        Args:
            x: input features with shape of (B, H, W, C)
            mask: to indicate padded positions (0), of size (B, H, W)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous() # [B, C + C + L+1, H, W]
        x = x * mask.unsqueeze(1)
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level + 1), 1)
        
        # context aggreation
        ctx_all = 0 
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx = ctx * mask.unsqueeze(1)
            ctx_all = ctx_all + ctx*self.gates[:, l:l+1]
        ctx_global = ctx.sum(2, keepdim=True).sum(3, keepdim=True) \
            / (mask.sum(1, keepdim=True).sum(2, keepdim=True).unsqueeze(1) + 1e-10)
        ctx_global = self.act(ctx_global)
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous() # [B, H, W, C]
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        
        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        x_out = x_out * mask.unsqueeze(3)
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class FocalNetBlock(nn.Module):
    r""" Focal Modulation Network Block.
    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): Number of focal levels. 
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                    focal_level=1, max_kh=8, focal_window=3,
                    use_layerscale=False, layerscale_value=1e-4, 
                    use_postln=False, use_postln_in_modulation=False, 
                    normalize_modulator=False):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln

        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim, proj_drop=drop, focal_window=focal_window, focal_level=self.focal_level, max_kh=max_kh,
            use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.H = None
        self.W = None

    def forward(self, x, mask):
        """
        x: (B, L, C)
        mask: (B, H, W)
        return: (B, L, C)
        """
        mask_flat = mask.flatten(1)

        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x

        # Focal Modulation
        x = x if self.use_postln else self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.modulation(x, mask).view(B, H * W, C)
        if self.use_postln:
            x = self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(
            self.gamma_2 * (self.norm2(self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x))))
        x = x * mask_flat.unsqueeze(2)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, mlp_ratio={self.mlp_ratio}"


class BasicConvStage(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """

    def __init__(self, dim, out_dim, depth,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False,
                 focal_level=1, stride=2, max_kh=7, focal_window=1,
                 is_stem=False,
                 use_layerscale=False, layerscale_value=1e-4,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # build blocks
        self.blocks = nn.ModuleList([
            FocalNetBlock(
                dim=dim, 
                mlp_ratio=mlp_ratio, 
                drop=drop, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                focal_level=focal_level,
                max_kh=max_kh,
                focal_window=focal_window, 
                use_layerscale=use_layerscale, 
                layerscale_value=layerscale_value,
                use_postln=use_postln, 
                use_postln_in_modulation=use_postln_in_modulation, 
                normalize_modulator=normalize_modulator, 
            ) for i in range(depth)])

        if downsample is not None:
            max_kh_ds = max_kh if max_kh > 3 else 2
            self.downsample = downsample(
                max_kh=max_kh_ds,
                stride=stride,
                in_chans=dim,
                embed_dim=out_dim,
                is_stem=is_stem,
                norm_layer=norm_layer, 
            )
        else:
            self.downsample = None

    def forward(self, x, H, W, mask):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, mask)
            else:
                x = blk(x, mask)

        if self.downsample is not None:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
            x, Ho, Wo, mask = self.downsample(x, mask)
        else:
            Ho, Wo = H, W        
        return x, Ho, Wo, mask

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class Stem(nn.Module):
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.conv0 = MaskedConv2d(in_chans, 32, kernel_size=3, stride=1, padding=1)
        self.norm0 = nn.LayerNorm(32)
        self.conv1_0 = MaskedConv2d(32, 32, kernel_size=5, stride=2, padding=2, groups=32)
        self.norm_1 = nn.LayerNorm(32)
        self.conv1_1 = MaskedConv2d(32, 64, kernel_size=1)
        self.conv2_0 = MaskedConv2d(64, 64, kernel_size=5, stride=2, padding=2, groups=64)
        self.norm2 = nn.LayerNorm(64)
        self.conv2_1 = MaskedConv2d(64, embed_dim, kernel_size=1)
        self.conv3 = nn.Linear(embed_dim, embed_dim)

        self.act = nn.GELU()
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, in_c, H, W)
            mask (torch.Tensor): (B, H, W)
        """
        x, mask = self.conv0(x, mask)
        x = self.norm0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x, mask = self.conv1_0(x, mask)
        x = self.norm_1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x, mask = self.conv1_1(x, mask)
        x = self.act(x)
        x, mask = self.conv2_0(x, mask)
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x, mask = self.conv2_1(x, mask)
        x = self.act(x)
        x = self.conv3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x, mask


class DownsamplingLayer(nn.Module):
    r""" Downsampling layer.

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, max_kh=3, stride=2, in_chans=3, embed_dim=96, is_stem=False, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if is_stem:
            self.proj = Stem(in_chans, embed_dim)
        else:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            kh, kw = min(max_kh, 3), 3
            if stride == 1:
                kh = max(3, kh)
            ph, pw = (kh - 1) // 2, kw // 2
            stride = (stride, 1) # only mapping
            self.proj = MaskedConv2d(in_chans, embed_dim, kernel_size=(kh, kw), stride=stride, padding=(ph, pw))
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, mask):
        B, C, H, W = x.shape

        x, mask = self.proj(x, mask)        
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        
        # if H != mask.shape[1]:
        #     mask = F.adaptive_max_pool2d(mask.unsqueeze(1), (H, W)).squeeze(1)
        return x, H, W, mask


class FocalNet(nn.Module):
    r""" Focal Modulation Networks (FocalNets)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1] 
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1] 
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layer_scale_init_value (float): Value for layer scale. Default: 1e-4 
        use_postln (bool): Whether use layernorm after modulation (it helps stablize training of large models)
    """
    def __init__(self,
                in_chans=3,
                embed_dim=96, 
                depths=[2, 2, 6, 2], 
                mlp_ratio=4., 
                drop_rate=0., 
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, 
                patch_norm=True,
                use_checkpoint=False,                 
                focal_levels=[2, 2, 2, 2],
                strides=[2, 2, 1],
                max_khs=[7, 7, 7, 7],
                focal_windows=[3, 3, 3, 3], 
                use_layerscale=False, 
                layer_scale_init_value=1e-4, 
                use_postln=False, 
                use_postln_in_modulation=False, 
                normalize_modulator=False, 
                **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers)]

        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        
        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = DownsamplingLayer(
            in_chans=in_chans, 
            embed_dim=embed_dim[0], 
            is_stem=True,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicConvStage(
                dim=embed_dim[i_layer],
                out_dim=embed_dim[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=DownsamplingLayer if (i_layer < self.num_layers - 1) else None,
                focal_level=focal_levels[i_layer],
                stride=strides[i_layer] if (i_layer < self.num_layers - 1) else 1,
                max_kh=max_khs[i_layer],
                focal_window=focal_windows[i_layer],
                is_stem=False,
                use_checkpoint=use_checkpoint,
                use_layerscale=use_layerscale,
                layerscale_value=layer_scale_init_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {''}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {''}

    def forward(self, x, mask):
        """
        x: (B, C, H, W)
        mask: (B, H, W)
        """
        # mask.fill_(1.0)
        x, H, W, mask = self.patch_embed(x, mask)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W, mask = layer(x, H, W, mask)
        x = self.norm(x)  # B L C
        B, _, C = x.size()
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = x * mask.unsqueeze(1)
        return x, mask


model_urls = {
    "focalnet_tiny_lrf": "",
    "focalnet_small_lrf": "",
    "focalnet_base_lrf": "",    
}

@register_model
def focalnet_tiny_lrf(pretrained=False, **kwargs):
    h_fm = kwargs['h_fm'] or 1
    assert h_fm in [1, 2, 4, 8]
    n_skip_stride = int(math.log2(h_fm))
    strides = [2] * (3 - n_skip_stride) + [1] * n_skip_stride
    if h_fm == 8:
        max_khs = [7, 7, 7, 7]
    elif h_fm == 4:
        max_khs = [7, 3, 3, 3]
    else:
        max_khs = [7, 3, 3, 1]
    model = FocalNet(depths=[2, 2, 6, 2], embed_dim=64, focal_levels=[3, 3, 3, 3],\
        strides=strides, max_khs=max_khs, use_layerscale=True, **kwargs)
    if pretrained:
        url = model_urls['focalnet_tiny_lrf']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def focalnet_base_lrf(pretrained=False, **kwargs):
    h_fm = kwargs['h_fm'] or 1
    assert h_fm in [1, 2, 4, 8]
    n_skip_stride = int(math.log2(h_fm))
    strides = [2] * (3 - n_skip_stride) + [1] * n_skip_stride
    if h_fm == 8:
        max_khs = [7, 7, 7, 7]
    elif h_fm == 4:
        max_khs = [7, 3, 3, 3]
    else:
        max_khs = [7, 3, 3, 1]
    model = FocalNet(depths=[2, 2, 9, 2], embed_dim=96, focal_levels=[3, 3, 3, 3],\
        strides=strides, max_khs=max_khs, use_layerscale=True, **kwargs)
    if pretrained:
        url = model_urls['focalnet_base_lrf']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == "__main__":
    img_size = (32, 128)
    x = torch.rand(2, 3, *img_size).cuda()
    mask = torch.ones(2, *img_size).cuda()
    mask[0, :, 96:] = 0
    mask[1, :, 72:] = 0
    model = focalnet_tiny_lrf(h_fm=2).cuda()
    print(model)
    y, mask = model(x, mask)
    print(y.size(), mask.size())
    # import ipdb;ipdb.set_trace()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters / 1e6} M")
