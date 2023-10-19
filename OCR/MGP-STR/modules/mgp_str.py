'''
Implementation of MGP-STR based on ViTSTR.

Copyright 2022 Alibaba
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
import torch.nn as nn
import logging
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from copy import deepcopy
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models import create_model
from .token_learner import TokenLearner

_logger = logging.getLogger(__name__)

__all__ = [
    'mgp_str_base_patch4_3_32_128',
    'mgp_str_large_patch4_3_32_128',
    'mgp_str_tiny_patch4_3_32_128',
    'mgp_str_small_patch4_3_32_128',
]

def create_mgp_str(batch_max_length, num_tokens, model=None, checkpoint_path=''):
    mgp_str = create_model(
        model,
        pretrained=True,
        num_classes=num_tokens,
        checkpoint_path=checkpoint_path,
        batch_max_length=batch_max_length)

    # might need to run to get zero init head for transfer learning
    mgp_str.reset_classifier(num_classes=num_tokens)

    return mgp_str

class MGPSTR(VisionTransformer):

    def __init__(self, batch_max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.batch_max_length = batch_max_length
        self.char_tokenLearner = TokenLearner(self.embed_dim, self.batch_max_length)

        self.bpe_tokenLearner = TokenLearner(self.embed_dim, self.batch_max_length)
        self.wp_tokenLearner = TokenLearner(self.embed_dim, self.batch_max_length)
        
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.char_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.bpe_head = nn.Linear(self.embed_dim, 50257) if num_classes > 0 else nn.Identity()
        self.wp_head = nn.Linear(self.embed_dim, 30522) if num_classes > 0 else nn.Identity()
        

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) 

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i,blk in enumerate(self.blocks):
            x = blk(x)
            
        attens = []

        # char
        char_attn, x_char = self.char_tokenLearner(x)
        x_char = self.char_head(x_char)
        char_out = x_char
        attens = [char_attn] 

        # bpe
        bpe_attn, x_bpe = self.bpe_tokenLearner(x)
        bpe_out = self.bpe_head(x_bpe)
        attens += [bpe_attn]

        # wp
        wp_attn, x_wp = self.wp_tokenLearner(x)
        wp_out = self.wp_head(x_wp)
        attens += [wp_attn]
        
        return attens, char_out, bpe_out, wp_out

    def forward(self, x, is_eval=False):
        attn_scores, char_out, bpe_out, wp_out = self.forward_features(x)
        if is_eval:
            return [attn_scores, char_out, bpe_out, wp_out]
        else:
            return [char_out, bpe_out, wp_out]

def load_pretrained(model, cfg=None, num_classes=1000, in_chans=1, filter_fn=None, strict=True):
    '''
    Loads a pretrained checkpoint
    From an older version of timm
    '''
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(cfg['url'], progress=True, map_location='cpu')
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    print("in_chans",in_chans)
    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        key = conv1_name + '.weight'
        if key in state_dict.keys():
            _logger.info('(%s) key found in state_dict' % key)
            conv1_weight = state_dict[conv1_name + '.weight']
        else:
            _logger.info('(%s) key NOT found in state_dict' % key)
            return
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        try:
        # completely discard fully connected for all other differences between pretrained and created model
            del state_dict[classifier_name + '.weight']
            del state_dict[classifier_name + '.bias']
        except:
            pass
        strict = False

    print("Loading pre-trained vision transformer weights from %s ..." % cfg['url'])
    model.load_state_dict(state_dict, strict=strict)

def _conv_filter(state_dict):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if not 'patch_embed' in k and  not 'pos_embed' in k :
            out_dict[k] = v
        else:
            print("not load",k) 
    return out_dict

@register_model
def mgp_str_large_patch4_3_32_128(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    # model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    # model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    kwargs['in_chans'] = 3
    model = MGPSTR(
        img_size=(32,128), patch_size=4, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
        # url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'
        url = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

@register_model
def mgp_str_base_patch4_3_32_128(pretrained=False, **kwargs):
    kwargs['in_chans'] = 3
    model = MGPSTR(
        img_size=(32,128), patch_size=4, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            #url='https://github.com/roatienza/public/releases/download/v0.1-deit-base/deit_base_patch16_224-b5f2ef4d.pth'
            url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

# below is work in progress
@register_model
def mgp_str_tiny_patch4_3_32_128(pretrained=False, **kwargs):
    kwargs['in_chans'] = 3
    model = MGPSTR(
        img_size=(32,128), patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'
    )

    load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def mgp_str_small_patch4_3_32_128(pretrained=False, **kwargs):
    kwargs['in_chans'] = 3
    model = MGPSTR(
        img_size=(32,128), patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth"
    )
    load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model