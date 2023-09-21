# Copyright (2023) Alibaba Group and its affiliates
import torch
import torch.nn as nn
from timm.models import create_model
from timm.models.layers import trunc_normal_
import math
import time

from .feature_extractor import focalnet_tiny_lrf, focalnet_base_lrf
from .nb_decoder import NeighborDecoder
from .pat_decoder import ParallelAttention
from .ctc_decoder import CTCDecoder
from .RNNDecoder import RNNAttention
from .la_utils import LocalAttentionModule


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass
    
    def update_with_loss(self, ret, labels=None, label_lens=None):
        if labels is not None:
            assert label_lens is not None
            loss_dict = self.get_loss(ret, labels, label_lens)
            ret.update(loss_dict)
        return ret
    
    def get_loss(self, **kwargs):
        raise NotImplementedError


class LISTER(BaseModel):
    def __init__(self, num_classes, max_ch=None, iters=0, enc='focalnet',
                 sa4enc=False, enc_version='tiny', **kwargs):
        super().__init__()
        if enc == 'focalnet':
            if enc_version == 'tiny':
                self.encoder = focalnet_tiny_lrf(**kwargs)
            elif enc_version == 'base':
                self.encoder = focalnet_base_lrf(**kwargs)
            else:
                raise NotImplementedError(f"encoder type {enc_version} is not supported for now.")
        else:
            raise NotImplementedError(f"{enc} is not supported for now.")
        feat_dim = self.encoder.embed_dim[-1]
        self.sa4enc = sa4enc
        if sa4enc:
            self.sa_layer = LocalAttentionModule(feat_dim, nhead=8, window_size=25, num_layers=2,
                drop_rate=0.1, proj_drop_rate=0.1)
        self.decoder = NeighborDecoder(num_classes, feat_dim, max_ch=max_ch, iters=iters, **kwargs)
        self.num_classes = num_classes

        self.apply(self._init_weights)

        self.celoss_fn = nn.CrossEntropyLoss(reduction='none')
        self.coef=(1.0, 0.01, 0.001) # for loss of rec, eos and ent respectively

        self.timer = 'timer' in kwargs and kwargs['timer']
    
    def forward(self, x, mask, max_char=None, labels=None, label_lens=None):
        t0 = time.time()
        feat, mask = self.encoder(x, mask)
        if self.sa4enc:
            B, C, H, W = feat.size()
            assert H == 1, 'the current version only supports 1-d sequence.'
            feat = feat.flatten(2).transpose(1, 2)
            mask_flt = mask.flatten(1)
            feat = self.sa_layer(feat, mask_flt)
            feat = feat.transpose(1, 2).contiguous().view(B, C, H, W)
        t1 = time.time()
        ret = self.decoder(feat, mask, max_char)
        t2 = time.time()
        ret = self.update_with_loss(ret, labels, label_lens)
        if self.timer:
            t_dict = {"t_enc": t1 - t0, "t_dec": t2 - t1, 't_total': t2 - t0}
            return ret, t_dict
        return ret
    
    def calc_rec_loss(self, logits, targets, target_lens, mask):
        """
        Args: 
            logits: [minibatch, C, T], not passed to the softmax func.
            targets, torch.cuda.LongTensor [minibatch, T]
            target_lens: [minibatch]
            mask: [minibatch, T]
        """
        losses = self.celoss_fn(logits, targets)
        losses = losses * mask
        loss = losses.sum(-1) / (target_lens + 1e-10)
        loss = loss.mean()
        return loss
    
    def calc_eos_loc_loss(self, char_maps, target_lens, eps=1e-10):
        max_tok = char_maps.shape[2]
        eos_idx = (target_lens - 1).contiguous().view(-1, 1, 1).expand(-1, 1, max_tok)
        eos_maps = torch.gather(char_maps, dim=1, index=eos_idx).squeeze(1) # (b, max_tok)
        loss = (eos_maps[:, -1] + eps).log().neg()
        return loss.mean()
    
    def calc_entropy(self, p:torch.Tensor, mask:torch.Tensor, eps=1e-10):
        """
        Args:
            p: probability distribution over the last dimension, of size (..., L, C)
            mask: (..., L)
        """
        p_nlog = (p + eps).log().neg()
        ent = p * p_nlog
        ent = ent.sum(-1) / math.log(p.size(-1) + 1)
        ent = (ent * mask).sum(-1) / (mask.sum(-1) + eps) # (...)
        ent = ent.mean()
        # ent.fill_(0)
        return ent
    
    def get_loss(self, model_output, labels, label_lens):
        batch_size, max_len = labels.size()
        seq_range = torch.arange(0, max_len, device=labels.device).long().unsqueeze(0).expand(
                batch_size, max_len)
        seq_len = label_lens.unsqueeze(1).expand_as(seq_range)
        mask = (seq_range < seq_len).float() # [batch_size, max_len]

        l_rec, l_eos, l_ent = [], [], []
        iters = len(model_output['logits'])
        for i in range(iters):
            l_rec.append(self.calc_rec_loss(model_output['logits'][i].transpose(1, 2), labels, label_lens, mask))
            l_eos.append(self.calc_eos_loc_loss(model_output['char_maps'][i], label_lens))
            l_ent.append(self.calc_entropy(model_output['char_maps'][i], mask))
        if all([li.item() > 2.1 for li in l_rec]): # avoid hard training in the start
            l_rec = l_rec[0] + sum(l_rec[1:]) / (iters - 1 + 1e-8) * 0.0
            l_eos = l_eos[0] + sum(l_eos[1:]) / (iters - 1 + 1e-8) * 0.0
            l_ent = l_ent[0] + sum(l_ent[1:]) / (iters - 1 + 1e-8) * 0.0
        else:
            l_rec = sum(l_rec) / iters
            l_eos = sum(l_eos) / iters
            l_ent = sum(l_ent) / iters

        loss = l_rec * self.coef[0] + l_eos * self.coef[1] + l_ent * self.coef[2]
        loss_dict = dict(
            loss = loss,
            l_rec = l_rec,
            l_eos = l_eos,
            l_ent = l_ent,
        )
        return loss_dict


class RNNAttnSTR(BaseModel):
    def __init__(self, num_classes, enc='focalnet', **kwargs):
        super().__init__()
        if enc == 'focalnet':
            self.encoder = focalnet_tiny_lrf(**kwargs)
        else:
            raise NotImplementedError(f"{enc} is not supported for now.")
        feat_dim = self.encoder.embed_dim[-1]
        self.decoder = RNNAttention(feat_dim, 512, num_classes, embed_lm=False)

        self.apply(self._init_weights)

        self.celoss_fn = nn.CrossEntropyLoss(reduction='none')
        self.coef=(1.0, 0.01)

        self.timer = 'timer' in kwargs and kwargs['timer']
    
    def forward(self, x, mask, max_char=None, labels=None, label_lens=None):
        t0 = time.time()
        feat, mask = self.encoder(x, mask)
        t1 = time.time()
        ret = self.decoder(feat, mask, num_steps=max_char)
        t2 = time.time()
        ret = self.update_with_loss(ret, labels, label_lens)
        if self.timer:
            t_dict = {"t_enc": t1 - t0, "t_dec": t2 - t1, 't_total': t2 - t0}
            return ret, t_dict
        return ret
    
    def calc_rec_loss(self, logits, targets, target_lens, mask):
        """
        Args: 
            logits: [minibatch, C, T], not passed to the softmax func.
            targets, torch.cuda.LongTensor [minibatch, T]
            target_lens: [minibatch]
            mask: [minibatch, T]
        """
        losses = self.celoss_fn(logits, targets)
        losses = losses * mask
        loss = losses.sum(-1) / (target_lens + 1e-10)
        loss = loss.mean()
        return loss
    
    def calc_eos_loc_loss(self, char_maps, target_lens, eps=1e-10):
        max_tok = char_maps.shape[2]
        eos_idx = (target_lens - 1).contiguous().view(-1, 1, 1).expand(-1, 1, max_tok)
        eos_maps = torch.gather(char_maps, dim=1, index=eos_idx).squeeze(1) # (b, max_tok)
        loss = (eos_maps[:, -1] + eps).log().neg()
        return loss.mean()
    
    def get_loss(self, model_output, labels, label_lens):
        batch_size, max_len = labels.size()
        seq_range = torch.arange(0, max_len, device=labels.device).long().unsqueeze(0).expand(
                batch_size, max_len)
        seq_len = label_lens.unsqueeze(1).expand_as(seq_range)
        mask = (seq_range < seq_len).float() # [batch_size, max_len]

        l_rec = self.calc_rec_loss(model_output['logits'].transpose(1, 2), labels, label_lens, mask)
        loss = l_rec
        loss_dict = dict(
            loss = loss,
            l_rec=l_rec,
        )
        return loss_dict


class PATModel(BaseModel):
    def __init__(self, num_classes, enc='focalnet', **kwargs):
        super().__init__()
        if enc == 'focalnet':
            self.encoder = focalnet_tiny_lrf(**kwargs)
        else:
            raise NotImplementedError(f"{enc} is not supported for now.")
        feat_dim = self.encoder.embed_dim[-1]
        self.decoder = ParallelAttention(num_classes, feat_dim)

        self.apply(self._init_weights)

        self.celoss_fn = nn.CrossEntropyLoss(reduction='none')
        self.coef=(1.0, 0.01)

        self.timer = 'timer' in kwargs and kwargs['timer']
    
    def forward(self, x, mask, labels=None, label_lens=None):
        t0 = time.time()
        feat, mask = self.encoder(x, mask)
        t1 = time.time()
        ret = self.decoder(feat, mask)
        t2 = time.time()
        ret = self.update_with_loss(ret, labels, label_lens)
        if self.timer:
            t_dict = {"t_enc": t1 - t0, "t_dec": t2 - t1, 't_total': t2 - t0}
            return ret, t_dict
        return ret
    
    def calc_rec_loss(self, logits, targets, target_lens, mask):
        """
        Args: 
            logits: [minibatch, C, T], not passed to the softmax func.
            targets, torch.cuda.LongTensor [minibatch, T]
            target_lens: [minibatch]
            mask: [minibatch, T]
        """
        logits = logits[:, :, :targets.size(1)]
        losses = self.celoss_fn(logits, targets)
        losses = losses * mask
        loss = losses.sum(-1) / (target_lens + 1e-10)
        loss = loss.mean()
        return loss

    def calc_eos_loc_loss(self, char_maps, target_lens, eps=1e-10):
        max_tok = char_maps.shape[2]
        eos_idx = (target_lens - 1).contiguous().view(-1, 1, 1).expand(-1, 1, max_tok)
        eos_maps = torch.gather(char_maps, dim=1, index=eos_idx).squeeze(1) # (b, max_tok)
        loss = (eos_maps[:, -1] + eps).log().neg()
        return loss.mean()

    def get_loss(self, model_output, labels, label_lens):
        batch_size, max_len = labels.size()
        seq_range = torch.arange(0, max_len, device=labels.device).long().unsqueeze(0).expand(
                batch_size, max_len)
        seq_len = label_lens.unsqueeze(1).expand_as(seq_range)
        mask = (seq_range < seq_len).float() # [batch_size, max_len]

        l_rec = self.calc_rec_loss(model_output['logits'].transpose(1, 2), labels, label_lens, mask)
        loss = l_rec

        loss_dict = dict(
            loss=loss,
            l_rec=l_rec,
        )
        return loss_dict


class CTCModel(BaseModel):
    def __init__(self, num_classes, blank_id, enc='focalnet', **kwargs):
        super().__init__()
        if enc == 'focalnet':
            self.encoder = focalnet_tiny_lrf(**kwargs)
        else:
            raise NotImplementedError(f"{enc} is not supported for now.")
        feat_dim = self.encoder.embed_dim[-1]
        self.decoder = CTCDecoder(num_classes, feat_dim, blank_id)

        self.apply(self._init_weights)

        self.loss_fn = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)

        self.timer = 'timer' in kwargs and kwargs['timer']
    
    def forward(self, x, mask, labels=None, label_lens=None):
        t0 = time.time()
        feat, mask = self.encoder(x, mask)
        t1 = time.time()
        ret = self.decoder(feat, mask)
        t2 = time.time()
        ret = self.update_with_loss(ret, labels, label_lens)
        if self.timer:
            t_dict = {"t_enc": t1 - t0, "t_dec": t2 - t1, 't_total': t2 - t0}
            return ret, t_dict
        return ret
    
    def calc_rec_loss(self, logits, targets, pred_sizes, target_lens):
        """
        Args: 
            logits: [b, T, C], not passed to the softmax func.
            targets: torch.IntTensor [T']
            pred_sizes: [b]
            target_lens: [b]
        """
        torch.backends.cudnn.enabled = False
        loss = self.loss_fn(logits.transpose(0, 1).log_softmax(2), targets, pred_sizes, target_lens)
        torch.backends.cudnn.enabled = True
        return loss

    def get_loss(self, model_output, labels, label_lens):
        pred_sizes = model_output['char_masks'].sum(1).round().int() # [b]
        loss = self.calc_rec_loss(model_output['logits'], labels, pred_sizes, label_lens)
        loss_dict = dict(
            loss=loss,
        )
        return loss_dict


if __name__ == "__main__":
    img_size = (32, 128)
    x = torch.rand(2, 3, *img_size).cuda()
    mask = torch.ones(2, *img_size).cuda()
    mask[0, :, 96:] = 0
    mask[1, :, 72:] = 0
    
    # model = LISTER(37, drop_path_rate=.1, iters=2).cuda()
    model = RNNAttnSTR(37, drop_path_rate=.1).cuda()
    ret = model(x, mask, max_char=5)
    # model = PATModel(37).cuda()
    # ret = model(x, mask)

    for k, v in ret.items():
        # print(k, len(v), v[0].size())
        print(k, v.size())
    
    labels = torch.tensor([[3, 5, 0, 0, 0], [2, 6, 8, 3, 0]]).cuda()
    label_lens = torch.tensor([3, 5]).cuda()
    loss_dict = model.get_loss(ret, labels, label_lens)
    for key, loss in loss_dict.items():
        print(key, loss.item())
