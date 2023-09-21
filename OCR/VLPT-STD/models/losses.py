import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import time


def compute_mlm(mlm_logits, mlm_labels, vocab_size):

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, vocab_size),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
    }

    return ret
    

def compute_image_text_contrast(image_features, text_features, logit_scale):

    bs_text = text_features.size(0)
    bs_image = image_features.size(0)

    rank = dist.get_rank()

    # distributed version   
    # gather image features
    all_image_features = [
        torch.zeros_like(image_features) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_image_features, image_features)
    all_image_features = torch.cat(all_image_features)
    # gather text features
    all_text_features = [
        torch.zeros_like(text_features) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_text_features, text_features)
    all_text_features = torch.cat(all_text_features)

    # calculate two-way logits
    logits_per_image = logit_scale * image_features @ all_text_features.t()
    logits_per_text = logit_scale * text_features @ all_image_features.t()

    # labels
    labels_text = torch.arange(rank*bs_image, (rank+1)*bs_image).to(logits_per_image.device)
    labels_image = torch.arange(rank*bs_text, (rank+1)*bs_text).to(logits_per_image.device)

    img_loss = F.cross_entropy(logits_per_image, labels_image)
    txt_loss = F.cross_entropy(logits_per_text, labels_text)

    ret = {}
    ret.update({'img_loss': 0.5*img_loss})
    ret.update({'txt_loss': 0.5*txt_loss})

    return ret


def compute_global_local_contrast(image_feats_tii, candidate_false_text_feats_tii, valid_text_masks, logit_scale):

    image_features = image_feats_tii
    text_features = candidate_false_text_feats_tii
    text_masks = valid_text_masks

    B, N_token, N_false_text, C = text_features.shape

    logits = logit_scale * torch.matmul(image_features, text_features.view(B,-1,C).permute(0,2,1))
    logits = logits.squeeze(0) # B, N_token*N_false_text
    logits = logits.view(B, N_token, N_false_text).permute(0, 2, 1)

    labels = torch.zeros(B, N_token).long().to(image_features.device)

    labels[~text_masks] = -100

    loss = F.cross_entropy(logits, labels, ignore_index=-100)

    ret = {}
    ret.update({'wip_contrast_loss': loss})
    
    return ret



