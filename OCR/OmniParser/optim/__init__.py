import torch
import torch.nn as nn

from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

def build_criterion(args):
    weight = torch.ones(args.num_classes)
    
    if args.vie_categories > 0:
        weight[-args.vie_categories:] = 4

    weight[args.pt_eos_index] = args.pt_eos_loss_coef
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=args.padding_index)

    device = torch.device('cuda')
    criterion = criterion.to(device)

    return criterion

def build_optimizer_scheduler(model, args):
    if args.distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    backbone_lr = args.lr * args.lr_backbone_ratio
    param_dict = [
        {'params': [p for n, p in model_without_ddp.named_parameters() if 'backbone' not in n and p.requires_grad],
         'lr': args.lr},
        {'params': [p for n, p in model_without_ddp.named_parameters() if 'backbone' in n and p.requires_grad],
         'lr': backbone_lr},
    ]
    
    optimizer = torch.optim.AdamW(param_dict, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        lr_end=args.end_lr,
        power=args.decay_power,
    )

    return optimizer, scheduler

