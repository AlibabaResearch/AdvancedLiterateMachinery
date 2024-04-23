import torch
from .omniparser import OmniParser
from .backbone import build_backbone
from .transformer import build_transformer


def build_model(args):
    from .transformer import build_transformer

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = OmniParser(backbone, transformer, args.num_classes, args.use_fpn)

    device = torch.device('cuda')
    model = model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    return model