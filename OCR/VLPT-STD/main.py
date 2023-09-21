import os
import os.path as osp
from tqdm import tqdm
import yaml
import argparse

from dataset import SynthTextDataset
from models import VLPT
from optimizer import set_schedule
from utils import record_statics

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import apex
from apex import amp
from apex.parallel import DistributedDataParallel 


def get_arguments():
    parser = argparse.ArgumentParser(description="VLPT-STD")
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--exp_name', default="base", type=str, 
                        help='name of current experiment')
    return parser.parse_args()
args = get_arguments()


def train(model, train_dataloader, global_step, optimizer, scheduler, writer, cfg):

    model.train()
    mlm_correct, mlm_total = 0, 0

    param_groups = optimizer.param_groups

    bar = tqdm(train_dataloader)
    for batch_data in bar:
        global_step += 1
        optimizer.zero_grad()

        ret = model(batch_data)
        total_loss = sum([v for k, v in ret.items() if "loss" in k])

        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        bar.set_description('epoch {}, loss {:.2f}'.format(global_step // len(train_dataloader), total_loss.item()))
        
        optimizer.step()
        scheduler.step()

        if global_step == cfg['max_steps']:
            return global_step

        result = record_statics(ret, writer, cfg, global_step, mode='train')
        mlm_correct += result[0]
        mlm_total += result[1]

        for i in range(len(param_groups)):
            writer.add_scalar('train/learning_rate_' + str(i), optimizer.param_groups[i]['lr'], global_step)

    mlm_accuracy = mlm_correct / mlm_total
    if cfg['local_rank'] == 0:
        writer.add_scalar('train/mlm_acc_epoch', mlm_accuracy, global_step)

    return global_step


def validate(model, val_dataloader, epoch, writer, cfg):

    model.eval()
    mlm_correct, mlm_total = 0, 0

    global_step = epoch * len(val_dataloader)
    with torch.no_grad():
        for batch_data in tqdm(val_dataloader):
            global_step += 1
            
            ret = model(batch_data)
        
            result = record_statics(ret, writer, cfg, global_step, mode='val')
            mlm_correct += result[0]
            mlm_total += result[1]

        mlm_accuracy = mlm_correct / mlm_total
        if cfg['local_rank'] == 0:
            writer.add_scalar('val/mlm_acc_epoch', mlm_accuracy, global_step)

    return mlm_accuracy


def main():

    with open('conf/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['local_rank'] = args.local_rank
    cfg['exp_name'] = args.exp_name
    save_dir = 'outputs/' + cfg['exp_name']
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(osp.join(save_dir, 'logs', str(cfg['local_rank'])))

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(cfg["local_rank"])

    model = VLPT(cfg)
    model = apex.parallel.convert_syncbn_model(model)
    model.cuda(cfg["local_rank"])

    optimizer, scheduler = set_schedule(model, cfg)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2', loss_scale='dynamic')
    model = DistributedDataParallel(model)

    train_dataset = SynthTextDataset(cfg, split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"], \
                                                    num_workers=cfg["num_workers"], pin_memory=True, sampler=train_sampler, collate_fn=train_dataset.collate)

    val_dataset = SynthTextDataset(cfg, split='val')
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg["batch_size"], \
                                                    num_workers=cfg["num_workers"], pin_memory=True, sampler=val_sampler, collate_fn=val_dataset.collate)

    global_step = 0
    for epoch in range(cfg["epoch"]):
        train_sampler.set_epoch(epoch)

        global_step = train(model, train_dataloader, global_step, optimizer, scheduler, writer, cfg)
        
        mlm_acc_val = validate(model, val_dataloader, epoch, writer, cfg)        
        
        if cfg['local_rank'] == 0:
            torch.save(model.state_dict(), osp.join(save_dir, str(epoch) + '_' + str(mlm_acc_val.item()) + '.pth'))
        
        if global_step == cfg['max_steps']:
            return 
            

if __name__ == "__main__":
    main()