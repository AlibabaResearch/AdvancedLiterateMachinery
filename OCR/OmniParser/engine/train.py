import sys
import math 
import torch
from utils.dist import reduce_dict
from utils.logger import MetricLogger, SmoothedValue
import numpy as np

scaler = torch.cuda.amp.GradScaler()


def train_one_epoch(model, dataloader, criterion, optimizer, lr_scheduler, epoch, global_step, checkpointer, checkpoint_folder, args):
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter='  ')    
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    device = torch.device('cuda')

    for iter_time, (samples, input_seqs, output_seqs) in metric_logger.log_every(dataloader, args.print_freq, header):

        samples = samples.to(device)
        input_seqs = [x.to(device) for x in input_seqs]
        output_seqs = [x.to(device) for x in output_seqs]

        with torch.cuda.amp.autocast():
            outputs = model(samples, input_seqs)
            
            if args.vie_categories > 0:
                outputs[1][:,:,-args.vie_categories:] = -np.Inf
                outputs[2][:,:,-args.vie_categories:] = -np.Inf
                
            pt_loss = criterion(outputs[0].transpose(1, 2), output_seqs[0])
            poly_loss = criterion(outputs[1].transpose(1, 2), output_seqs[1])
            rec_loss  = criterion(outputs[2].transpose(1, 2), output_seqs[2])

        loss_dict = {'pt_loss': pt_loss, 'poly_loss': poly_loss, 'rec_loss': rec_loss}
        weight_dict = {'pt_loss': args.pt_loss_weight, 'poly_loss': args.poly_loss_weight, 'rec_loss': args.rec_loss_weight}

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced = sum(loss_dict_reduced_scaled.values()).item()

        if not math.isfinite(losses_reduced):
            print(f'Loss is {losses_reduced}, stopping training')
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(losses).backward()

        if args.max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        global_step += 1
        checkpointer.save(checkpoint_folder, model, optimizer, lr_scheduler, epoch, global_step, args)

        if global_step == args.max_steps:
            return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step

    metric_logger.synchronize_between_processes()
    print('Averaged stats', metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step