import torch
import torch.distributed as dist

def compute_acc(logits, labels):
    
    preds = logits.argmax(dim=-1)
    preds = preds[labels != -100]
    targets = labels[labels != -100]
    correct = torch.sum(preds == targets)
    total = (targets != -100).sum()
    
    return correct, total

def record_statics(ret, writer, cfg, global_step, mode):

    mlm_loss = ret['mlm_loss'].data
    wip_contrast_loss = ret['wip_contrast_loss'].data

    img_loss = ret['img_loss'].data
    txt_loss = ret['txt_loss'].data

    total_loss = mlm_loss + img_loss + txt_loss + wip_contrast_loss

    dist.all_reduce(total_loss.data, op=dist.ReduceOp.SUM)
    dist.all_reduce(mlm_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(wip_contrast_loss, op=dist.ReduceOp.SUM)

    dist.all_reduce(img_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(txt_loss, op=dist.ReduceOp.SUM)

    record_total_loss = total_loss.item() / dist.get_world_size()
    record_mlm_loss = mlm_loss.item() / dist.get_world_size()
    record_wip_contrast_loss = wip_contrast_loss.item() / dist.get_world_size()

    record_img_loss = img_loss.item() / dist.get_world_size()
    record_txt_loss = txt_loss.item() / dist.get_world_size()

    mlm_correct, mlm_total = compute_acc(ret['mlm_logits'].detach(), ret['mlm_labels'].detach())

    dist.all_reduce(mlm_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(mlm_total, op=dist.ReduceOp.SUM)

    if cfg['local_rank'] == 0:
        writer.add_scalar(mode + '/total_loss', record_total_loss, global_step)
        writer.add_scalar(mode + '/mlm_loss', record_mlm_loss, global_step)
        writer.add_scalar(mode + '/wip_contrast_loss', record_wip_contrast_loss, global_step)

        writer.add_scalar(mode + '/img_loss', record_img_loss, global_step)
        writer.add_scalar(mode + '/txt_loss', record_txt_loss, global_step)
                
        writer.add_scalar(mode + '/mlm_acc', mlm_correct/mlm_total, global_step)

    return mlm_correct, mlm_total
