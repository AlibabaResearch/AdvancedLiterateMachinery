import torch
import os
from utils.dist import is_main_process

class Checkpointer(object):
    def __init__(self, distributed):
        self.distributed = distributed

    def resize_embedding(self, new_state_dict, old_state_dict, replace_keys, vie_categories):
        keys = new_state_dict.keys()
        for key in keys:
            if key in replace_keys:
                old_weight = old_state_dict[key]
                new_state_dict[key][:-vie_categories] = old_weight
            else:
                new_state_dict[key] = old_state_dict[key]
        return new_state_dict

    def load(self, checkpoint_path, model, args, optimizer=None, lr_scheduler=None):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if not args.continue_train:
            new_checkpoint = {}
            new_checkpoint['model'] = checkpoint['model']
            checkpoint = new_checkpoint

        if self.distributed:
            model = model.module

        if args.train_vie and not args.continue_train:
            new_state_dict = model.state_dict()
            old_state_dict = checkpoint['model']
            replace_keys = ['transformer.embedding.word_embeddings.weight', 
                            'transformer.pt_pred_layer.layers.2.weight',
                            'transformer.pt_pred_layer.layers.2.bias',
                            'transformer.poly_pred_layer.layers.2.weight',
                            'transformer.poly_pred_layer.layers.2.bias',
                            'transformer.rec_pred_layer.layers.2.weight',
                            'transformer.rec_pred_layer.layers.2.bias']
            
            new_state_dict = self.resize_embedding(new_state_dict, old_state_dict, replace_keys, args.vie_categories)
            model.load_state_dict(new_state_dict)
        else:
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
        if (not optimizer is None) and ('optimizer' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer']) 
        
        if (not lr_scheduler is None) and ('lr_scheduler' in checkpoint):
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) 

        if 'epoch' in checkpoint:
            last_epoch = checkpoint['epoch']
        else:
            last_epoch = -1

        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        else:
            global_step = 0
        
        return last_epoch, global_step
    
    def save(self, checkpoint_folder, model, optimizer, lr_scheduler, epoch, global_step, args):
        if global_step == args.max_steps or global_step % args.checkpoint_freq == 0:
            checkpoint_name = f'checkpoint_step{global_step}.pth'

            checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)
            save_dict = {
                'model': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'args': args,
            }
            save_dict_last = {
                'model': model.module.state_dict() if args.distributed else model.state_dict()
            }
            if is_main_process():
                if global_step == args.max_steps:
                    torch.save(save_dict_last, checkpoint_path)
                else:
                    torch.save(save_dict, checkpoint_path)

