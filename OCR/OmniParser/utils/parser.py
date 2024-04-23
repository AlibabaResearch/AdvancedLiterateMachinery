import argparse 

class DefaultParser(object):

    def __init__(self):
        parser = argparse.ArgumentParser()

        # Data
        parser.add_argument('--data_root', type=str)
        parser.add_argument('--train_dataset', type=str, nargs='+')
        parser.add_argument('--val_dataset', type=str, nargs='+')
        parser.add_argument('--train_min_size', type=int, nargs='+', default=[640, 672, 704, 736, 768, 800, 832, 864, 896])
        parser.add_argument('--train_max_size', type=int, default=1920)
        parser.add_argument('--test_min_size', type=int, default=1200)
        parser.add_argument('--test_max_size', type=int, default=1920)
        parser.add_argument('--chars', type=str, default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')

        # Sequence Construction
        parser.add_argument('--num_bins', type=int, default=1000)
        parser.add_argument('--rec_length', type=int, default=25)
        parser.add_argument('--pt_seq_length', type=int, default=1024)

        # Data Augmentation
        parser.add_argument('--crop_min_size_ratio', type=float, default=0.3)
        parser.add_argument('--crop_max_size_ratio', type=float, default=1.0)
        parser.add_argument('--crop_prob', type=float, default=1.0)
        parser.add_argument('--rotate_max_angle', type=int, default=90)
        parser.add_argument('--rotate_prob', type=float, default=0.5)
        parser.add_argument('--dist_brightness', type=float, default=0.5)
        parser.add_argument('--dist_contrast', type=float, default=0.5)
        parser.add_argument('--dist_saturation', type=float, default=0.5)
        parser.add_argument('--dist_hue', type=float, default=0.5)
        parser.add_argument('--distortion_prob', type=float, default=0.5)

        # Model Configuration
        parser.add_argument('--backbone', type=str, default='swin_transformer')
        parser.add_argument('--pretrained_file', type=str, default='./pretrained_weights/swin_base_patch4_window7_224_22k.pth')
        parser.add_argument('--position_embedding', type=str, default='sine')
        parser.add_argument('--tfm_hidden_dim', type=int, default=512)
        parser.add_argument('--tfm_dropout', type=float, default=0.1)
        parser.add_argument('--tfm_nheads', type=int, default=8)
        parser.add_argument('--tfm_dim_feedforward', type=int, default=2048)
        parser.add_argument('--tfm_dec_layers', type=int, default=4)
        parser.add_argument('--tfm_pre_norm', action='store_true') # True

        # Training Parameters
        parser.add_argument('--use_char_window_prompt', action='store_true')
        parser.add_argument('--global_prob', type=float, default=0.4)
        parser.add_argument('--use_fpn', action='store_true')
        parser.add_argument('--train_vie', action='store_true')
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--vie_categories', type=int, default=0)

        parser.add_argument('--lr', type=float, default=0.0005)
        parser.add_argument('--end_lr', type=float, default=0)
        parser.add_argument('--decay_power', type=float, default=1)
        parser.add_argument('--warmup_steps', type=int, default=10000)
        parser.add_argument('--max_steps', type=int, default=400000)
        parser.add_argument('--lr_backbone_ratio', type=float, default=0.1) 
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--pt_eos_loss_coef', type=float, default=0.01)
        parser.add_argument('--pt_loss_weight', type=float, default=1)
        parser.add_argument('--poly_loss_weight', type=float, default=1)
        parser.add_argument('--rec_loss_weight', type=float, default=1)
        parser.add_argument('--epochs', type=int, default=1000000000000000)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--resume', type=str, default='')
        parser.add_argument('--output_folder', type=str)
        parser.add_argument('--print_freq', type=int, default=10)
        parser.add_argument('--checkpoint_freq', type=int, default=1)
        parser.add_argument('--max_norm', type=float, default=0.1)

        # Inference Parameters
        parser.add_argument('--visualize', action='store_true')
        parser.add_argument('--infer_vie', action='store_true')

        # Distributed Parameters 
        parser.add_argument('--local_rank', type=int, default=0)
                     
        self.parser = parser

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)
    
    def parse_args(self):
        args = self.parser.parse_args()

        num_char_classes = len(args.chars) + 1 # +unknown, total 96
        args.recog_pad_index = args.num_bins + num_char_classes
        
        args.pt_eos_index = args.recog_pad_index + 1 
        args.poly_eos_index = args.pt_eos_index + 1 
        args.rec_eos_index = args.poly_eos_index + 1 

        args.pt_sos_index = args.rec_eos_index + 1
        args.poly_sos_index = args.pt_sos_index + 1
        args.rec_sos_index = args.poly_sos_index + 1

        args.padding_index = args.rec_sos_index + 1 
        args.num_classes = args.padding_index + 1 + args.vie_categories

        return args
        