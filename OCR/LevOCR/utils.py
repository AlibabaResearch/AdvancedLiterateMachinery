import torch
import numpy as np
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TokenLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, src_dict):
        # character (str): set of the possible characters.
        self.character = list(src_dict.keys())
        self.dict = src_dict

    def encode_vision(self, text, batch_max_length=26, device='cuda:0'):
        length = [len(s) for s in text]

        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length, ignore_spec_char=False):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                try:
                    if t[i] != 0: 
                        char_list.append(self.character[t[i]])
                    else:
                        break
                except:
                    print(self.character)
                    print(t[i])
                    raise Exception('failed!')
            text = ''.join(char_list)
            texts.append(text)
        return texts
    
    def encode_levt(self, text, tgt_dict, batch_max_length=26, device='cuda:0'):
        length = [len(s) for s in text]
        pad_index = tgt_dict.pad()
        bos_index = tgt_dict.bos_index
        eos_index = tgt_dict.eos_index

        batch_text = torch.LongTensor(len(text), batch_max_length+2).fill_(pad_index)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            text = [bos_index] + text + [eos_index]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device)) 


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def get_device(verbose=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if verbose:
        print("Device:", device)
    return device
    

def get_args(is_train=True):
    parser = argparse.ArgumentParser(description='STR')

    # for test
    parser.add_argument('--eval_data', help='path to evaluation dataset')
    parser.add_argument('--flops', action='store_true', help='calculates approx flops (may not work)')

    # for train
    parser.add_argument('--exp_name', default='LevOCR', help='Where to store logs and models')
    parser.add_argument('--train_data', required=is_train, help='path to training dataset')
    parser.add_argument('--valid_data', help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers. Use -1 to use all cores.', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_path', default='./saved_models', help="path to save")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--sgd', action='store_true', help='Whether to use SGD (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=26, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    
    """ Model Architecture """
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')

    # use cosine learning rate decay
    parser.add_argument('--scheduler', action='store_true', help='Use lr scheduler')

    # orig paper uses this for fast benchmarking
    parser.add_argument('--fast_acc', action='store_true', help='Fast average accuracy computation')
   
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--device', default='cuda',
                        # help='device to use for training / testing')
    
    # for eval
    parser.add_argument('--eval_img', action='store_true', help='eval imgs dataset')
    parser.add_argument('--range', default=None, help="start-end for example(800-1000)")
    parser.add_argument('--model_dir', default='') 
    parser.add_argument('--demo_imgs', default='')
    
    # this is for levt
    parser.add_argument('--_name', default='levenshtein_transformer')
    parser.add_argument('--activation_dropout', default=0.0)
    parser.add_argument('--activation_fn', default='relu')
    parser.add_argument('--apply_bert_init', default=True)
    parser.add_argument('--attention_dropout', default=0.0)
    parser.add_argument('--cross_self_attention', default=False)
    parser.add_argument('--decoder_attention_heads', default=8)
    parser.add_argument('--decoder_embed_dim', default=512)
    parser.add_argument('--decoder_embed_path', default=None)
    parser.add_argument('--decoder_ffn_embed_dim', default=2048)
    parser.add_argument('--decoder_layers', default=6)
    parser.add_argument('--decoder_normalize_before', default=False)
    parser.add_argument('--decoder_output_dim', default=512)
    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--embed_len_text', default=28)
    parser.add_argument('--embed_len_img', default=96)
    parser.add_argument('--encoder_attention_heads', default=8)
    parser.add_argument('--encoder_embed_dim', default=512)
    parser.add_argument('--encoder_ffn_embed_dim', default=2048)
    parser.add_argument('--encoder_layers', default=6)
    parser.add_argument('--encoder_normalize_before', default=False)
    parser.add_argument('--eos', default=2)
    parser.add_argument('--label_smoothing', default=0.1)
    parser.add_argument('--noise', default='random_delete')
    parser.add_argument('--quant_noise_pq', default=0)
    parser.add_argument('--quant_noise_pq_block_size', default=8)
    parser.add_argument('--sampling_for_deletion', default=False)  

    parser.add_argument('--dataset_charset_path', default='charset/charset_36.txt')
    parser.add_argument('--vis_model', default='') 
    parser.add_argument('--levt_model', default='') 
    parser.add_argument('--device', default='cuda:0') 
    parser.add_argument('--max_iter', default=2)
    parser.add_argument('--post_process', default='subword_nmt')
    parser.add_argument('--th', type=float, default=0.5)
    
    args = parser.parse_args()
    return args
