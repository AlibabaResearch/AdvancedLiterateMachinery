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

    def encode(self, text, batch_max_length=25, device='cuda:0'):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

#=======================TODO: 4 is represent for * ====================================
        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            #print('text:', text)
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
    
    def decode_new(self, text_index):
        """ convert text-index into text-label. """

        char_list = []
        for i in range(text_index.size(0)):
            if text_index[i] < len(self.character) and text_index[i] != 0:
                char_list.append(self.character[text_index[i]])
        text = ''.join(char_list)

        return text     
    
    def encode_levt_tgt(self, text, tgt_dict, batch_max_length=25, device='cuda:0'):
        """convert text-label into text-index for levt. different length with pad!!!
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        #batch_max_length = max(length)
        #batch_max_length -= 2 
        pad_index = tgt_dict.pad()
        bos_index = tgt_dict.bos_index
        eos_index = tgt_dict.eos_index

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length+2).fill_(pad_index)
        for i, t in enumerate(text):
            text = list(t)
            #print('text:', text)
            text = [self.dict[char] for char in text]
            text = [bos_index] + text + [eos_index]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device)) 

    def encode_only_eosbos(self, text, tgt_dict, batch_max_length=25, device='cuda:0'):
        """convert text-label into text-index for levt. different length with pad!!!
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        #batch_max_length = max(length)
        #batch_max_length -= 2 
        pad_index = tgt_dict.pad()
        bos_index = tgt_dict.bos_index
        eos_index = tgt_dict.eos_index

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length+2).fill_(pad_index)
        for i, t in enumerate(text):
            text = [bos_index] + [eos_index]
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
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=is_train, help='path to training dataset')
    parser.add_argument('--valid_data', required=is_train, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers. Use -1 to use all cores.', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_path', default='./saved_models', help="path to save")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--sgd', action='store_true', help='Whether to use SGD (default is Adadelta)')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
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
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    
    """ Model Architecture """
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    # selective augmentation 
    # can choose specific data augmentation
    parser.add_argument('--issel_aug', action='store_true', help='Select augs')
    parser.add_argument('--sel_prob', type=float, default=1., help='Probability of applying augmentation')
    parser.add_argument('--pattern', action='store_true', help='Pattern group')
    parser.add_argument('--warp', action='store_true', help='Warp group')
    parser.add_argument('--geometry', action='store_true', help='Geometry group')
    parser.add_argument('--weather', action='store_true', help='Weather group')
    # parser.add_argument('--noise', action='store_true', help='Noise group')
    parser.add_argument('--blur', action='store_true', help='Blur group')
    parser.add_argument('--camera', action='store_true', help='Camera group')
    parser.add_argument('--process', action='store_true', help='Image processing routines')

    # use cosine learning rate decay
    parser.add_argument('--scheduler', action='store_true', help='Use lr scheduler')

    parser.add_argument('--intact_prob', type=float, default=0.5, help='Probability of not applying augmentation')
    parser.add_argument('--isrand_aug', action='store_true', help='Use RandAug')
    parser.add_argument('--augs_num', type=int, default=3, help='Number of data augment groups to apply. 1 to 8.')
    parser.add_argument('--augs_mag', type=int, default=None, help='Magnitude of data augment groups to apply. None if random.')

    # for comparison to other augmentations
    parser.add_argument('--issemantic_aug', action='store_true', help='Use Semantic')
    parser.add_argument('--isrotation_aug', action='store_true', help='Use ')
    parser.add_argument('--isscatter_aug', action='store_true', help='Use ')
    parser.add_argument('--islearning_aug', action='store_true', help='Use ')

    # orig paper uses this for fast benchmarking
    parser.add_argument('--fast_acc', action='store_true', help='Fast average accuracy computation')
   
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--device', default='cuda',
                        # help='device to use for training / testing')
    
    # mask train
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument("--patch_size", type=int, default=4)

    # for eval
    parser.add_argument('--eval_img', action='store_true', help='eval imgs dataset')
    parser.add_argument('--range', default=None, help="start-end for example(800-1000)")
    parser.add_argument('--model_dir', default='') 
    parser.add_argument('--demo_imgs', default='')
    
    # this is for levt
    parser.add_argument('--_name', default='levenshtein_transformer')
    parser.add_argument('--activation_dropout', default=0.0)
    parser.add_argument('--activation_fn', default='relu')
    parser.add_argument('--adam_betas', default='(0.9,0.98)')
    parser.add_argument('--adam_eps', default=1e-08)
    parser.add_argument('--adaptive_input', default=False)
    parser.add_argument('--adaptive_softmax_cutoff', default=None)
    parser.add_argument('--adaptive_softmax_dropout', default=0)
    parser.add_argument('--all_gather_list_size', default=16384)
    parser.add_argument('--amp', default=False)
    parser.add_argument('--amp_batch_retries', default=2)
    parser.add_argument('--amp_init_scale', default=128)
    parser.add_argument('--amp_scale_window', default=None)
    parser.add_argument('--apply_bert_init', default=True)
    parser.add_argument('--arch', default='levenshtein_transformer')
    parser.add_argument('--attention_dropout', default=0.0)
    parser.add_argument('--azureml_logging', default=False)
    parser.add_argument('--batch_size_valid', default=None)
    parser.add_argument('--best_checkpoint_metric', default='loss')
    parser.add_argument('--bf16', default=False)
    parser.add_argument('--bpe', default=None)
    parser.add_argument('--broadcast_buffers', default=False)
    parser.add_argument('--bucket_cap_mb', default=25)
    parser.add_argument('--checkpoint_activations', default=False)
    parser.add_argument('--checkpoint_shard_count', default=1)
    parser.add_argument('--checkpoint_suffix', default='')
    parser.add_argument('--clip_norm', default=0.0)
    parser.add_argument('--combine_valid_subsets', default=None)
    parser.add_argument('--cpu', default=False)
    parser.add_argument('--cpu_offload', default=False)
    parser.add_argument('--criterion', default='nat_loss')
    parser.add_argument('--cross_self_attention', default=False)
    parser.add_argument('--curriculum', default=0)
    parser.add_argument('--ddp_backend', default='legacy_ddp')
    parser.add_argument('--ddp_comm_hook', default='none')
    parser.add_argument('--decoder_attention_heads', default=8)
    parser.add_argument('--decoder_embed_dim', default=512)
    parser.add_argument('--decoder_embed_path', default=None)
    parser.add_argument('--decoder_ffn_embed_dim', default=2048)
    parser.add_argument('--decoder_input_dim', default=512)
    parser.add_argument('--decoder_layerdrop', default=0)
    parser.add_argument('--decoder_layers', default=6)
    parser.add_argument('--decoder_layers_to_keep', default=None)
    parser.add_argument('--decoder_learned_pos', default=True)
    parser.add_argument('--decoder_normalize_before', default=False)
    parser.add_argument('--decoder_output_dim', default=512)
    parser.add_argument('--device_id', default=0)
    parser.add_argument('--disable_validation', default=False)
    parser.add_argument('--distributed_backend', default='nccl')
    parser.add_argument('--distributed_init_method', default=None)
    parser.add_argument('--distributed_no_spawn', default=False)
    parser.add_argument('--distributed_num_procs', default=1)
    parser.add_argument('--distributed_port', default=-1)
    parser.add_argument('--distributed_rank', default=0)
    parser.add_argument('--distributed_world_size', default=1)
    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--early_exit', default='6,6,6')
    parser.add_argument('--empty_cache_freq', default=0)
    parser.add_argument('--encoder_attention_heads', default=8)
    parser.add_argument('--encoder_embed_dim', default=512)
    parser.add_argument('--encoder_ffn_embed_dim', default=2048)
    parser.add_argument('--encoder_layerdrop', default=0)
    parser.add_argument('--encoder_layers', default=6)
    parser.add_argument('--encoder_learned_pos', default=True)
    parser.add_argument('--encoder_normalize_before', default=False)
    parser.add_argument('--eos', default=2)
    parser.add_argument('--eval_bleu', default=False)
    parser.add_argument('--eval_bleu_args', default='{}')
    parser.add_argument('--eval_bleu_detok', default='space')
    parser.add_argument('--eval_bleu_detok_args', default='{}')
    parser.add_argument('--eval_bleu_print_samples', default=False)
    parser.add_argument('--eval_bleu_remove_bpe', default=None)
    parser.add_argument('--eval_tokenized_bleu', default=False)
    parser.add_argument('--fast_stat_sync', default=False)
    parser.add_argument('--find_unused_parameters', default=False)
    parser.add_argument('--finetune_from_model', default=None)
    parser.add_argument('--fix_batches_to_gpus', default=False)
    parser.add_argument('--fixed_validation_seed', default=7)
    parser.add_argument('--fp16', default=False)
    parser.add_argument('--fp16_init_scale', default=128)
    parser.add_argument('--fp16_no_flatten_grads', default=False)
    parser.add_argument('--fp16_scale_tolerance', default=0.0)
    parser.add_argument('--fp16_scale_window', default=None)
    parser.add_argument('--fp32_reduce_scatter', default=False)
    parser.add_argument('--gen_subset', default='test')
    parser.add_argument('--heartbeat_timeout', default=-1)
    parser.add_argument('--ignore_unused_valid_subsets', default=False)
    parser.add_argument('--keep_best_checkpoints', default=-1)
    parser.add_argument('--keep_interval_updates', default=-1)
    parser.add_argument('--keep_interval_updates_pattern', default=-1)
    parser.add_argument('--keep_last_epochs', default=-1)
    parser.add_argument('--label_smoothing', default=0.1)
    parser.add_argument('--layernorm_embedding', default=False)
    parser.add_argument('--left_pad_source', default=True)
    parser.add_argument('--left_pad_target', default=False)
    parser.add_argument('--load_alignments', default=False)
    parser.add_argument('--load_checkpoint_on_all_dp_ranks', default=False)
    parser.add_argument('--localsgd_frequency', default=3)
    parser.add_argument('--log_format', default='simple')
    parser.add_argument('--log_interval', default=100)
    parser.add_argument('--lr_scheduler', default='inverse_sqrt')
    parser.add_argument('--max_epoch', default=0)
    parser.add_argument('--max_source_positions', default=1024)
    parser.add_argument('--max_target_positions', default=1024)
    parser.add_argument('--max_tokens', default=8000)
    parser.add_argument('--max_tokens_valid', default=8000)
    parser.add_argument('--max_update', default=300000)
    parser.add_argument('--max_valid_steps', default=None)
    parser.add_argument('--maximize_best_checkpoint_metric', default=False)
    parser.add_argument('--memory_efficient_bf16', default=False)
    parser.add_argument('--memory_efficient_fp16', default=False)
    parser.add_argument('--min_loss_scale', default=0.0001)
    parser.add_argument('--min_params_to_wrap', default=100000000)
    parser.add_argument('--model_parallel_size', default=1)
    parser.add_argument('--no_cross_attention', default=False)
    parser.add_argument('--no_epoch_checkpoints', default=False)
    parser.add_argument('--no_last_checkpoints', default=False)
    parser.add_argument('--no_progress_bar', default=False)
    parser.add_argument('--no_reshard_after_forward', default=False)
    parser.add_argument('--no_save', default=False)
    parser.add_argument('--no_save_optimizer_state', default=False)
    parser.add_argument('--no_scale_embedding', default=False)
    parser.add_argument('--no_seed_provided', default=False)
    parser.add_argument('--no_share_discriminator', default=False)
    parser.add_argument('--no_share_last_layer', default=False)
    parser.add_argument('--no_share_maskpredictor', default=False)
    parser.add_argument('--no_token_positional_embeddings', default=False)
    parser.add_argument('--noise', default='random_delete')
    parser.add_argument('--nprocs_per_node', default=1)
    parser.add_argument('--num_batch_buckets', default=0)
    parser.add_argument('--num_shards', default=1)
    parser.add_argument('--num_workers', default=1)
    parser.add_argument('--offload_activations', default=False)
    parser.add_argument('--on_cpu_convert_precision', default=False)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--optimizer_overrides', default='{}')
    parser.add_argument('--pad', default=1)
    parser.add_argument('--patience', default=-1)
    parser.add_argument('--pipeline_balance', default=None)
    parser.add_argument('--pipeline_checkpoint', default='never')
    parser.add_argument('--pipeline_chunks', default=0)
    parser.add_argument('--pipeline_decoder_balance', default=None)
    parser.add_argument('--pipeline_decoder_devices', default=None)
    parser.add_argument('--pipeline_devices', default=None)
    parser.add_argument('--pipeline_encoder_balance', default=None)
    parser.add_argument('--pipeline_encoder_devices', default=None)
    parser.add_argument('--pipeline_model_parallel', default=False)
    parser.add_argument('--plasma_path', default='/tmp/plasma')
    parser.add_argument('--profile', default=False)
    parser.add_argument('--quant_noise_pq', default=0)
    parser.add_argument('--quant_noise_pq_block_size', default=8)
    parser.add_argument('--quant_noise_scalar', default=0)
    parser.add_argument('--quantization_config_path', default=None)
    parser.add_argument('--required_batch_size_multiple', default=8)
    parser.add_argument('--required_seq_len_multiple', default=1)
    parser.add_argument('--reset_dataloader', default=False)
    parser.add_argument('--reset_logging', default=False)
    parser.add_argument('--reset_lr_scheduler', default=False)
    parser.add_argument('--reset_meters', default=False)
    parser.add_argument('--reset_optimizer', default=False)
    parser.add_argument('--restore_file', default='checkpoint_last.pt')
    parser.add_argument('--sampling_for_deletion', default=False)
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--save_interval', default=1)
    parser.add_argument('--save_interval_updates', default=10000)
    parser.add_argument('--scoring', default='bleu')
    parser.add_argument('--seed', default=1)
    parser.add_argument('--sentence_avg', default=False)
    parser.add_argument('--shard_id', default=0)
    parser.add_argument('--share_decoder_input_output_embed', default=False)
    parser.add_argument('--share_discriminator_maskpredictor', default=False)
    parser.add_argument('--simul_type', default=None)
    parser.add_argument('--skip_invalid_size_inputs_valid_test', default=False)
    parser.add_argument('--slowmo_algorithm', default='LocalSGD')
    parser.add_argument('--slowmo_momentum', default=None)
    parser.add_argument('--stop_min_lr', default=1e-09)
    parser.add_argument('--stop_time_hours', default=0)
    parser.add_argument('--suppress_crashes', default=False)
    parser.add_argument('--task', default='translation_lev')
    parser.add_argument('--tensorboard_logdir', default=None)
    parser.add_argument('--threshold_loss_scale', default=None)
    parser.add_argument('--tie_adaptive_weights', default=False)
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--tpu', default=False)
    parser.add_argument('--train_subset', default='train')
    parser.add_argument('--truncate_source', default=False)
    parser.add_argument('--unk', default=3)
    parser.add_argument('--update_freq', default=[1])
    parser.add_argument('--upsample_primary', default=-1)
    parser.add_argument('--use_bmuf', default=False)
    parser.add_argument('--use_old_adam', default=False)
    parser.add_argument('--use_plasma_view', default=False)
    parser.add_argument('--use_sharded_state', default=False)
    parser.add_argument('--user_dir', default=None)
    parser.add_argument('--valid_subset', default='valid')
    parser.add_argument('--validate_after_updates', default=0)
    parser.add_argument('--validate_interval', default=1)
    parser.add_argument('--validate_interval_updates', default=0)
    parser.add_argument('--wandb_project', default=None)
    parser.add_argument('--warmup_init_lr', default=1e-07)
    parser.add_argument('--warmup_updates', default=10000)
    parser.add_argument('--weight_decay', default=0.01)
    parser.add_argument('--write_checkpoints_asynchronously', default=False)
    parser.add_argument('--zero_sharding', default='none')    

    parser.add_argument('--train_ctc_only', default=False)
    parser.add_argument('--dataset_charset_path', default='charset/charset_36.txt')
    parser.add_argument('--ctc_model', default='') 
    parser.add_argument('--abi_model', default='') 
    parser.add_argument('--levt_model', default='') 
    parser.add_argument('--freeze_vision', action='store_true') 
    parser.add_argument('--freeze_levt', default=False) 
    parser.add_argument('--device', default='cuda:0') 
    parser.add_argument('--img_feature', action='store_true') 
    parser.add_argument('--ctc_max', action='store_true')
    parser.add_argument('--max_iter', default=2)
    parser.add_argument('--post_process', default='subword_nmt')
    parser.add_argument('--th', type=float, default=0.5)
    
    args = parser.parse_args()
    return args
