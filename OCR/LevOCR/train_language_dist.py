import os
import sys
import time
import random
import string
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import Averager, TokenLabelConverter
from dataset import TextDataset
from models import LevOCRModel
from utils import get_args
import utils_dist as utils
from levt import utils as utils_levt
from abinet.utils import CharsetMapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def fix_bn(m):
#     classname = m.__class__.__name__
#     if classname.find("BatchNorm") !=-1:
#         m.eval()

def train(opt):
    """ character configuration """
    charset = CharsetMapper(opt.dataset_charset_path, max_length=opt.batch_max_length)
    opt.num_class = charset.num_classes
    print('num_class:', opt.num_class)

    indices = charset.char_to_label
    src_dict = utils_levt.build_dict(indices)

    if opt.rgb:
        opt.input_channel = 3

    model = LevOCRModel(opt, src_dict)
    print(model)

    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.eval = False
    train_dataset = TextDataset(opt.train_data, opt=opt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True, 
        num_workers=int(opt.workers), pin_memory=True, drop_last=True)

    log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')
    
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    """ model configuration """
    converter = TokenLabelConverter(src_dict.indices)

    # data parallel for multi-GPU
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
    model.train()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss().to(device)  # ignore [GO] token = ignore index 0
        
    # loss averager
    loss_avg = Averager()

    # setup optimizer
    optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    scheduler = CosineAnnealingLR(optimizer, T_max=int(opt.num_iter))
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000000)

    """ final options """
    # print(opt)
    with open(f'{opt.saved_path}/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        #print(opt_log)
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)

    """ start training """
    start_iter = 0
    iteration = start_iter  
    while(True):
        # train part
        for labels, labels_noise in train_loader:
            tgt_tokens, _ = converter.encode_levt(labels, src_dict, device=device, batch_max_length=opt.batch_max_length)
            text_levt_noise, _ = converter.encode_levt(labels_noise, src_dict, device=device, batch_max_length=opt.batch_max_length)

            loss_levt, _, _, preds, logging_output = model(None, text_levt_noise, None, tgt_tokens, criterion)
            cost = loss_levt 

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)

            if utils.is_main_process() and ((iteration + 1) % opt.valInterval == 0 or iteration == 0): # To see training progress, we also conduct validation when 'iteration == 0' 
                # for log
                with open(f'{opt.saved_path}/{opt.exp_name}/log_train.txt', 'a') as log:
                    loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}'
                    loss_avg.reset()
                    print(loss_log)
                    log.write(loss_log + '\n')

            # save model per 1e+5 iter.
            if utils.is_main_process() and (iteration + 1) % 5e+3 == 0:
                torch.save(
                    model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/iter_{iteration+1}.pth')

            if (iteration + 1) == opt.num_iter:
                print('end the training')
                sys.exit()
            iteration += 1
            if scheduler is not None:
                scheduler.step()

if __name__ == '__main__':

    opt = get_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}' if opt.Transformer else f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'

    opt.exp_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'{opt.saved_path}/{opt.exp_name}', exist_ok=True)


    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    
    utils.init_distributed_mode(opt)

    print(opt)
    
    """ Seed and GPU setting """
    
    seed = opt.manualSeed + utils.get_rank()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    train(opt)

