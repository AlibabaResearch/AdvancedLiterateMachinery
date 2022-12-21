import os
import sys
import time
import random
import string
import copy
import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from models import LevOCRModel, inject_noise
from eval import validation
from utils import get_args
import utils_dist as utils
from levt import utils as utils_levt
from abinet.utils import CharsetMapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    train_dataset = Batch_Balanced_Dataset(opt)
    log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')
    val_opt = copy.deepcopy(opt)
    val_opt.eval = True
    
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
    valid_dataset, _ = hierarchical_dataset(root=opt.valid_data, opt=val_opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    """ model configuration """
    converter = TokenLabelConverter(src_dict.indices)

    # data parallel for multi-GPU
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
    model.train()
    
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        model.load_state_dict(torch.load(opt.saved_model), strict=False)    
    # load vision part and levt part model
    elif opt.vis_model != '' and opt.levt_model != '':
        print('load parameter from visual model and levt model!!!')
        model.module.vision.load_state_dict(torch.load(opt.vis_model, map_location=device)['model'], strict=True) 
        new_state_dict = OrderedDict()
        state_dict = torch.load(opt.levt_model, map_location=device)
        for k, v in state_dict.items():
            name = k
            if k.startswith('module.'):
                name = k[7:] # remove `module.`
            if 'levt' in name:
                name = name[5:]
                new_state_dict[name] = v    
        model.module.levt.load_state_dict(new_state_dict, strict=False) 

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss().to(device)
        
    # loss averager
    loss_avg = Averager()
    v_loss_avg = Averager()
    lev_loss_avg = Averager()

    # setup optimizer
    optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    scheduler = CosineAnnealingLR(optimizer, T_max=int(opt.num_iter))

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
    start_time = time.time()
    best_accuracy = -1
    iteration = start_iter
            
    print("LR",scheduler.get_last_lr()[0])
        
    while(True):
        # train part
        image_tensors, labels, labels_noise = train_dataset.get_batch()
        image = image_tensors.to(device)
        tgt_vision, length = converter.encode_vision(labels, batch_max_length=opt.batch_max_length, device=device)

        text_levt, _ = converter.encode_levt(labels, src_dict, device=device, batch_max_length=opt.batch_max_length)
        tgt_tokens = text_levt.type(torch.LongTensor).to(device)

        text_input, _ = converter.encode_levt(labels_noise, src_dict, device=device, batch_max_length=opt.batch_max_length)
        if random.random() > 0.5:
            text_input = inject_noise(opt, text_input, src_dict).to(device)
        else:
            text_input = text_input
        
        loss_levt, loss_vison, vison_pred, preds, _ = model(image, text_input, tgt_vision, tgt_tokens, criterion)

        cost = loss_levt + loss_vison 
        
        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)
        v_loss_avg.add(loss_vison)
        lev_loss_avg.add(loss_levt)

        # validation part
        if utils.is_main_process() and ((iteration + 1) % opt.valInterval == 0 or iteration == 0): # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            with open(f'{opt.saved_path}/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                
                with torch.no_grad():
                    valid_loss, current_accuracys, char_preds, labels, infer_time, _, _ = validation(
                        model, criterion, valid_loader, converter, src_dict, opt)
                    char_accuracy = current_accuracys[0]
                model.train()

                loss_log = f'[{iteration+1}/{opt.num_iter}] LR: {scheduler.get_last_lr()[0]:0.5f}, Train loss: {loss_avg.val():0.5f}, Valid vision loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()
                current_model_log = f'{"char_accuracy":17s}: {char_accuracy:0.3f}'

                # keep best accuracy model (on valid dataset)
                if char_accuracy > best_accuracy:
                    best_accuracy = char_accuracy
                    torch.save(model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/best_accuracy.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred in zip(labels[:5], char_preds[:5]):
                    predicted_result_log += f'{gt:25s} | {pred:25s}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 5e+3 iter.
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

