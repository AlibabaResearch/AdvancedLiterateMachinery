import os
import time
import string

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from utils import Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate
from models import LevOCRModel
from utils import get_args
from levt import utils as utils_levt
from levt.dictionary import Dictionary
from abinet.utils import CharsetMapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def benchmark_all_eval(model, criterion, converter, src_dict, opt):
    if opt.fast_acc:
    # # To easily compute the total accuracy of our paper.
        eval_data_list = ['IC13_857', 'SVT', 'IIIT5k_3000', 'IC15_1811', 'SVTP', 'CUTE80']
    else:
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                          'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    evaluation_batch_size = opt.batch_size

    char_list_accuracy = []
    vision_list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    char_total_correct_number = 0
    vision_total_correct_number = 0
    dashed_line = '-' * 80
    print(dashed_line)
    for eval_data in eval_data_list:

        eval_data_path = os.path.join(opt.eval_data, eval_data)
        print(eval_data_path)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)

        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracys, _, _, infer_time, length_of_data, accur_numbers = validation(
            model, criterion, evaluation_loader, converter, src_dict, opt)
        char_list_accuracy.append(f'{accuracys[0]:0.3f}')
        vision_list_accuracy.append(f'{accuracys[1]:0.3f}')

        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        char_total_correct_number += accur_numbers[0]
        vision_total_correct_number += accur_numbers[1]
        print(f'levocr_Acc {accuracys[0]:0.3f}\t vision_Acc {accuracys[1]:0.3f}\t')
        print(dashed_line)

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    char_total_accuracy = round(char_total_correct_number/total_evaluation_data_number*100,3)
    vision_total_accuracy = round(vision_total_correct_number/total_evaluation_data_number*100,3)
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: ' + '\n'
    evaluation_log += 'levocr_total_Acc:'+str(char_total_accuracy)+'\t' + 'vision_total_Acc:'+str(vision_total_accuracy)+'\t'+'th:'+str(opt.th)+'\n'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    return [char_total_accuracy, vision_total_accuracy]

def validation(model, criterion, evaluation_loader, converter, src_dict, opt):
    """ validation or evaluation """
    char_n_correct = 0
    vision_n_correct = 0

    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels, _) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        text, length = converter.encode_vision(labels, batch_max_length=opt.batch_max_length, device=device)
        start_time = time.time()
        
        forward_time = time.time() - start_time

        out = model.module.vision(image)
        pred_logit = out['logits']
        features = out['features']
        pred_vision = F.log_softmax(pred_logit, dim=-1)
        pred_vision_max = pred_vision.max(2)[1]

        cost = criterion(pred_vision.contiguous().view(-1, pred_vision.shape[-1]), text.contiguous().view(-1))

        vision_preds_size = torch.IntTensor([pred_logit.size(1)] * batch_size)
        vision_preds_str = converter.decode(pred_vision_max, vision_preds_size, ignore_spec_char=True)
        vision_final_pred, _ = converter.encode_levt(vision_preds_str, src_dict, device=device, batch_max_length=pred_vision.size(1))

        img_feature_new = model.module.extract_img_feature(features)
        
        preds = generate(model, vision_final_pred, img_feature_new, batch_size, src_dict.pad(), max_iter=int(opt.max_iter))

        char_preds_str = []
        for i in range(batch_size):
            vision_str = vision_preds_str[i]
            target_str = labels[i]
            for j, hypo in enumerate(preds[i]):
                hypo_tokens, hypo_str, alignment = utils_levt.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=vision_str,
                    alignment=hypo["alignment"],
                    align_dict=None,
                    tgt_dict=src_dict,
                    remove_bpe=opt.post_process,
                    extra_symbols_to_ignore={src_dict.eos()}
                )
                hypo_str = hypo_str.replace(" ", "")
                hypo_str = hypo_str.replace(",", "")
                char_preds_str.append(hypo_str)                
                if hypo_str == target_str:
                    char_n_correct += 1
                if vision_str == target_str:
                    vision_n_correct += 1
        infer_time += forward_time
        valid_loss_avg.add(cost)

    char_accuracy = char_n_correct/float(length_of_data) * 100
    vision_accuracy = vision_n_correct/float(length_of_data) * 100
    return valid_loss_avg.val(), [char_accuracy, vision_accuracy], char_preds_str, labels, infer_time, length_of_data, [char_n_correct, vision_n_correct]

def generate(
    model, 
    vision_final_pred, 
    img_feature, 
    batch_size, 
    pad, 
    eos_penalty=0.0, 
    max_iter=10,
    max_ratio=2,
    decoding_format=None,
):
    bsz = batch_size
    prev_decoder_out = model.module.levt.initialize_output_tokens(vision_final_pred)
    prev_output_tokens = prev_decoder_out.output_tokens.clone()
    sent_idxs = torch.arange(bsz)
    finalized = [[] for _ in range(bsz)]

    def finalized_preds(step, prev_out_token, prev_out_score, prev_out_attn):
        cutoff = prev_out_token.ne(pad)
        tokens = prev_out_token[cutoff]
        if prev_out_score is None:
            scores, score = None, None
        else:
            scores = prev_out_score[cutoff]
            score = scores.mean()

        if prev_out_attn is None:
            hypo_attn, alignment = None, None
        else:
            hypo_attn = prev_out_attn[cutoff]
            alignment = hypo_attn.max(dim=1)[1]
        return {
            "steps": step,
            "tokens": tokens,
            "positional_scores": scores,
            "score": score,
            "hypo_attn": hypo_attn,
            "alignment": alignment,
        }

    for step in range(max_iter + 1):
        decoder_options = {
            "eos_penalty": eos_penalty,
            "max_ratio": max_ratio,
            "decoding_format": decoding_format,
        }
        prev_decoder_out = prev_decoder_out._replace(
            step=step,
            max_step=max_iter + 1,
        )
        
        decoder_out = model.module.levt.forward_decoder(
            prev_decoder_out, img_feature, **decoder_options
        )

        # for next step
        prev_decoder_out = decoder_out._replace(
            output_tokens=decoder_out.output_tokens,
            output_scores=decoder_out.output_scores,
            attn=decoder_out.attn
            if (decoder_out.attn is not None and decoder_out.attn.size(0) > 0)
            else None,
        )
        prev_output_tokens = prev_decoder_out.output_tokens.clone()  

        if step == max_iter:  # reach last iteration
            # collect finalized sentences
            finalized_tokens = decoder_out.output_tokens
            finalized_scores = decoder_out.output_scores
            finalized_attn = (
                None
                if (decoder_out.attn is None or decoder_out.attn.size(0) == 0)
                else decoder_out.attn
            )
            for i in range(bsz):
                finalized[i] = [
                    finalized_preds(
                        step,
                        finalized_tokens[i],
                        finalized_scores[i],
                        None if finalized_attn is None else finalized_attn[i],
                    )
                ]
    return finalized


def test(opt):
    """ model configuration """
    charset = CharsetMapper(opt.dataset_charset_path, max_length=opt.batch_max_length)
    opt.num_class = charset.num_classes
    print('num_class:', opt.num_class)
    
    indices = charset.char_to_label
    src_dict = utils_levt.build_dict(indices)
    converter = TokenLabelConverter(src_dict.indices)
    opt.num_class = len(converter.character)
    
    if opt.rgb:
        opt.input_channel = 3
    model = LevOCRModel(opt, src_dict)
    model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss().to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    opt.eval = True
    with torch.no_grad():
        return benchmark_all_eval(model, criterion, converter, src_dict, opt)

if __name__ == '__main__':
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    from tabulate import tabulate
    if opt.range is not None:
        start_range, end_range = sorted([int(e) for e in opt.range.split('-')])
        print("eval range: ",start_range,end_range)
    
    if os.path.isdir(opt.model_dir):
        result = []
        model_list = os.listdir(opt.model_dir)
        model_list = [model for model in model_list if model.startswith('iter_')]
        model_list = sorted(model_list, key=lambda x: int(x.split('.')[0].split('_')[-1]), reverse=True)
        err_list = []
        for model in model_list:
            if opt.range is not None:
                num_epoch = int(str(model).split('_')[1].split('.')[0])
                if not (num_epoch>=start_range and num_epoch <=end_range):
                    continue
            opt.saved_model = os.path.join(opt.model_dir, model)
            result.append(test(opt)+[opt.saved_model])
            print('opt.model_path :', opt.saved_model)
        tab_title = ['levocr_acc', 'model']
        result = sorted(result, key=lambda x: x[0], reverse=True)
        print(tabulate(result, tab_title, numalign='right'))
    else:
        for th in range(int(opt.th*100), 51, 1):
            opt.th = th/100
            opt.saved_model = opt.model_dir
            test(opt)
