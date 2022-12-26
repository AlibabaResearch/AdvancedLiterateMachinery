import os
import time
import string

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from utils import Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollateTest, RawDataset
from models import LevOCRModel
from utils import get_args
from levt import utils as utils_levt
from levt.dictionary import Dictionary
from abinet.utils import CharsetMapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    retain_history=True
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

        # collect finalized sentences
        finalized_tokens = decoder_out.output_tokens
        finalized_scores = decoder_out.output_scores
        finalized_attn = (
            None
            if (decoder_out.attn is None or decoder_out.attn.size(0) == 0)
            else decoder_out.attn
        )

        finalized_history_tokens = decoder_out.history
        for i in range(bsz):
            finalized[i] = [
                finalized_preds(
                    step,
                    finalized_tokens[i],
                    finalized_scores[i],
                    None if finalized_attn is None else finalized_attn[i],
                )
            ]

            finalized[i][0]["history"] = []
            for j in range(len(finalized_history_tokens)):
                finalized[i][0]["history"].append(
                    finalized_preds(
                        step, finalized_history_tokens[j][i], None, None
                    )
                )

        # for next step
        prev_decoder_out = decoder_out._replace(
            output_tokens=decoder_out.output_tokens,
            output_scores=decoder_out.output_scores,
            attn=decoder_out.attn
            if (decoder_out.attn is not None and decoder_out.attn.size(0) > 0)
            else None,
            history=decoder_out.history
            if decoder_out.history is not None
            else None,
        )
        prev_output_tokens = prev_decoder_out.output_tokens.clone()  

    return finalized


def test(opt):
    opt.eval = True
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

    AlignCollate_demo = AlignCollateTest(imgH=opt.imgH, imgW=opt.imgW)
    demo_data = RawDataset(root=opt.demo_imgs, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    """ evaluation """
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            out = model.module.vision(image)
            pred_logit = out['logits']
            features = out['features']
            pred_vision = F.log_softmax(pred_logit, dim=-1)
            pred_vision_max = pred_vision.max(2)[1]
            vision_preds_size = torch.IntTensor([pred_logit.size(1)] * batch_size)
            vision_preds_str = converter.decode(pred_vision_max, vision_preds_size, ignore_spec_char=True)
            vision_final_pred, _ = converter.encode_levt(vision_preds_str, src_dict, device=device, batch_max_length=pred_vision.size(1))

            img_feature_new = model.module.extract_img_feature(features)
            preds = generate(model, vision_final_pred, img_feature_new, batch_size, src_dict.pad(), max_iter=int(opt.max_iter))

            for i in range(batch_size):
                print('=======================================')
                print(image_path_list[i])
                vision_str = vision_preds_str[i]
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

                    # print history
                    print("Visual Prediction:", vision_str)

                    historys = hypo["history"]
                    his_result = []
                    for i,history in enumerate(historys):
                        step_ = history["steps"]
                        _, his_str, _ = utils_levt.post_process_prediction(
                            hypo_tokens=history["tokens"].int().cpu(),
                            src_str=vision_str,
                            alignment=history["alignment"],
                            align_dict=None,
                            tgt_dict=src_dict,
                            remove_bpe=opt.post_process,
                            extra_symbols_to_ignore={src_dict.eos()}
                        )
                        his_str = his_str.replace(" ", "")
                        his_result.append(his_str)

                    his_result_str = ""
                    iteration = 0
                    for index_ in range(0, len(his_result), 3):
                        iteration += 1
                        tmp = 'Iteration{}:Del-Plh-Ins {}-{}-{}'.format(str(iteration), his_result[index_], his_result[index_+1], his_result[index_+2])
                        if his_result_str == '':
                            his_result_str = tmp
                        else:
                            his_result_str += '\n' 
                            his_result_str += tmp
                    print(his_result_str)
                    

if __name__ == '__main__':
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    opt.saved_model = opt.model_dir
    test(opt)
