import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np
from utils.misc import decode_seq
from utils.visualize import visualize_decoded_result


@torch.no_grad()
def validate(model, dataloader, epoch, args):
    model.eval()
    device = torch.device('cuda')
    output_folder = os.path.join(args.output_folder, 'results', f'ep{epoch:03d}')

    results = []
    step = 0
    for samples, targets in tqdm(dataloader):

        step += 1
        assert(len(targets) == 1) # Only support inference with batch size = 1
        samples = samples.to(device)
 
        if args.use_char_window_prompt:
            pt_seq = torch.tensor([[0, 0, args.num_bins - 1, args.num_bins - 1, args.num_bins, args.num_bins + len(args.chars), args.pt_sos_index]], dtype=torch.long).to(device)
        else:
            pt_seq = torch.tensor([[0, 0, args.num_bins - 1, args.num_bins - 1, args.pt_sos_index]], dtype=torch.long).to(device)

        poly_seq = torch.ones(1, 1, dtype=torch.long).to(device) * args.poly_sos_index
        rec_seq = torch.ones(1, 1, dtype=torch.long).to(device) * args.rec_sos_index

        seqs = [pt_seq, poly_seq, rec_seq, targets[0]['orig_size']]

        output = model(samples, seqs)
        if not output:
            continue

        if args.vie_categories > 0:
            json_path = os.path.join(output_folder, targets[0]['file_name'] +'.json')
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(output, f)
        else:
            pred_seqs, probs = output

            pred_seqs = [pred_seq[0].cpu() for pred_seq in pred_seqs]
            probs = probs[0].cpu()
            
            result = decode_pred_seq(pred_seqs, probs, targets[0], args)
            results.extend(result)

            if args.visualize:
                image = cv2.imread(os.path.join(targets[0]['image_folder'], targets[0]['file_name']))
                if 'polygons' in targets[0].keys():
                    image = visualize_decoded_result(image, result, targets[0]['polygons'])
                else:
                    image = visualize_decoded_result(image, result, None)
                save_path = os.path.join(output_folder, 'vis', targets[0]['file_name'])
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, image)

    if args.vie_categories == 0:
        json_path = os.path.join(output_folder, targets[0]['dataset_name']+'.json')
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            f.write(json.dumps(results, indent=4))


def decode_pred_seq(index_seqs, prob_seqs, target, args):

    pt_index_seq = index_seqs[0]
    if len(pt_index_seq) % 2 != 0:
        pt_index_seq = pt_index_seq[:-len(pt_index_seq)%2]
    
    poly_index_seq = index_seqs[1]
    rec_index_seq = index_seqs[2]

    pt_decode_results = decode_seq(pt_index_seq, args, 'pt', 'none')
    poly_decode_results = decode_seq(poly_index_seq, args, 'poly', 'none')
    rec_decode_results, confs = decode_seq(rec_index_seq, args, 'rec', prob_seqs)

    image_id = target['file_name']
    image_h, image_w = target['orig_size']
    results = []
    for pt_decode_result, poly_decode_result, rec_decode_result, conf in zip(pt_decode_results, poly_decode_results, rec_decode_results, confs):
        point_x = pt_decode_result['point'][0] * image_w 
        point_y = pt_decode_result['point'][1] * image_h 
        poly_decode_result = poly_decode_result['polygon'] * torch.tensor([image_w, image_h] * 16)
        rec = rec_decode_result['rec']
        result = {
            'image_id': image_id,
            'pts': [[point_x.item(), point_y.item()]],
            'score': conf,
            'polys': poly_decode_result.reshape(-1,2).tolist(),
            'rec': rec,
        }
        results.append(result)
    
    return results
