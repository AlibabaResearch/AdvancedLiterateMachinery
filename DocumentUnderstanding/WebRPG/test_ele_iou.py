from tqdm import tqdm 
import os
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
import argparse

def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def compute_iou(box_1, box_2, epsilon=1e-10):
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, torch.Tensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max),
                   lib.zeros_like(a1[0]))

    au = a1 + a2 - ai

    # Ensure that au is not zero
    au = lib.where(au > 0, au, epsilon)
    
    iou = ai / au
    iou = lib.clip(iou, 0, 1) 

    return iou

def __compute_elements_iou(layout_1, layout_2):
    score = 0.
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)

    for idx in range(N):
        if li[idx] == lj[idx]:
            iou = compute_iou(bi[idx].reshape(1, 4), bj[idx].reshape(1, 4))
            score += iou

    return score / N if N != 0 else 0


def compute_elements_iou(layouts_1, layouts_2):
    assert len(layouts_1) == len(layouts_2), "The two layout lists must have the same length"
    
    scores = [
        __compute_elements_iou(layouts_1[i], layouts_2[i])
        for i in tqdm(range(len(layouts_1)))
    ]

    return np.mean(scores).item()

def process_file(args):
    file, pt_dir = args
    res = torch.load(os.path.join(pt_dir, file))
    
    element_num = np.sum(res["element_mask"])
    layout_pred = res["pred"][:element_num, :4]
    layout_gt = res["gt"][:element_num, :4]

    layout_label = np.zeros(128)
    layout_label = res["unique_tids"]
    layout_label = layout_label[:element_num]

    return file,(layout_pred, layout_label), (layout_gt, layout_label)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--pt_dir', type=str, required=True, help='The directory containing the files to be processed.')
    
    args = parser.parse_args()
    pt_dir = args.pt_dir

    file_lis = os.listdir(pt_dir)
    
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, [(f, pt_dir) for f in file_lis]), total=len(file_lis)))

    files, pred_layouts, gt_layouts = zip(*results)

    print("Elements IoU: ", compute_elements_iou(pred_layouts, gt_layouts))


    
    
