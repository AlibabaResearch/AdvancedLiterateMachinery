# Copyright (2023) Alibaba Group and its affiliates

""" multi-scale ensemble
"""
# nums = [3000, 647, 857, 1811, 645, 288]# IIIT5K, SVT, IC13, IC15, SVTP, CUTE
nums = [4800] # TUL

###
res_fn_candidates = [
    "data/ocr_res_lister_base_1.txt",
    "data/ocr_res_lister_base_2.txt",
    "data/ocr_res_lister_base_3.txt",
]
###


def parse_txt(fn):
    gts, preds, probs = [], [], []
    with open(fn, 'r') as f:
        for line in f:
            _, gt, pred, prob = line.strip('\n').split('\t')
            gts.append(gt)
            preds.append(pred)
            probs.append(prob)
    return gts, preds, probs

preds_mul, probs_mul = [], []
for fn in res_fn_candidates:
    gts, preds, probs = parse_txt(fn)
    preds_mul.append(preds)
    probs_mul.append(probs)

cor, total = 0, 0
cor_sub, total_sub = 0, 0
cnt = 0
for i, prob_mul in enumerate(zip(*probs_mul)):
    idx = prob_mul.index(max(prob_mul))
    pred_final = preds_mul[idx][i]
    total += 1
    total_sub += 1
    cnt += 1
    if pred_final == gts[i]:
        cor += 1
        cor_sub += 1
    if cnt == nums[0]:
        acc_sub = cor_sub / total_sub
        print(f"sub-> {acc_sub:.2%} ({total_sub})")
        cor_sub, total_sub, cnt = 0, 0, 0
        del nums[0]
acc = cor / total
print(f"total-> {acc:.2%} ({total})")
