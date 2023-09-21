# Copyright (2023) Alibaba Group and its affiliates
import sys
import Levenshtein
import re
from collections import defaultdict

assert len(sys.argv) == 2
res_fp = sys.argv[1]
# res_fp = 'data/ocr_res.txt'

total_dict = defaultdict(int)
correct_dict = defaultdict(int)
cr_dict = defaultdict(float)
total = 0
correct = 0
cr = 0.0


def equal(s1, s2):
    s1 = s1.lower()
    s1 = re.sub(r'[^0-9a-z]', '', s1)
    s2 = s2.lower()
    s2 = re.sub(r'[^0-9a-z]', '', s2)
    return s1 == s2

with open(res_fp, 'r') as f:
    for line in f:
        _, gt, pred, *_ = line.strip('\n').split('\t')
        # if len(gt) > 31: continue
        nned = 1 - Levenshtein.distance(gt, pred) / max(len(gt), len(pred))
        total += 1
        cr += nned
        total_dict[len(gt)] += 1
        cr_dict[len(gt)] += nned
        if equal(gt, pred):
            correct += 1
            correct_dict[len(gt)] += 1

acc_all = float(correct) / total
cr_all = cr / total
print(f'acc_all: {acc_all:.2%} cr_all: {cr_all:.2%}')

len_list = []
acc_i_list = []
cr_i_list = []
cnt_list = []
for i in range(1, 60):
    if total_dict[i] == 0: continue
    len_list.append(i)
    cnt_list.append(total_dict[i])
    acc_i = float(correct_dict[i]) / total_dict[i]
    acc_i_list.append(acc_i)
    cr_i = cr_dict[i] / total_dict[i]
    cr_i_list.append(cr_i)
    print(f'{i} -> cnt: {total_dict[i]} acc: {acc_i:.2f} cr: {cr_i:.2f}')

print("The following len is length-wise")
for le in len_list:
    print(f'{le}', end=',')
print("avg/total")

print("The following cnt is length-wise and the total:")
for cnt in cnt_list:
    print(f'{cnt}', end=',')
print(sum(cnt_list))

print("The following acc. is length-wise and the total:")
for acc_i in acc_i_list:
    print(f'{acc_i:.4f}', end=',')
print(f"{acc_all:.4f}")

print("\nThe following cr. is length-wise and the total:")
for cr_i in cr_i_list:
    print(f'{cr_i:.4f}', end=',')
print(f"{cr_all:.4f}")

# acc_seen = sum(acc_i_list[:15]) / 15
# acc_unseen = sum(acc_i_list[15:]) / 9
# print(f'acc_seen: {acc_seen:.1%}; acc_unseen: {acc_unseen:.1%}')
