import json
import os
import pandas as pd
import pyarrow as pa
import scipy.io as sio
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict


dataset_root = './data'

gts = sio.loadmat(f"{dataset_root}/SynthText/gt.mat")
img_names = gts['imnames'][0]

bs = []
for i in tqdm(range(len(img_names))):
    if i > 1000:
        break
    img_name = img_names[i][0]
    
    word_ann = []
    txt_ann = gts['txt'][0][i]

    for j in range(len(txt_ann)):
        bbox_ann = txt_ann[j].split('\n')
        for k in range(len(bbox_ann)):
            word_ann.extend(bbox_ann[k].strip().split(' '))
    
    word_ann = ' '.join(word_ann)

    # 4% for val
    if random.uniform(0,1) > 0.04:
        split = 'train'
    else:
        split = 'val'

    with open(f"{dataset_root}/SynthText/{img_name}", "rb") as fp:
        binary = fp.read()
    
    bs.append([binary, [word_ann], img_name, split])

for split in ["train", "val"]:
    batches = [b for b in bs if b[-1] == split]

    dataframe = pd.DataFrame(
        batches, columns=["image", "caption", "image_id", "split"],
    )

    table = pa.Table.from_pandas(dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
        f"{dataset_root}/synthtext_{split}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
