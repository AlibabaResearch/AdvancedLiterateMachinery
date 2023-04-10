from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import argparse

import numpy as np
from tqdm import tqdm

from lib.utils.eval_utils import coco_into_labels, Table, pairTab

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type = str)
    parser.add_argument('--predict_dir', type = str)
    args = parser.parse_args()

    coco_into_labels(args.dataset_dir, args.predict_dir)

    gt_bbox_path = os.path.join(args.predict_dir, 'gt_center')
    gt_logi_path = os.path.join(args.predict_dir, 'gt_logi')
    bbox_path = os.path.join(args.predict_dir, 'center')
    logi_path = os.path.join(args.predict_dir, 'logi')

    table_dict = []
    for file_name in tqdm(os.listdir(os.path.join(args.predict_dir, 'gt_center'))):
        if 'txt' in file_name:
            pred_table = Table(bbox_path, logi_path, file_name)
            gt_table = Table(gt_bbox_path, gt_logi_path, file_name)
            table_dict.append({'file_name': file_name, 'pred_table': pred_table, 'gt_table': gt_table})
    
    acs = []
    bbox_recalls = []
    bbox_precisions = []
    for i in tqdm(range(len(table_dict))):
        pair = pairTab(table_dict[i]['pred_table'], table_dict[i]['gt_table'])
        #Acc of Logical Locations
        ac = pair.evalAxis()
        if ac != 'null':
            acs.append(ac)

        #Recall of Cell Detection 
        # recall = pair.evalBbox('recall')
        # bbox_recalls.append(recall)
        
        # #Precision of Cell Detection 
        # precision = pair.evalBbox('precision')
        # bbox_precisions.append(precision)
    
    # det_precision =  np.array(bbox_precisions).mean()
    # det_recall =  np.array(bbox_recalls).mean()
    # f = 2 * det_precision * det_recall / (det_precision + det_recall)

    print('Evaluation Results | Accuracy of Logical Location: {:.2f}.'.format(np.array(acs).mean()))