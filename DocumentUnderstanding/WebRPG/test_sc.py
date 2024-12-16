from typing import List, Set
from tqdm import tqdm 
import torch
import os
from multiprocessing import Pool
import argparse
import numpy as np

def jaccard_similarity(set_a: Set[int], set_b: Set[int]) -> float:
    """
    Compute the Jaccard similarity between two sets.
    """
    intersection_len = len(set_a.intersection(set_b))
    union_len = len(set_a.union(set_b))
    return intersection_len / union_len if union_len != 0 else 0

def weighted_sc_single(original_set: List[Set[int]], generated_set: List[Set[int]]) -> float:
    total_weighted_similarity = 0
    total_elements = sum(len(s) for s in original_set)  # Total number of elements in all original sets

    for s_original in original_set:
        max_similarity = 0
        set_weight = len(s_original) / total_elements  # Weight of the set based on its size
        for s_generated in generated_set:
            similarity = jaccard_similarity(s_original, s_generated)
            max_similarity = max(max_similarity, similarity)
        total_weighted_similarity += max_similarity * set_weight  # Multiply similarity by the set's weight

    return total_weighted_similarity

def weighted_sc_overall(original_pages: List[List[Set[int]]], generated_pages: List[List[Set[int]]]) -> float:
    total_weighted_score = 0
    num_pages = len(original_pages)
    # all_sc = []
    
    for i in tqdm(range(num_pages)):
        sc_single = weighted_sc_single(original_pages[i], generated_pages[i])
        total_weighted_score += sc_single

    return total_weighted_score / num_pages


def get_unique_indices(result,mask):
    unique_sets = []
    indices = []
    result = result[:np.sum(mask)]
    
    for i, row in enumerate(result):
        row_set = set(row)
        if row_set in unique_sets:
            index = unique_sets.index(row_set)
            indices[index].add(i)
        else:
            unique_sets.append(row_set)
            indices.append(set([i]))
            
    return indices


def process_file(args):
    file, pt_dir = args
    res = torch.load(os.path.join(pt_dir, file))
    return get_unique_indices(res["gt"][:,4:],res["element_mask"]), get_unique_indices(res["pred"][:,4:],res["element_mask"]),file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pt_dir argument.')
    parser.add_argument('--pt_dir', type=str, required=True, help='The directory containing the files to be processed.')
    
    args = parser.parse_args()
    pt_dir = args.pt_dir

    file_lis = os.listdir(pt_dir)
    gt_style = []
    pred_style = []

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_file, [(f, pt_dir) for f in file_lis]), total=len(file_lis)))

    gt_style = [res[0] for res in tqdm(results)]
    pred_style = [res[1] for res in tqdm(results)]

    print("Style Consistency Score",weighted_sc_overall(gt_style,pred_style))