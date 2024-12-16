import os
import random
import json
import numpy as np
import torch
import h5py
import collections
import math
from tqdm import tqdm
from bs4 import BeautifulSoup, Comment
import json
from markuplm.tokenization_markuplm import MarkupLMTokenizer
import argparse
from multiprocessing import Pool
from css_utils.utils import *
from css_utils.color_process import *
import argparse

property_list = [
    "left",
    "top",
    "width",
    "height",
    "font-style",
    "font-weight",
    "font-size",
    "line-height",
    "color",
    "text-align",
    "text-decoration",
    "text-transform",
    "background-color",
]


tokenizer = MarkupLMTokenizer.from_pretrained("markuplm-large")
CLS_TOKEN = tokenizer.cls_token
SEP_TOKEN = tokenizer.sep_token
MASK_TOKEN = tokenizer.mask_token
PAD_TOKEN = tokenizer.pad_token

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch_size",
    type=int,
    default=2000,
    help="Batch size for processing data in groups.",
)

parser.add_argument(
    "--random_seed", type=int, default=12345, help="Random seed for reproducibility."
)

parser.add_argument("--num_process", type=int, default=50, help="Number of processes.")

parser.add_argument("--pad_id", type=int, default=1992, help="Padding token ID.")

parser.add_argument(
    "--max_elements_length", type=int, default=128, help="Maximum number of elements."
)

parser.add_argument(
    "--meta_data_len", type=int, default=13, help="Number of rendering parameters."
)

parser.add_argument(
    "--token_2_index_path",
    type=str,
    default="./token_2_index.json",
    help="Path to the token_2_index file.",
)

parser.add_argument(
    "--output_dir", type=str, default="./output", help="Directory to save the output."
)

parser.add_argument(
    "--root_dir",
    type=str,
    default="./root",
    help="Root directory containing the files listed in file_json.",
)

parser.add_argument(
    "--file_json",
    type=str,
    default="./text2webpage_0405/klarna_data/klerna_filter_LC_file_lis_train_1002.json",
    help="Path to the JSON file containing the file list.",
)

parser.add_argument(
    "--start", default=0, type=int, help="Starting index of the data group to process"
)

parser.add_argument(
    "--end", default=1, type=int, help="Ending index of the data group to process"
)

# Markuplm-related arguments
parser.add_argument(
    "--max_seq_length", type=int, default=512, help="Maximum sequence length."
)

parser.add_argument(
    "--max_title_length", type=int, default=64, help="Maximum title length."
)

parser.add_argument(
    "--max_depth", type=int, default=50, help="Maximum depth of the DOM tree."
)

FLAGS = parser.parse_args()


data_config = {
    "min_layout": 0,
    "max_layout": 1920,
    "min_color": 0,
    "max_color": 45,
    "min_font_size": 0,
    "max_font_size": 32,
    "min_line_height": 0,
    "max_line_height": 50,
}


class TrainingExample(object):
    def __init__(
        self,
        doc_tokens=None,
        file_id=None,
        offset=None,
        tag_num=None,
        html_code=None,
        tok_to_orig_index=None,
        orig_to_tok_index=None,
        all_doc_tokens=None,
        tok_to_tags_index=None,
        xpath_tag_map=None,
        xpath_subs_map=None,
        html_title=None,
        e_id_to_text_dict=None,
        e_id_to_t_id_dict=None,
        e_id_start_end_dict=None,
        meta_data_map=None,
        processed_meta_data_map=None,
        unique_tids=None,
    ):
        self.doc_tokens = doc_tokens
        self.file_id = file_id
        self.offset = offset
        self.tag_num = tag_num
        self.html_code = html_code
        self.tok_to_orig_index = tok_to_orig_index
        self.orig_to_tok_index = orig_to_tok_index
        self.all_doc_tokens = all_doc_tokens
        self.tok_to_tags_index = tok_to_tags_index
        self.xpath_tag_map = xpath_tag_map
        self.xpath_subs_map = xpath_subs_map
        self.html_title = html_title
        self.e_id_to_text_dict = e_id_to_text_dict
        self.e_id_to_t_id_dict = e_id_to_t_id_dict
        self.e_id_start_end_dict = e_id_start_end_dict
        self.meta_data_map = meta_data_map
        self.processed_meta_data_map = processed_meta_data_map
        self.unique_tids = unique_tids

    def __str__(self):
        return self.__repr__()


def process_meta_data_map(meta_data_map, unique_tids, token_2_index):
    processed_mete_data_map = {}
    for tid in unique_tids:
        tag_style = meta_data_map[tid]
        tmp_meta_data = []
        for key in property_list:
            value = tag_style[key]
            value = value[0]
            if key in ["left", "top", "width", "height"]:
                value = float(value[:-2])
                value = math.floor(value)
                value = max(
                    data_config["min_layout"], min(value, data_config["max_layout"])
                )
                value = f"{value}px"
                tmp_meta_data.append(value)
            elif key in ["font-size"]:
                value = float(value[:-2])
                value = math.floor(value)
                value = max(
                    data_config["min_font_size"],
                    min(value, data_config["max_font_size"]),
                )
                value = f"{value}px"
                tmp_meta_data.append(value)
            elif key in ["line-height"]:
                if value != "normal":
                    value = float(value[:-2])
                    value = math.floor(value)
                    value = max(
                        data_config["min_line_height"],
                        min(value, data_config["max_line_height"]),
                    )
                    value = f"{value}px"
                else:
                    value = f"{value}-line-height"
                tmp_meta_data.append(value)
            elif key in ["color", "background-color"]:
                idx = procee_color(value)
                tmp_meta_data.append(f"{idx}-color")
            elif key in ["text-decoration"]:
                value = value.split(" ")[0]
                tmp_meta_data.append(f"{value}-text-decoration")
            else:
                tmp_meta_data.append(f"{value}-{key}")
        processed_mete_data_map[tid] = tmp_meta_data
    PAD_index = token_2_index["PAD"]
    for key in processed_mete_data_map.keys():
        for i, value in enumerate(processed_mete_data_map[key]):
            processed_mete_data_map[key][i] = token_2_index.get(value, PAD_index)
    return processed_mete_data_map


def create_training_examples(
    input_file,
    tokenizer,
    token_2_index,
    max_depth=50,
    max_title_length=64,
    max_elements_length=512,
):
    file_id = input_file.split("_")[-2]
    offset = input_file.split(".")[0].split("_")[-1]

    html_fn = os.path.join(FLAGS.root_dir, "css_mod_{}_{}.html".format(file_id, offset))
    html_file = open(html_fn).read()
    html_code = BeautifulSoup(str(html_file), "html.parser")

    json_fn = os.path.join(
        FLAGS.root_dir, "css_dict_{}_{}.json".format(file_id, offset)
    )
    with open(json_fn, "r", encoding="utf-8") as reader:
        metadata_dict = json.load(reader)

    if html_code.head.title != None:
        title = html_code.head.title.get_text()
        tokenize_title = []
        for a in title.strip().split(" "):
            tokenize_title += tokenizer.tokenize(a)
        tokenize_title = tokenize_title[:max_title_length]
    else:
        tokenize_title = ""

    html_code = BeautifulSoup(
        '<html class="cls-1">{} </html>'.format(str(html_code.body)), "html.parser"
    )

    ele_lis = get_all_element_lis(html_code)
    ele_lis = get_all_element_lis(html_code)

    raw_text_list, tag_num = html_to_text_list(html_code, ele_lis)

    e_id_to_text_dict = get_e_id_to_text_dict(html_code, ele_lis)
    e_id_to_t_id_dict = get_e_id_to_t_id_dict(html_code, ele_lis)

    doc_tokens = []

    for page_text in raw_text_list:
        doc_tokens.extend(get_doc_tokens(page_text))

    tag_list = []
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    for i, token in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if token in tag_list:
            sub_tokens = [token]
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_to_tags_index, unique_tids, e_id_start_end_dict = subtoken_tag_offset(
        html_code, tok_to_orig_index, orig_to_tok_index, ele_lis
    )

    elements = get_elements(str(html_code))

    if len(elements) > max_elements_length:
        return None
    unique_tids = []
    for element in elements:
        unique_tids.append(element["class"][0])

    xpath_tag_map, xpath_subs_map = get_xpath_and_treeid4tokens(
        html_code, unique_tids, max_depth=max_depth
    )

    processed_meta_data_map = process_meta_data_map(
        metadata_dict, unique_tids, token_2_index
    )

    example = TrainingExample(
        doc_tokens=doc_tokens,
        file_id=file_id,
        offset=offset,
        tag_num=tag_num,
        html_code=str(html_code),
        tok_to_orig_index=tok_to_orig_index,
        orig_to_tok_index=orig_to_tok_index,
        all_doc_tokens=all_doc_tokens,
        tok_to_tags_index=tok_to_tags_index,
        xpath_tag_map=xpath_tag_map,
        xpath_subs_map=xpath_subs_map,
        html_title=tokenize_title,
        e_id_to_text_dict=e_id_to_text_dict,
        e_id_to_t_id_dict=e_id_to_t_id_dict,
        e_id_start_end_dict=e_id_start_end_dict,
        unique_tids=unique_tids,
        processed_meta_data_map=processed_meta_data_map,
    )

    return example


def write_instance_to_torch_example_file(
    instances,
    tokenizer,
    max_seq_length,
    max_title_length,
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    tag_pad=-1,
    mask_padding_with_zero=True,
    max_depth=50,
    max_elements_length=512,
    meta_data_len=17,
    element_pad=2202,
):
    num_instances = len(instances)
    pad_x_tag_seq = [216] * max_depth
    pad_x_subs_seq = [1001] * max_depth
    pad_meta_data_seq = [element_pad] * meta_data_len
    pad_bos = 0

    features = collections.OrderedDict()
    features["input_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["input_mask"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["segment_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")

    features["xpath_tags_seq"] = np.zeros(
        [num_instances, max_seq_length, max_depth], dtype="int32"
    )
    features["xpath_subs_seq"] = np.zeros(
        [num_instances, max_seq_length, max_depth], dtype="int32"
    )

    features["all_xpath_tags_seq"] = np.zeros(
        [num_instances, max_elements_length, max_depth], dtype="int32"
    )
    features["all_xpath_subs_seq"] = np.zeros(
        [num_instances, max_elements_length, max_depth], dtype="int32"
    )
    features["element_mask"] = np.zeros(
        [num_instances, max_elements_length], dtype="int32"
    )
    features["meta_data_seq"] = np.zeros(
        [num_instances, max_elements_length, meta_data_len], dtype="int32"
    )
    features["element_bos"] = np.zeros(
        [num_instances, max_elements_length, max_seq_length], dtype="int32"
    )
    features["element_text_len"] = np.zeros(
        [num_instances, max_elements_length], dtype="int32"
    )

    features["file_id"] = np.zeros([num_instances, 1], dtype="int32")
    features["offset"] = np.zeros([num_instances, 1], dtype="int32")
    features["unique_tids"] = np.zeros(
        [num_instances, max_elements_length], dtype="int32"
    )

    for inst_index, instance in tqdm(enumerate(instances)):

        xpath_tag_map = instance.xpath_tag_map
        xpath_subs_map = instance.xpath_subs_map

        title_tokens = instance.html_title
        if len(title_tokens) > max_title_length:
            title_tokens = title_tokens[0:max_title_length]

        max_tokens_for_doc = max_seq_length - len(title_tokens) - 3

        tokens = []
        segment_ids = []
        token_to_tag_index = []
        doc_start = -1

        tokens.append(CLS_TOKEN)
        segment_ids.append(cls_token_segment_id)
        token_to_tag_index.append(tag_pad)

        tokens += title_tokens
        segment_ids += [sequence_a_segment_id] * len(title_tokens)
        token_to_tag_index += [tag_pad] * len(title_tokens)

        tokens.append(SEP_TOKEN)
        segment_ids.append(sequence_a_segment_id)
        token_to_tag_index.append(tag_pad)

        doc_start = len(tokens)

        for i in range(len(instance.all_doc_tokens[:max_tokens_for_doc])):
            tokens.append(instance.all_doc_tokens[i])
            segment_ids.append(sequence_b_segment_id)
            token_to_tag_index.append(instance.tok_to_tags_index[i])

        tokens.append(SEP_TOKEN)
        segment_ids.append(sequence_b_segment_id)
        token_to_tag_index.append(tag_pad)

        input_mask = [1 if mask_padding_with_zero else 0] * len(tokens)

        while len(tokens) < max_seq_length:
            tokens.append(PAD_TOKEN)
            input_mask.append(0 if mask_padding_with_zero else 1)
            segment_ids.append(pad_token_segment_id)
            token_to_tag_index.append(tag_pad)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_to_tag_index) == max_seq_length

        xpath_tags_seq = [
            xpath_tag_map.get(tid, pad_x_tag_seq) for tid in token_to_tag_index
        ]  # ok # token-level
        xpath_subs_seq = [
            xpath_subs_map.get(tid, pad_x_subs_seq) for tid in token_to_tag_index
        ]  # ok

        unique_tids = instance.unique_tids
        element_mask = [1] * len(unique_tids)
        while len(unique_tids) < max_elements_length:
            unique_tids.append(tag_pad)
            element_mask.append(0)

        all_xpath_tags_seq = [
            xpath_tag_map.get(tid, pad_x_tag_seq) for tid in unique_tids
        ]  # ok # element-level
        all_xpath_subs_seq = [
            xpath_subs_map.get(tid, pad_x_tag_seq) for tid in unique_tids
        ]  # ok

        processed_meta_data_map = instance.processed_meta_data_map
        meta_data_seq = [
            processed_meta_data_map.get(tid, pad_meta_data_seq) for tid in unique_tids
        ]  # ok

        element_bos = []
        element_text_len = []
        e_id_start_end_dict = instance.e_id_start_end_dict
        for tid in unique_tids:
            if tid in e_id_start_end_dict.keys():
                start = min(
                    e_id_start_end_dict[tid]["start"] + doc_start, max_seq_length - 1
                )
                end = min(
                    e_id_start_end_dict[tid]["end"] + doc_start, max_seq_length - 1
                )

                element_bos.append([i for i in range(start, end + 1)])

                if start != doc_start or end != doc_start:
                    element_text_len.append(min(end - start + 1, max_tokens_for_doc))
                else:
                    element_text_len.append(0)
            else:
                element_bos.append([pad_bos] * max_seq_length)
                element_text_len.append(0)

        for i in range(len(element_bos)):
            while len(element_bos[i]) < max_seq_length:
                element_bos[i].append(pad_bos)

        features["input_ids"][inst_index] = input_ids
        features["input_mask"][inst_index] = input_mask
        features["segment_ids"][inst_index] = segment_ids
        features["xpath_tags_seq"][inst_index] = xpath_tags_seq
        features["xpath_subs_seq"][inst_index] = xpath_subs_seq
        features["all_xpath_tags_seq"][inst_index] = all_xpath_tags_seq
        features["all_xpath_subs_seq"][inst_index] = all_xpath_subs_seq
        features["meta_data_seq"][inst_index] = meta_data_seq
        features["element_mask"][inst_index] = element_mask
        features["element_bos"][inst_index] = element_bos
        features["element_text_len"][inst_index] = element_text_len
        features["file_id"][inst_index] = instance.file_id
        features["offset"][inst_index] = instance.offset
        features["unique_tids"][inst_index] = [
            int(x.split("cls")[-1]) if isinstance(x, str) and "cls" in x else -1
            for x in unique_tids
        ]

    return features


def save_features_to_hdf5(features, output_file, file_name):
    f = h5py.File(os.path.join(output_file, file_name), "w")
    f.create_dataset(
        "input_ids", data=features["input_ids"], dtype="i4", compression="gzip"
    )
    f.create_dataset(
        "input_mask", data=features["input_mask"], dtype="i4", compression="gzip"
    )
    f.create_dataset(
        "segment_ids", data=features["segment_ids"], dtype="i4", compression="gzip"
    )

    f.create_dataset(
        "xpath_tags_seq",
        data=features["xpath_tags_seq"],
        dtype="i4",
        compression="gzip",
    )
    f.create_dataset(
        "xpath_subs_seq",
        data=features["xpath_subs_seq"],
        dtype="i4",
        compression="gzip",
    )

    f.create_dataset(
        "all_xpath_tags_seq",
        data=features["all_xpath_tags_seq"],
        dtype="i4",
        compression="gzip",
    )
    f.create_dataset(
        "all_xpath_subs_seq",
        data=features["all_xpath_subs_seq"],
        dtype="i4",
        compression="gzip",
    )
    f.create_dataset(
        "meta_data_seq", data=features["meta_data_seq"], dtype="i4", compression="gzip"
    )
    f.create_dataset(
        "element_mask", data=features["element_mask"], dtype="i4", compression="gzip"
    )
    f.create_dataset(
        "element_bos", data=features["element_bos"], dtype="i4", compression="gzip"
    )
    f.create_dataset(
        "element_text_len",
        data=features["element_text_len"],
        dtype="i4",
        compression="gzip",
    )

    f.create_dataset(
        "file_id", data=features["file_id"], dtype="i4", compression="gzip"
    )
    f.create_dataset("offset", data=features["offset"], dtype="i4", compression="gzip")
    f.create_dataset(
        "unique_tids", data=features["unique_tids"], dtype="i4", compression="gzip"
    )

    f.flush()
    f.close()

    print("Saving at {}".format(str(os.path.join(output_file, file_name))))


def multi_create_training_examples(args):
    input_file, tokenizer, token_2_index = args
    return create_training_examples(
        input_file,
        tokenizer,
        token_2_index,
        FLAGS.max_depth,
        FLAGS.max_title_length,
        FLAGS.max_elements_length,
    )


def main():
    batch_size = FLAGS.batch_size

    with open(FLAGS.file_json, "r") as f:
        file_lis = json.load(f)
    count = len(file_lis)
    iters = int(count / batch_size) + 1

    i = FLAGS.start

    print(f"All the data to be processed is divided into {iters} groups.")

    with open(FLAGS.token_2_index_path, "r", encoding="utf-8") as f:
        json_str = f.read()
        token_2_index = json.loads(json_str)

    rng = random.Random(FLAGS.random_seed)

    while i < iters and i != FLAGS.end:
        print("==========i : {}============".format(i))
        start = i * batch_size
        records = file_lis[start : start + batch_size]

        args_lis = [(input_file, tokenizer, token_2_index) for input_file in records]

        with Pool(processes=FLAGS.num_process) as pool:
            instances = list(
                tqdm(
                    pool.imap(multi_create_training_examples, args_lis),
                    total=len(args_lis),
                )
            )

        features = write_instance_to_torch_example_file(
            instances,
            tokenizer,
            FLAGS.max_seq_length,
            FLAGS.max_title_length,
            max_elements_length=FLAGS.max_elements_length,
            meta_data_len=FLAGS.meta_data_len,
            element_pad=FLAGS.pad_id,
        )

        file_name = "t2w_2000_features_1_{}.hdf5".format(i)
        save_features_to_hdf5(features, FLAGS.output_dir, file_name)

        i += 1

    return 0


if __name__ == "__main__":
    main()
