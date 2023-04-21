import json
import os
from glob import glob

import imagesize
from tqdm import tqdm
from transformers import BertTokenizer

MAX_SEQ_LENGTH = 512
MODEL_TYPE = "bert"
VOCA = "bert-base-uncased"

CLASSES = ["O", "HEADER", "QUESTION", "ANSWER"]
CLASSES_VALID = ["HEADER", "QUESTION", "ANSWER"]
INPUT_PATH = "path/to/dataset/funsd/"
anno_dir = 'annotations'
# if not os.path.exists(INPUT_PATH):
#     os.system("wget https://guillaumejaume.github.io/FUNSD/dataset.zip")
#     os.system("unzip dataset.zip")
#     os.system("rm -rf dataset.zip __MACOSX")

OUTPUT_PATH = "path/to/dataset/funsd_geo"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "preprocessed"), exist_ok=True)


def main():
    tokenizer = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)

    for dataset_split in ["train", "val"]:
        print(f"dataset_split: {dataset_split}")
        do_preprocess(tokenizer, dataset_split)

    os.system(f"cp -r {os.path.join(INPUT_PATH, 'training_data')} {OUTPUT_PATH}")
    os.system(f"cp -r {os.path.join(INPUT_PATH, 'testing_data')} {OUTPUT_PATH}")
    save_class_names()


def do_preprocess(tokenizer, dataset_split):
    if dataset_split == "train":
        dataset_root_path = os.path.join(INPUT_PATH, "training_data")
    elif dataset_split == "val":
        dataset_root_path = os.path.join(INPUT_PATH, "testing_data")
    else:
        raise ValueError(f"Invalid dataset_split={dataset_split}")

    json_files = glob(os.path.join(dataset_root_path, anno_dir, "*.json"))
    preprocessed_fnames = []
    for json_file in tqdm(json_files):
        in_json_obj = json.load(open(json_file, "r", encoding="utf-8"))

        out_json_obj = {}
        out_json_obj['blocks'] = {'first_token_idx_list': [], 'boxes': []}
        out_json_obj["words"] = []
        out_json_obj["parse"] = {"class": {}}
        for c in CLASSES:
            out_json_obj["parse"]["class"][c] = []
        out_json_obj["parse"]["relations"] = []

        form_id_to_word_idx = {} # record the word index of the first word of each block, starting from 0
        other_seq_list = {}
        num_tokens = 0

        # words
        for form_idx, form in enumerate(in_json_obj["form"]):
            form_id = form["id"]
            form_text = form["text"].strip()
            form_label = form["label"].upper()
            if form_label.startswith('O'):
                form_label = "O"
            form_linking = form["linking"]
            form_box = form["box"]

            if len(form_text) == 0:
                continue # filter text blocks with empty text

            word_cnt = 0
            class_seq = []
            real_word_idx = 0
            for word_idx, word in enumerate(form["words"]):
                word_text = word["text"]
                bb = word["box"]
                bb = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
                tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_text))

                word_obj = {"text": word_text, "tokens": tokens, "boundingBox": bb}
                if len(word_text) != 0: # filter empty words
                    out_json_obj["words"].append(word_obj)
                    if real_word_idx == 0:
                        out_json_obj['blocks']['first_token_idx_list'].append(num_tokens + 1)
                    num_tokens += len(tokens)

                    word_cnt += 1
                    class_seq.append(len(out_json_obj["words"]) - 1) # word index
                    real_word_idx += 1
            if real_word_idx > 0:
                out_json_obj['blocks']['boxes'].append(form_box)

            is_valid_class = False if form_label not in CLASSES else True
            if is_valid_class:
                out_json_obj["parse"]["class"][form_label].append(class_seq)
                form_id_to_word_idx[form_id] = len(out_json_obj["words"]) - word_cnt
            else:
                other_seq_list[form_id] = class_seq

        # parse
        for form_idx, form in enumerate(in_json_obj["form"]):
            form_id = form["id"]
            form_text = form["text"].strip()
            form_linking = form["linking"]

            if len(form_linking) == 0:
                continue

            for link_idx, link in enumerate(form_linking):
                if link[0] == form_id:
                    if (
                        link[1] in form_id_to_word_idx
                        and link[0] in form_id_to_word_idx
                    ):
                        relation_pair = [
                            form_id_to_word_idx[link[0]],
                            form_id_to_word_idx[link[1]],
                        ]
                        out_json_obj["parse"]["relations"].append(relation_pair)

        # meta
        out_json_obj["meta"] = {}
        image_file = (
            os.path.splitext(json_file.replace(f"/{anno_dir}/", "/images/"))[0] + ".png"
        )
        if dataset_split == "train":
            out_json_obj["meta"]["image_path"] = image_file[
                image_file.find("training_data/") :
            ]
        elif dataset_split == "val":
            out_json_obj["meta"]["image_path"] = image_file[
                image_file.find("testing_data/") :
            ]
        width, height = imagesize.get(image_file)
        out_json_obj["meta"]["imageSize"] = {"width": width, "height": height}
        out_json_obj["meta"]["voca"] = VOCA

        this_file_name = os.path.basename(json_file)

        # Save file name to list
        preprocessed_fnames.append(os.path.join("preprocessed", this_file_name))

        # Save to file
        data_obj_file = os.path.join(OUTPUT_PATH, "preprocessed", this_file_name)
        with open(data_obj_file, "w", encoding="utf-8") as fp:
            json.dump(out_json_obj, fp, ensure_ascii=False)

    # Save file name list file
    preprocessed_filelist_file = os.path.join(
        OUTPUT_PATH, f"preprocessed_files_{dataset_split}.txt"
    )
    with open(preprocessed_filelist_file, "w", encoding="utf-8") as fp:
        fp.write("\n".join(preprocessed_fnames))


def save_class_names():
    with open(
        os.path.join(OUTPUT_PATH, "class_names.txt"), "w", encoding="utf-8"
    ) as fp:
        fp.write("\n".join(CLASSES))


if __name__ == "__main__":
    main()
