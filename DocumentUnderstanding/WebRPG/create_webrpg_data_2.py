import os
import random
import numpy as np
import torch
import h5py
import collections
from tqdm import tqdm
from markuplm.tokenization_markuplm import MarkupLMTokenizer
from markuplm import MarkupLMConfig, MarkupLMModel
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataloader_batch_size",
    type=int,
    default=8,
    help="Batch size for the MarkupLM dataloader",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help='Device to use for running MarkupLM (e.g., "cuda" or "cpu")',
)
parser.add_argument(
    "--random_seed", type=int, default=12345, help="Random seed for reproducibility"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./output",
    help="Directory to save the output",
)
parser.add_argument(
    "--input_dir",
    type=str,
    default="./input",
    help="Directory containing the input data",
)
parser.add_argument(
    "--markuplm_model_name_or_path",
    type=str,
    default="./markuplm-large",
    help="Path to the MarkupLM model checkpoint",
)
parser.add_argument(
    "--start", default=0, type=int, help="Starting index of the data group to process"
)
parser.add_argument(
    "--end", default=1, type=int, help="Ending index of the data group to process"
)

FLAGS = parser.parse_args()


def write_instance_to_torch_example_file(
    h5py_f, all_sequence_output, all_pooled_output, all_xpath_embeddings
):

    features = collections.OrderedDict()

    keys = [
        "input_ids",
        "input_mask",
        "segment_ids",
        "xpath_tags_seq",
        "xpath_subs_seq",
        "all_xpath_tags_seq",
        "all_xpath_subs_seq",
        "meta_data_seq",
        "element_mask",
        "element_bos",
        "element_text_len",
        "file_id",
        "offset",
        "unique_tids",
    ]

    for key in keys:
        features[key] = np.asarray(h5py_f[key][:])

    features["all_sequence_embeddings"] = all_sequence_output
    print("Finished writing 'all_sequence_embeddings' to features.")

    features["all_pooled_embeddings"] = all_pooled_output
    print("Finished writing 'all_pooled_embeddings' to features.")

    features["all_xpath_embeddings"] = all_xpath_embeddings
    print("Finished writing 'all_xpath_embeddings' to features.")

    return features


def save_features_to_hdf5(features, output_file, file_name):
    f = h5py.File(os.path.join(output_file, file_name), "w")
    f.create_dataset(
        "input_ids", data=features["input_ids"], dtype="i4", compression="gzip"
    )
    print("Created dataset: input_ids")

    f.create_dataset(
        "input_mask", data=features["input_mask"], dtype="i4", compression="gzip"
    )
    print("Created dataset: input_mask")

    f.create_dataset(
        "segment_ids", data=features["segment_ids"], dtype="i4", compression="gzip"
    )
    print("Created dataset: segment_ids")

    f.create_dataset(
        "xpath_tags_seq",
        data=features["xpath_tags_seq"],
        dtype="i4",
        compression="gzip",
    )
    print("Created dataset: xpath_tags_seq")

    f.create_dataset(
        "xpath_subs_seq",
        data=features["xpath_subs_seq"],
        dtype="i4",
        compression="gzip",
    )
    print("Created dataset: xpath_subs_seq")

    f.create_dataset(
        "all_xpath_tags_seq",
        data=features["all_xpath_tags_seq"],
        dtype="i4",
        compression="gzip",
    )
    print("Created dataset: all_xpath_tags_seq")

    f.create_dataset(
        "all_xpath_subs_seq",
        data=features["all_xpath_subs_seq"],
        dtype="i4",
        compression="gzip",
    )
    print("Created dataset: all_xpath_subs_seq")

    f.create_dataset(
        "meta_data_seq", data=features["meta_data_seq"], dtype="i4", compression="gzip"
    )
    print("Created dataset: meta_data_seq")

    f.create_dataset(
        "element_mask", data=features["element_mask"], dtype="i4", compression="gzip"
    )
    print("Created dataset: element_mask")

    f.create_dataset(
        "element_bos", data=features["element_bos"], dtype="i4", compression="gzip"
    )
    print("Created dataset: element_bos")

    f.create_dataset(
        "element_text_len",
        data=features["element_text_len"],
        dtype="i4",
        compression="gzip",
    )
    print("Created dataset: element_text_len")

    f.create_dataset(
        "all_sequence_embeddings",
        data=features["all_sequence_embeddings"],
        dtype="f8",
        compression="gzip",
    )
    print("Created dataset: all_sequence_embeddings")
    f.create_dataset(
        "all_pooled_embeddings",
        data=features["all_pooled_embeddings"],
        dtype="f8",
        compression="gzip",
    )
    print("Created dataset: all_pooled_embeddings")
    f.create_dataset(
        "all_xpath_embeddings",
        data=features["all_xpath_embeddings"],
        dtype="f8",
        compression="gzip",
    )
    print("Created dataset: all_xpath_embeddings")

    f.create_dataset(
        "file_id", data=features["file_id"], dtype="f8", compression="gzip"
    )
    f.create_dataset("offset", data=features["offset"], dtype="f8", compression="gzip")
    f.create_dataset(
        "unique_tids", data=features["unique_tids"], dtype="f8", compression="gzip"
    )

    f.flush()
    f.close()
    print("Saving at {}".format(str(os.path.join(output_file, file_name))))


class pretraining_dataset(Dataset):
    def __init__(self, input_file, f):

        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "xpath_tags_seq",
            "xpath_subs_seq",
            "all_xpath_tags_seq",
            "all_xpath_subs_seq",
            "meta_data_seq",
            "element_mask",
            "element_bos",
        ]

        self.inputs = [np.asarray(f[key][:]) for key in keys]

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        input_ids = torch.from_numpy(self.inputs[0][index].astype(np.int64))
        input_mask = torch.from_numpy(self.inputs[1][index].astype(np.int64))
        segment_ids = torch.from_numpy(self.inputs[2][index].astype(np.int64))
        xpath_tags_seq = torch.from_numpy(self.inputs[3][index].astype(np.int64))
        xpath_subs_seq = torch.from_numpy(self.inputs[4][index].astype(np.int64))
        all_xpath_tags_seq = torch.from_numpy(self.inputs[5][index].astype(np.int64))
        all_xpath_subs_seq = torch.from_numpy(self.inputs[6][index].astype(np.int64))
        meta_data_seq = torch.from_numpy(self.inputs[7][index].astype(np.int64))
        element_mask = torch.from_numpy(self.inputs[8][index].astype(np.int64))
        element_bos = torch.from_numpy(self.inputs[9][index].astype(np.int64))

        return [
            input_ids,
            input_mask,
            segment_ids,
            xpath_tags_seq,
            xpath_subs_seq,
            all_xpath_tags_seq,
            all_xpath_subs_seq,
            meta_data_seq,
            element_mask,
            element_bos,
        ]


def create_pretraining_dataset(input_file, args, back_files=None):
    read_finish = False
    range_ids = None

    idx = 0

    h5py_f = h5py.File(os.path.join(args.input_dir, input_file), "r")
    train_data = pretraining_dataset(input_file=input_file, f=h5py_f)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=args.dataloader_batch_size,
        num_workers=1,
        pin_memory=True,
    )
    return train_dataloader, input_file, h5py_f


def main():
    file_lis = os.listdir(FLAGS.input_dir)
    count = len(file_lis)
    iters = count

    print(f"All the data to be processed is divided into {iters} groups.")

    i = FLAGS.start

    markuplm = MarkupLMModel.from_pretrained(FLAGS.markuplm_model_name_or_path)
    markuplm.to(FLAGS.device)
    xpath = markuplm.embeddings.xpath_embeddings
    xpath.to(FLAGS.device)
    markuplm.eval()

    tokenizer = MarkupLMTokenizer.from_pretrained(FLAGS.markuplm_model_name_or_path)

    rng = random.Random(FLAGS.random_seed)

    while i < iters and i != FLAGS.end:
        print("==========i : {}============".format(i))
        records = file_lis[i]

        train_dataloader, input_file, h5py_f = create_pretraining_dataset(
            records, FLAGS
        )
        train_iter = tqdm(train_dataloader, desc="Iteration")

        all_sequence_output = []
        all_pooled_output = []
        all_xpath_embeddings = []

        for step, batch in enumerate(train_iter):
            batch = [t.to(FLAGS.device) for t in batch]
            (
                input_ids,
                input_mask,
                segment_ids,
                xpath_tags_seq,
                xpath_subs_seq,
                all_xpath_tags_seq,
                all_xpath_subs_seq,
                meta_data_seq,
                element_mask,
                element_bos,
            ) = batch
            markuplm.eval()
            with torch.no_grad():
                sequence_output, pooled_output = markuplm(
                    input_ids,
                    xpath_tags_seq=xpath_tags_seq,
                    xpath_subs_seq=xpath_subs_seq,
                    attention_mask=input_mask,
                    return_dict=False,
                )

                masks = torch.zeros(
                    [sequence_output.shape[0], 1, sequence_output.shape[-1]],
                    device=sequence_output.device,
                )
                sequence_output = sequence_output.scatter(
                    dim=1,
                    index=torch.zeros(
                        sequence_output.shape[0],
                        1,
                        sequence_output.shape[-1],
                        dtype=torch.int64,
                        device=sequence_output.device,
                    ),
                    src=masks,
                )
                sequence_output = sequence_output.unsqueeze(1).expand(
                    sequence_output.shape[0],
                    element_bos.shape[1],
                    sequence_output.shape[1],
                    sequence_output.shape[2],
                )
                output = torch.gather(
                    sequence_output,
                    dim=2,
                    index=element_bos.unsqueeze(-1).expand(
                        element_bos.size(0),
                        element_bos.size(1),
                        element_bos.size(2),
                        sequence_output.shape[-1],
                    ),
                )
                count = torch.sum(output != 0, dim=2)
                count[count == 0] = 1
                output = torch.sum(output, 2) / count

                xpath_output = xpath(all_xpath_tags_seq, all_xpath_subs_seq)

            all_sequence_output.append(output.cpu().numpy())
            all_pooled_output.append(torch.unsqueeze(pooled_output, 1).cpu().numpy())
            all_xpath_embeddings.append(xpath_output.cpu().numpy())

        all_sequence_output = np.concatenate(all_sequence_output, axis=0)
        all_pooled_output = np.concatenate(all_pooled_output, axis=0)
        all_xpath_embeddings = np.concatenate(all_xpath_embeddings, axis=0)

        features = write_instance_to_torch_example_file(
            h5py_f, all_sequence_output, all_pooled_output, all_xpath_embeddings
        )

        file_name = "t2w_websrc_features_2_{}.hdf5".format(i)
        save_features_to_hdf5(features, FLAGS.output_dir, file_name)

        i += 1

    return 0


if __name__ == "__main__":
    main()
