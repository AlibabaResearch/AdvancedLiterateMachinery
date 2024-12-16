import os
import numpy as np
import torch
import h5py
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from multiprocessing import Pool, Manager


parser = argparse.ArgumentParser()

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
    "--start", default=0, type=int, help="Starting index of the data group to process"
)
parser.add_argument(
    "--end", default=1, type=int, help="Ending index of the data group to process"
)

FLAGS = parser.parse_args()


def process_file(args):
    index, key, input_data, output_dir, file_id, offset_id = args
    dir_name = "t2w_features_split_{}_{}".format(file_id, offset_id)
    output_path = os.path.join(output_dir, dir_name)
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except FileExistsError:
            pass
    torch.save(input_data, os.path.join(output_path, f"{key}.pt"))


def main():

    file_lis = os.listdir(FLAGS.input_dir)
    count = len(file_lis)
    iters = count

    i = FLAGS.start

    print(f"The total number of HDF5 files to be processed is {iters}.")

    while i < iters and i != FLAGS.end:
        print("==========i : {}============".format(i))
        h5py_f = h5py.File(os.path.join(FLAGS.input_dir, file_lis[i]), "r")

        keys = list(h5py_f.keys())
        file_id_id = keys.index("file_id")
        offset_id = keys.index("offset")

        inputs = [np.asarray(h5py_f[key][:]) for key in tqdm(keys)]

        num_files = len(h5py_f[keys[0]])

        with Pool(processes=64) as pool:
            tasks = []
            for index in tqdm(range(num_files)):
                for id, key in enumerate(keys):
                    input_data = inputs[id][index]
                    file_id = int(inputs[file_id_id][index][0])
                    offset = int(inputs[offset_id][index][0])
                    tasks.append(
                        (index, key, input_data, FLAGS.output_dir, file_id, offset)
                    )
            results = list(
                tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks))
            )
        h5py_f.close()
        i += 1


if __name__ == "__main__":

    main()
