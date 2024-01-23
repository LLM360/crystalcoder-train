# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import glob
import torch
import numpy as np
import multiprocessing
import h5py
from glob import glob
import json
import os
from tqdm import tqdm

from megatron.data import indexed_dataset


def main(source_root, target_dir, sample_count=None):
    
    print(f'Convert: {source_root} --> {target_dir}')
    print(f'max sample count: {sample_count}')
    all_data_files = sorted( glob(os.path.join(source_root, '**/*.h5'), recursive=True))
    print("all_data_files", all_data_files)
    write_count = 0
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    key = 'token_ids'
    output_bin_files[key] = os.path.join(target_dir ,f"{key}.bin")
    output_idx_files[key] = os.path.join(target_dir ,f"{key}.idx")
    builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(
        output_bin_files[key],
        dtype=indexed_dataset.DType.optimal_dtype(40000),
    )

    for filename in tqdm(all_data_files, position=0, leave=True):
        f = h5py.File(filename, 'r')

        # print(f.keys())

        data = f['data']
        local_sample_total = data.shape[0]
        for local_idx in tqdm(range(local_sample_total ), position=1, leave=False):
            sample = data[local_idx]
            token_idx = sample[0]
            # token_idx = torch.Tensor(token_idx)
            # mask = sample[1]
            # label_idx = sample[2]

            write_count += 1
            if sample_count is not None and write_count >= sample_count:
                break
            builders[key].add_item_np(token_idx)

    builders[key].finalize(output_idx_files[key])
    print(f'Write {write_count} samples to {target_dir}')
    tokens_cnt = write_count * 2048
    print(f'Total tokens: {tokens_cnt}')

if __name__ == '__main__':
    # COMPLETE

    ########

    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/starcoder_3/train/html', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/starcoder_3/train/html' )
    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/SlimPajama/ArXiv_train_packed', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/SlimPajama/ArXiv_train_packed' )
    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/starcoder_3/train/javascript', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/starcoder_3/train/javascript' )
    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/starcoder_3/train/python', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/starcoder_3/train/python' )
    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/starcoder_2_mix', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/starcoder_2_mix' )
    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/starcoder_3/train/css', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/starcoder_3/train/css' )

    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/SlimPajama/C4_train_packed', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/SlimPajama/C4_train_packed' )
    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/SlimPajama/Wikipedia_train_packed', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/SlimPajama/Wikipedia_train_packed' )
    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/SlimPajama/Book_train_packed', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/SlimPajama/Book_train_packed' )
    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/SlimPajama/CommonCrawl_train_packed', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/SlimPajama/CommonCrawl_train_packed' )
    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/SlimPajama/StackExchange_train_packed', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/SlimPajama/StackExchange_train_packed' )

    # main('/home/tianhua.tao/scratch/manaslu/tmp/h5/SlimPajama/Github_train_packed', target_dir = '/home/tianhua.tao/scratch/manaslu/tmp/m/SlimPajama/Github_train_packed' )
    ############################ TODO



    pass
