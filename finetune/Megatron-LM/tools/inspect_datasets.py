import os
import sys
import json
import argparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from megatron.data.indexed_dataset import (
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)



def main():
    # data_prefix = '/home/tianhua/lm4sci/datasets/tokenized-eod/dolma/wiki-en-simple/data_text_document'
    data_prefix = '/home/tianhua/lm4sci/datasets/tokenized-eod/peS2o/data_text_document'
    
    indexed_dataset = MMapIndexedDataset(path=data_prefix, skip_warmup=True)

    print(indexed_dataset)
    doc_num = indexed_dataset.sizes.shape[0]
    tokens_num = indexed_dataset.sizes.sum().item()
    print(f'Number of documents: {doc_num}')
    print(f'Number of tokens: {tokens_num}')

if __name__ == '__main__':

    main()