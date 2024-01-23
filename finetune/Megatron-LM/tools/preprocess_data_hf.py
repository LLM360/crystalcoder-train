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
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
import datasets

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, data_sample):
        data = data_sample
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens



def print_processing_stats(log_interval, count, proc_start, total_sample_count):
    if count % log_interval == 0:
        current = time.time()
        elapsed = current - proc_start
        print(f"Processed {count}/{total_sample_count} ({count/total_sample_count*100:.2f}%) documents",
                f"({count/elapsed} docs/s).",
                  file=sys.stderr)


def process_json_file(args):
    input_file_name, output_prefix = args.input, args.output_prefix

    ds = datasets.load_dataset(input_file_name, split='train', num_proc=args.workers)
    total_sample_count = len(ds)

    startup_start = time.time()
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, ds, 1)

    level = "document"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                        key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                        key, level)
        # check parent dir exists
        parent_dir = os.path.dirname(output_bin_files[key])
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

    startup_end = time.time()
    proc_start = time.time()

    print("Time to startup:", startup_end - startup_start)
    for i, (doc, sentence_lens) in enumerate(encoded_docs, start=1):
        for key in doc.keys():
            builders[key].add_doc(doc[key], sentence_lens[key])
        print_processing_stats(args.log_interval, i, proc_start, total_sample_count)

    builders[key].finalize(output_idx_files[key])


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='YTTM tokenizer model.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))

    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args



def main():
    args = get_args()

    process_json_file(args)



if __name__ == '__main__':

    main()

