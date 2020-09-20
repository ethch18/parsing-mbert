# parses wikiextractor output into a format that is compatible with BERT et. al
import argparse
import math
import os
import random

from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from tqdm import tqdm

NO_PERIOD = ['.', ':']

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str, help='Input file',
        required=True)
parser.add_argument('--output-dir', type=str, help='Output file',
        required=True)
args = parser.parse_args()

splitter = SpacyWordSplitter(language='xx_ent_wiki_sm')
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(args.input_file, 'r') as f:
    for line in f:
        raw = line.strip()
        sentences = raw.split('. ')
        doc = []
        for sentence in sentences:
            s = sentence.strip()
            if s:
                if s[-1] not in NO_PERIOD:
                    # add back the period
                    s = s + '.'

                # tokenize
                tokenized = ' '.join([tkn.text \
                        for tkn in splitter.split_words(s)]).strip()
                doc.append(tokenized)

        partitioner = random.random()
        if partitioner < 0.1:
            outf = open(os.path.join(args.output_dir, 'valid.txt'), 'a')
        elif partitioner < 0.2:
            outf = open(os.path.join(args.output_dir, 'test.txt'), 'a')
        else:
            outf = open(os.path.join(args.output_dir, 'train.txt'), 'a')
        for sentence in doc:
            outf.write(sentence)
            outf.write('\n')
        # end of doc
        outf.write('\n')

