"""
1. Get all the words that yield UNK in the corpus
2. If one is UNK and the other is not, add the unique wordpieces
3. Pad to args.count with the most common wordpieces in the new vocab
"""
import argparse
from collections import Counter, OrderedDict

import torch
from pytorch_transformers import BertTokenizer, WordpieceTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True, help='Reference corpus file')
parser.add_argument('--base-vocab', type=str, required=True, help='Base vocab file')
parser.add_argument('--new-vocab', type=str, required=True, help='New vocab file')
parser.add_argument('--count', type=int, default=99, help='How many values to return')
parser.add_argument('--output-file', type=str, required=True, help='Where to output')
parser.add_argument('--lower-case', action='store_true',
        help='Whether to lowercase inputs (for uncased models)')

args = parser.parse_args()

def count(tkn, lst):
    return sum(1 if tkn == item else 0 for item in lst)

base_tokenizer = BertTokenizer.from_pretrained(
    args.base_vocab,
    do_lower_case=args.lower_case,
    do_basic_tokenize=True,
)

existing_vocab = set()
with open(args.base_vocab) as base_file:
    for line in base_file:
        entry = line.strip()
        existing_vocab.add(entry)

new_vocab_words = []
unk = '[UNK]'
with open(args.new_vocab) as eval_file:
    for line in eval_file:
        entry = line.strip()
        if entry != unk:
            wordpieces = base_tokenizer.tokenize(entry)
            # if unk in wordpieces:
            if True:
                new_vocab_words.append(entry)

# if len(new_vocab_words) > args.count:
if True:
    # create a "vocabulary" with the new vocab words
    fake_vocab = OrderedDict()
    for index, vocab in enumerate(new_vocab_words):
        fake_vocab[vocab] = index

    fake_tokenizer = WordpieceTokenizer(fake_vocab, unk)

    unk_words = []
    fake_words = []
    with open(args.corpus) as corpus:
        for line in corpus:
            orig_tokens = line.strip().split()
            for token in orig_tokens:
                wordpieces = base_tokenizer.tokenize(token)
                fake_wordpieces = fake_tokenizer.tokenize(token)
                fake_words.extend(fake_wordpieces)
                if count(unk, wordpieces) > count(unk, fake_wordpieces):
                    unk_words.extend([fwp for fwp in fake_wordpieces if fwp not in wordpieces]) 
                # if unk in wordpieces:
                #     unk_words.extend(fake_tokenizer.tokenize(token))

    vocab_counter = Counter(unk_words)
    selection = vocab_counter.most_common()
    selection = [tup[0] for tup in selection if tup[0] not in existing_vocab]
    selection = selection[:args.count]

    fake_counter = Counter(fake_words).most_common()
    fake_counter = [tup[0] for tup in fake_counter if tup[0] not in existing_vocab]

    i = 0
    while len(selection) < args.count and i < len(fake_counter):
        if fake_counter[i] not in selection:
            selection.append(fake_counter[i])
        i += 1
        

with open(args.output_file, 'w') as ouf:
    for item in selection:
        ouf.write(item)
        ouf.write('\n')
