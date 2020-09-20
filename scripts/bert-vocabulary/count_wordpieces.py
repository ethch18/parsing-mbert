"""
Compute wordpiece statistics for a given file
"""
import argparse

import torch
from pytorch_transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', type=str, required=True,
        help='Vocabulary file')
parser.add_argument('--lower-case', action='store_true',
        help='Whether to lowercase inputs (for uncased models)')
parser.add_argument('--input-file', type=str, action='append', default=[],
        help='Files to count wordpieces of')

args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.vocab_file,
        do_lower_case=args.lower_case,
        do_basic_tokenize=True)

unk = '[UNK]'


for infile in args.input_file:
    token_count = 0
    wordpiece_count = 0
    unk_count = 0
    token_count_unks = 0
    token_count_nounks = 0
    wordpiece_count_unks = 0
    wordpiece_count_nounks = 0
    with open(infile) as inf:
        for line in inf:
            orig_tokens = line.strip().split()
            for token in orig_tokens:
                token_count += 1
                wordpieces = tokenizer.tokenize(token)
                wordpiece_count += len(wordpieces)
                local_unk_count = sum(1 if piece == unk else 0 for piece in wordpieces)
                unk_count += local_unk_count

                if local_unk_count > 0:
                    token_count_unks += 1
                    wordpiece_count_unks += len(wordpieces)
                else:
                    token_count_nounks += 1
                    wordpiece_count_nounks += len(wordpieces)

    if token_count_unks == 0:
        token_count_unks = 1
    if token_count_nounks == 0:
        token_count_nounks = 1

    print(f"{infile}: {token_count} tokens, {wordpiece_count} wordpieces, "
          f"{wordpiece_count / token_count} wordpieces/token on average, "
          f"{unk_count} unknown wordpieces")
    print(f"\tUNK tokens: {token_count_unks} tokens, {wordpiece_count_unks} "
          f"wordpieces, {wordpiece_count_unks/token_count_unks} ratio")
    print(f"\tNo UNK tokens: {token_count_nounks} tokens, {wordpiece_count_nounks} "
          f"wordpieces, {wordpiece_count_nounks/token_count_nounks} ratio")

