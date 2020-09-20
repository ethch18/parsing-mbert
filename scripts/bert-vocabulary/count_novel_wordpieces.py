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

novel = set()
with open(args.vocab_file) as vf:
    for i, line in enumerate(vf):
        if i == 0:
            # [PAD]
            continue
        elif i > 99:
            break
        novel.add(line.strip())


for infile in args.input_file:
    token_count = 0
    tokens_with_novel = 0
    wordpiece_count = 0
    novel_wordpiece_count = 0
    with open(infile) as inf:
        for line in inf:
            orig_tokens = line.strip().split()
            for token in orig_tokens:
                token_count += 1
                wordpieces = tokenizer.tokenize(token)
                wordpiece_count += len(wordpieces)
                novels = sum(1 if wp in novel else 0 for wp in wordpieces)
                novel_wordpiece_count += novels
                tokens_with_novel += (1 if novels > 0 else 0)

    print(f"{infile}: {token_count} tokens, {wordpiece_count} wordpieces, "
          f"{novel_wordpiece_count} novel, {tokens_with_novel} novel tokens")

