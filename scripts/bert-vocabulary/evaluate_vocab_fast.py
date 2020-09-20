"""
Compare a base and new vocab file
"""
# TODO: there's still something wrong here where it doesn't line up exactly with BERT
import argparse

import torch
# from pytorch_transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file-base', type=str, required=True,
        help='Vocabulary file')
parser.add_argument('--vocab-file-eval', type=str, required=True,
        help='Vocabulary file to evaluate')
parser.add_argument('--lower-case', action='store_true',
        help='Whether to lowercase inputs (for uncased models)')
parser.add_argument('--input-file', type=str, action='append', default=[],
        help='Files to count wordpieces of')

args = parser.parse_args()

# base_tokenizer = BertTokenizer.from_pretrained(args.vocab_file_base,
#         do_lower_case=args.lower_case,
#         do_basic_tokenize=True)
# eval_tokenizer = BertTokenizer.from_pretrained(args.vocab_file_eval,
#         do_lower_case=args.lower_case,
#         do_basic_tokenize=True)
base_tokenizer = BertWordPieceTokenizer(
        vocab_file=args.vocab_file_base,
        clean_text=False,
        handle_chinese_chars=True,
        strip_accents=args.lower_case,
        lowercase=args.lower_case
)
eval_tokenizer = BertWordPieceTokenizer(
        vocab_file=args.vocab_file_eval,
        clean_text=False,
        handle_chinese_chars=True,
        strip_accents=args.lower_case,
        lowercase=args.lower_case
)

unk = '[UNK]'

# there will be an UNK token in the eval file anyways
eval_unk_count = -1
with open(args.vocab_file_eval) as eval_file:
    for line in eval_file:
        entry = line.strip()
        # wordpieces = base_tokenizer.tokenize(entry)
        wordpieces = base_tokenizer.encode(entry, add_special_tokens=False).tokens
        if unk in wordpieces:
            eval_unk_count += 1

print(f"{eval_unk_count} subwords with UNK in new vocab file")

def compute_statistics(infile, tokenizer):
    token_count = 0
    wordpiece_count = 0
    unk_count = 0
    with open(infile) as inf:
        for line in inf:
            orig_tokens = line.strip().split()
            for token in orig_tokens:
                token_count += 1
                # wordpieces = tokenizer.tokenize(token)
                wordpieces = tokenizer.encode(token, add_special_tokens=False).tokens
                wordpiece_count += len(wordpieces)
                local_unk_count = sum(1 if piece == unk else 0 for piece in wordpieces)
                unk_count += local_unk_count

    print(f"{infile}: {token_count} tokens, {wordpiece_count} wordpieces, "
          f"{wordpiece_count / token_count} wordpieces/token on average, "
          f"{unk_count} unknown wordpieces")

for infile in args.input_file:
    print('base tokenizer')
    compute_statistics(infile, base_tokenizer)
    print('eval tokenizer')
    compute_statistics(infile, eval_tokenizer)
