"""
Simple script for asserting that pytorch_transformers gives a good tokenization
"""
import argparse

import tokenization
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

base_tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file,
        do_lower_case=args.lower_case)
torch_tokenizer = BertTokenizer.from_pretrained(args.vocab_file,
        do_lower_case=args.lower_case,
        do_basic_tokenize=True)


for infile in args.input_file:
    with open(infile) as inf:
        for line in inf:
            orig_tokens = line.strip()
            wordpieces = base_tokenizer.tokenize(orig_tokens)
            torch_wordpieces = torch_tokenizer.tokenize(orig_tokens)
            if wordpieces != torch_wordpieces:
                print(f"{orig_tokens} became {wordpieces} and {torch_wordpieces}")
            for token in orig_tokens.split():
                wordpieces = base_tokenizer.tokenize(token)
                torch_wordpieces = torch_tokenizer.tokenize(token)
                if wordpieces != torch_wordpieces:
                    print(f"{token} became {wordpieces} and {torch_wordpieces}")

