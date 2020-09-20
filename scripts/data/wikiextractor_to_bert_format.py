# parses wikiextractor output into a format that is compatible with BERT et. al
import argparse
import math
import os
import random

from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from tqdm import tqdm

NO_PERIOD = ['.', ':']

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, help='Base input directory',
        required=True)
parser.add_argument('--output-dir', type=str, help='Output file',
        required=True)
parser.add_argument('--downsample', action='store_true', help='Whether to sample')
parser.add_argument('--output-listings', type=str, help='Output for listings '
        '(for reproducibility)')
parser.add_argument('--listings', type=str, help='Directory listings')
parser.add_argument('--fraction', type=float, help='Fraction of listings to take')
args = parser.parse_args()

splitter = SpacyWordSplitter(language='xx_ent_wiki_sm')
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.downsample:
    listings = []
    with open(args.listings, 'r') as listings_file:
        for line in listings_file:
            listings.extend(line.strip().split())
    sample_size = math.floor(len(listings) * args.fraction)
    subset = random.sample(listings, sample_size)
    with open(args.output_listings, 'w') as listings_output:
        listings_output.write(f"Sample of {args.fraction} ({len(subset)} of the "
                f"total {len(listings)}\n")
        for listing in subset:
            listings_output.write(listing)
            listings_output.write('\n')
    file_source = subset
else:
    file_source = os.listdir(args.input_dir)

after_title_line = False
for filename in tqdm(file_source):
    fqname = os.path.join(args.input_dir, filename)
    with open(fqname, 'r') as f:
        document = []
        for line in f:
            if line.startswith('<doc id='):
                if len(document) > 1:
                    partitioner = random.random()
                    if partitioner < 0.1:
                        outf = open(os.path.join(args.output_dir, 'valid.txt'), 'a')
                    elif partitioner < 0.2:
                        outf = open(os.path.join(args.output_dir, 'test.txt'), 'a')
                    else:
                        outf = open(os.path.join(args.output_dir, 'train.txt'), 'a')
                    for sentence in document:
                        outf.write(sentence)
                        outf.write('\n')
                    # end of doc
                    outf.write('\n')

                after_title_line = True
                document = []
                continue
            elif after_title_line:
                after_title_line = False
                continue
            elif (line.startswith('=') \
                    or line.startswith('[[') \
                    or line.startswith('<')):
                continue

            raw = line.strip()
            raw = raw.replace('<br>', '')
            sentences = raw.split('. ')
            for sentence in sentences:
                s = sentence.strip()
                if s:
                    if s[-1] not in NO_PERIOD:
                        # add back the period
                        s = s + '.'

                    # tokenize
                    tokenized = ' '.join([tkn.text \
                            for tkn in splitter.split_words(s)]).strip()
                    document.append(s)

# Remove two lines at the start of each doc (<doc id=... and the title)
# Remove section headers: =HEAD=
# Remove categories [[...]]
# Remove doc end </doc>
# Remove html <br>
# Split sentences at period (. => .\n)
# Strip lines
# Remove blanks
# Randomly partition documents
# Remove stubs (single sentence)
# Print a blank line at the end of each doc
# Tokenize with spacy multiling
