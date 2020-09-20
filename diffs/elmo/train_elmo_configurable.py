# -*- coding: utf-8 -*-
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50, variable=args.variable)
    if args.variable:
        vocab.save_vocab(args.save_dir)

    # define the options
    if args.batch_size > 0:
        batch_size = args.batch_size
    else:
        batch_size = 128  # batch size for each GPU

    if args.n_epochs > 0:
        n_epochs = args.n_epochs
    else:
        n_epochs = 10

    n_gpus = 1

    if args.lang == 'ga':
        n_train_tokens = 3573002
    elif args.lang == 'mt':
        n_train_tokens = 1045392
    elif args.lang == 'sg':
        n_train_tokens = 1196930
    elif args.lang == 'vi':
        n_train_tokens = 5552361
    else:
        raise f'Unrecognized language: {args.lang}'


    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': vocab.n_chars if args.variable else 261,
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': n_epochs,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs',
            default=0)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=0)
    parser.add_argument('--variable', action='store_true',
                        help='Whether to use variable-length vocab')
    parser.add_argument('--lang', choices=['ga', 'mt', 'sg', 'vi'],
                        help='Language to train on')

    args = parser.parse_args()
    main(args)

