# Parsing with Multilingual BERT, a Small Corpus, and a Small Treebank

This repo contains code and data for the paper "Parsing with Multilingual BERT, a
Small Corpus, and a Small Treebank" by Ethan C. Chau, Lucy H. Lin, and Noah A. Smith.

## Setting up
1. Clone the repo: `git clone git@github.com:ethch18/parsing-mbert.git`
2. Install macro dependencies:
    ```bash
    # PyTorch, CPU-only
    pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    # PyTorch, CUDA 9.2
    pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
    # tensorflow-gpu 1.12 (conda installation may be easier)
    pip install tensorflow-gpu==1.12
    # AllenNLP, v0.9.0
    pip install allennlp==0.9.0
    # others
    pip install h5py
    ```
3. Clone allentune, bilm-tf, bert.  You may need to check out the commit that
   was the most recent as of June 1, 2020.
    ```bash
    git clone git@github.com:allenai/allentune.git
    git clone git@github.com:allenai/bilm-tf.git
    git clone git@github.com:google-research/bert.git
    ```
4. Install allentune:
    ```bash
    pip install --editable .
    ```
5. Install bilm-tf locally:
    ```bash
    python setup.py develop
    ```
6. Apply diffs to AllenNLP, bilm-tf, and bert
    * AllenNLP: see `diffs/allennlp.txt`
    * bilm-tf: see `diffs/elmo`.  Copy `data.py` and `training.py` into `bilm/`
        and `train_elmo_configurable.py` into `bin/`.
    * bert: see `diffs/bert`.  Copy everything into the root directory.
7. Edit the directory paths in `config/`, `scripts/`, and the various top-level
   scripts to match your directory structure.

## Getting Data
You can download the UD 2.5 treebanks
[here](http://hdl.handle.net/11234/1-3105).  We used the Irish (IDT), Maltese
(MUDT), and Vietnamese (VTB) treebanks.  The Singlish treebank is in `data/sing`.

We've provided the fastText vectors that we trained as tarballs, but they were
too large for git.  You can find links to them under `data/fasttext`.

We provide the raw unlabeled data in `data/unlabeled`.  Here's how you can
generate BERT-compatible shards (use either the base mBERT vocab or a VA vocab
as the vocabulary file).

A subset of the trained models is available [here](https://drive.google.com/drive/folders/1AZi33t6u-xSA4L_cWQC7tz9VuZUxP9FP?usp=sharing).

```bash
python create_pretraining_data.py --do_lower_case=False --max_seq_length=128 \
    --random_seed 13370 --dupe_factor=5 --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 --input_file=INPUT.txt --output_file=OUTPUT.tfrecord \
    --vocab_file=/path/to/vocab/file.txt
```

Download the original mBERT checkpoint from
[here](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip).

To generate ELMo-compatible shards, see `scripts/data/bert_to_elmo_format.py`
and `scripts/data/create_elmo_vocab.py`.

## LAPT

```bash
#python run_pretraining_nonsp.py --input_file=INPUT.tfrecord --output_dir=/output/dir \
    --do_train=True --do_eval=True \
    --bert_config_file=/path/to/mbert/bert_config.json \
    --init_checkpoint=/path/to/mbert/bert_model.ckpt --train_batch_size=BATCH \
    --num_train_steps=STEPS --save_checkpoint_steps=SAVE_STEPS \
    --num_warmup_steps=1000 --learning_rate=2e-5
```
where `STEPS` is num_epochs * len(INPUT.tfrecord) / batch_size.

You can convert the checkpoints using `scripts/modeling/convert-tf-checkpoint`.

## VA

Obtain your corpus for vocabulary creation. Run
`python scripts/bert-vocabulary/train_wordpiece_vocab.py` and
`python scripts/bert-vocabulary/select_wordpieces_for_injection.py` to generate the vocab.

With the new vocab, run the same training command for LAPT.

## TVA

Use `run_pretraining_nonsp_discrim.py` and add `--secondary_learning_rate=1e-4`,
but otherwise follow the steps for VA.

## ELMo

Follow the steps in `train_elmo_variable`.  Note that due to various
environment-related reasons, this script doesn't actually work.  Instead,
you'll need to:
1. Run `python bin/train_elmo_configurable.py --variable` with the relevant
   params (language, # epochs, train data, vocab, save dir).  The `--variable`
   flag ensures that you use a variable-length vocab.
2. Dump the weights (`python bin/dump_weights.py`) of the model
3. Update the `"n_characters"` field in the outputted `options.json` to be one
   more than it was during training (see [here](https://github.com/allenai/bilm-tf#whats-the-deal-with-n_characters-and-padding)
   for more details).

## Running downstream experiments

Run `bert-allentune-search` with the config name (see `config/`) minus the
`.jsonnet` suffix.

## Training Scripts

We've provided our training scripts (`./bert*` and `./train-elmo-variable`) for
reference.  These were used to run training (`./bert-allentune-search`),
summarize results (`./bert-allentune-report`), and evaluate
(`./bert-allentune-evaluate`).  Results are reported using the `mail` command;
feel free to comment or edit this to your preferred notification system.

