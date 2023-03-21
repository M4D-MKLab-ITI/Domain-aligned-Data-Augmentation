#!/usr/bin/env bash

SRC=src
RAWDATADIR=data/domain/datasets/mixed/bart/
RAWDATAFILE=data/domain/datasets/mixed/100/tune_imdb_100_lower.txt
RAWDATAFILEDEV=data/domain/datasets/mixed/bart/tune_imdb_5_lower.txt
DATABIN=data/domain/datasets/mixed/bart/imdb100/
BART_PATH=bart.base
TASK=rt #doesn't matter - it's only used to create labels correctly for fine-tuning in tasks and not here.

python $SRC/utils/bpe_encoder.py \
            --encoder-json $SRC/utils/gpt2_bpe/encoder.json \
            --vocab-bpe $SRC/utils/gpt2_bpe/vocab.bpe \
            --inputs $RAWDATAFILE  \
            --outputs $RAWDATADIR/train_bpe.src \
            --workers 1 --keep-empty --dataset $TASK


python $SRC/utils/bpe_encoder.py \
            --encoder-json $SRC/utils/gpt2_bpe/encoder.json \
            --vocab-bpe $SRC/utils/gpt2_bpe/vocab.bpe \
            --inputs $RAWDATAFILEDEV  \
            --outputs $RAWDATADIR/dev_bpe.src \
            --workers 1 --keep-empty --dataset $TASK

fairseq-preprocess --user-dir=$SRC/bart_aug --only-source \
                    --task mask_s2s \
                    --trainpref $RAWDATADIR/train_bpe.src \
                    --validpref $RAWDATADIR/dev_bpe.src \
                    --destdir $DATABIN \
                    --srcdict $BART_PATH/dict.txt