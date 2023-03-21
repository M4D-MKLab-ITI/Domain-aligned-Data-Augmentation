#!/usr/bin/env bash

WARMUP_UPDATES=300
LR=5e-06  # Peak LR for polynomial LR scheduler.
SRC=src
CACHE=CACHE
BART_PATH=models/bart.base
MAXEPOCH=10
MASKLEN=span
MRATIO=40

MODELDIR=models/ft_bart_imdb_100_lr5e-6_10ep_masks2s
DATABIN=data/domain/datasets/mixed/bart/imdb100/

mkdir $MODELDIR

CUDA_VISIBLE_DEVICES=0 fairseq-train  $DATABIN \
                    --skip-invalid-size-inputs-valid-test\
                    --user-dir=$SRC/bart_aug \
                    --restore-file $BART_PATH/model.pt \
                    --arch bart_base \
                    --task mask_s2s \
                    --bpe gpt2 \
                    --gpt2_encoder_json $SRC/utils/gpt2_bpe/encoder.json \
                    --gpt2_vocab_bpe $SRC/utils/gpt2_bpe/vocab.bpe \
                    --layernorm-embedding \
                    --share-all-embeddings \
                    --save-dir $MODELDIR\
                    --share-decoder-input-output-embed \
                    --reset-optimizer --reset-dataloader --reset-meters \
                    --required-batch-size-multiple 1 \
                    --max-tokens 2000 \
                    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
                    --dropout 0.1 --attention-dropout 0.1 \
                    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
                    --clip-norm 0.0 \
                    --lr-scheduler polynomial_decay --lr $LR \
                    --warmup-updates $WARMUP_UPDATES \
                    --replace-length 1 --mask-length $MASKLEN --mask $MRATIO --fp16 --update-freq 1 \
                    --max-epoch $MAXEPOCH --no-epoch-checkpoints > $MODELDIR/bart.log