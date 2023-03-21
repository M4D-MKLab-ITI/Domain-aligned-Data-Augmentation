#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

SRC=src
CACHE=CACHE
TASK=rt
MODEL=gpt2
BERTLR=4e-5

for NUMEXAMPLES in 10;
do
    for i in {0..14};
        do
        RAWDATADIR=data/task/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}
        MODEL=gpt2

       # Baseline classifier
        python $SRC/task/classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log

        #######################
        # GPT2 Classifier
        #######################
        cat $RAWDATADIR/train.tsv > $RAWDATADIR/train_original.tsv

        GPT2DIR=$RAWDATADIR/gpt2
        mkdir $GPT2DIR
        python3 $SRC/compare/cg_clm.py --model $MODEL --data_dir $RAWDATADIR --output_dir $GPT2DIR \
        --task_name $TASK  --num_train_epochs 25 --seed ${i} --top_p 0.9 --temp 1.0 --cache $CACHE --train_batch_size 2\
        --block_size 512 --max_seq_length 512 --eosn_index 512 --sample_num 5
        cat $RAWDATADIR/train_original.tsv $GPT2DIR/cmodgpt2_aug_3.tsv > $GPT2DIR/train.tsv
        cp $RAWDATADIR/test.tsv $GPT2DIR/test.tsv
        cp $RAWDATADIR/dev.tsv $GPT2DIR/dev.tsv
        python $SRC/task/classifier.py --task $TASK --data_dir $GPT2DIR --seed ${i} --cache $CACHE > $RAWDATADIR/bert_gpt2_3.log --max_seq_length 512

        MODEL=models/ft_gpt2_imdb_100_lr5e-6_5ep
        GPT2DIR=$RAWDATADIR/gpt2tuned
        mkdir $GPT2DIR
        python3 $SRC/compare/cg_clm.py --model $MODEL --data_dir $RAWDATADIR --output_dir $GPT2DIR \
        --task_name $TASK  --num_train_epochs 25 --seed ${i} --top_p 0.9 --temp 1.0 --cache $CACHE --train_batch_size 2\
        --block_size 512 --max_seq_length 512 --eosn_index 512 --sample_num 5
        cat $RAWDATADIR/train_original.tsv $GPT2DIR/cmodgpt2_aug_3.tsv > $GPT2DIR/train.tsv
        cp $RAWDATADIR/test.tsv $GPT2DIR/test.tsv
        cp $RAWDATADIR/dev.tsv $GPT2DIR/dev.tsv
        python $SRC/task/classifier.py --task $TASK --data_dir $GPT2DIR --seed ${i} --cache $CACHE > $RAWDATADIR/bert_gpt2_turned_3.log --max_seq_length 512
    done
done


