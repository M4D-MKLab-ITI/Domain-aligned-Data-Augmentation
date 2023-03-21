#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

SRC=src
CACHE=CACHE
TASK=rt
MODEL=bert-base-uncased
BERTLR=4e-5

for NUMEXAMPLES in 10;
do
    for i in {0..14};
        do
        RAWDATADIR=data/task/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}
        MODEL=bert-base-uncased

        # # Baseline classifier
        python $SRC/task/classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log --max_seq_length 512

        # #######################
        # # BERT Classifier
        # #######################
        cat $RAWDATADIR/train.tsv > $RAWDATADIR/train_original.tsv

        CMODBERTPDIR=$RAWDATADIR/cmodbertp
        mkdir $CMODBERTPDIR
        python $SRC/compare/cg_mlm.py --model $MODEL --data_dir $RAWDATADIR --output_dir $CMODBERTPDIR --task_name $TASK  \
        --num_train_epochs 10 --seed ${i} --cache $CACHE > $RAWDATADIR/cmodbertp.log --train_batch_size 2 --max_seq_length 512 --masking_prob 0.15 --max_preds 256 --sample_num 5
        cat $RAWDATADIR/train_original.tsv $CMODBERTPDIR/cmodbertp_aug.tsv > $CMODBERTPDIR/train.tsv
        cp $RAWDATADIR/test.tsv $CMODBERTPDIR/test.tsv
        cp $RAWDATADIR/dev.tsv $CMODBERTPDIR/dev.tsv
        python $SRC/task/classifier.py --task $TASK --data_dir $CMODBERTPDIR --seed ${i}  --cache $CACHE > $RAWDATADIR/bert_cmodbertp.log --max_seq_length 512

        MODEL=models/ft_bert_imdb_100
        CMODBERTPDIR=$RAWDATADIR/cmodbertptuned
        mkdir $CMODBERTPDIR
        python $SRC/compare/cg_mlm.py --model $MODEL --data_dir $RAWDATADIR --output_dir $CMODBERTPDIR --task_name $TASK  \
        --num_train_epochs 10 --seed ${i} --cache $CACHE > $RAWDATADIR/cmodbertp.log --train_batch_size 2 --max_seq_length 512 \
        --masking_prob 0.15 --max_preds 256 --sample_num 5
        cat $RAWDATADIR/train_original.tsv $CMODBERTPDIR/cmodbertp_aug.tsv > $CMODBERTPDIR/train.tsv
        cp $RAWDATADIR/test.tsv $CMODBERTPDIR/test.tsv
        cp $RAWDATADIR/dev.tsv $CMODBERTPDIR/dev.tsv
        python $SRC/task/classifier.py --task $TASK --data_dir $CMODBERTPDIR --seed ${i}  --cache $CACHE > $RAWDATADIR/bert_cmodbertp_tuned.log --max_seq_length 512

    done
done


