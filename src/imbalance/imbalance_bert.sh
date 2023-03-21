#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

SRC=src
CACHE=CACHE
TASK=stsa
MODEL=bert-base-uncased
BERTLR=4e-5
n_samples=2

for fold in {0..4};
do
    labels_imb=( Positive Negative )
    for split in "${labels_imb[@]}";
    do
        percentage=( 10 30 50 )    
        for percentage in "${percentage[@]}";
        do
            RAWDATADIR=data/task/datasets/imbalanced/${TASK}/${fold}_${split}_${percentage}/
            MODEL=bert-base-uncased 

            #making sure that train file is always the correct one in multiple runs.
            if [ -f $RAWDATADIR/train_original.tsv ]; 
            then
                mv $RAWDATADIR/train_original.tsv $RAWDATADIR/train.tsv
            fi

            # Baseline classifier
            python $SRC/task/classifier.py --task $TASK  --data_dir $RAWDATADIR --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log

            cat $RAWDATADIR/train.tsv > $RAWDATADIR/train_original.tsv
            cat $RAWDATADIR/pruned.tsv > $RAWDATADIR/train.tsv
            
            
            #######################
            # BERT Classifier 
            #######################
            CMODBERTPDIR=$RAWDATADIR/cmodbertp
            mkdir $CMODBERTPDIR
            python $SRC/compare/cg_mlm.py --model $MODEL --data_dir $RAWDATADIR --output_dir $CMODBERTPDIR --task_name $TASK --num_train_epochs 10 --cache $CACHE > $RAWDATADIR/cmodbertp.log --masking_prob 0.15 --max_preds 256 --sample_num $n_samples
            cat $RAWDATADIR/train_original.tsv $CMODBERTPDIR/cmodbertp_aug.tsv > $CMODBERTPDIR/train.tsv
            cp $RAWDATADIR/test.tsv $CMODBERTPDIR/test.tsv
            cp $RAWDATADIR/dev.tsv $CMODBERTPDIR/dev.tsv
            python $SRC/task/classifier.py --task $TASK --data_dir $CMODBERTPDIR --cache $CACHE > $RAWDATADIR/bert_cmodbertp.log


            #######################
            # Aligned BERT Classifier 
            #######################
            MODEL=models/ft_bert_imdb_25
            CMODBERTPDIR=$RAWDATADIR/cmodbertptuned
            mkdir $CMODBERTPDIR
            python $SRC/compare/cg_mlm.py --model $MODEL --data_dir $RAWDATADIR --output_dir $CMODBERTPDIR --task_name $TASK  --num_train_epochs 10 --cache $CACHE > $RAWDATADIR/cmodbertp.log --masking_prob 0.15 --max_preds 256 --sample_num $n_samples
            cat $RAWDATADIR/train_original.tsv $CMODBERTPDIR/cmodbertp_aug.tsv > $CMODBERTPDIR/train.tsv
            cp $RAWDATADIR/test.tsv $CMODBERTPDIR/test.tsv
            cp $RAWDATADIR/dev.tsv $CMODBERTPDIR/dev.tsv
            python $SRC/task/classifier.py --task $TASK --data_dir $CMODBERTPDIR --cache $CACHE > $RAWDATADIR/bert_cmodbertp_tuned.log
        done
    done 
done