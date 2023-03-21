#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

SRC=src
CACHE=CACHE
TASK=rt
MODEL=gpt2
BERTLR=4e-5

for fold in {0..4};
do
    labels_imb=( Positive Negative )
    for split in "${labels_imb[@]}";
    do
        percentage=( 10 30 50 )    
        for percentage in "${percentage[@]}";
        do
            if [ $percentage -eq 10 ]
            then
                echo "Creating 10 samples"
                n_samples=10
            fi
            if [ $percentage -eq 30 ]
            then
                echo "Creating 5 samples"
                n_samples=5
            fi
            if [ $percentage -eq 50 ]
            then
                echo "Creating 2 samples"
                n_samples=2
            fi

            RAWDATADIR=data/task/datasets/imbalanced/${TASK}/${fold}_${split}_${percentage}/
            MODEL=gpt2

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
            # GPT2 Classifier
            #######################
            GPT2DIR=$RAWDATADIR/gpt2
            mkdir $GPT2DIR
            python3 $SRC/compare/cg_clm.py --model $MODEL --data_dir $RAWDATADIR --output_dir $GPT2DIR --task_name $TASK  --num_train_epochs 25 --top_p 0.9 --temp 1.0 --cache $CACHE --train_batch_size 2 --block_size 512 --max_seq_length 512 --eosn_index 512 --prefix 250 --sample_num $n_samples
            cat $RAWDATADIR/train_original.tsv $GPT2DIR/cmodgpt2_aug_250.tsv > $GPT2DIR/train.tsv
            cp $RAWDATADIR/test.tsv $GPT2DIR/test.tsv
            cp $RAWDATADIR/dev.tsv $GPT2DIR/dev.tsv
            python $SRC/task/classifier.py --task $TASK --data_dir $GPT2DIR --cache $CACHE > $RAWDATADIR/bert_gpt2_250.log --max_seq_length 512


            #######################
            # Aligned GPT2 Classifier 
            #######################
            MODEL=model/ft_gpt2_imdb_50
            GPT2DIR=$RAWDATADIR/gpt2tuned
            mkdir $GPT2DIR
            python3 $SRC/compare/cg_clm.py --model $MODEL --data_dir $RAWDATADIR --output_dir $GPT2DIR --task_name $TASK  --num_train_epochs 25 --top_p 0.9 --temp 1.0 --cache $CACHE --train_batch_size 2 --block_size 512 --max_seq_length 512 --eosn_index 512 --prefix 250 --sample_num $n_samples
            cat $RAWDATADIR/train_original.tsv $GPT2DIR/cmodgpt2_aug_250.tsv > $GPT2DIR/train.tsv
            cp $RAWDATADIR/test.tsv $GPT2DIR/test.tsv
            cp $RAWDATADIR/dev.tsv $GPT2DIR/dev.tsv
            python $SRC/task/classifier.py --task $TASK --data_dir $GPT2DIR --cache $CACHE > $RAWDATADIR/bert_gpt2_turned_250.log --max_seq_length 512
        done
    done 
done