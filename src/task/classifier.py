# Modifications Copyright Nikolaos Stylianou (nstylia@iti.gr)
# Original Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0


import torch
import argparse

from data_processors import get_data
from engine import Classifier
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    examples, label_list = get_data(
        task=args.task,
        data_dir=args.data_dir,
        data_seed=args.seed)

    t_total = len(examples['train']) // args.epochs

    classifier = Classifier(label_list=label_list, device=device, cache_dir=args.cache)
    classifier.get_optimizer(learning_rate=args.learning_rate,
                             warmup_steps=args.warmup_steps,
                             t_total=t_total)

    classifier.load_data(
        'train', examples['train'], args.batch_size, max_length=args.max_seq_length, shuffle=True)
    classifier.load_data(
        'dev', examples['dev'], args.batch_size, max_length=args.max_seq_length, shuffle=False)
    classifier.load_data(
        'test', examples['test'], args.batch_size, max_length=args.max_seq_length, shuffle=False)

    print('=' * 60, '\n', 'Training', '\n', '=' * 60, sep='')
    best_dev_acc, final_test_acc = -1., -1.
    best_dev_gmean, best_dev_f1 = -1., -1.
    for epoch in range(args.epochs):
        classifier.train_epoch()
        dev_acc, _, _, dev_gmean, dev_f1 = classifier.evaluate('dev')

        if epoch >= args.min_epochs:
            do_test = (dev_f1 > best_dev_f1) #this is for imbalanced
            # best_dev_acc = max(best_dev_acc, dev_acc) #this is for lowresource
            # best_dev_gmean = max(best_dev_gmean, dev_gmean) 
            best_dev_f1 = max(best_dev_f1, dev_f1) #this is for imbalanced
        else:
            do_test = False

        print('Epoch {}, Dev F1: {:.4f}, Best Ever: {:.4f}'.format(
            epoch, 100. * dev_f1, 100. * best_dev_f1))

        if do_test:
            final_test_acc, report, auc, gmean, f1 = classifier.evaluate('test')
            print('Test Acc: {:.4f}'.format(100. * final_test_acc))
            print(report)
            print(f'AUC: {auc}')
            print(f'Gmean: {gmean}')
            print(f'F1: {f1}')

    print('Epoch {}, Dev F1: {:.4f}, Best Ever: {:.4f}'.format(
            epoch, 100. * dev_f1, 100. * best_dev_f1))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', choices=['stsa', 'snips', 'trec', 'rt'])
    parser.add_argument('--data_dir', type=str, help="Data dir path with {train, dev, test}.tsv")
    parser.add_argument('--seed', default=159, type=int)
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")

    parser.add_argument('--cache', default="transformers_cache", type=str)

    parser.add_argument('--epochs', default=8, type=int)
    parser.add_argument('--min_epochs', default=0, type=int)
    parser.add_argument("--learning_rate", default=4e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)

    args = parser.parse_args()
    print(args)
    main(args)

