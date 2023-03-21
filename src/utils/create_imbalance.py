import argparse
import os
import random

CURRENT_DIR = os.getcwd()
TASK_DATA_DIR = 'data/task/datasets/'
OUTPUT_DATA_DIR = 'data/task/datasets/imbalanced/'

def save_imbalanced(data, outpath, split):
    split+='.tsv'
    out_path = os.path.join(outpath, split)
    print(f'Saving data in: {out_path}')

    with open(out_path, 'w') as fout:
        for sample in data: 
            fout.write(sample)

    return True

def create_imbalance(data, label, percent, seed):
    
    if label == 'Positive':
        sample = data['pos']
        imbalanced = data['neg']
    else:
        sample = data['neg']
        imbalanced = data['pos']
    
    random.Random(seed).shuffle(sample)
    
    end_index = (len(sample)*percent)//100
    pruned_label_samples = sample[0:end_index]
    print(f'Label {"Positive" if label=="Positive" else "Negative"} has been downsized to {len(pruned_label_samples)} samples' )

    imbalanced.extend(pruned_label_samples)
    print(f'Final dataset is contains {len(imbalanced)} samples')

    random.Random(seed).shuffle(imbalanced)

    return imbalanced, pruned_label_samples, sample

def read_dataset(dataset, split, lower):
    split+='.tsv'
    filepath = os.path.join(dataset,split)
    print(f'Reading data from {filepath}..')
    with open(filepath, 'r') as fin: 
        lines = fin.readlines()

    dataset={'pos':[],'neg':[]}
    for line in lines:
        lp = line.strip().split('\t')
        if 'Positive'==lp[0]:
            dataset['pos'].append(line.lower()) if lower else dataset['pos'].append(line)
        else:
            dataset['neg'].append(line.lower()) if lower else dataset['neg'].append(line)

    print(f'Reading complete. Data consists of: \n {len(dataset["pos"])} Positive samples and {len(dataset["neg"])} samples')
    return dataset

def main(): 
    parser = argparse.ArgumentParser(description='Create a imbalanced dataset')
    #data selection
    parser.add_argument('--collection', metavar='--C', help='Data collection to be chose from',
                         type=str, default='all', required=True, choices=['stsa', 'rt'])
    parser.add_argument('--imbalance_label', metavar='--L', help='Label to be undersampled',
                         type=str, required=True, default='Positive', choices=['Positive', 'Negative'])
    parser.add_argument('--percentage', metavar='-P', help='Sample percentage of the imbalanced class (e.g. 30' 
                        'is 30 percent of all samples in the original dataset from the selected class', required=True, type=int, default=30)
    parser.add_argument('--seed', metavar='-S', help='Seed number for random sample selection.', required=True, type=int, default=1337)                        
    parser.add_argument('--folds', metavar='-F', help='Number of testing folds', required=True, type=int, default=5)
    # data pre-processing steps
    parser.add_argument('--lower', action='store_true', default=False)

    args = parser.parse_args()
    print(f'Creating {args.percentage}% imbalance in {args.imbalance_label} class from {args.collection} \n')
    
    seed = args.seed
    
    out_path = os.path.join(OUTPUT_DATA_DIR,args.collection)
    
    if not os.path.exists(os.path.join(out_path)):
        print(f'Creating directory {out_path}')
        os.mkdir(os.path.join(out_path))

    for fold in range(args.folds):

        fold_path = os.path.join(out_path,'_'.join([str(fold), args.imbalance_label, str(args.percentage)]))
        if not os.path.exists(os.path.join(fold_path)):
            print(f'Creating directory {fold_path}')
            os.mkdir(os.path.join(fold_path))

        dataset = os.path.join(TASK_DATA_DIR, args.collection)
        for split in ['train','dev']: 
            data = read_dataset(dataset, split, args.lower)
            imbalanced, pruned_label_samples, unpruned_label_samples = create_imbalance(data, args.imbalance_label, args.percentage, seed)
            save_imbalanced(imbalanced, fold_path, split)
            if split == 'train':
                save_imbalanced(imbalanced, fold_path, 'pruned')
            print('\n')
        
        full_percent = 100
        for split in ['test']:
            data = read_dataset(dataset, split, args.lower)
            imbalanced, _, _ = create_imbalance(data, args.imbalance_label, full_percent, seed)
            save_imbalanced(imbalanced, fold_path, split)
    
    return 

if __name__=='__main__':
    main()
