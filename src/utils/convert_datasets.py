'''
Copyright Nikolaos Stylianou (nstylia@iti.gr)
Convert task-specific labeled datasets to pure-text (non-labeled) txt files for
Language Model consumption and evaluation of pretrained and finetuned models.
'''
import os 
import random

DATA_DIR = 'data/task/datasets/'
SEED = 2310 

def to_tsv(data_path:str, output_path:str, fold_train:list, fold_val:list, fold_test:list): 
    os.makedirs(output_path, exist_ok=True)
    
    train_data = []
    for fname in fold_train:
        with open(os.path.join(data_path, fname), 'r') as f:
            point = f.readlines()
        text = ''.join(point).replace('\n', ' ')
        label = 'Positive' if 'pos' in fname else 'Negative'
        train_data.append([label, text])
    
    with open(os.path.join(output_path, 'train.tsv'), 'w') as f:
        for data in train_data:
            f.write('\t'.join(data)+'\n')
    
    val_data = []
    for fname in fold_val:
        with open(os.path.join(data_path, fname), 'r') as f:
            point = f.readlines()
        text = ''.join(point).replace('\n', ' ')
        label = 'Positive' if 'pos' in fname else 'Negative'
        val_data.append([label, text])
    
    with open(os.path.join(output_path, 'dev.tsv'), 'w') as f:
        for data in val_data:
            f.write('\t'.join(data)+'\n')

    test_data = []
    for fname in fold_test:
        with open(os.path.join(data_path, fname), 'r') as f:
            point = f.readlines()
        text = ''.join(point).replace('\n', ' ')
        label = 'Positive' if 'pos' in fname else 'Negative'
        test_data.append([label, text])
    
    with open(os.path.join(output_path, 'test.tsv'), 'w') as f:
        for data in test_data:
            f.write('\t'.join(data)+'\n')
    return True 

def rt_for_task(data_path:str, output_path:str, percent:int=20, samples=10, folds=15):
    '''
    creates a given number of folds - randomly picking from shuffled examples 
    the exampels are not class mixed to maintain balance between labels for LM task-specific fine-tuning.
    '''
    
    fnames_pos = []
    pos_dir = os.path.join(data_path, 'pos')
    for fname in os.listdir(pos_dir):
        fnames_pos.append('pos/'+str(fname))

    fnames_neg = []
    neg_dir = os.path.join(data_path, 'neg')
    for fname in os.listdir(neg_dir):
        fnames_neg.append('neg/'+str(fname))

    random.Random(SEED).shuffle(fnames_pos)



    return True

def rt_for_task_per_fold(data_path:str, output_path:str, percent:int=20, samples=10):
    '''
    Based on readme: 
    fold 1: files tagged cv000 through cv099, in numerical order
    fold 2: files tagged cv100 through cv199, in numerical order
    etc. 
    Creates 10 folds of data - based on the pre-split fold of the dataset
    '''
    
    fnames_pos = []
    pos_dir = os.path.join(data_path, 'pos')
    for fname in os.listdir(pos_dir):
        fnames_pos.append('pos/'+str(fname))

    fnames_neg = []
    neg_dir = os.path.join(data_path, 'neg')
    for fname in os.listdir(neg_dir):
        fnames_neg.append('neg/'+str(fname))
    
    folds = [] 
    for i in range(10):
        folds.append([[],[]])
    
    for fname in fnames_pos:
        fold = int([d for d in fname.split('_')[0] if d.isdigit()][0]) #get first number - indicating fold
        folds[fold][0].append(fname)
    
    for fname in fnames_neg:
        fold = int([d for d in fname.split('_')[0] if d.isdigit()][0]) #get first number - indicating fold
        folds[fold][1].append(fname)
    
    for fold in folds:
        random.Random(SEED).shuffle(fold[0]) #mix positives within the same fold
        random.Random(SEED).shuffle(fold[1]) #mix negatives within the same fold
        
    #having built the folds, we will create 10 experiment files with train/dev/test tsv files and
    
    for i in range(10):
        train_data, dev_data, test_data = [], [], []
        cur_split = [train_data, dev_data, test_data]
        if samples:
            for split in cur_split:
                if samples < (len(folds[i][0])+len(folds[i][1])):
                    pos_samples = random.sample(folds[i][0], samples)
                    neg_samples = random.sample(folds[i][1], samples)
                else: 
                    pos_samples = folds[i][0]
                    neg_samples = folds[i][1]
                split.extend(pos_samples)
                split.extend(neg_samples)
            output_fold_path = os.path.join(output_path,f'exp_{i}_{samples}')
            to_tsv(data_path, output_fold_path, train_data, dev_data, test_data)

    return True 

def rt_for_lm_eval(data_path:str, output_path:str):
    '''
    Will only take the first 5 folds of the dataset (as no clear train set). 
    '''
    fnames = []
    pos_dir = os.path.join(data_path, 'pos')
    for fname in os.listdir(pos_dir):
        fold = int([d for d in fname.split('_')[0] if d.isdigit()][0]) #get first number - indicating fold
        if fold <= 5:
            fnames.append('pos/'+str(fname))

    neg_dir = os.path.join(data_path, 'neg')
    for fname in os.listdir(neg_dir):
        fold = int([d for d in fname.split('_')[0] if d.isdigit()][0]) #get first number - indicating fold
        if fold <= 5:
            fnames.append('neg/'+str(fname))
     
    data = []
    for fname in fnames:
        with open(os.path.join(data_path,fname), 'r') as f:
            point = f.readlines()
        text = ''.join(point).replace('\n', ' ')
        data.append(text)
    with open(output_path, 'w') as f:
        for point in data: 
            f.write(point+'\n')

    return True

def for_lm_eval(input_path:str, output_path:str):
    new_data = []
    with open(input_path, 'r') as f:
        data = f.readlines()
    for line in data: 
        point = line.strip().split('\t')
        if len(point)==2: 
            text = point[-1]
            new_data.append(text)

    with open(output_path, 'w') as f:
        for d in data: 
            f.write(d+'\n')
    return True

if __name__=='__main__':
    simple_tasks = ['rt', 'trec', 'stsa', 'snips']
    for task in simple_tasks: 
        print(f'Creating LM evaluation file for {task}')
        current_dir = os.path.join(DATA_DIR,task)
        if task == 'rt':
            current_output_dir = os.path.join(current_dir, 'train.txt')
            data_dir = os.path.join(current_dir,'txt_sentoken')
            rt_for_task_per_fold(data_dir,current_dir)
        else:
            #only do train set -otherwise loop from os.listdir on files only 
            current_input_dir = os.path.join(current_dir,'train.tsv')
            current_output_dir = os.path.join(current_dir, 'train.txt')
            for_lm_eval(current_input_dir,current_output_dir)

    