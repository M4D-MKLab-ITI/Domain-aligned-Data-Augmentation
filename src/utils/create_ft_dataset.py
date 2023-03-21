import os
import argparse
import random

from datasets import percent

CURRENT_DIR = os.getcwd()
DOMAIN_DATA_DIR = 'data/domain/datasets/'
OUTPUT_DATA_DIR = 'data/domain/datasets/mixed/'
SEED = 2310

def create_dataset(output_path:str,out_file_name:str):
    if not os.path.exists(output_path):
        print(f'Creating {output_path} directory')
        os.mkdir(output_path)
    with open(os.path.join(output_path,out_file_name), 'w') as f:
        print(f'Creating {out_file_name} dataset file')
        pass
    return True

def extend_dataset(output_path:str,out_file_name:str, data:list):
    print(f'Adding {len(data)} to {out_file_name}')
    with open(os.path.join(output_path,out_file_name), 'a') as f:
        for d in data: 
            f.write(d+'\n')
    return True

def prepare_amazon(percent:int,lower:bool,outfile:str,use_summary:bool=False):
    import json
    data = []
    path = os.path.join(DOMAIN_DATA_DIR,'amazon')
    files=os.listdir(path)
    #sampling equally the percentage specified from all files/categories. 
    for file in files:
        print(f'Processing: {file}')
        assert '.gz' not in file, 'Unziped file detected. Exiting...'
        with open(os.path.join(path,file), 'r') as json_file:
            lines = json_file.readlines()

        random.Random(SEED).shuffle(lines)
        
        end_index = (len(lines)*percent)//100
        i = 0
        while i < end_index and i < len(lines):
            point = json.loads(lines[i])
            try: 
                review=point['reviewText'].strip('\n')
                text = ' '.join([review,point['summary']]).strip('\n') if use_summary else review
                if lower:
                    text = text.lower()
                data.append(text.replace('\n', ' '))
                i+=1
            except KeyError:
                i+=1
                end_index+=1
        extend_dataset(OUTPUT_DATA_DIR,outfile,data)
        data = []
    return data

def prepare_yelp(percent:int,lower:bool):
    import json 

    data = []
    path = os.path.join(DOMAIN_DATA_DIR,'yelp')
    file = os.listdir(path)[0]
    with open(os.path.join(path,file), 'r') as json_file:
        lines = json_file.readlines()

    random.Random(SEED).shuffle(lines)
    end_index = (len(lines)*percent)//100

    for i in range(end_index):
        line = lines[i]
        point = json.loads(line)
        data.append(point['text'].strip().lower().replace('\n', ' ')) if lower else data.append(point['text'].strip().replace('\n', ' '))
    return data

def prepare_imdb(percent:int,lower:bool,split:str='train',balanced:bool=True):
    '''
    Balanced and unbalanced option return different number of data. 
    '''
    data = []
    path = os.path.join(DOMAIN_DATA_DIR,'imdb/aclImdb')
    path = os.path.join(path,split)
    options = ['pos', 'neg', 'unsup']
    if not balanced:
        print('Creating randomly picked - unbalanced selectiong of IMDB text')
        for option in options: 
            fnames = [option+'/'+str(file) for file in os.listdir(os.path.join(path,option))]
            
        random.Random(SEED).shuffle(fnames)

        end_index = (len(fnames)*percent)//100
        for i in range(end_index):
            with open(os.path.join(path,fnames[i]), 'r') as f:
                text = f.readline()
            data.append(text.strip().lower().replace('\n', ' ')) if lower else data.append(text.strip().replace('\n', ' '))
        
    else: 
        print('Creating balanced selectiong of IMDB text')
        pos_fnames = ['pos/'+str(file) for file in os.listdir(os.path.join(path,'pos'))]
        neg_fnames = ['neg/'+str(file) for file in os.listdir(os.path.join(path,'neg'))]
        unsup_fnames = ['unsup/'+str(file) for file in os.listdir(os.path.join(path,'unsup'))]
        
        random.Random(SEED).shuffle(pos_fnames)
        random.Random(SEED).shuffle(neg_fnames)
        random.Random(SEED).shuffle(unsup_fnames)

        options = [pos_fnames, neg_fnames, unsup_fnames]

        for option in options:
            end_opt_index = (len(option)*percent)//100
            for i in range(end_opt_index):
                with open(os.path.join(path,option[i]), 'r') as f:
                    text = f.readline()
                data.append(text.strip().lower().replace('\n', ' ')) if lower else data.append(text.strip().replace('\n', ' '))
        
        random.Random(SEED).shuffle(data) #shuffling data to mix different options.

    return data

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create a fine-tuning dataset from a collection of in-domain datapoints')
    #data selection
    parser.add_argument('--collection', metavar='--C', help='Data collection to be chose from', type=str, default='all', required=True)
    collection_options=['amazon', 'mdsd', 'yelp', 'imdb' ,'all']

    parser.add_argument('--percentage', metavar='-P', help='Percentage to sample from each file in the dataset', required=True, type=int, default=20)

    # data pre-processing steps
    parser.add_argument('--lower', action='store_true', default=False)

    args = parser.parse_args()

    assert args.collection in collection_options, "Invalid dataset collection selected"
    assert args.percentage in range(0,101), "Invalid percentage selected"
    
    out = 'tune_'+args.collection+'_'+str(args.percentage)+'_'
    out+= 'lower.txt' if args.lower else '.txt'

    OUTPUT_DATA_DIR = os.path.join(OUTPUT_DATA_DIR,str(args.percentage)+"/")
    create_dataset(OUTPUT_DATA_DIR, out)
    if args.collection == 'all':
        data = prepare_amazon(percent=args.percentage,lower=args.lower, outfile=out)
        data = prepare_yelp(percent=args.percentage,lower=args.lower)
        extend_dataset(OUTPUT_DATA_DIR, out, data)
        data = prepare_imdb(percent=args.percentage,lower=args.lower)
        extend_dataset(OUTPUT_DATA_DIR, out, data)  
    elif args.collection == 'amazon':
        data = prepare_amazon(percent=args.percentage,lower=args.lower, outfile=out)
        extend_dataset(OUTPUT_DATA_DIR, out, data)
    elif args.collection == 'mdsd':
        data = prepare_mdsd(percent=args.percentage,lower=args.lower)
        extend_dataset(OUTPUT_DATA_DIR, out, data)
    elif args.collection == 'yelp':
        data = prepare_yelp(percent=args.percentage,lower=args.lower)
        extend_dataset(OUTPUT_DATA_DIR, out, data)
    elif args.collection == 'imdb':
        data = prepare_imdb(percent=args.percentage,lower=args.lower)
        extend_dataset(OUTPUT_DATA_DIR, out, data)

    print(f'Fine-tune dataset collection finished and stored in: {OUTPUT_DATA_DIR}{out}')
