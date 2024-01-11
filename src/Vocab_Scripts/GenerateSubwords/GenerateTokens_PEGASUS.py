import re
from transformers import PegasusTokenizer
from collections import Counter, defaultdict
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-dataset',type=str)
parser.add_argument('-path_pubmed',type=str)
parser.add_argument('-path_data',type=str)
parser.add_argument('-csv_path',type=str)


tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')

train_pubmed = open(args.path_pubmed,'r').read()
train_pubmed = re.sub(r'[^\w\s]', ' ', train_pubmed)

train_data = open(args.path_data,'r').read()
train_data = re.sub(r'[^\w\s]', ' ', train_data)

def checkAllNum(chars):
    a= re.search('[0-9]',chars)
    b = ')' in chars or '(' in chars
    if a or b: return True
    
    return False

train_pubmed = train_pubmed.split()
train_pubmed = [tok for tok in train_pubmed if not checkAllNum(tok)]

train_data = train_data.split()
train_data = [tok for tok in train_data if not checkAllNum(tok)]

freq_pubmed = Counter(train_pubmed)
freq_data = Counter(train_data)

list_datatype, list_toks, list_splits,list_freq = list(), list(), list(), list()

for tok,freq in freq_pubmed.items():
    tokenized_ids = tokenizer('the '+tok)['input_ids']
    if len(tokenized_ids) >= 4:  
        list_datatype.append('PubMed')
        list_toks.append(tok)
        list_splits.append(len(tokenized_ids)-2)
        list_freq.append(freq)
        
for tok,freq in freq_data.items():
    tokenized_ids = tokenizer('the '+tok)['input_ids']
    if len(tokenized_ids) >= 4:  
        list_datatype.append('CHQ')
        list_toks.append(tok)
        list_splits.append(len(tokenized_ids)-2)
        list_freq.append(freq)

df = pd.DataFrame({'Token':list_toks, \
                   'SplitSize': list_splits, \
                   'Frequency': list_freq, \
                   'TokenFrom': list_datatype})

df.to_csv(args.csv_path,index=False)

