from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dataset',type=str)
parser.add_argument('-path_pubmed',type=str)
parser.add_argument('-path_data',type=str)
parser.add_argument('-csv_path',type=str)

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

train_data = open(args.path_data,'r').read().lower()
train_data = re.sub(r'[^\w\s]', ' ', train_data)

train_pubmed = open(args.pubmed_data,'r').read().lower()
train_pubmed = re.sub(r'[^\w\s]', ' ', train_pubmed)

def checkAllNum(chars):
    a= re.search('[0-9]',chars)
    b = ')' in chars or '(' in chars
    if a or b: return True
    
    return False

train_data = train_data.split()
train_data = [tok for tok in train_data if not checkAllNum(tok)]

train_pubmed = train_pubmed.split()
train_pubmed = [tok for tok in train_pubmed if not checkAllNum(tok)]

train_data

from collections import Counter
freq_toks = Counter(train_data)
print(len(freq_toks))

freq_pubmed = Counter(train_pubmed)
print(len(freq_pubmed))

import pandas as pd
import re

list_datatype, list_toks, list_splits,list_freq = list(), list(), list(), list()

for tok in freq_toks:
    tokenized = tokenizer_bert.tokenize(tok)
    if len(tokenized) == 1: continue
    list_datatype.append(args.dataset)
    list_toks.append(tok)
    list_splits.append(len(tokenized))
    list_freq.append(freq_toks[tok])
        

for tok in freq_pubmed:
    tokenized = tokenizer_bert.tokenize(tok)
    if len(tokenized) == 1: continue
    list_datatype.append('PubMed')
    list_toks.append(tok)
    list_splits.append(len(tokenized))
    list_freq.append(freq_pubmed[tok])

df = pd.DataFrame({'Token':list_toks, \
                   'SplitSize': list_splits, \
                   'Frequency': list_freq, \
                   'TokenFrom': list_datatype})
df.to_csv(args.csv_path,index=False)
 
