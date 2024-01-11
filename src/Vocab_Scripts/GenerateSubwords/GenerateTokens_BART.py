from transformers import BartTokenizer
import pandas as pd      
import re
from collections import Counter
import argparse

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

parser = argparse.ArgumentParser()
parser.add_argument('-dataset',type=str)
parser.add_argument('-path_pubmed',type=str)
parser.add_argument('-path_data',type=str)
parser.add_argument('-csv_path',type=str)

train_data = open(args.path_data,'r').read().lower()
train_data = re.sub(r'[^\w\s]', ' ', train_data)

train_pubmed = open(args.pubmed_data,'r').read().lower()
train_pubmed = re.sub(r'[^\w\s]', ' ', train_pubmed)

def checkAllNum(chars):
    a= re.search('[0-9]',chars)
    b = ')' in chars or '(' in chars
    if a or b: return True
    
    return False

train_pubmed = train_pubmed.split()
train_pubmed = [tok for tok in train_pubmed if not checkAllNum(tok)]
freq_pubmed = Counter(train_pubmed)
print(len(train_pubmed),len(freq_pubmed))

train_data = train_data.split()
train_data = [tok for tok in train_data if not checkAllNum(tok)]
train_data_Freq = Counter(train_data)

list_datatype, list_toks, list_splits,list_freq = list(), list(), list(), list()

for tok,freq in freq_pubmed.items():
    tokenized_ids = tokenizer('the '+tok)['input_ids']
    if len(tokenized_ids) >=5 :
        list_datatype.append('PubMed')
        list_toks.append(tok)
        list_splits.append(len(tokenized_ids)-3)
        list_freq.append(freq)
        

for tok,freq in train_data_Freq.items():
    tokenized_ids = tokenizer('the '+tok)['input_ids']
    if len(tokenized_ids) >=5 :
        list_datatype.append(args.dataset)
        list_toks.append(tok)
        list_splits.append(len(tokenized_ids)-3)
        list_freq.append(freq)


df = pd.DataFrame({'Token':list_toks, \
                   'SplitSize': list_splits, \
                   'Frequency': list_freq,\
                   'TokenFrom': list_datatype})
df.to_csv(args.csv_path,index=False)
 
