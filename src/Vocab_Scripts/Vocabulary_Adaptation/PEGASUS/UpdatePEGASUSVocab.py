#!/usr/bin/env python
import argparse
from transformers import PegasusTokenizer
import sentencepiece as spm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-csv_path',type=str,required=True,description='Path to CSV file containing token data')
parser.add_argument('-v_size_PM_GT4', type=int,required=True,description='Vocab Size to be used')
parser.add_argument('-v_size_PM_LT4', type=int,required=True,description='Vocab Size to be used')
parser.add_argument('-v_size_PM_ALL', type=int,required=True,description='Vocab Size to be used')
parser.add_argument('-v_size_TGT_GT4', type=int,required=True,description='Vocab Size to be used')
parser.add_argument('-v_size_TGT_LT4', type=int,required=True,description='Vocab Size to be used')
parser.add_argument('-dataset',type=str,required=True,description='Target Dataset Name')
parser.add_argument('-frac',type=float,required=True,description='Fraction of Vocab to be used')

args = parser.parse_args()
csv_path = args.csv_path

df=pd.read_csv(csv_path[args.dataset])
df = df[df['Consider']==1]

##--PAC-SUmm Files
df_PM = df[df['TokenFrom']=='PubMed'].dropna()

df_PM_GT4 = df_PM[df_PM['SplitSize']>=4]
df_PM_GT4 = df_PM_GT4.drop(columns=['SplitSize','TokenFrom','Consider'])
df_PM_GT4.to_csv(f'VocabFiles_PEGASUS/{args.dataset}_PM_GT4_Freq.tsv',index=False,sep='\t',header=False)

df_PM_LT4 = df_PM[df_PM['SplitSize']<4]
df_PM_LT4 = df_PM_LT4.drop(columns=['SplitSize','TokenFrom','Consider'])
df_PM_LT4.to_csv(f'VocabFiles_PEGASUS/{args.dataset}_PM_LT4_Freq.tsv',index=False,sep='\t',header=False)

df_PM_ALL = df_PM
df_PM_ALL = df_PM_ALL.drop(columns=['SplitSize','TokenFrom','Consider'])
df_PM_ALL.to_csv(f'VocabFiles_PEGASUS/{args.dataset}_PM_ALL_Freq.tsv',index=False,sep='\t',header=False)

##--TGT Files
df_TGT = df[df['TokenFrom']==args.dataset].dropna()
df_TGT = df_TGT[df_TGT['Consider']==1]
df_TGT = df_TGT.dropna()

df_TGT_GT4 = df_TGT[df_TGT['SplitSize']>=4]
df_TGT_GT4 = df_TGT_GT4.drop(columns=['SplitSize','TokenFrom','Consider'])
df_TGT_GT4.to_csv(f'VocabFiles_PEGASUS/{args.dataset}_GT4_Freq.tsv',index=False,sep='\t',header=False)

df_TGT_LT4 = df_TGT[df_TGT['SplitSize']<4]
df_TGT_LT4 = df_TGT_LT4.drop(columns=['SplitSize','TokenFrom','Consider'])
df_TGT_LT4.to_csv(f'VocabFiles_PEGASUS/{args.dataset}_LT4_Freq.tsv',index=False,sep='\t',header=False)


## -- Training Target
spm.SentencePieceTrainer.train(f'--input=VocabFiles_PEGASUS/{args.dataset}_LT4_Freq.tsv --input_format=tsv \
        --model_prefix=VocabFiles_PEGASUS/{args.dataset}_LT4 --vocab_size={args.v_size_TGT_LT4}')

spm.SentencePieceTrainer.train(f'--input=VocabFiles_PEGASUS/{args.dataset}_GT4_Freq.tsv --input_format=tsv \
         --model_prefix=VocabFiles_PEGASUS/{args.dataset}_GT4 --vocab_size={args.v_size_TGT_GT4}')

##-- Training PAC-Summ

spm.SentencePieceTrainer.train(f'--input=VocabFiles_PEGASUS/EBM_PM_GT4_Freq.tsv --input_format=tsv \
                                --model_prefix=VocabFiles_PEGASUS/EBM_PM_GT4 --vocab_size={args.v_size_PM_GT4}')

spm.SentencePieceTrainer.train(f'--input=VocabFiles_PEGASUS/EBM_PM_ALL_Freq.tsv --input_format=tsv \
                                --model_prefix=VocabFiles_PEGASUS/EBM_PM_ALL --vocab_size={args.v_size_PM_ALL}')

spm.SentencePieceTrainer.train(f'--input=VocabFiles_PEGASUS/EBM_PM_LT4_Freq.tsv --input_format=tsv \
                                --model_prefix=VocabFiles_PEGASUS/EBM_PM_LT4 --vocab_size={args.v_size_PM_LT4}')



tok = PegasusTokenizer.from_pretrained('google/pegasus-large')
org_vocab = tok.get_vocab()

list_PM_GT4, list_PM_LT4, list_PM_All = list(), list(), list()

with open(f'VocabFiles_PEGASUS/EBM_PM_LT4.vocab','r') as f:
    for idx,line in enumerate(f):
        if idx < 4: continue
        term = line.split()[0]
        if term in org_vocab: continue
        list_PM_LT4.append(term)


with open(f'VocabFiles_PEGASUS/EBM_PM_GT4.vocab','r') as f:
    for idx,line in enumerate(f):
        if idx < 4: continue
        term = line.split()[0]
        if term in org_vocab: continue
        list_PM_GT4.append(term)
        

with open(f'VocabFiles_PEGASUS/EBM_PM_ALL.vocab','r') as f:
    for idx,line in enumerate(f):
        if idx < 4: continue
        term = line.split()[0]
        if term in org_vocab: continue
        list_PM_All.append(line)


# In[ ]:


list_TGT_GT4, list_TGT_LT4 = list(), list()
with open(f'VocabFiles_PEGASUS/{args.dataset}_GT4.vocab','r') as f:
    for idx,line in enumerate(f):
        if idx < 4: continue
        term = line.split()[0]
        if term in org_vocab: continue
        if term in list_PM_GT4: list_TGT_GT4.append(line)
    

with open(f'VocabFiles_PEGASUS/{args.dataset}_LT4.vocab','r') as f:
    for idx,line in enumerate(f):
        if idx < 4: continue
        term = line.split()[0]
        if term in org_vocab: continue
        if term in list_PM_LT4: list_TGT_LT4.append(line)

def get_Union(l1,l2):
    ret_list = [x for x in l1]
    ret_list_keys = [x.split()[0] for x in ret_list]
    for row in l2:
        if row.split()[0]  not in ret_list_keys: ret_list.append(row)
    
    return ret_list

TGT_Vocab = get_Union(list_TGT_GT4,list_TGT_LT4)

print('GT4_LT4 Size:',len(TGT_Vocab))

def get_Union_PM(l1,l2,frac):
    ret_list = [x for x in l1]
    ret_list_keys = [x.split()[0] for x in ret_list]
    new_list = list()
    
    v_size = int(frac*len(ret_list))
    
    added = 0
    for row in l2:
        if added >= v_size: break
        
        if row.split()[0]  not in ret_list_keys:
            if row.split()[0] not in org_vocab:
                new_list.append(row)
                added +=1
    
    return new_list+ret_list

FINAL_Vocab = get_Union_PM(TGT_Vocab,list_PM_All,args.frac)
print('Final_Vocab:', len(FINAL_Vocab))

with open(dump_path,'w') as f:  #put dump path as directory of your choice
    f.write(''.join(FINAL_Vocab))
f.close()

