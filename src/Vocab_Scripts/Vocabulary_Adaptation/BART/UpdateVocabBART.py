#!/usr/bin/env python
# coding: utf-8
import argparse
from transformers import BartTokenizer
from tokenizers import ByteLevelBPETokenizer
import os
import pickle as pkl
import json

parser = argparse.ArgumentParser()
parser.add_argument('-v_size', type=int,required=True,description='Vocab Size to be used')
parser.add_argument('-dataset',type=str,required=True,description='Dataset Name')
parser.add_argument('-frac',type=float,required=True,description='Fraction of Vocab to be used')
parser.add_argument('-csv_path',type=str,required=True,description='Path to CSV file')

args = parser.parse_args()

df_PM = pd.read_csv(args.csv_path)
df_consider = df_PM[df_PM['Consider']==1]
df_PM = df_consider[df_consider['TokenFrom']=='PubMed']

df_TGT = pd.read_csv(args.csv_path)
df_consider = df_TGT[df_TGT['Consider']==1]
df_TGT = df_consider[df_consider['TokenFrom']==args.dataset]

list_PM_GT4, list_PM_LT4 = list(), list()

for idx in range(df_PM.shape[0]):
    if df_PM.iloc[idx,1] >=4: 
        list_PM_GT4.append(' '.join([df_PM.iloc[idx,0]]*df_PM.iloc[idx,2]))
    else: list_PM_LT4.append(' '.join([df_PM.iloc[idx,0]]*df_PM.iloc[idx,2]))

with open(f'VocabFiles_BART/PM_LT4','w') as f:
    f.write('\n'.join(list_PM_LT4))
f.close()

with open('VocabFiles_BART/PM_GT4','w') as f:
     f.write('\n'.join(list_PM_GT4))
f.close()

list_PM_All = list()
for idx in range(df_PM.shape[0]):
    list_PM_All.append(' '.join([df_PM.iloc[idx,0]]*df_PM.iloc[idx,2]))

with open('VocabFiles_BART/PM_All','w') as f:
     f.write('\n'.join(list_PM_All))
f.close()

           
list_TGT_LT4, list_TGT_GT4 = list(), list()

for idx in range(df_TGT.shape[0]):
    if df_TGT.iloc[idx,1] >=4: list_TGT_GT4.extend([df_TGT.iloc[idx,0]]*df_TGT.iloc[idx,2])
    else: list_TGT_LT4.extend([df_TGT.iloc[idx,0]]*df_TGT.iloc[idx,2])

with open(f'VocabFiles_BART/{args.dataset}_GT4','w') as f:
    f.write('\n'.join(list_TGT_GT4))
f.close()

with open(f'VocabFiles_BART/{args.dataset}_LT4','w') as f:
    f.write('\n'.join(list_TGT_LT4))
f.close()

pretrained_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
pretrained_vocab = pretrained_tokenizer.get_vocab()

domain_tokenizer_PM_GT4 = ByteLevelBPETokenizer()
domain_tokenizer_PM_GT4.train('VocabFiles_BART/PM_GT4',vocab_size=args.v_size)

domain_tokenizer_PM_LT4 = ByteLevelBPETokenizer()
domain_tokenizer_PM_LT4.train('VocabFiles_BART/PM_LT4',vocab_size=args.v_size)

domain_tokenizer_PM_All = ByteLevelBPETokenizer()
domain_tokenizer_PM_All.train('VocabFiles_BART/PM_ALL',vocab_size=args.v_size)

domain_tokenizer_TGT_GT4 = ByteLevelBPETokenizer()
domain_tokenizer_TGT_GT4.train(f'VocabFiles_BART/{args.dataset}_GT4')

domain_tokenizer_TGT_LT4 = ByteLevelBPETokenizer()
domain_tokenizer_TGT_LT4.train(f'VocabFiles_BART/{args.dataset}_LT4')

def get_domain_traits(domain_tokenizer, vocab_path):
    
    if not os.path.exists(vocab_path): os.mkdir(vocab_path)
    
    domain_tokenizer.save_model(vocab_path, prefix="custom")
    custom_merges_path = os.path.join(vocab_path,"custom-merges.txt")

    domain_vocab = domain_tokenizer.get_vocab()
    sorted_bpe = sorted(domain_vocab.items(), key=lambda x: x[-1])

    with open(custom_merges_path, "r") as reader:
        merges_file = reader.readlines()
    start_index = len(sorted_bpe) - len(merges_file) + 1
    vocab_merge_pair = []

    for bpe, merge in zip(sorted_bpe[start_index:], merges_file[1:]):
        vocab_merge_pair.append({"vocab": bpe, "merge": merge})

    vocab_merges = vocab_merge_pair

    complement_pairs = [example for example in vocab_merges if
                        example["vocab"][0] not in pretrained_vocab]
    return complement_pairs

complement_pairs_PM_GT4 = get_domain_traits(domain_tokenizer_PM_GT4,f'BART-Config/BART-PM-GT4-{args.v_size}-{args.dataset}')
complement_pairs_PM_LT4 = get_domain_traits(domain_tokenizer_PM_LT4,f'BART-Config/BART-PM-LT4-{args.v_size}-{args.dataset}')
complement_pairs_PM_All = get_domain_traits(domain_tokenizer_PM_All,f'BART-Config/BART-PM-All-{args.v_size}-{args.dataset}')

complement_pairs_TGT_GT4 = get_domain_traits(domain_tokenizer_TGT_GT4,f'BART-Config/BART-TGT-GT4-{args.v_size}-{args.dataset}')
complement_pairs_TGT_LT4 = get_domain_traits(domain_tokenizer_TGT_LT4,f'BART-Config/BART-TGT-LT4-{args.v_size}-{args.dataset}')

def retain_common_set(cp1,cp2): #cp1: pm, cp2: tgt
    l_cp2 = [x for x in cp2]
    vocab_cp1 = [d['vocab'][0] for d in cp1]
    vocab_cp2 = [[idx,d['vocab'][0]] for idx,d in enumerate(cp2)]
    
    list_del = list()
    
    for idx,key in enumerate(vocab_cp2):
        if key[1] not in vocab_cp1: list_del.append(idx)
    
    for i in  list_del[::-1]: del l_cp2[i]
        
    return l_cp2

common_part_GT4 = retain_common_set(complement_pairs_PM_GT4, complement_pairs_TGT_GT4)
common_part_LT4 = retain_common_set(complement_pairs_PM_LT4, complement_pairs_TGT_LT4)

print(len(complement_pairs_TGT_GT4),len(complement_pairs_TGT_LT4))
print(len(common_part_GT4), len(common_part_LT4))

vocab_common_GT4 = [x['vocab'][0] for x in common_part_GT4]
check_terms_GT4 = [x['merge'].strip().split() for x in common_part_GT4]
check_term_f_GT4 = [x for sub in check_terms_GT4 for x in sub if \
                   (x not in pretrained_vocab and x not in vocab_common_GT4)]

vocab_common_LT4 = [x['vocab'][0] for x in common_part_LT4]
check_terms_LT4 = [x['merge'].strip().split() for x in common_part_LT4]
check_term_f_LT4 = [x for sub in check_terms_LT4 for x in sub if \
                   (x not in pretrained_vocab and x not in vocab_common_LT4)]

pairs_GT4 = [x for x in common_part_GT4]
pairs_LT4 = [x for x in common_part_LT4]

def add_term_GT4(term):
    if (term in vocab_common_GT4) or (term in pretrained_vocab) : 
        if (term in vocab_common_GT4) : print(f'{term} in GT4: {vocab_common_GT4.index(term)}')
        else: print(f'{term} in PRETRAINED_VOCAB: {pretrained_vocab[term]}')
        return
        
    for tup in complement_pairs_TGT_GT4:
        if tup['vocab'][0] == term:
            print(tup)
            pairs_GT4.append(tup)
            vocab_common_GT4.append(term)
            
            m1,m2 = tup['merge'].strip().split()
            add_term_GT4(m1)
            add_term_GT4(m2)

for term in check_term_f_GT4:
    print('----------')
    add_term_GT4(term)
    print('----------')


def add_term_LT4(term):
    if (term in vocab_common_LT4) or (term in pretrained_vocab) : 
        if (term in vocab_common_LT4) : print(f'{term} in LT4: {vocab_common_LT4.index(term)}')
        else: print(f'{term} in PRETRAINED_VOCAB: {pretrained_vocab[term]}')
        return
    
    for tup in complement_pairs_TGT_LT4:
        if tup['vocab'][0] == term:
            print(tup)
            pairs_LT4.append(tup)
            vocab_common_LT4.append(term)
            
            m1,m2 = tup['merge'].strip().split()

            add_term_LT4(m1)
            add_term_LT4(m2)

print('*******LT4**********')
for term in check_term_f_LT4:
    print('------------')
    add_term_LT4(term)
    print('-------------')

print(len(pairs_GT4), len(pairs_LT4), len(vocab_common_LT4), len(vocab_common_GT4))

def return_union(cp1, cp2):
    cp1_c = [x for x in cp1]
    cp2_c = [x for x in cp2]
    vocab_cp1 = [d['vocab'][0] for d in cp1]
    vocab_cp2 = [[idx,d['vocab'][0]] for idx,d in enumerate(cp2)]
    
    list_del = list()
    
    for idx,key in enumerate(vocab_cp2):
        if key[1] in vocab_cp1: list_del.append(idx)
    
    for i in  list_del[::-1]: del cp2_c[i]
    
    cp1_c.extend(cp2_c)
    
    return cp1_c

union_GT4_LT4 = return_union(pairs_GT4, pairs_LT4)

def return_topk_PM(PM, GT4_LT4,frac):
    PM_C = [x for x in PM]
    vocab_cp1 = [d['vocab'][0] for d in GT4_LT4]
    vocab_cp2 = [[idx,d['vocab'][0]] for idx,d in enumerate(PM)]
    
    list_del = list()
    
    for idx,key in enumerate(vocab_cp2):
        if key[1] in vocab_cp1: list_del.append(idx)
    
    for i in  list_del[::-1]: del PM_C[i]
        
    return PM_C[:int(frac*len(GT4_LT4))]

PM_all = return_topk_PM(complement_pairs_PM_All, union_GT4_LT4,args.frac)

pairs_All = [x for x in PM_all]

vocab_common_All = [x['vocab'][0] for x in PM_all]
check_terms_All = [x['merge'].strip().split() for x in PM_all]
check_term_f_All = [x for sub in check_terms_All for x in sub if \
                   (x not in pretrained_vocab and x not in vocab_common_All)]

def add_term_All(term):
    if (term in vocab_common_All) or (term in pretrained_vocab): 
        if (term in vocab_common_All) : print(f'{term} in ALL: {vocab_common_All.index(term)}')
        else: print(f'{term} in PRETRAINED_VOCAB: {pretrained_vocab[term]}')
        return
    
    for tup in complement_pairs_PM_All:
        if tup['vocab'][0] == term:
            print(tup)
            pairs_All.append(tup)
            vocab_common_All.append(term)
            
            m1,m2 = tup['merge'].strip().split()
            add_term_All(m1)
            add_term_All(m2)
        
print('**********************ALL************************')
for term in check_term_f_All:
    print('--------')
    add_term_All(term)
    print('-------------')

complement_pairs =return_union(pairs_All,union_GT4_LT4)
pretrained_tokenizer.save_pretrained(f'BART-{args.dataset}-NewOOV/{args.v_size}-{args.frac}')
original_vocab = pretrained_tokenizer.get_vocab()

complement_pairs = sorted(complement_pairs, key = lambda x:x['vocab'][1])
print('Final Vocab Size:',len(complement_pairs))

writer = open(f'BART-{args.dataset}-NewOOV/{args.v_size}-{args.frac}/merges.txt', "a")
for vocab_merge_pair in complement_pairs:
    #print(vocab_merge_pair)
    vocab = vocab_merge_pair["vocab"]
    merge = vocab_merge_pair["merge"]
    original_vocab.update({vocab[0]: len(original_vocab)})
    writer.write(merge)
writer.close()

with open(f'BART-{args.dataset}-NewOOV/{args.v_size}-{args.frac}/vocab.json', "w") as writer:
    json.dump(original_vocab, writer)
