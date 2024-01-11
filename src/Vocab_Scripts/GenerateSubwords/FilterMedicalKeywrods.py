#!/usr/bin/env python
# coding: utf-8
import argparse
from quickumls import QuickUMLS
import pandas as pd
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-umls_path',type=str)
parser.add_argument('-csv_path',type=str)

matcher = QuickUMLS(umls_path,similarity_name='cosine') #put the default directory of UMLS here.

print(f'Starting for {fname}.....')
df = pd.read_csv(fname)
list_consider = list()

for idx in range(df.shape[0]):
    if idx%10000 == 0: print(f'Processed till {idx+1}...')
    
    if df.iloc[idx,1] >= 4:
        list_consider.append(1)
        continue

    flag = 0
    d = matcher.match(df.iloc[idx,0], best_match=True, ignore_syntax=False)

    if len(d) == 0:
        list_consider.append(0)
        continue

    for l in d[0]:
        if l['preferred'] == 1:
            if df.iloc[idx,1] == 2:
                if l['similarity'] >= 0.9 : 
                    flag = 1
                    break

            if df.iloc[idx,1] == 3:
                if l['similarity'] >= 0.9 : 
                    flag = 1
                    break

    if flag == 1: 
        list_consider.append(1)

    else: list_consider.append(0)

df['Consider'] = list_consider
df.to_csv('%sWithConsiderFlag.csv'%fname[:-4],index=False)




