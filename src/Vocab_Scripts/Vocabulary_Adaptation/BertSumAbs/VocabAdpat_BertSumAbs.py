from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-dataset',type=str,required=True,description='Dataset Name')
parser.add_argument('-csv_path',type=str,required=True,description='Path to CSV file')

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
base_vocabs = tokenizer_bert.get_vocab()
if not os.path.exists(args.dataset):
    os.mkdir(args.dataset)


df_PM = pd.read_csv(args.csv_path)
df_consider = df_PM[df_PM['Consider']==1]
df_PM = df_consider[df_consider['TokenFrom']=='PubMed']

df_TGT = pd.read_csv(args.csv_path)
df_consider = df_TGT[df_TGT['Consider']==1]
df_TGT = df_consider[df_consider['TokenFrom']==args.dataset]

list_PM_GT4, list_PM_LT4 = list(), list()

for idx in range(df_PM.shape[0]):
    if df_PM.iloc[idx,1] >=4: 
        list_PM_GT4.extend([df_PM.iloc[idx,0]]*df_PM.iloc[idx,2])
    else: list_PM_LT4.extend([df_PM.iloc[idx,0]]*df_PM.iloc[idx,2])

with open(f'VocabFiles_BERT/PM_BertSumAbs_LT4','w') as f:
    f.write('\n'.join(list_PM_LT4))
f.close()

with open('VocabFiles_BERT/PM_BertSumAbs_GT4','w') as f:
     f.write('\n'.join(list_PM_GT4))
f.close()

list_PM_All = list()
for idx in range(df_PM.shape[0]):
    list_PM_All.extend([df_PM.iloc[idx,0]]*df_PM.iloc[idx,2])

with open('VocabFiles_BERT/PM_BertSumAbs_All','w') as f:
     f.write('\n'.join(list_PM_All))
f.close()


            
list_TGT_LT4, list_TGT_GT4 = list(), list()

for idx in range(df_TGT.shape[0]):
    if df_TGT.iloc[idx,1] >=4: list_TGT_GT4.extend([df_TGT.iloc[idx,0]]*df_TGT.iloc[idx,2])
    else: list_TGT_LT4.extend([df_TGT.iloc[idx,0]]*df_TGT.iloc[idx,2])

with open('VocabFiles_BERT/{args.dataset}_BertSumAbs_GT4','w') as f:
    f.write('\n'.join(list_TGT_GT4))
f.close()

with open('VocabFiles_BERT/{args.dataset}_BertSumAbs_LT4','w') as f:
    f.write('\n'.join(list_TGT_LT4))
f.close()
    


for v_size in [5000, 10000, 15000, 20000, 25000, 30000]:
    for alpha in [0.25,0.5,0.75,1]:
        tokenizer_rs = BertWordPieceTokenizer()
        tokenizer_rs.train(f'VocabFiles_BERT/{args.dataset}_BertSumAbs_GT4',show_progress=False)
        RS_vocab = [x[0] for x in sorted(tokenizer_rs.get_vocab().items(),key=lambda x:x[1])]
        RS_abs = [y for y in RS_vocab if y not in base_vocabs]

        tokenizer_pubmed = BertWordPieceTokenizer()
        tokenizer_pubmed.train('VocabFiles_BERT/PM_WithMed_BertSumAbs_GT4',vocab_size=v_size,show_progress=False)
        vocab_pubmed = tokenizer_pubmed.get_vocab()
        RS_abs_imp = [y for y in RS_abs if y in vocab_pubmed]

        tokenizer_rs = BertWordPieceTokenizer()
        tokenizer_rs.train(f'VocabFiles_BERT/{args.dataset}_BertSumAbs_LT4',show_progress=False)
        RS_vocab = [x[0] for x in sorted(tokenizer_rs.get_vocab().items(),key=lambda x:x[1])]
        RS_abs = [y for y in RS_vocab if y not in base_vocabs]
        
        tokenizer_pubmed = BertWordPieceTokenizer()
        tokenizer_pubmed.train('VocabFiles_BERT/PM_WithMed_BertSumAbs_LT4',vocab_size=v_size,show_progress=False)
        vocab_pubmed = tokenizer_pubmed.get_vocab()
        RS_abs_imp_LT4 = [y for y in RS_abs if y in vocab_pubmed]

        mark_common = list()
        for idx in range(len(RS_abs_imp_LT4)):
            if RS_abs_imp_LT4[idx] in RS_abs_imp: mark_common.append(idx)

        for i in mark_common[::-1]: del RS_abs_imp_LT4[i]

        final_vocab = RS_abs_imp + RS_abs_imp_LT4


        tokenizer_pubmed = BertWordPieceTokenizer()
        tokenizer_pubmed.train('VocabFiles_BERT/PM_BertSumAbs_All',vocab_size=v_size,show_progress=False)
        vocab_pubmed = tokenizer_pubmed.get_vocab()
        vocab_pubmed = [y for y in vocab_pubmed if y not in base_vocabs]
        vocab_pubmed = [y for y in vocab_pubmed if y not in final_vocab]

        final_vocab_new = final_vocab + vocab_pubmed[:int(alpha*len(final_vocab))+1]
        
        with open(f'{args.dataset}/vocab_DRIFT_{v_size//1000}K_{alpha}.txt','w') as f:
            f.write('\n'.join(base_vocabs)+'\n'+'\n'.join(final_vocab_new))
        f.close()
        
        print(f'Done for {v_size//1000}K_{alpha}: {len(final_vocab_new)}')

