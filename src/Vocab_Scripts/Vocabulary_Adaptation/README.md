The primary input to the files is a csv containing token description, which you will generate using the scripts in Generate Subwords folder.

For ```BertSumAbs``` please use the following shell script:
```
python BertSumAbs/VocabAdpat_BertSumAbs.py \
       -dataset #name of dataset
       -csv_path #csv file containing token information
```

For ```BART``` please use the following shell script:
```
python BART/UpdateVocabBART.py \
       -dataset #name of dataset \
       -csv_path #csv file containing token information \
       -v_size #the size of V-PAC \
       -frac #fraction of V-PAC to sample
```

For ```PEGASUS``` please use the following shell command:
```
python UpdatePEGASUSVocab.py  \
        -v_size_PM_GT4 35000 \
        -v_size_PM_LT4 12565 \
        -v_size_PM_ALL 35000 \
        -v_size_TGT_GT4 1242 \
        -v_size_TGT_LT4 501 \
        -dataset BioASQ \
        -frac 10 \
        -csv_path #path to csv containing token information
```