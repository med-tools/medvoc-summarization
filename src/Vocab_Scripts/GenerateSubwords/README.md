1. Before running the scripts kindly make a preprocessing step. For each of the dataset (include PAC-Summ), create a file containg reference summary seperated by newline (```\n```) character and store it in a file of your preferred location.

2. All the GenerateTokens*.py takes as input four cmd args and generate CSVs that contains token information. You can run the follwing shell script to get the token description:
```
python GenerateTokens_BART.py  -dataset EBM \
                               -path_pubmed #location of preprocessed file for PubMed \
                               -path_data #location of preprocessed file for EBM \
                               -csv_path #location in which you want store the data
```
3. Once the CSVs are generate you can run FilterMedicalKeywords.py to get only medically relevant words as follows:
```
python FilterMedicalKeywrods -umls_path #path to QuickUMLS directory \
                             -csv_path #path to any csv generated above
```
