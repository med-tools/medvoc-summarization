```Model_Scripts``` contains the modified scripts for BertSumAbs and BART/PEGASUS model. We provide shell scripts in the relevant folder to run the scripts.

```Vocab_Scripts``` contains the codebase for obtaining the modified vocabulary.

Ideally you should follow this sequential steps:
```
1. Organize the data into train/test/val splits as csvs/jsons (for BART and PEGASUS). For BertSumAbs please follow the preprocessing steps as discussed in their codebase.

2. Run the shell script in Vocab_Scripts to obtain the DRIFT vocabulary for a model and vocabulary
  2a. First run the shell scripts in Generate Subwords to get the tokens fit for subwords consideration
  2b. Then run the script in Vocabulary_Adaptation folder.

3. Run the shell script in Model_Scripts to obtain the final trained models ready for inference.
```
