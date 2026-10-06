# PLMF-MDA
## Paper [**[Paper link]**](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013968 "Click") 
## Cite
Sheng N, Liu Y, Gao L, et al. A pre-trained language model-based cross-modal fusion framework for predicting miRNA-drug resistance and sensitivity associations[J]. PLOS Computational Biology, 2026, 22(2): e1013968.
## 1. Overview
The code for paper "A pre-trained language model-based cross-modal fusion framework for predicting miRNA-drug resistance and sensitivity associations". The repository is organized as follows:

+ `data/MDR/MDS` contains the data in the paper:
  * `miRNA_drug_R_matrix.xlsx/miRNA_drug_S_matrix.xlsx` contains known miRNA-drug resistance and sensitivity associations;
  * `drug_ID_R.xlsx/drug_ID_S.xlsx` contains drug ID, name, smiles ;
  * `miRNA_ID_R.xlsx/miRNA_ID_S.xlsx` contains miRNA ID, name, sequences ;
  * `pos_MDA_R.edgelist/pos_MDA_S.edgelist` contains positive sample pairs.
  * `neg_MDA_R.edgelist/pos_MDA_S.edgelist` contains negative sample pairs.
    
+ `code/`
  * `parms_setting.py`contains hyperparmeters;
  * `data_preprocess.py` contains the preprocess of data;
  * `layer.py` contains MGCNA's model layer;
  * `train.py` contains training and testing code;
  * `DeepChem` contains ChemBERTa files;
  
## 2. Dependencies
* torch == 2.1.2+cu118
* torch-geometric == 2.4.0
* numpy == 1.24.4
* RDKit v2023.9.6

## 3. Quick Start
Here we provide a example:

1. Download and upzip our data and code files
2. Run "train.py" 

## 4. Reminder
The dataset was partitioned into training and testing sets with an 8:2 ratio.

## 5. Contacts
If you have any questions, please email Nan Sheng (shengnan@jlu.edu.cn)
