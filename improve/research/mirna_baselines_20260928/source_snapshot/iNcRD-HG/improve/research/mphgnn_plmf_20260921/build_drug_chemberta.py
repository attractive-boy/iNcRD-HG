# -*- coding: utf-8 -*-
"""Drug node features for MPHGNN on PLMF-MDA: ChemBERTa (seyonec/ChemBERTa-zinc-base-v1)
mean-pooled last hidden state, 768-d -- the same encoder + pooling incrd-hg used to
build drug_features-chemBertA.csv, which is exactly what MPHGNN's released
MiDrug_data_end.pth drug.x is (row-wise min-max of that file; verified max|diff|=0).
Model weights come from hf-mirror.com (huggingface.co is unreachable here)."""
import os, sys
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HOME', '/data1/licheng/incrd-hg/.external/models/hf_home')
import numpy as np, pandas as pd, torch
from rdkit import Chem, RDLogger
from transformers import AutoModel, AutoTokenizer
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
PLMF = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/repos/PLMF-MDA-main/data/unpacked2/data'
OUT = os.path.join(HERE, 'cache', 'drug_chemberta_zinc.npz')
MODEL = 'seyonec/ChemBERTa-zinc-base-v1'


def main():
    dr = pd.concat([pd.read_excel(f'{PLMF}/MDR/drug_ID_R.xlsx'), pd.read_excel(f'{PLMF}/MDS/drug_ID_S.xlsx')])
    dr = dr.drop_duplicates('DrugBank_ID').reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    feats = []
    with torch.no_grad():
        for smi in dr.SMILES.astype(str):
            assert Chem.MolFromSmiles(smi) is not None, smi
            inp = tok([smi], return_tensors='pt', padding=True, truncation=True, max_length=512)
            feats.append(model(**inp).last_hidden_state.mean(1).squeeze(0).numpy())
    X = np.stack(feats).astype(np.float32)
    np.savez_compressed(OUT, drugbank_id=dr.DrugBank_ID.to_numpy().astype(str), x=X, smiles=dr.SMILES.to_numpy().astype(str))
    print('wrote', OUT, X.shape, 'model', MODEL, 'pooling mean(last_hidden_state)')


if __name__ == '__main__':
    main()
