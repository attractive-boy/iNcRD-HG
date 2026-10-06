"""Immutable ID-based splits; no labels are used to construct content features."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[2]
PLMF = BASE/'new5_repro_20260919/repos/PLMF-MDA-main/data/unpacked2/data'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fasta(path):
    out = {}; key = None
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            key = line[1:].split()[0].lower(); out[key] = ''
        elif key:
            out[key] += line.strip().upper().replace('T', 'U')
    return out

def kmer(seq):
    vs = []
    for k in (1, 2, 3):
        v = np.zeros(4**k, np.float32)
        for start in range(len(seq)-k+1):
            word = seq[start:start+k]
            if any(c not in 'ACGU' for c in word): continue
            idx = 0
            for c in word: idx = idx*4+'ACGU'.index(c)
            v[idx] += 1
        vs.append(v/max(1, v.sum()))
    return np.concatenate(vs)

def content(smiles, seqs):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    xd = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: raise ValueError('Invalid SMILES: '+s)
        arr = np.zeros(2048, np.float32)
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), arr)
        xd.append(arr)
    return np.stack([kmer(s) for s in seqs]), np.stack(xd)

def check_split(parts):
    sets = []
    report = {}
    for name, (p, y) in parts.items():
        keys = set(map(tuple, p.tolist()))
        assert len(keys) == len(p), (name, 'duplicate ID pair')
        assert set(np.unique(y)) == {0, 1}
        report[name] = {'n':len(p), 'positive':int(y.sum()), 'negative':int((y == 0).sum())}
        for old in sets: assert not keys & old, (name, 'split overlap')
        sets.append(keys)
    return report

def prepare():
    (HERE/'data').mkdir(exist_ok=True)
    manifest = {'protocol':'val_auc_train_graph_only_v1', 'sources':{}, 'datasets':{}}
    mature_path = HERE/'data/mirbase_mature.fa'
    mature = fasta(mature_path)
    manifest['sources'][str(mature_path)] = {'sha256':sha(mature_path), 'url':'https://www.mirbase.org/download/mature.fa'}
    # Preserve historical mature names in the benchmark. Sequence tables contain no labels.
    sequence_source={n:'miRBase downloaded 2026-09-28' for n in mature}
    for ds,suffix in [('MDR','R'),('MDS','S')]:
        path=PLMF/ds/f'miRNA_ID_{suffix}.xlsx'
        table=pd.read_excel(path)
        for n,s in zip(table.miRNA_Name,table.miRNA_Sequence):
            n=n.lower(); mature[n]=s; sequence_source[n]=str(path)
    # miRBase now gives this previously unqualified mature entry its arm suffix.
    mature['hsa-mir-302f']=mature['hsa-mir-302f-3p']
    sequence_source['hsa-mir-302f']='miRBase MIMAT0005932: hsa-miR-302f -> hsa-miR-302f-3p'
    for ds in ('MiDrug', 'MDR', 'MDS'):
        sources = []
        if ds == 'MiDrug':
            own = ROOT/'incrd-run'
            names_path = own/'2.construct_network/miRNA.csv'
            drug_path = own/'2.construct_network/drug_smiles.csv'
            names = pd.read_csv(names_path).iloc[:,0].str.lower().tolist()
            drugs = pd.read_csv(drug_path)
            smiles = drugs.smiles.tolist(); drug_ids = drugs.iloc[:,0].astype(str).tolist()
            missing = [n for n in names if n not in mature]
            if missing: raise ValueError(f'Missing mature miRNAs: {missing}')
            seqs = [mature[n] for n in names]
            (HERE/'data/MiDrug_sequence_provenance.json').write_text(json.dumps(
                {n:sequence_source[n] for n in names},indent=2))
            parts = {}
            dedup={}
            for s in ('train','val','test'):
                path = own/f'3.heter_data/MiDrug_{s}_data.pth'; sources.append(path)
                graph = torch.load(path, map_location='cpu')
                e = graph['miRNA','MiDrug','drug']
                pp=e.edge_label_index.t().numpy(); yy=e.edge_label.numpy().astype(np.int64)
                mapping={}
                for key,label in zip(map(tuple,pp),yy):
                    if key in mapping: assert mapping[key]==label, 'Conflicting labels'
                    mapping[key]=label
                _,idx=np.unique(pp,axis=0,return_index=True); idx=np.sort(idx)
                dedup[s]=len(pp)-len(idx)
                parts[s]=(pp[idx],yy[idx])
            manifest['MiDrug_identical_duplicate_rows_removed']=dedup
            sources += [names_path, drug_path]
            folds = [(0, parts)]
        else:
            suffix = 'R' if ds == 'MDR' else 'S'
            sources = [PLMF/ds/f'miRNA_ID_{suffix}.xlsx', PLMF/ds/f'drug_ID_{suffix}.xlsx',
                       PLMF/ds/f'pos_MDA_{suffix}.edgelist', PLMF/ds/f'neg_MDA_{suffix}.edgelist']
            mir = pd.read_excel(sources[0]); drug = pd.read_excel(sources[1])
            names = mir.miRNA_Name.str.lower().tolist(); seqs = mir.miRNA_Sequence.tolist()
            smiles = drug.SMILES.tolist(); drug_ids = drug.iloc[:,0].astype(str).tolist()
            pos, neg = [np.loadtxt(p, dtype=np.int64) for p in sources[2:]]
            assert not set(map(tuple,pos)) & set(map(tuple,neg))
            # Bit-identical outer folds to plmf_data._balanced_pairs(seed=42).
            rng = np.random.RandomState(42)
            neg = neg[rng.permutation(len(neg))[:len(pos)]]
            p = np.concatenate([pos,neg]); y = np.r_[np.ones(len(pos)),np.zeros(len(pos))].astype(np.int64)
            order = rng.permutation(len(p)); p,y = p[order],y[order]
            chunks = np.array_split(np.arange(len(p)),5); folds=[]
            for f in range(5):
                outer_train = np.concatenate([chunks[j] for j in range(5) if j!=f])
                tr, va = train_test_split(outer_train,test_size=0.1,random_state=1042+f,stratify=y[outer_train])
                folds.append((f,{'train':(p[tr],y[tr]),'val':(p[va],y[va]),'test':(p[chunks[f]],y[chunks[f]])}))
        seqs = [s.upper().replace('T','U') for s in seqs]
        xr, xd = content(smiles,seqs)
        np.savez_compressed(HERE/f'data/{ds}_content.npz', xr=xr, xd=xd)
        entities = {'mirna_names':names,'drug_ids':drug_ids,'sequences':seqs,'smiles':smiles}
        (HERE/f'data/{ds}_entities.json').write_text(json.dumps(entities,ensure_ascii=False,indent=2))
        info = {'n_mirna':len(names),'n_drug':len(smiles),'n_unique_sequences':len(set(seqs)),
                'sequence_length_range':[min(map(len,seqs)),max(map(len,seqs))], 'folds':{}}
        for f,parts in folds:
            info['folds'][str(f)] = check_split(parts)
            arrays = {}
            for s,(p,y) in parts.items(): arrays[s+'_pairs']=p; arrays[s+'_y']=y
            path = HERE/f'data/{ds}_fold{f}.npz'; np.savez_compressed(path,**arrays)
            info['folds'][str(f)]['sha256'] = sha(path)
            # Content-identical entities are retained as distinct author IDs; audit separately.
            keys = {s:set((seqs[i],smiles[j]) for i,j in p) for s,(p,y) in parts.items()}
            info['folds'][str(f)]['content_pair_train_test_overlap'] = len(keys['train'] & keys['test'])
        manifest['datasets'][ds]=info
        for p in sources: manifest['sources'][str(p)]={'sha256':sha(p)}
    (HERE/'data/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest['datasets'],indent=2))

def load(ds, fold):
    entities=json.loads((HERE/f'data/{ds}_entities.json').read_text())
    features=dict(np.load(HERE/f'data/{ds}_content.npz'))
    split=dict(np.load(HERE/f'data/{ds}_fold{fold}.npz'))
    return {**entities,**features,**split}

if __name__ == '__main__': prepare()
