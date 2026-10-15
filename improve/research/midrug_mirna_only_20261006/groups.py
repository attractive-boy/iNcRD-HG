"""Label-free name/scaffold groups, using original miRNA names and SMILES."""
import json,re
import numpy as np
import pandas as pd
import torch
from rdkit import Chem,rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from data import HERE,sha256

ORIGINAL=HERE.parents[3]/'incrd-orig/2.construct_network'


def equal_kernel(groups):
    groups=np.asarray(groups);K=(groups[:,None]==groups[None,:]).astype(np.float32)
    np.fill_diagonal(K,0)
    return K


def prepare():
    names=pd.read_csv(ORIGINAL/'miRNA.csv').iloc[:,0].str.upper().tolist()
    ids=pd.read_csv(ORIGINAL/'Drug.csv')['CID'].astype(int).tolist()
    smiles=pd.read_csv(ORIGINAL/'drug_smiles.csv').set_index('drug_cid')
    # The original table indexes CIDs; enforce exact order before molecular parsing.
    smiles.index=smiles.index.astype(int)
    if set(ids).issubset(set(smiles.index)):
        strings=smiles.loc[ids,'smiles'].astype(str).tolist()
    else:
        raise ValueError('SMILES index must align to original CID order')
    assert len(names)==605 and len(ids)==216
    clean=torch.load(HERE/'graph.pt',map_location='cpu')
    expression=pd.read_csv(ORIGINAL/'MiExp.csv',index_col=0).T
    assert expression.index.str.upper().tolist()==names
    expression=expression[sorted(expression.columns,key=lambda s:int(s.split('-')[-1]))]
    X=expression.values.astype(np.float32)
    X=(X-X.min(1,keepdims=True))/(np.ptp(X,axis=1,keepdims=True)+1e-8)
    np.testing.assert_array_equal(X,clean['miRNA'].x.numpy())
    drug_features=pd.read_csv(ORIGINAL/'drug_features-chemBertA.csv',index_col=0)
    assert drug_features.index.astype(int).tolist()==ids
    X=drug_features.values.astype(np.float32)
    X=(X-X.min(1,keepdims=True))/(np.ptp(X,axis=1,keepdims=True)+1e-8)
    np.testing.assert_array_equal(X,clean['drug'].x.numpy())
    rna_stem=[re.sub(r'-[35]P$','',name) for name in names]
    numeric=[]
    for name in names:
        match=re.match(r'HSA-(MIR|LET)-(\d+)',name)
        assert match,name
        numeric.append('-'.join(match.groups()))
    parents,scaffolds,generic=[],[],[]
    for index,string in enumerate(strings):
        mol=Chem.MolFromSmiles(string);assert mol is not None
        pieces=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True)
        parent=max(pieces,key=lambda m:m.GetNumHeavyAtoms())
        parents.append(Chem.MolToSmiles(parent,isomericSmiles=False))
        scaffold=MurckoScaffold.GetScaffoldForMol(parent)
        if scaffold.GetNumAtoms():
            scaffolds.append(Chem.MolToSmiles(scaffold,isomericSmiles=False))
            generic.append(Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold)))
        else:
            # Acyclic molecules are not all made equivalent by an empty scaffold.
            scaffolds.append(f'acyclic_singleton_{index}');generic.append(f'acyclic_singleton_{index}')
    sets=dict(rna_stem=rna_stem,rna_numeric=numeric,drug_parent=parents,drug_scaffold=scaffolds,drug_generic=generic)
    arrays={name:equal_kernel(group) for name,group in sets.items()}
    for K in arrays.values():assert np.diag(K).sum()==0
    np.savez_compressed(HERE/'groups.npz',**arrays)
    info=dict(status='prepared',source_sha256=sha256(__file__),groups_sha256=sha256(HERE/'groups.npz'),rdkit_version=rdBase.rdkitVersion,
        original_source_hashes={file:sha256(ORIGINAL/file) for file in ('miRNA.csv','MiExp.csv','Drug.csv','drug_smiles.csv','drug_features-chemBertA.csv')},
        entity_order_verified_against_clean_graph=True,
        groups={name:dict(distinct_groups=len(set(group)),nodes_with_partners=int((arrays[name].sum(1)>0).sum()),directed_pairs=int(arrays[name].sum())) for name,group in sets.items()},
        interpretation='Name-derived groups are not verified sequence families; molecular scaffold equality is not identical drug response',
        lncrna_used=False,labels_used=False)
    (HERE/'groups_provenance.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)


class GroupEvidence:
    def __init__(self,data,device):
        meta=json.loads((HERE/'groups_provenance.json').read_text());assert sha256(HERE/'groups.npz')==meta['groups_sha256']
        with np.load(HERE/'groups.npz') as z:groups={k:torch.as_tensor(z[k],device=device) for k in z.files}
        self.A=torch.zeros((605,216),device=device);self.N=torch.zeros_like(self.A)
        pos=torch.as_tensor(data['train_pairs'][data['train_y']==1],device=device)
        neg=torch.as_tensor(data['train_pairs'][data['train_y']==0],device=device)
        self.A[pos[:,0],pos[:,1]]=1;self.N[neg[:,0],neg[:,1]]=1
        self.channels={}
        def make(prefix,P,N,size):
            # All five statistics exclude the query via zero-diagonal kernels.
            values=[torch.log1p(P),torch.log1p(N),P/(P+N+1),torch.log1p(P+N),P/(size+1)]
            self.channels[prefix]=torch.stack(values,dim=-1)
        rg={k:v for k,v in groups.items() if k.startswith('rna_')};dg={k:v for k,v in groups.items() if k.startswith('drug_')}
        for name,K in rg.items():make(name,K@self.A,K@self.N,K.sum(1)[:,None])
        for name,K in dg.items():make(name,self.A@K,self.N@K,K.sum(0)[None,:])
        for rn,R in rg.items():
            for dn,D in dg.items():make(rn+'__'+dn,R@self.A@D,R@self.N@D,R.sum(1)[:,None]*D.sum(0)[None,:])

    def features(self,r,d,mode):
        if mode=='rna':keys=[k for k in self.channels if k.startswith('rna_') and '__' not in k]
        elif mode=='drug':keys=[k for k in self.channels if k.startswith('drug_')]
        elif mode=='combined':keys=list(self.channels)
        else:raise ValueError(mode)
        return torch.cat([self.channels[k][r,d] for k in keys],dim=1)

if __name__=='__main__':prepare()
