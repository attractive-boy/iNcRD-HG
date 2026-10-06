"""PLMF author fusion network with cached, frozen eval-mode language encoders."""
import sys,json
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from bench_data import BASE,HERE,sha

REPRO=BASE/'new5_repro_20260919'
sys.path.insert(0,str(REPRO/'venvs/plmf_site'))
sys.path.insert(0,str(REPRO/'repos/PLMF-MDA-main/code'))
import layers
from data_preprocess import smiles_to_graph,encode_sequence

def safe_drug_graph(smiles):
    from rdkit import Chem
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid SMILES: '+smiles)
    # The author function truncates x at 100 but leaves all bond indices intact.
    # Keep its padding for small molecules and retain ALL atoms in larger ones.
    graph=smiles_to_graph(smiles,max_num_nodes=max(100,mol.GetNumAtoms()))
    if graph.edge_index.numel()==0:graph.edge_index=torch.empty((2,0),dtype=torch.long)
    assert graph.edge_index.numel()==0 or int(graph.edge_index.max())<graph.num_nodes
    return graph

def build_cache(dataset,device):
    entities=json.loads((HERE/f'data/{dataset}_entities.json').read_text())
    path=HERE/f'data/{dataset}_plmf_cache.pt'
    if path.exists(): return
    seqs=list(dict.fromkeys(entities['sequences'])); smis=list(dict.fromkeys(entities['smiles']))
    rna=layers.RNAFMFeatureExtractor(128).to(device).eval(); r=[]
    with torch.no_grad():
        for start in range(0,len(seqs),32):
            _,_,tokens=rna.batch_converter([(str(j),s) for j,s in enumerate(seqs[start:start+32])])
            z=rna.model(tokens.to(device),repr_layers=[rna.model_layer])['representations'][rna.model_layer]
            r.append(z[:,0,:].cpu())
    del rna; torch.cuda.empty_cache()
    drug=layers.ChemBERTaFeatureExtractor(128).to(device).eval(); d=[]
    with torch.no_grad():
        for start in range(0,len(smis),32):
            inp=drug.tokenizer(smis[start:start+32],padding=True,truncation=True,max_length=512,return_tensors='pt')
            z=drug.model(**{k:v.to(device) for k,v in inp.items()}).last_hidden_state[:,0,:]
            d.append(z.cpu())
    torch.save(dict(sequences=seqs,smiles=smis,rna=torch.cat(r),drug=torch.cat(d),
                    entities_sha256=sha(HERE/f'data/{dataset}_entities.json')),path)
    print('Cached frozen PLMF features',dataset,flush=True)

class FrozenCached(nn.Module):
    def __init__(self,values,keys,output_dim):
        super().__init__(); self.register_buffer('values',values); self.index={v:i for i,v in enumerate(keys)}
        self.projection=nn.Linear(values.shape[1],output_dim); self.dropout=nn.Dropout(0.2)
    def forward(self,keys):
        idx=torch.tensor([self.index[k] for k in keys],device=self.values.device)
        return self.dropout(self.projection(self.values[idx]))

class PLMFAdapter:
    def __init__(self,data,args):
        self.data=data; self.args=args; self.dev=args.device
        path=HERE/f'data/{args.dataset}_plmf_cache.pt'
        if not path.exists(): build_cache(args.dataset,args.device)
        cache=torch.load(path,map_location='cpu')
        assert cache['entities_sha256']==sha(HERE/f'data/{args.dataset}_entities.json')
        layers.RNAFMFeatureExtractor=lambda output_dim:FrozenCached(cache['rna'],cache['sequences'],output_dim)
        layers.ChemBERTaFeatureExtractor=lambda output_dim:FrozenCached(cache['drug'],cache['smiles'],output_dim)
        from engine import seed_all
        seed_all(args.seed)
        cfg=SimpleNamespace(mirna_feature_dim=128,drug_feature_dim=128,drug_input_dim=69,drug_hidden_dim=128)
        self.model=layers.DrugMiRNAInteractionModel(cfg).to(self.dev)
        self.graphs=[]
        for s in data['smiles']:
            g=safe_drug_graph(s)
            assert g is not None
            # PyG accepts empty edge lists only when shaped (2, 0).
            if g.edge_index.numel()==0:g.edge_index=torch.empty((2,0),dtype=torch.long)
            self.graphs.append(g)
        self.encoded=torch.stack([encode_sequence(s) for s in data['sequences']]).to(self.dev)
        self.note='PLMF-MDA author CNN/GCN and fusion; frozen eval-mode RNA-FM/ChemBERTa-2 caches; preserve 100-node padding for small drugs, retain all atoms above 100 to fix author truncated-x/untrimmed-bond bug; train on shared pairs.'

    def batches(self,part,shuffle=False):
        n=len(self.data[part+'_y']); order=np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0,n,self.args.batch):
            ix=order[start:start+self.args.batch]; pairs=self.data[part+'_pairs'][ix]
            graph=Batch.from_data_list([self.graphs[j] for i,j in pairs]).to(self.dev)
            smi=[self.data['smiles'][j] for i,j in pairs]; seq=[self.data['sequences'][i] for i,j in pairs]
            encoded=self.encoded[torch.tensor(pairs[:,0],device=self.dev)]
            yield ix,(graph,smi,encoded,seq)

    def train_epoch(self,opt,epoch):
        total=0.; n=0
        for ix,inp in self.batches('train',True):
            opt.zero_grad(); p=self.model(*inp).flatten()
            y=torch.tensor(self.data['train_y'][ix],dtype=torch.float,device=self.dev)
            loss=nn.functional.binary_cross_entropy(p,y); loss.backward(); opt.step()
            total+=float(loss.detach())*len(ix); n+=len(ix)
        return total/n
    def predict(self,part):
        return np.concatenate([self.model(*inp).flatten().cpu().numpy() for _,inp in self.batches(part)])

if __name__=='__main__':
    from engine import seed_all
    seed_all(42)
    for ds in ('MiDrug','MDR','MDS'):build_cache(ds,'cuda:1')
