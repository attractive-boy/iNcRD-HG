"""Meaningful benchmark checks: split identity, leakage boundary, selection, result integrity."""
import argparse,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from bench_data import HERE,PLMF,load,check_split,sha
from engine import fit,metrics,seed_all

def check_data():
    manifest=json.loads((HERE/'data/manifest.json').read_text())
    for ds,info in manifest['datasets'].items():
        for fold,record in info['folds'].items():
            d=load(ds,int(fold)); parts={p:(d[p+'_pairs'],d[p+'_y']) for p in ('train','val','test')}
            check_split(parts)
            assert record['sha256']==sha(HERE/f'data/{ds}_fold{fold}.npz')
            assert all(s and set(s)<=set('ACGU') for s in d['sequences'])
            assert len(d['mirna_names'])==len(d['xr'])
            if ds!='MiDrug':
                su='R' if ds=='MDR' else 'S'
                pos=np.loadtxt(PLMF/ds/f'pos_MDA_{su}.edgelist',dtype=int)
                neg=np.loadtxt(PLMF/ds/f'neg_MDA_{su}.edgelist',dtype=int)
                rng=np.random.RandomState(42); neg=neg[rng.permutation(len(neg))[:len(pos)]]
                p=np.r_[pos,neg];y=np.r_[np.ones(len(pos)),np.zeros(len(pos))];order=rng.permutation(len(p))
                ix=np.array_split(np.arange(len(p)),5)[int(fold)]
                np.testing.assert_array_equal(p[order][ix],d['test_pairs'])
                np.testing.assert_array_equal(y[order][ix],d['test_y'])
    print('PASS: split disjointness, content coverage, hashes, PLMF outer-fold identity')

def check_selection():
    # The best test epoch would be epoch 3, but only validation epoch 2 may be selected.
    class Sentinel:
        note='test sentinel'
        def __init__(self):
            self.model=torch.nn.Linear(1,1,bias=False);self.test_calls=0
        def train_epoch(self,opt,epoch):
            self.model.weight.data.fill_(epoch);return 1.
        def predict(self,part):
            ep=int(self.model.weight.item())
            if part=='test':self.test_calls+=1;return np.array([.1,.9]) if ep==3 else np.array([.8,.2])
            return np.array([.1,.9]) if ep==2 else np.array([.9,.1])
    with tempfile.TemporaryDirectory() as out:
        adapter=Sentinel();data={p+'_y':np.array([0,1]) for p in ('val','test')}
        data.update({p+'_pairs':np.array([[0,0],[1,1]]) for p in ('val','test')})
        args=SimpleNamespace(model='sentinel',dataset='MiDrug',fold=0,seed=42,lr=.01,wd=0.,epochs=3,
                             eval_every=1,min_epochs=3,patience=5,output=out)
        r=fit(adapter,data,args)
        assert r['best_epoch']==2 and r['auc']==0 and adapter.test_calls==1
    print('PASS: validation selects checkpoint; test evaluated once after selection')

def check_graph_boundary():
    from graph_adapters import GraphAdapter
    d=load('MDR',0); changed=dict(d)
    for p in ('val','test'):changed[p+'_y']=1-d[p+'_y']
    a=SimpleNamespace(device='cpu',model='ammgc')
    seed_all(123);m1=GraphAdapter(d,a)
    seed_all(123);m2=GraphAdapter(changed,a)
    torch.testing.assert_close(m1.adj.to_dense(),m2.adj.to_dense())
    for x,y in zip(m1.views,m2.views):torch.testing.assert_close(x,y)
    # Held-out labels do not affect any training GIP/view.
    print('PASS: changing held-out labels leaves graph and all GIP views unchanged')

def check_results(require_complete=False):
    from run import DEFAULTS
    registry=json.loads((HERE/'registry.json').read_text()); expected=set()
    for m in registry:
        for ds in ('MiDrug','MDR','MDS'):
            runs=[(0,s) for s in (42,43,44)] if ds=='MiDrug' else [(f,42) for f in range(5)]
            expected.update(f'{m["id"]}_{ds}_f{f}_s{s}' for f,s in runs)
    found=set()
    for path in (HERE/'results').glob('*.json'):
        if path.name.endswith('_curves.json'):continue
        r=json.loads(path.read_text());tag=path.stem;found.add(tag)
        assert tag in expected,tag
        assert tag==f'{r["model"]}_{r["dataset"]}_f{r["fold"]}_s{r["seed"]}'
        config=r['config']
        for key in ('model','dataset','fold','seed'):assert config[key]==r[key],(tag,key)
        for key,value in zip(('epochs','lr','wd','eval_every','min_epochs','patience'),DEFAULTS[r['model']]):
            assert config[key]==value,(tag,key,'unexpected formal-run override')
        assert config['batch']==(0 if r['model']=='incrd_original' else 128) and r['threshold']==0.5
        if r['model']=='incrd_original':
            assert config['optimizer']=='RMSprop' and config['optimizer_alpha']==0.99
            assert config['hidden']==256 and config['hetero_aggr']=='mean'
        assert config['min_epochs']<=r['epochs_run']<=config['epochs']
        d=load(r['dataset'],r['fold']); p=np.load(path.with_name(tag+'_predictions.npz'))
        for part in ('val','test'):
            np.testing.assert_array_equal(p[part+'_pairs'],d[part+'_pairs'])
            np.testing.assert_array_equal(p[part+'_y'],d[part+'_y'])
            probability=p[part+'_probability']
            assert probability.shape==p[part+'_y'].shape
            assert np.isfinite(probability).all() and ((0<=probability)&(probability<=1)).all()
        assert r['selection']=='validation_auc' and r['graph']=='train_only'
        for k,v in metrics(p['test_y'],p['test_probability']).items():assert abs(v-r[k])<1e-10,(tag,k)
        curves=json.loads(path.with_name(tag+'_curves.json').read_text())
        best=max(curves,key=lambda c:c['val_auc'])
        assert best['epoch']==r['best_epoch']
        saved_val_auc=metrics(p['val_y'],p['val_probability'])['auc']
        assert abs(saved_val_auc-r['val_auc'])<1e-10
        # GPU reductions can slightly change tied rankings after state reload.
        assert abs(saved_val_auc-best['val_auc'])<1e-5,(tag,'selected validation score mismatch')
        assert sha(path.with_suffix('.pt'))==r['checkpoint_sha256']
        assert sha(HERE/f'data/{r["dataset"]}_fold{r["fold"]}.npz')==r['split_sha256']
    missing=sorted(expected-found)
    if require_complete:assert not missing,missing
    report={'completed':len(found),'expected':len(expected),'missing':missing,'verified':True}
    (HERE/'verification.json').write_text(json.dumps(report,indent=2))
    print('PASS: saved predictions, metrics, checkpoints, selected epochs;',len(found),'/',len(expected))

def check_compact_subknet():
    import types
    from subknet_adapter import onehot_linear,compact_rw_forward,module,CODE
    if not torch.cuda.is_available():return
    torch.manual_seed(42)
    x=torch.tensor([[1,6,31,61],[5,61,0,61]],device='cuda:0')
    first=torch.nn.Linear(61,16).cuda();second=torch.nn.Linear(61,16).cuda()
    second.load_state_dict(first.state_dict());second.forward=types.MethodType(onehot_linear,second)
    dense=torch.nn.functional.one_hot(x,62)[...,:61].float()
    a,b=first(dense),second(x);torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    a.square().sum().backward();b.square().sum().backward()
    for p,q in zip(first.parameters(),second.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
    net=module('subknet_equivalence',CODE/'model.py')
    model=net.RW_layer(50,32,hidden_dim=16,max_step=2,size_graph_filter=12,dropout=.2).cuda().eval()
    feat=torch.rand(8,36,50).cuda();adj=torch.randint(0,2,(8,36,36)).float().cuda()
    with torch.no_grad():a,d=model(adj,feat);b,e=compact_rw_forward(model,adj,feat)
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5);torch.testing.assert_close(d,e)
    print('PASS: SubKNet one-hot forward/backward and evaluation contraction equivalence')

def check_plmf_graphs():
    from plmf_adapter import safe_drug_graph
    from torch_geometric.data import Batch
    from rdkit import Chem
    for ds in ('MiDrug','MDR','MDS'):
        data=load(ds,0);graphs=[]
        for smiles in data['smiles']:
            mol=Chem.MolFromSmiles(smiles);g=safe_drug_graph(smiles)
            assert g.num_nodes>=mol.GetNumAtoms()
            assert g.edge_index.shape[1]==2*mol.GetNumBonds()
            graphs.append(g)
        batch=Batch.from_data_list(graphs)
        assert int(batch.edge_index.max())<batch.num_nodes
        assert torch.equal(batch.batch[batch.edge_index[0]],batch.batch[batch.edge_index[1]])
    print('PASS: PLMF retains all atoms/bonds; batched edges never cross molecule boundaries')

def check_original_incrd():
    from incrd_adapter import build_graph,OriginalINcRDAdapter,RELEASE_GRAPH
    pinned=json.loads((HERE/'original_source/SOURCE.json').read_text())
    for record in pinned['files']:
        assert sha(HERE/'original_source'/record['path'])==record['sha256']
    manifest=json.loads((HERE/'data/incrd_original_manifest.json').read_text())
    for ds in ('MiDrug','MDR','MDS'):
        data=load(ds,0);other=dict(data)
        for part in ('val','test'):other[part+'_y']=1-data[part+'_y']
        graph=build_graph(data,ds);perturbed=build_graph(other,ds)
        for node in graph.node_types:torch.testing.assert_close(graph[node].x,perturbed[node].x)
        for edge in graph.edge_types:torch.testing.assert_close(graph[edge].edge_index,perturbed[edge].edge_index)
        positive=data['train_pairs'][data['train_y']==1].T
        np.testing.assert_array_equal(graph['miRNA','MiDrug','drug'].edge_index.numpy(),positive)
        np.testing.assert_array_equal(graph['drug','rev_MiDrug','miRNA'].edge_index.numpy(),positive[::-1])
        record=manifest['datasets'][ds]
        for source in record['sources']:assert sha(source['path'])==source['sha256']
        if ds=='MiDrug':
            released=torch.load(RELEASE_GRAPH,map_location='cpu')
            assert set(graph.node_types)=={'lncRNA','miRNA','drug'} and len(graph.edge_types)==9
            for node in graph.node_types:torch.testing.assert_close(graph[node].x,released[node].x,rtol=0,atol=0)
            for edge in graph.edge_types:
                if edge[1] not in ('MiDrug','rev_MiDrug'):
                    torch.testing.assert_close(graph[edge].edge_index,released[edge].edge_index)
        else:assert set(graph.node_types)=={'miRNA','drug'} and len(graph.edge_types)==4
    args=SimpleNamespace(dataset='MDR',device='cpu',lr=1e-4,wd=0.)
    adapter=OriginalINcRDAdapter(load('MDR',0),args)
    optimizer=adapter.make_optimizer(args)
    assert isinstance(optimizer,torch.optim.RMSprop) and optimizer.defaults['alpha']==.99
    print('PASS: original source hashes, released MiDrug auxiliary inputs, target train-only graph, held-out label isolation, RMSprop recipe')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');p.add_argument('--results-only',action='store_true');a=p.parse_args()
    if not a.results_only:check_data();check_selection();check_graph_boundary();check_compact_subknet();check_plmf_graphs();check_original_incrd()
    check_results(a.require_complete)
