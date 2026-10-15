"""Frozen original M123 plus additive logistic trees; strictly original feature coordinates."""
import json
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from sklearn.decomposition import PCA
from common import HERE,Adapter,make_graph,Reconstruction,sha256
CONFIGS={n:dict(features=n) for n in ['raw','pca32','structure']}
PARAMS=dict(objective='binary:logistic',tree_method='hist',device='cpu',max_depth=2,min_child_weight=10.,reg_lambda=10.,
 learning_rate=.03,subsample=.85,colsample_bytree=.85,seed=42,nthread=2,disable_default_eval_metric=1)
REFERENCE=HERE.parent/'midrug_training_size_20261007/results/plan.json'

def source(fold):return json.loads(REFERENCE.read_text())['controls']['unmasked10'][str(fold)]
def make_adapter(data,name,device,fold=None):
 a=Adapter(data,make_graph(data),device)
 if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
 if fold is not None:
  s=source(fold);p=Path(s['path']);assert sha256(p.with_suffix('.pt'))==s['checkpoint_sha256']
  a.model.load_state_dict(torch.load(p.with_suffix('.pt'),map_location=device,weights_only=True))
 a.model.eval();return a

def pca_coordinates(a):
 # Labels do not enter feature fitting. All original covariates transductively available.
 out=[]
 for x in a.dense[:2]:
  xx=x.detach().cpu().numpy();pc=PCA(n_components=32,svd_solver='full').fit_transform(xx).astype(np.float32);out.append(pc)
 return out

def features(a,pairs,name,pc):
 r,d=torch.as_tensor(pairs.T,device=a.y.device)
 structure=torch.cat([a.ps.qis(r,d),a.ps.ease(r,d)],-1).detach().cpu().numpy()
 columns=[structure]
 if name=='raw':columns.extend([a.dense[0][r].detach().cpu().numpy(),a.dense[1][d].detach().cpu().numpy()])
 if name=='pca32':columns.extend([pc[0][pairs[:,0]],pc[1][pairs[:,1]]])
 x=np.concatenate(columns,1).astype(np.float32);assert np.isfinite(x).all();return x

def predict_margin(a,pairs):
 r,d=torch.as_tensor(pairs.T,device=a.y.device)
 with torch.no_grad():return a.model(torch.stack([r,d]),a.ps,(a.ps.qis(r,d)-a.mu)/a.sd,a.ps.ease(r,d),a.dense).detach().cpu().numpy()

def independent_tree_margin(booster,x,base):
 trees=[json.loads(t) for t in booster.get_dump(dump_format='json')];out=np.array(base,dtype=np.float64).copy()
 for tree in trees:
  for row in range(len(x)):
   node=tree
   while 'leaf' not in node:
    col=int(node['split'].removeprefix('f')) if hasattr(str,'removeprefix') else int(node['split'][1:])
    v=x[row,col];target=node['missing'] if np.isnan(v) else node['yes'] if v<node['split_condition'] else node['no']
    node=next(c for c in node['children'] if c['nodeid']==target)
   out[row]+=node['leaf']
 return out
