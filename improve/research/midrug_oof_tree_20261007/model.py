"""Same frozen full300 inference base and tree inputs; OOF versus in-sample training margins."""
import json
from pathlib import Path
import torch
from common import HERE,PREV,BASES,Adapter,make_graph,Reconstruction,sha256,read_module
prior=read_module('oof_integer_features',PREV/'model.py')
features,predict_margin,pca_coordinates=prior.features,prior.predict_margin,prior.pca_coordinates
PARAMS=prior.PARAMS
CONFIGS={'oof_raw':dict(features='raw',offset='oof'),'insample_raw':dict(features='raw',offset='in_sample'),'oof_pca32':dict(features='pca32',offset='oof')}

def source(fold,k=-1):return BASES/'bases'/f'base_f{fold}_k{k}.json'
def make_adapter(data,name,device,fold=None,nested=-1):
 a=Adapter(data,make_graph(data),device)
 if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
 if fold is not None:
  p=source(fold,nested);r=json.loads(p.read_text());assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
  a.model.load_state_dict(torch.load(p.with_suffix('.pt'),map_location=device,weights_only=True))
 a.model.eval();return a
