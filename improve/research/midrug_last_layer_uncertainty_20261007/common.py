"""Strict miRNA/drug source and immutable frozen last-layer feature extraction."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module('strict_uncertainty',HERE.parent/'midrug_subgraph_20261006/common.py')
Adapter,make_graph=base.Adapter,base.make_graph
recon=module('reconstruction_uncertainty',HERE.parent/'midrug_masked_reconstruction_20261006/model.py')
CLEAN,RELEASE=base.CLEAN,base.RELEASE
load_inner,sha256,seed_all=base.load_inner,base.sha256,base.seed_all
SOURCE_PLAN=HERE.parent/'midrug_training_size_20261007/results/plan.json'

def source_record(fold):
    record=json.loads(SOURCE_PLAN.read_text())['controls']['unmasked10'][str(fold)]
    path=Path(record['path']);assert sha256(path)==record['metadata_sha256']
    assert sha256(path.with_suffix('.pt'))==record['checkpoint_sha256']
    assert sha256(path.with_name(path.stem+'_inner.npz'))==record['probability_sha256']
    return path,record

def make_adapter(data,name,device):
    assert name=='unmasked10'
    a=base.Adapter(data,base.make_graph(data),device)
    a.model.add_module('reconstruction',recon.Reconstruction(a.graph,a.cfg.hidden).to(device))
    a.model.requires_grad_(False);a.model.eval();return a

def initialize(data,fold,device):
    a=make_adapter(data,'unmasked10',device);path,record=source_record(fold)
    a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location=device,weights_only=True),strict=True)
    return a,path,record

@torch.no_grad()
def features(a,part):
    a.model.eval();captured=[]
    hook=a.model.lin3.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().cpu().double().numpy()))
    try:z=a.logits(part).detach().cpu().double().numpy()
    finally:hook.remove()
    assert len(captured)==1
    h=captured[0];assert h.shape==(len(z),256) and np.isfinite(h).all()
    return h,z

def controls():
    return json.loads(SOURCE_PLAN.read_text())['controls']

def verify_plan():
    p=HERE/'results/plan.json';plan=json.loads(p.read_text())
    for f,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(f)==h,f
    assert sha256(CLEAN/'graph.pt')==plan['graph_sha256']
    assert sha256(CLEAN/'inner_data.npz')==plan['data_sha256']
    return plan
