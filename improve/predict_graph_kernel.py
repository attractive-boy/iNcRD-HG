"""Use a verified GNN + product-kernel model from the resplit experiment.

Run in the existing ncRNAdrug (PyG) environment. The selected split identifies
one frozen research model; these models have not replaced the strongest
tabular reference. CSV columns: rna_index, drug_index.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from improve import backbone_replacement as gnn
from improve import non_tabular_experiment as core
from improve import confirm_non_tabular_kernel as experiment
from improve import validate_transfer_splits as partitions


def load(task,split,device):
    p=experiment.protocol(experiment.OUT)
    if task not in p['tasks'] or split not in p['splits']:raise ValueError('Unknown frozen model')
    g=partitions.protocol(partitions.DEFAULT_OUT)
    root=partitions.DEFAULT_OUT/'inputs'/task/str(split)
    with np.load(root/'arrays.npz',allow_pickle=False) as z:
        a={k:z[k].copy() for k in ('pairs_train','y_train','pairs_val','y_val')}
    with np.load(core.OUT/'inputs'/task/'arrays.npz',allow_pickle=False) as z:
        a.update({k:z[k].copy() for k in ('ur','ud','sr','sd')})
    kd=experiment.OUT/'runs'/task/str(split)/'kernel_1'
    kr=json.loads((kd/'result.json').read_text())
    for n,h in kr['files'].items():
        if core.digest(kd/n)!=h:raise ValueError('Kernel artifact changed')
    gd=partitions.DEFAULT_OUT/'runs'/task/str(split)/'gnn'
    gr=json.loads((gd/'result.json').read_text())
    for n,h in gr['files'].items():
        if core.digest(gd/n)!=h:raise ValueError('GNN artifact changed')
    graph=torch.load(root/'graph.pt',map_location='cpu',weights_only=False)
    weights={k:torch.ones(v.shape[1]) for k,v in graph.edge_index_dict.items()}
    model,graph,weights=gnn.make_model(graph,weights,task,'base',g['gnn'][task]['config'],g['model_seed'],device)
    model.load_state_dict(torch.load(gd/'model.pt',map_location=device,weights_only=False));model.eval()
    km=core.model_for(a,p['fixed_configs']['kernel_1']).to(device)
    km.load_state_dict(torch.load(kd/'model.pt',map_location=device,weights_only=False));km.eval()
    manifest=dict(task=task,split=split,model_seed=p['seed'],
                  gnn_checkpoint_sha256=core.digest(gd/'model.pt'),kernel_checkpoint_sha256=core.digest(kd/'model.pt'),
                  score_definition='sigmoid(0.5*GNN_logit+0.5*kernel_logit); sampled association score',
                  reference_data='Frozen research model trained on one development-pool resplit')
    return a,model,graph,weights,km,g,manifest


def predict(bundle,pairs,device):
    a,model,graph,weights,km,g,_=bundle
    pairs=np.ascontiguousarray(pairs,dtype=np.int64)
    if pairs.ndim!=2 or pairs.shape[0]!=2 or (pairs<0).any():raise ValueError('Invalid pair array')
    if (pairs[0]>=len(a['ur'])).any() or (pairs[1]>=len(a['ud'])).any():raise ValueError('Entity index out of range')
    gl=gnn.predict(model,graph,weights,pairs,None,None,g['graph_training']['batch_size'])
    with torch.no_grad():kl=km(torch.as_tensor(pairs,device=device)).cpu().numpy()
    return .5*gl+.5*kl


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=core.TASKS)
    parser.add_argument('--split',type=int,default=42,choices=(42,2026,31415))
    parser.add_argument('--pairs');parser.add_argument('--out');parser.add_argument('--verify',action='store_true')
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bundle=load(args.task,args.split,args.device);a=bundle[0]
    if args.verify:
        q=a['pairs_val'];prediction=predict(bundle,q,args.device)
        path=experiment.OUT/'runs'/args.task/str(args.split)/'kernel_1/scores.npz'
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(q,z['pairs']);target=z['gnn_composed']
        reordered=predict(bundle,q[:,::-1],args.device)[::-1]
        chunks=np.concatenate([predict(bundle,part,args.device) for part in np.array_split(q,7,axis=1)])
        for p in (prediction,reordered,chunks):np.testing.assert_allclose(p,target,atol=5e-5,rtol=5e-5)
        report=dict(**bundle[-1],rows=len(prediction),replay_max_abs=float(max(np.abs(p-target).max() for p in (prediction,reordered,chunks))),
                    **core.metrics(a['y_val'],prediction,q),query_reordering_and_batching_checked=True)
        destination=experiment.OUT/'inference_replay'/args.task/(str(args.split)+'.json')
        core.write_json(destination,report);print(json.dumps(report,ensure_ascii=False));return
    if not args.pairs or not args.out:parser.error('--pairs and --out are required unless --verify is used')
    destination=Path(args.out)
    if destination.exists() or destination.with_suffix(destination.suffix+'.json').exists():raise ValueError('Output exists')
    frame=pd.read_csv(args.pairs)
    if not {'rna_index','drug_index'}.issubset(frame.columns):raise ValueError('Missing RNA/drug index columns')
    values=frame[['rna_index','drug_index']].to_numpy()
    if not np.isfinite(values).all() or not (values==np.floor(values)).all():raise ValueError('Indices must be finite integers')
    q=values.astype(np.int64).T
    known=set(map(tuple,a['pairs_train'].T))
    if known.intersection(map(tuple,q.T)):raise ValueError('Use pairs outside this model\'s fitting observations')
    prediction=predict(bundle,q,args.device)
    output=pd.DataFrame(dict(rna_index=q[0],drug_index=q[1],association_score=expit(prediction)))
    destination.parent.mkdir(parents=True,exist_ok=True);output.to_csv(destination,index=False)
    core.write_json(destination.with_suffix(destination.suffix+'.json'),dict(**bundle[-1],rows=len(output)))
    print('WROTE',destination,len(output))


if __name__=='__main__':main()
