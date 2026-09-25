"""Score new pairs with a frozen graph-transfer research model.

CSV inputs use rna_index,drug_index from the frozen task dictionary. Scores
refer to the sampled association problem and are not biological probabilities.
Use the graph-transfer environment documented in NON_TABULAR_FRONTIER_FINDINGS.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from improve import backbone_replacement as gnn
from improve import confirm_reification_kernel as experiment
from improve import non_tabular_experiment as base
from improve import reification_kernel as method
from improve import reification_transfer as graph
from improve import validate_transfer_splits as partitions


class Predictor:
    def __init__(self,task,split,device='cuda:0',out=experiment.OUT):
        self.out=Path(out);self.device=device
        self.p=experiment.protocol(self.out)
        if task not in self.p['tasks'] or split not in self.p['splits']:
            raise ValueError('Unknown frozen task/split')
        self.a=experiment.load_inputs(self.out,task,split)
        dest=self.out/'runs'/task/str(split)/'pretrained'
        result=json.loads((dest/'result.json').read_text());graph.verify_files(dest,result)
        if result['protocol_sha256']!=base.digest(self.out/'protocol.json'):
            raise ValueError('Training protocol changed')
        self.encoder=experiment.encoder('pretrained',self.p,device)
        self.readout=method.TransferKernel(self.a,np.zeros((1,192),dtype=np.float32)).to(device)
        self.readout.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True),strict=True)
        self.readout.eval()
        self.gp=partitions.protocol(partitions.DEFAULT_OUT)
        gd=partitions.DEFAULT_OUT/'runs'/task/str(split)/'gnn'
        gr=json.loads((gd/'result.json').read_text());graph.verify_files(gd,gr)
        path=partitions.DEFAULT_OUT/'inputs'/task/str(split)/'graph.pt'
        g=torch.load(path,map_location='cpu',weights_only=False)
        weights={k:torch.ones(v.shape[1]) for k,v in g.edge_index_dict.items()}
        self.gnn,self.g,self.weights=gnn.make_model(g,weights,task,'base',
            self.gp['gnn'][task]['config'],self.gp['model_seed'],device)
        self.gnn.load_state_dict(torch.load(gd/'model.pt',map_location=device,weights_only=True),strict=True)
        self.gnn.eval()
        self.known=set((self.a['pairs_train'][0]*int(self.a['nd'])+self.a['pairs_train'][1]).tolist())
        self.manifest=dict(task=task,split=split,source_sha256=base.digest(__file__),
            protocol_sha256=base.digest(self.out/'protocol.json'),
            readout_sha256=base.digest(dest/'model.pt'),encoder_sha256=base.digest(self.p['encoder_paths']['pretrained']),
            gnn_sha256=base.digest(gd/'model.pt'),
            score_definition='sigmoid(0.5*GNN_logit+0.5*graph_augmented_kernel_logit)',
            context='Average representations from the same five training-fold-masked support graphs used during fitting.',
            interpretation='Frozen development research model; sampled association score, not an intervention-effect estimate.')

    def validate_queries(self,pairs):
        q=np.asarray(pairs)
        if q.ndim!=2 or q.shape[0]!=2 or q.shape[1]==0:
            raise ValueError('Expected nonempty 2 x N query array')
        if not np.issubdtype(q.dtype,np.number) or not np.isfinite(q).all() or not (q==np.floor(q)).all():
            raise ValueError('Entity indices must be finite integers')
        if (q<0).any() or (q[0]>=int(self.a['nr'])).any() or (q[1]>=int(self.a['nd'])).any():
            raise ValueError('Query outside frozen entity dictionary')
        q=np.ascontiguousarray(q,dtype=np.int64)
        codes=q[0]*int(self.a['nd'])+q[1]
        if any(c in self.known for c in codes):raise ValueError('Query contains this model\'s fitting observations')
        return q

    @torch.no_grad()
    def predict(self,pairs,encoder_batch_size=None):
        q=self.validate_queries(pairs)
        features=[]
        for fold in range(self.p['folds']):
            omitted=self.a['pairs_train'][:,self.a['folds']==fold]
            ctx=graph.context_graph(self.a,omitted[0]*int(self.a['nd'])+omitted[1],self.device)
            z,_=method.extract(self.encoder,ctx,self.a,q,encoder_batch_size or self.p['batch_size'])
            features.append(z);del ctx
        z=torch.as_tensor(np.stack(features).mean(0),device=self.device)
        transfer=self.readout(torch.as_tensor(q,device=self.device),z).cpu().numpy()
        original=gnn.predict(self.gnn,self.g,self.weights,q,None,None,self.gp['graph_training']['batch_size'])
        return dict(standalone=transfer,gnn=original,gnn_composed=.5*original+.5*transfer)


def verify(model,rows):
    a=model.a
    ix=np.arange(len(a['y_val'])) if rows==0 else np.unique(np.linspace(0,len(a['y_val'])-1,min(rows,len(a['y_val'])),dtype=int))
    q=a['pairs_val'][:,ix]
    first=model.predict(q)
    second=model.predict(q[:,::-1],encoder_batch_size=3)
    chunks=[model.predict(part,encoder_batch_size=5) for part in np.array_split(q,3,axis=1) if part.shape[1]]
    path=model.out/'runs'/model.manifest['task']/str(model.manifest['split'])/'pretrained'/'scores.npz'
    errors={}
    with np.load(path,allow_pickle=False) as z:
        for name in first:
            expected=z[name][ix]
            values=[first[name],second[name][::-1],np.concatenate([c[name] for c in chunks])]
            for value in values:np.testing.assert_allclose(value,expected,atol=5e-4,rtol=5e-4)
            errors[name]=float(max(np.max(np.abs(value-expected)) for value in values))
    result=dict(**model.manifest,verified_rows=len(ix),total_saved_validation_rows=len(a['y_val']),
                selection='Uniformly spaced fixed indices; no selection by outcome or score.',
                max_abs_errors=errors,query_reordering_and_batching_checked=True,
                encoder_and_gnn_recomputed=True)
    dest=model.out/'inference_replay'/model.manifest['task']/(str(model.manifest['split'])+'.json')
    base.write_json(dest,result);print(json.dumps(result,ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task',required=True,choices=base.TASKS)
    ap.add_argument('--split',type=int,default=42,choices=(42,2026,31415))
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--out-dir',default=str(experiment.OUT))
    ap.add_argument('--pairs');ap.add_argument('--out');ap.add_argument('--verify',action='store_true')
    ap.add_argument('--verify-rows',type=int,default=31,help='0 checks all saved evaluation queries; default is 31 fixed positions.')
    args=ap.parse_args()
    if args.verify_rows<0:ap.error('--verify-rows must be nonnegative')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Predictor(args.task,args.split,args.device,args.out_dir)
    if args.verify:verify(model,args.verify_rows);return
    if not args.pairs or not args.out:ap.error('--pairs and --out are required')
    dest=Path(args.out)
    if dest.exists() or dest.with_suffix(dest.suffix+'.json').exists():raise ValueError('Output exists')
    frame=pd.read_csv(args.pairs)
    if not {'rna_index','drug_index'}.issubset(frame.columns):raise ValueError('Missing RNA/drug index columns')
    q=model.validate_queries(frame[['rna_index','drug_index']].to_numpy().T)
    score=model.predict(q)
    output=pd.DataFrame(dict(rna_index=q[0],drug_index=q[1],association_score=expit(score['gnn_composed'])))
    dest.parent.mkdir(parents=True,exist_ok=True);output.to_csv(dest,index=False)
    base.write_json(dest.with_suffix(dest.suffix+'.json'),dict(**model.manifest,rows=len(output)))
    print('WROTE',dest,len(output))


if __name__=='__main__':main()
