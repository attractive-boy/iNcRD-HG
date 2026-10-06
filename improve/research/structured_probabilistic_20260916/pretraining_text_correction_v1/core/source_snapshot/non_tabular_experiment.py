"""Evidence-preserving screen of non-tabular bipartite graph models.

Only the reused historical INNER arrays are used for screening. New resplits
are reserved for fixed-configuration confirmation; no original test access.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.nn import functional as F

from improve import non_tabular_models as models

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'non_tabular_20260914'
GRAPH = ROOT / 'results' / 'backbone_replacement_20260910'
EVIDENCE = ROOT / 'results' / 'tabicl_pathway_context_inputs_20260913'
REF = ROOT / 'results' / 'tabicl_pathway_context_20260913'
TASKS = ('MiDrug', 'LncDrug')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def metrics(y, pred, pairs):
    ap = []
    for d in np.unique(pairs[1]):
        mask = pairs[1] == d
        if len(np.unique(y[mask])) == 2:
            ap.append(average_precision_score(y[mask], pred[mask]))
    return dict(auc=float(roc_auc_score(y, pred)), ap=float(average_precision_score(y, pred)),
                macro_drug_ap=float(np.mean(ap)), macro_drug_count=len(ap))


def sparse_cosine(x):
    x = sparse.csr_matrix(x, dtype=np.float64)
    # Downweight ubiquitous genes using training-independent annotation counts.
    idf = np.log1p(x.shape[0] / (1 + np.asarray((x != 0).sum(0)).ravel()))
    x = x.multiply(idf).tocsr()
    norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    x = sparse.diags(1/np.maximum(norm, 1e-12)) @ x
    k = (x @ x.T).toarray()
    np.fill_diagonal(k, 1.)
    return models.normalize_kernel(k)


def prepare(out):
    out = Path(out)
    if out.exists():
        raise ValueError('Choose a new output directory')
    source_names = ('non_tabular_models.py', 'non_tabular_experiment.py')
    protocol = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=list(TASKS), seeds=[666, 0, 1], configs=models.configurations(),
        selection='Historical inner validation only; one checkpoint maximizes mean of GLOBAL AUC and average precision. Each family and all controls reported.',
        confirmation='Select using historical inner only, then refit frozen settings on previously prepared split seeds 42/2026/31415; report all outcomes. These are overlapping new partitions, not new biological data.',
        primary_comparator='Fair query-masked original GNN; strong TabICLv2 and TabDPT references reported separately. Passing the GNN gate does not imply surpassing either tabular model.',
        lnc_tolerance_absolute=.001,
        gate='MiDrug mean global AUC and AP improve and >=2/3 confirmation splits improve each; LncDrug each metric loses <=0.001 absolute in mean and on each split. Descriptive engineering gate, not a noninferiority significance test.',
        innovation='Three hypotheses (product-kernel posterior, graph-prior factorization, masked association heat denoising). Parameter variants and ablations are never counted as independent innovations. Success also requires the relevant mechanism control.',
        target_policy='Only provided sampled pairs enter likelihood. Missing pairs are not negatives. Every denoiser training query is absent from both support channels before graph smoothing.',
        side_policy='Frozen unsupervised node attributes and annotation-only gene kernels; no target edges or held-out outcome labels used in node kernels.',
        uncertainty='Report per-seed/split results and paired endpoint-cluster resampling for frozen final candidates; do not present initialization seeds as independent biological experiments.',
        original_test_access=False, original_outer_query_labels_access=False,
        source_sha256={name:digest(ROOT/name) for name in source_names}, input_sha256={})
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in source_names:
        shutil.copy2(ROOT/name, out/'source_snapshot'/name)
    for task in TASKS:
        source = GRAPH/'inputs'/task/'inner.npz'
        with np.load(source, allow_pickle=False) as z:
            names = ('pairs_train', 'y_train', 'pairs_val', 'y_val')
            a = {name:z[name].copy() for name in names}
            xr = z['pca_x_'+('miRNA' if task=='MiDrug' else 'lncRNA')].copy()
            xd = z['x_drug'].copy()
        protocol['input_sha256'][str(source)] = digest(source)
        if set(map(tuple, a['pairs_train'].T)) & set(map(tuple, a['pairs_val'].T)):
            raise ValueError('Train/validation pair overlap')
        for split in ('train', 'val'):
            if len(set(map(tuple,a['pairs_'+split].T))) != len(a['y_'+split]):
                raise ValueError('Duplicate pair')
        # Kernels combine node attributes with annotation identity. No y is read here.
        gene = EVIDENCE/(task+'_weighted.npz')
        drug_gene = EVIDENCE/'drug_gene.npz'
        rg = sparse.load_npz(gene)
        dg = sparse.load_npz(drug_gene)
        kr = models.normalize_kernel(.5*models.feature_kernel(xr)+.5*sparse_cosine(rg))
        kd = models.normalize_kernel(.5*models.feature_kernel(xd)+.5*sparse_cosine(dg))
        a.update(kr=kr.astype(np.float32), kd=kd.astype(np.float32),
                 sr=models.graph_operator(kr), sd=models.graph_operator(kd),
                 ur=models.spectral_features(kr,64), ud=models.spectral_features(kd,64))
        for path in (gene,drug_gene):
            protocol['input_sha256'][str(path)] = digest(path)
        a['folds'] = np.empty(len(a['y_train']),dtype=np.int64)
        for f,(_,query) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20260914).split(a['pairs_train'].T,a['y_train'])):
            a['folds'][query] = f
        models.masked_contexts(a['pairs_train'],a['y_train'],kr.shape[:1]+kd.shape[:1],a['folds'])
        dest=out/'inputs'/task
        dest.mkdir(parents=True)
        np.savez_compressed(dest/'arrays.npz', **a)
        protocol['input_sha256'][str(dest/'arrays.npz')] = digest(dest/'arrays.npz')
    protocol['sha256'] = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    write_json(out/'protocol.json',protocol)
    print('PREPARED',out,flush=True)


def load_protocol(out):
    p=json.loads((Path(out)/'protocol.json').read_text())
    content={k:v for k,v in p.items() if k!='sha256'}
    if hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()!=p['sha256']:
        raise ValueError('Protocol seal mismatch')
    for name,h in p['source_sha256'].items():
        if digest(ROOT/name)!=h: raise ValueError('Implementation changed: '+name)
    for path,h in p['input_sha256'].items():
        if digest(path)!=h: raise ValueError('Input changed: '+path)
    return p


def model_for(a,cfg):
    rate=float(a['y_train'].mean())
    rank=cfg['rank']
    if cfg['family']=='kernel':
        return models.KernelLogistic(a['ur'][:,:rank],a['ud'][:,:rank],rate,cfg.get('interaction',True))
    if cfg['family']=='factor':
        return models.GraphFactorization(a['sr'],a['sd'],rate,rank)
    return models.AssociationDenoiser(a['ur'][:,:rank],a['ud'][:,:rank],a['sr'],a['sd'],rate,
                                     cfg['width'],cfg['graph'],cfg['zeros'])


def train_one(a,cfg,seed,device,epochs=None,select=True):
    start=time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=model_for(a,cfg).to(device)
    n=len(a['y_train'])
    pairs=torch.as_tensor(a['pairs_train'],device=device)
    yp=torch.as_tensor(a['y_train'],device=device)
    vp=torch.as_tensor(a['pairs_val'],device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg.get('weight_decay',0.))
    epochs=epochs or cfg['epochs']
    contexts=[]
    if cfg['family']=='denoise':
        for f,pos,neg in models.masked_contexts(a['pairs_train'],a['y_train'],(len(a['ur']),len(a['ud'])),a['folds']):
            ix=torch.as_tensor(np.flatnonzero(a['folds']==f),device=device)
            contexts.append((torch.as_tensor(pos,device=device),torch.as_tensor(neg,device=device),ix))
        positive=np.zeros((len(a['ur']),len(a['ud'])),dtype=np.float32)
        zero=np.zeros_like(positive)
        for label,matrix in ((1,positive),(0,zero)):
            z=a['pairs_train'][:,a['y_train']==label]
            matrix[z[0],z[1]]=1.
        full_context=(torch.as_tensor(positive,device=device),torch.as_tensor(zero,device=device))
    best,best_epoch,best_state=-np.inf,None,None
    trace=[]
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        if cfg['family']=='denoise':
            pos,neg,ix=contexts[(epoch-1)%len(contexts)]
            heat_time=float(np.random.default_rng(seed+epoch).uniform(0,1))
            score=model.full(pos,neg,heat_time,True)
            logits=score[pairs[0,ix],pairs[1,ix]]
            loss=F.binary_cross_entropy_with_logits(logits,yp[ix])
        else:
            loss=F.binary_cross_entropy_with_logits(model(pairs),yp)
            if cfg['family']=='kernel':
                loss=loss+model.penalty(cfg['node_reg'],cfg['interaction_reg'])/n
            else:
                loss=loss+model.penalty(cfg['node_reg'],cfg['interaction_reg'],cfg['graph_reg'])/n
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
        if epoch%10==0 or epoch==epochs:
            model.eval()
            with torch.no_grad():
                if cfg['family']=='denoise':
                    pred=model.full(*full_context,time=.5)[vp[0],vp[1]].cpu().numpy()
                else:
                    pred=model(vp).cpu().numpy()
            m=metrics(a['y_val'],pred,a['pairs_val'])
            criterion=(m['auc']+m['ap'])/2
            trace.append(dict(epoch=epoch,loss=float(loss),**m))
            if (select and criterion>best) or not select:
                best,best_epoch=criterion,epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        full=model.full(*full_context,time=.5) if cfg['family']=='denoise' else model.full()
        pred=full[vp[0],vp[1]].cpu().numpy()
    result=dict(seed=seed,config=cfg,epoch=best_epoch,seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in model.parameters()),**metrics(a['y_val'],pred,a['pairs_val']))
    return result,pred,best_state,trace


def run(args):
    out=Path(args.out_dir);p=load_protocol(out)
    torch.set_num_threads(4)
    with np.load(out/'inputs'/args.task/'arrays.npz',allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    cfgs=[c for c in p['configs'] if not args.config or c['id'] in args.config]
    seeds=args.seeds if args.seeds is not None else p['seeds']
    for cfg in cfgs:
        for seed in seeds:
            dest=out/'screen'/args.task/cfg['id']/str(seed)
            if (dest/'result.json').exists():
                r=json.loads((dest/'result.json').read_text())
                if r['protocol_sha256']!=p['sha256']:raise ValueError('Wrong prior run')
                for name,h in r['files'].items():
                    if digest(dest/name)!=h:raise ValueError('Prior result changed')
                print('EXISTS',args.task,cfg['id'],seed,flush=True);continue
            dest.mkdir(parents=True,exist_ok=True)
            row,pred,state,trace=train_one(a,cfg,seed,args.device)
            torch.save(state,dest/'model.pt')
            # Replay from the saved checkpoint and verify query partition invariance.
            model=model_for(a,cfg).to(args.device)
            model.load_state_dict(torch.load(dest/'model.pt',map_location=args.device,weights_only=True))
            model.eval()
            with torch.no_grad():
                if cfg['family']=='denoise':
                    pos=np.zeros((len(a['ur']),len(a['ud'])),dtype=np.float32);zero=np.zeros_like(pos)
                    for label,matrix in ((1,pos),(0,zero)):
                        ids=a['pairs_train'][:,a['y_train']==label];matrix[ids[0],ids[1]]=1.
                    full=model.full(torch.as_tensor(pos,device=args.device),torch.as_tensor(zero,device=args.device),time=.5).cpu().numpy()
                else:full=model.full().cpu().numpy()
            q=a['pairs_val'];replay=full[q[0],q[1]]
            np.testing.assert_array_equal(replay,pred)
            reversed_score=full[q[0,::-1],q[1,::-1]][::-1]
            chunks=np.concatenate([full[q[0,ix],q[1,ix]] for ix in np.array_split(np.arange(len(pred)),7)])
            np.testing.assert_array_equal(replay,reversed_score);np.testing.assert_array_equal(replay,chunks)
            np.savez_compressed(dest/'scores.npz',pred=pred,y=a['y_val'],pairs=q)
            write_json(dest/'trace.json',trace)
            row.update(task=args.task,protocol_sha256=p['sha256'],replay_max_abs=0.,
                       query_permutation_and_batches_exact=True,
                       files={name:digest(dest/name) for name in ('model.pt','scores.npz','trace.json')})
            write_json(dest/'result.json',row)
            print('VERIFIED',args.task,cfg['id'],seed,row['epoch'],row['auc'],row['ap'],row['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=('prepare','run'))
    ap.add_argument('--out-dir',default=str(OUT))
    ap.add_argument('--task',choices=TASKS)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--config',nargs='+')
    ap.add_argument('--seeds',type=int,nargs='+')
    args=ap.parse_args()
    if args.command=='prepare':prepare(args.out_dir)
    else:run(args)


if __name__=='__main__':main()
