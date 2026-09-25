"""BioPathNet-inspired residual reasoning over an evidence-typed background graph.

The accepted frozen REIFM / attribute-kernel score is the starting function.
Only the path correction is trained; no ncRNA intervention claim is made.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.nn import functional as F

from improve import non_tabular_experiment as base
from improve import biopath_models as method

OUT=base.ROOT/'results/biopath_residual_20260914'
PARENT=base.ROOT/'results/reification_kernel_20260914'
PHYSICAL=base.ROOT/'results/typed_sheaf_20260914'
EVIDENCE=base.ROOT/'results/tabicl_pathway_context_inputs_20260913'
RESEARCH=base.ROOT/'research/path_geometry_20260914'
KINDS=('full','no_brg','shuffled_brg')


def parent_scores(task,a):
    """Reconstruct the frozen readout, including the original OOF normalization."""
    root=PARENT/'runs'/task/'pretrained';path=root/'model.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    x=np.full((len(a['y_train']),len(state['w'])),np.nan,dtype=np.float32)
    values=[];paths=[path]
    for fold in range(5):
        source=root/('features_'+str(fold)+'.npz')
        with np.load(source,allow_pickle=False) as z:
            expected=np.flatnonzero(a['folds']==fold)
            np.testing.assert_array_equal(z['oof_indices'],expected)
            x[expected]=z['oof'];values.append(z['val'].copy())
        paths.append(source)
    if not np.isfinite(x).all():raise ValueError('Incomplete parent OOF features')
    outputs={}
    with torch.no_grad():
        kernel=state['kernel.bias']+state['kernel.row'][:,None]+state['kernel.col'][None,:]
        kernel=kernel+state['kernel.ur']@state['kernel.w']@state['kernel.ud'].T
        for part,features in [('train',x),('val',np.stack(values).mean(0))]:
            q=a['pairs_'+part]
            z=(torch.from_numpy(features)-state['mean'])/state['scale']/(len(state['w'])**.5)
            outputs['base_'+part]=(kernel[q[0],q[1]]+z@state['w']).numpy()
    source=root/'scores.npz'
    with np.load(source,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['pairs'],a['pairs_val']);np.testing.assert_array_equal(z['y'],a['y_val'])
        np.testing.assert_allclose(outputs['base_val'],z['pred'],atol=5e-5,rtol=5e-5)
        error=float(np.max(np.abs(outputs['base_val']-z['pred'])))
    paths.append(source)
    return outputs,paths,error


def background(seed):
    dictionary=json.loads((EVIDENCE/'dictionary.json').read_text())
    matrices={name:sparse.load_npz(EVIDENCE/(name+'.npz')).tocsr() for name in
              ('MiDrug_functional','LncDrug_functional','drug_gene','gene_pathway')}
    active=np.asarray(matrices['MiDrug_functional'].sum(0)+matrices['LncDrug_functional'].sum(0)+matrices['drug_gene'].sum(0)).ravel()>0
    genes=np.flatnonzero(active)
    gp=matrices['gene_pathway'][genes]
    pathways=np.flatnonzero(np.asarray(gp.sum(0)).ravel()>0)
    matrices={k:v[:,genes] for k,v in matrices.items() if k!='gene_pathway'}
    matrices['gene_pathway']=gp[:,pathways]
    shuffled={};audits={}
    for i,task in enumerate(base.TASKS):
        name=task+'_functional'
        shuffled[name],audits[task]=method.degree_preserving_shuffle(matrices[name],seed+77+i)
    return dictionary,matrices,shuffled,genes,pathways,audits


def prepare(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new path-reasoning experiment directory')
    acceptance=json.loads((PARENT/'ASSESSMENT.json').read_text())
    if not acceptance['shared_gate']:raise ValueError('Frozen parent did not pass its development screen')
    names=('biopath_experiment.py','biopath_models.py','test_biopath_models.py','test_biopath_experiment.py','non_tabular_experiment.py')
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),seed=20260914,tasks=list(base.TASKS),
           kinds=list(KINDS),width=32,depth=6,steps=1500,lr=.005,weight_decay=.001,
           residual_penalty=.5,batch_drugs=4,eval_batch=4,fold_count=5,
           primary='0.5 GNN logit + 0.5 (frozen REIFM-attribute-kernel logit + learned path residual); same modification and settings on both tasks.',
           literature='Hu et al. Enhancing link prediction in biomedical knowledge graphs with BioPathNet. Nature Biomedical Engineering, 2026-01-20. DOI 10.1038/s41551-025-01598-z.',
           architecture='Portable query-conditioned DistMult, six width-32 PNA layers with source boundary, LayerNorm, ReLU, residual shortcuts, and a two-layer MLP. Final output layer initialized to zero. Drug-to-RNA direction only.',
           adaptation='New residual extension, not official BioPathNet reproduction. No newly generated or hard-mined negatives, no symmetric scoring, no external BioPathNet weights. Existing sampled-unobserved pairs retain their original labels.',
           background='Frozen functional RNA-gene evidence, unsigned drug-gene targets, and gene-pathway membership. Weak binding/association-only RNA-gene edges are not added. Background edges enter propagation, not the supervised loss.',
           control='no_brg removes background edges; shuffled_brg switches RNA-gene edges separately by RNA type, exactly preserving every RNA degree and every gene degree. Original graph, drug targets, pathways, readout, losses and budgets stay fixed.',
           sampling='Sample a fold proportional to its query count; sample four drugs with replacement proportional to fold query count per drug. Each sampled drug contributes mean loss over its fold queries. This estimates uniform loss over the frozen training pairs, not macro-drug loss.',
           masking='All positives in the selected OOF query fold removed from the physical graph before reverse edges, PNA degree computation and propagation. Evaluation averages the same five masked contexts.',
           training='AdamW for path parameters only. BCE on frozen parent logit plus residual, plus 0.5 mean squared residual; gradient norm clipped at 5. Final step only, no validation early stopping or parameter selection.',
           gate='MiDrug mean AUC/AP must exceed the frozen parent composition and both matched controls; each MiDrug GNN-seed comparison must be positive. LncDrug mean and each GNN-seed comparison may lose at most 0.001 absolute AUC/AP relative to the frozen parent composition.',
           seed_interpretation='One neural fit per kind/task. Three original GNN seeds supply reference predictions; they are not three path-model fits or independent data partitions.',
           development_boundary='Historical inner screen already used in development. Other overlapping resplits were used for other methods. No independent biological generalization claim.',
           original_test_access=False,outer_query_labels_access=False,
           source_sha256={n:base.digest(base.ROOT/n) for n in names},input_sha256={})
    dictionary,matrices,shuffled,genes,pathways,shuffle_audit=background(p['seed'])
    if any(x['changed_fraction']<.5 for x in shuffle_audit.values()):raise ValueError('Insufficient effective background shuffle')
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    for name in names:shutil.copy2(base.ROOT/name,out/'source_snapshot'/name)
    paths=[EVIDENCE/'dictionary.json',EVIDENCE/'INPUTS.json',PARENT/'protocol.json',PARENT/'ASSESSMENT.json']
    paths += [EVIDENCE/(n+'.npz') for n in ('MiDrug_functional','LncDrug_functional','drug_gene','gene_pathway')]
    paths += [RESEARCH/n for n in ('author_biopathnet_layer.py','author_biopathnet_model.py','author_config_lnctardppi_lnctardppi.yaml','biopathnet_crossref.json','biopathnet_text.txt')]
    paths += [base.ROOT/'vendor/biopathnet_20260914'/n for n in ('LICENSE','PROVENANCE.json')]
    metadata=dict(gene_dictionary_indices=genes.tolist(),pathway_dictionary_indices=pathways.tolist(),
                  gene_names=[dictionary['genes'][i] for i in genes],shuffle_audit=shuffle_audit,
                  gene_count=len(genes),pathway_count=len(pathways),
                  semantics='Database functional-support classification is not a measured per-cell ncRNA intervention effect. Unsigned drug targets are not gene knockout effects. Pathway membership is an annotation relationship.')
    base.write_json(out/'background.json',metadata);paths.append(out/'background.json')
    for task in p['tasks']:
        node_path=PHYSICAL/'inputs'/task/'arrays.npz'
        with np.load(node_path,allow_pickle=False) as z:physical={n:z[n].copy() for n in z.files}
        schema_path=PHYSICAL/'inputs'/task/'graph.json';schema=json.loads(schema_path.read_text())
        parent_path=PARENT/'inputs'/task/'arrays.npz'
        with np.load(parent_path,allow_pickle=False) as z:
            a={n:z[n].copy() for n in ('pairs_train','y_train','pairs_val','y_val','folds','pair_offset','nr','nd','gnn_666','gnn_0','gnn_1')}
        for name in ('pairs_train','y_train','pairs_val','y_val'):np.testing.assert_array_equal(a[name],physical[name])
        offsets={};cursor=0
        for name,count in zip(schema['node_types'],physical['node_counts']):offsets[name]=cursor;cursor+=int(count)
        if cursor!=schema['nodes']:raise ValueError('Original node dictionary mismatch')
        rna='miRNA' if task=='MiDrug' else 'lncRNA'
        np.testing.assert_array_equal(a['pair_offset'],[offsets[rna],offsets['drug']])
        if int(a['nr'])!=len(dictionary['rna_names'][task]) or int(a['nd'])!=len(dictionary['drug_names']):
            raise ValueError('Evidence and original entity dictionaries differ in size')
        # Verify the index-order contract against original graph construction inputs.
        net=base.ROOT.parents[1]/'incrd-run/2.construct_network'
        rna_file=net/('MiExp.csv' if task=='MiDrug' else 'LncExp.csv')
        np.testing.assert_array_equal(pd.read_csv(rna_file,index_col=0,nrows=0).columns.str.upper(),dictionary['rna_names'][task])
        np.testing.assert_array_equal(pd.read_csv(net/'Drug.csv')['Drug_Name'].str.upper(),dictionary['drug_names'])
        paths += [rna_file,net/'Drug.csv']
        gene_offset=cursor;pathway_offset=cursor+len(genes)
        keys=schema['physical_relations'].copy();edges=[physical['edges']];random_edges=[physical['edges']]
        relations=[physical['relation']];codes=[physical['target_codes']];bg=[np.zeros(len(physical['relation']),dtype=bool)]
        additions=[('MiDrug_functional',offsets['miRNA'],gene_offset,['miRNA','functional_gene','gene']),
                   ('LncDrug_functional',offsets['lncRNA'],gene_offset,['lncRNA','functional_gene','gene']),
                   ('drug_gene',offsets['drug'],gene_offset,['drug','target_gene','gene']),
                   ('gene_pathway',gene_offset,pathway_offset,['gene','member_of','pathway'])]
        for name,start,end,key in additions:
            m=matrices[name].tocoo();s=shuffled.get(name,matrices[name]).tocoo()
            edges.append(np.stack([m.row+start,m.col+end]));random_edges.append(np.stack([s.row+start,s.col+end]))
            relations.append(np.full(m.nnz,len(keys),dtype=np.int64));codes.append(np.full(m.nnz,-1,dtype=np.int64));bg.append(np.ones(m.nnz,dtype=bool));keys.append(key)
        a.update(edges=np.concatenate(edges,axis=1),shuffled_edges=np.concatenate(random_edges,axis=1),
                 relation=np.concatenate(relations),target_codes=np.concatenate(codes),background=np.concatenate(bg),
                 base_nodes=np.array(cursor),nodes=np.array(pathway_offset+len(pathways)),
                 physical_relations=np.array(len(keys)),symmetric_relations=np.array([k[0]==k[2] for k in keys]),
                 query_relation=np.array([k[1] for k in keys].index(task)+len(keys)))
        scores,pp,error=parent_scores(task,a);a.update(scores);paths+=pp
        dest=out/'inputs'/task;dest.mkdir(parents=True)
        np.savez_compressed(dest/'arrays.npz',**a)
        base.write_json(dest/'graph.json',dict(node_types=schema['node_types']+['gene','pathway'],physical_relations=keys,
                         base_nodes=cursor,nodes=int(a['nodes']),physical_edges=a['edges'].shape[1],
                         background_edges=int(a['background'].sum()),parent_readout_replay_max_abs=error))
        paths += [node_path,schema_path,parent_path,dest/'arrays.npz',dest/'graph.json']
    p['input_sha256']={str(path):base.digest(path) for path in paths}
    base.write_json(out/'protocol.json',p);print('FROZEN',out,flush=True)


def protocol(out):
    p=json.loads((Path(out)/'protocol.json').read_text())
    for name,h in p['source_sha256'].items():
        if base.digest(base.ROOT/name)!=h:raise ValueError('Source changed: '+name)
    for name,h in p['input_sha256'].items():
        if base.digest(name)!=h:raise ValueError('Input changed: '+name)
    return p


def load_arrays(out,task):
    with np.load(Path(out)/'inputs'/task/'arrays.npz',allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}


@torch.no_grad()
def score(model,a,contexts,pairs,device,batch):
    ids=np.unique(pairs[1]);views=[]
    for ctx in contexts:
        values=np.full(pairs.shape[1],np.nan,dtype=np.float32)
        for start in range(0,len(ids),batch):
            drugs=ids[start:start+batch]
            source=torch.as_tensor(drugs+int(a['pair_offset'][1]),device=device)
            output=model(ctx,source,int(a['query_relation']),int(a['pair_offset'][0]),int(a['nr'])).cpu().numpy()
            for j,drug in enumerate(drugs):
                ix=np.flatnonzero(pairs[1]==drug);values[ix]=output[j,pairs[0,ix]]
        if not np.isfinite(values).all():raise ValueError('Nonfinite or uncovered path prediction')
        views.append(values)
    return np.stack(views).mean(0)


def fit(a,p,kind,device):
    torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    rng=np.random.default_rng(p['seed']+811)
    model=method.PathResidual(2*int(a['physical_relations']),p['width'],p['depth']).to(device)
    contexts=[method.context(a,f,kind,device) for f in range(p['fold_count'])]
    opt=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    base_train=torch.as_tensor(a['base_train'],device=device);y=torch.as_tensor(a['y_train'],device=device)
    fold_sizes=np.bincount(a['folds'],minlength=p['fold_count']);fold_prob=fold_sizes/fold_sizes.sum()
    groups=[]
    for f in range(p['fold_count']):
        ix=np.flatnonzero(a['folds']==f);counts=np.bincount(a['pairs_train'][1,ix],minlength=int(a['nd']))
        groups.append((ix,counts,counts/counts.sum()))
    trace=[];start=time.perf_counter()
    for step in range(1,p['steps']+1):
        fold=int(rng.choice(p['fold_count'],p=fold_prob));ix,counts,prob=groups[fold]
        selected=rng.choice(int(a['nd']),size=p['batch_drugs'],replace=True,p=prob)
        drugs,multiplicity=np.unique(selected,return_counts=True)
        query=ix[np.isin(a['pairs_train'][1,ix],drugs)]
        which=np.searchsorted(drugs,a['pairs_train'][1,query])
        weights=multiplicity[which]/(p['batch_drugs']*counts[a['pairs_train'][1,query]])
        if not np.isclose(weights.sum(),1.):raise ValueError('Incorrect pair-loss sampling weights')
        model.train();opt.zero_grad(set_to_none=True)
        source=torch.as_tensor(drugs+int(a['pair_offset'][1]),device=device)
        output=model(contexts[fold],source,int(a['query_relation']),int(a['pair_offset'][0]),int(a['nr']))
        path=output[torch.as_tensor(which,device=device),torch.as_tensor(a['pairs_train'][0,query],device=device)]
        qi=torch.as_tensor(query,device=device);w=torch.as_tensor(weights,dtype=torch.float32,device=device)
        bce=F.binary_cross_entropy_with_logits(base_train[qi]+path,y[qi],reduction='none')
        loss=(w*(bce+p['residual_penalty']*path.square())).sum()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite path loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        if step%100==0 or step==p['steps']:
            row=dict(step=step,loss=float(loss),bce=float((w*bce).sum()),
                     residual_rms=float((w*path.square()).sum().sqrt()),elapsed=time.perf_counter()-start)
            trace.append(row);print('TRAIN',kind,step,round(row['loss'],6),round(row['elapsed'],1),flush=True)
    model.eval();return model,contexts,trace,time.perf_counter()-start


def verify_files(dest,protocol_hash):
    row=json.loads((dest/'result.json').read_text())
    if row['protocol_sha256']!=protocol_hash:raise ValueError('Run protocol differs')
    for name,h in row['files'].items():
        if base.digest(dest/name)!=h:raise ValueError('Artifact changed: '+str(dest/name))
    return row


def run(out,task,device,kind=None):
    out=Path(out);p=protocol(out);a=load_arrays(out,task)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol_hash=base.digest(out/'protocol.json')
    for arm in ([kind] if kind else p['kinds']):
        if arm not in p['kinds']:raise ValueError('Unknown path arm')
        dest=out/'runs'/task/arm
        if (dest/'result.json').exists():verify_files(dest,protocol_hash);continue
        model,contexts,trace,seconds=fit(a,p,arm,device)
        residual=score(model,a,contexts,a['pairs_val'],device,p['eval_batch'])
        dest.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),dest/'model.pt')
        loaded=method.PathResidual(2*int(a['physical_relations']),p['width'],p['depth']).to(device)
        loaded.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True),strict=True);loaded.eval()
        ix=np.unique(np.linspace(0,len(residual)-1,min(31,len(residual))).astype(int))[::-1]
        replay=score(loaded,a,contexts,a['pairs_val'][:,ix],device,3)
        np.testing.assert_allclose(replay,residual[ix],atol=2e-4,rtol=2e-4)
        pred=a['base_val']+residual
        scores=dict(pred=pred,residual=residual,base=a['base_val'],y=a['y_val'],pairs=a['pairs_val'])
        metrics={}
        for seed in (666,0,1):
            scores['composed_'+str(seed)]=.5*a['gnn_'+str(seed)]+.5*pred
            metrics[str(seed)]=base.metrics(a['y_val'],scores['composed_'+str(seed)],a['pairs_val'])
        np.savez_compressed(dest/'scores.npz',**scores);base.write_json(dest/'trace.json',trace)
        row=dict(task=task,kind=arm,seed=p['seed'],seconds=seconds,protocol_sha256=protocol_hash,
                 parameters=sum(v.numel() for v in model.parameters()),metrics=metrics,
                 replay_rows=len(ix),replay_max_abs=float(np.max(np.abs(replay-residual[ix]))),
                 residual_rms=float(np.sqrt(np.mean(residual.astype(float)**2))),
                 contexts=len(contexts),files={n:base.digest(dest/n) for n in ('model.pt','scores.npz','trace.json')})
        base.write_json(dest/'result.json',row);print('DONE',task,arm,json.dumps(metrics),flush=True)
        del model,loaded,contexts


def assessment(table):
    means=table.groupby(['task','kind'])[['auc','ap','macro_drug_ap']].mean()
    comparisons={};passed=True
    for task in base.TASKS:
        p=table[(table.task==task)&(table.kind=='full')].sort_values('gnn_seed')
        b=table[(table.task==task)&(table.kind=='parent')].sort_values('gnn_seed')
        comparisons[task]={}
        for metric in ('auc','ap'):
            delta=p[metric].to_numpy()-b[metric].to_numpy()
            okay=bool(np.all(delta>0)) if task=='MiDrug' else bool(delta.mean()>=-.001 and delta.min()>=-.001)
            comparisons[task][metric]=dict(mean_delta=float(delta.mean()),minimum_delta=float(delta.min()),positive_gnn_seeds=int((delta>0).sum()),passed=okay)
            passed &= okay
    mechanism={}
    for control in ('no_brg','shuffled_brg'):
        mechanism[control]={}
        for metric in ('auc','ap'):
            delta=float(means.loc[('MiDrug','full'),metric]-means.loc[('MiDrug',control),metric])
            mechanism[control][metric]=delta;passed &= delta>0
    return dict(shared_gate=bool(passed),performance=bool(all(v['passed'] for t in comparisons.values() for v in t.values())),
                comparisons=comparisons,mechanism_deltas=mechanism,
                mechanism_supported=bool(all(v>0 for c in mechanism.values() for v in c.values())),
                independent_biological_validation=False,neural_fits=6,
                newly_confirmed_innovations=0,
                interpretation='Historical, reused development screen. Three GNN predictions do not create three neural path models. A passing screen requires a separate frozen resplit confirmation; no causal-path claim.')


def report(out):
    out=Path(out);p=protocol(out);protocol_hash=base.digest(out/'protocol.json');rows=[];max_error=0.
    for task in p['tasks']:
        a=load_arrays(out,task)
        for seed in (666,0,1):
            for kind,pred in [('gnn',a['gnn_'+str(seed)]),('parent',.5*a['gnn_'+str(seed)]+.5*a['base_val'])]:
                rows.append(dict(task=task,kind=kind,gnn_seed=seed,**base.metrics(a['y_val'],pred,a['pairs_val'])))
        for kind in p['kinds']:
            dest=out/'runs'/task/kind;row=verify_files(dest,protocol_hash);max_error=max(max_error,row['replay_max_abs'])
            with np.load(dest/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['y'],a['y_val']);np.testing.assert_array_equal(z['pairs'],a['pairs_val'])
                for seed in (666,0,1):
                    expected=.5*a['gnn_'+str(seed)]+.5*(a['base_val']+z['residual'])
                    np.testing.assert_allclose(z['composed_'+str(seed)],expected,atol=1e-6,rtol=1e-6)
                    m=base.metrics(a['y_val'],expected,a['pairs_val'])
                    for n,value in m.items():np.testing.assert_allclose(value,row['metrics'][str(seed)][n],atol=1e-12,rtol=1e-12)
                    rows.append(dict(task=task,kind=kind,gnn_seed=seed,**m))
    table=pd.DataFrame(rows);table.to_csv(out/'metrics.csv',index=False)
    summary=table.groupby(['task','kind'])[['auc','ap','macro_drug_ap']].mean().reset_index();summary.to_csv(out/'summary.csv',index=False)
    result=assessment(table);result['protocol_sha256']=protocol_hash;base.write_json(out/'ASSESSMENT.json',result)
    base.write_json(out/'COMPLETION.json',dict(neural_fits=6,source_input_artifact_hashes_verified=True,
                    metrics_recomputed=True,full_encoder_replay_rows_per_fit=31,replay_max_abs=max_error,
                    accepted_default_replaced=False,original_test_access=False))
    print(summary.to_string(index=False));print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','run','report'])
    ap.add_argument('--out-dir',default=str(OUT));ap.add_argument('--task',choices=base.TASKS)
    ap.add_argument('--kind',choices=KINDS);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    if args.command=='prepare':prepare(args.out_dir)
    elif args.command=='run':run(args.out_dir,args.task,args.device,args.kind)
    else:report(args.out_dir)


if __name__=='__main__':main()
