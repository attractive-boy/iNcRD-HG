"""FORGE-inspired shared latent regression on real functional assays.

RNA expression predicts CRISPR gene effects and PRISM viability jointly. These
are predictive expression associations, not causal effects of RNA intervention.
This is an independent adaptation, not the original FORGE implementation.
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
from torch import nn

from improve import non_tabular_experiment as base

INPUTS=base.ROOT/'results/functional_response_inputs_20260914'
OUT=base.ROOT/'results/functional_response_transfer_20260914'
KINDS=('joint','drug_only','random_encoder','cell_shuffled')


def cell_folds(groups,seed,count=5):
    rng=np.random.default_rng(seed);folds=np.empty(len(groups),dtype=np.int64)
    for group in np.unique(groups):
        ix=np.flatnonzero(groups==group);order=rng.permutation(ix)
        folds[order]=(np.arange(len(ix))+int(rng.integers(count)))%count
    return folds


def within_lineage_permutation(groups,seed):
    rng=np.random.default_rng(seed);permutation=np.arange(len(groups))
    for group in np.unique(groups):
        ix=rng.permutation(np.flatnonzero(groups==group))
        if len(ix)>1:permutation[ix]=np.roll(ix,1)
    return permutation


def target_norm(y):
    observed=np.isfinite(y);count=observed.sum(0)
    mean=np.where(observed,y,0.).sum(0,dtype=np.float64)/np.maximum(count,1)
    variance=np.where(observed,(y-mean)**2,0.).sum(0,dtype=np.float64)/np.maximum(count,1)
    return mean.astype(np.float32),np.maximum(np.sqrt(variance),.1).astype(np.float32),count>=8


def normalization(a,indices):
    x=a['x'][indices]
    stats=dict(x_mean=x.mean(0,dtype=np.float64).astype(np.float32),
               x_scale=np.maximum(x.std(0,dtype=np.float64),1e-5).astype(np.float32))
    for name in ('gene','drug'):
        mean,scale,eligible=target_norm(a[name][indices])
        stats.update({name+'_mean':mean,name+'_scale':scale,name+'_eligible':eligible})
    return stats


def standard_targets(y,mean,scale,eligible):
    mask=np.isfinite(y)&eligible[None,:]
    z=np.where(mask,(y-mean)/scale,0.).astype(np.float32)
    return z,mask


def masked_mse(pred,target,mask):
    residual=torch.where(mask,pred-target,torch.zeros_like(pred))
    return residual.square().sum()/mask.sum().clamp_min(1)


class SharedFunctionalModel(nn.Module):
    def __init__(self,stats,rank=32,freeze_encoder=False):
        super().__init__()
        for key,value in stats.items():self.register_buffer(key,torch.as_tensor(value))
        nx,ng,nd=len(stats['x_mean']),len(stats['gene_mean']),len(stats['drug_mean'])
        self.encoder=nn.Linear(nx,rank,bias=False)
        self.gene_head=nn.Linear(rank,ng)
        self.drug_head=nn.Linear(rank,nd)
        nn.init.normal_(self.encoder.weight,std=nx**-.5)
        for head in (self.gene_head,self.drug_head):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.encoder.requires_grad_(not freeze_encoder)

    def forward(self,x):
        state=self.encoder((x-self.x_mean)/self.x_scale)
        return self.gene_head(state),self.drug_head(state)

    def penalty(self,kind):
        rank=self.encoder.out_features
        penalty=self.encoder.weight.square().sum()/rank+self.drug_head.weight.square().sum()/self.drug_head.out_features
        if kind!='drug_only':penalty=penalty+self.gene_head.weight.square().sum()/self.gene_head.out_features
        return penalty


def assay_metrics(y,pred,mean,scale,eligible):
    if pred.shape!=y.shape or not np.isfinite(pred).all():raise ValueError('Invalid assay prediction')
    truth,mask=standard_targets(y,mean,scale,eligible)
    count=int(mask.sum())
    if not count:raise ValueError('No eligible held-out assay observations')
    error=np.where(mask,pred-truth,0.).astype(np.float64)
    normalized_mse=float(np.square(error).sum()/count)
    baseline=float(np.square(np.where(mask,truth,0.).astype(np.float64)).sum()/count)
    raw_mse=float(np.square(error*scale).sum()/count)
    n=mask.sum(0);den=np.maximum(n,1)
    pm=np.where(mask,pred,0.).sum(0,dtype=np.float64)/den
    tm=np.where(mask,truth,0.).sum(0,dtype=np.float64)/den
    pc=np.where(mask,pred-pm,0.);tc=np.where(mask,truth-tm,0.)
    pv=np.square(pc).sum(0);tv=np.square(tc).sum(0)
    valid=(n>=5)&(pv>1e-12)&(tv>1e-12)
    corr=(pc*tc).sum(0)/np.sqrt(np.maximum(pv*tv,1e-30))
    return dict(normalized_mse=normalized_mse,training_mean_baseline_mse=baseline,
                relative_mse_reduction=1-normalized_mse/baseline,raw_mse=raw_mse,
                macro_pearson=float(corr[valid].mean()) if valid.any() else 0.,
                correlation_outputs=int(valid.sum()),observations=count)


def fit(a,train,kind,p,device,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    stats=normalization(a,train)
    model=SharedFunctionalModel(stats,p['rank'],kind=='random_encoder').to(device)
    if kind=='drug_only':model.gene_head.requires_grad_(False)
    permutation=within_lineage_permutation(a['lineages'][train],seed+913) if kind=='cell_shuffled' else np.arange(len(train))
    x=torch.as_tensor(a['x'][train[permutation]],device=device)
    targets={}
    for name in ('gene','drug'):
        y,mask=standard_targets(a[name][train],stats[name+'_mean'],stats[name+'_scale'],stats[name+'_eligible'])
        targets[name]=(torch.as_tensor(y,device=device),torch.as_tensor(mask,device=device))
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['lr'],weight_decay=0.)
    trace=[];start=time.perf_counter()
    for epoch in range(1,p['epochs']+1):
        opt.zero_grad(set_to_none=True)
        gene,drug=model(x)
        gl=masked_mse(gene,*targets['gene']);dl=masked_mse(drug,*targets['drug'])
        # Preserve the drug-head loss weight in every arm. Only the auxiliary
        # gene loss changes in the drug-only ablation.
        loss=dl+(gl if kind!='drug_only' else 0.)+p['regularization']*model.penalty(kind)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite functional regression loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if epoch%20==0 or epoch==p['epochs']:
            trace.append(dict(epoch=epoch,objective=float(loss),gene_mse=float(gl),drug_mse=float(dl)))
    model.eval()
    return model,stats,trace,time.perf_counter()-start,permutation


def load_arrays():
    with np.load(INPUTS/'arrays.npz',allow_pickle=False) as z:a={n:z[n].copy() for n in z.files}
    if len(np.unique(a['cells']))!=len(a['cells']):raise ValueError('Duplicate cell identity')
    if not np.isfinite(a['x']).all() or np.isinf(a['gene']).any() or np.isinf(a['drug']).any():
        raise ValueError('Invalid expression or assay values')
    if any(len(a[n])!=len(a['cells']) for n in ('x','gene','drug','lineages')):
        raise ValueError('Cell dimensions differ')
    a['matched_drug_outputs']=np.asarray(sparse.load_npz(INPUTS/'drug_assay_map.npz').sum(0)).ravel()>0
    return a


def prepare(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new functional-experiment directory')
    record=json.loads((INPUTS/'INPUTS.json').read_text())
    if record['association_labels_read']:raise ValueError('Unexpected association supervision')
    if base.digest(base.ROOT/'functional_response_data.py')!=record['source_sha256']:
        raise ValueError('Input builder changed after alignment')
    paths=[INPUTS/'INPUTS.json']
    for name,h in record['files'].items():
        path=INPUTS/name
        if base.digest(path)!=h:raise ValueError('Aligned input changed: '+name)
        paths.append(path)
    for name,h in record['input_sha256'].items():
        if base.digest(name)!=h:raise ValueError('Original input changed: '+name)
    a=load_arrays()
    names=('functional_response_transfer.py','functional_response_data.py',
           'test_functional_response_transfer.py','non_tabular_experiment.py')
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),seed=20260914,
           rank=32,epochs=200,lr=.01,regularization=.1,fold_count=5,kinds=list(KINDS),
           primary='joint: one linear RNA-expression encoder and separate CRISPR gene-effect and PRISM viability heads.',
           objective='Mean masked standardized drug MSE + mean masked standardized gene MSE + 0.1 dimension-normalized squared-weight penalty. Drug-only removes only the gene loss and gene-head penalty.',
           optimizer='Full-batch AdamW, weight_decay=0, gradient norm clipped at 10; final epoch only.',
           normalization='Training-cell means and population SD only; RNA SD floor 1e-5, target SD floor 0.1. Outputs require >=8 training observations. Missing assays contribute no loss or gradient.',
           cell_split='Five folds, randomized within lineage. All RNA profiles and both assay types for a held-out ModelID are excluded from fitting, normalization and permutation.',
           shuffle='Derange RNA expression identities within each training lineage, preserving assay values and missingness. Singleton lineages cannot be deranged. Evaluate on correct held-out expression identities.',
           controls='Drug-only and joint use the same drug loss weight; random-encoder freezes the same initialized encoder; both functional heads are fitted for random and shuffled controls.',
           gate=dict(mean_mse_advantage=0.,min_positive_folds=3,
                     baseline_assays=['gene','drug','matched_drug'],
                     control_assays={'drug':['drug_only','random_encoder','cell_shuffled'],
                                     'gene':['random_encoder','cell_shuffled']}),
           gate_description='Joint normalized MSE must beat the training-mean predictor for gene, all drug, and exactly matched drug outputs in the mean and >=3/5 folds. Joint drug MSE must beat drug-only, random and shuffled controls in mean and >=3/5 folds; gene MSE must beat random and shuffled controls by the same rule.',
           followup='Fit four full-data functional models only if the external gate passes. Any downstream association experiment requires a separate frozen protocol and mechanism controls.',
           validation_boundary='Held-out cells within archived DepMap/PRISM cohorts, not an independent biological cohort; no RNA-drug association labels are read in this experiment.',
           phenotype_semantics=record['phenotype_semantics'],
           source_sha256={n:base.digest(base.ROOT/n) for n in names},
           input_sha256={str(path):base.digest(path) for path in paths},
           association_labels_read=False,model_selection_on_association_outcomes=False,
           selection_history='Recipe frozen before any functional-model fits or held-out assay metrics. Input dimensions and coverage were inspected; no hyperparameter search.',
           versions=dict(python=__import__('platform').python_version(),numpy=np.__version__,torch=torch.__version__))
    folds=cell_folds(a['lineages'],p['seed'],p['fold_count'])
    if set(folds)!=set(range(p['fold_count'])):raise ValueError('Empty cell fold')
    p['fold_cells']={str(i):int((folds==i).sum()) for i in range(p['fold_count'])}
    p['matched_assay_outputs']=int(a['matched_drug_outputs'].sum())
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    for name in names:shutil.copy2(base.ROOT/name,out/'source_snapshot'/name)
    np.savez_compressed(out/'folds.npz',folds=folds,cells=a['cells'])
    p['input_sha256'][str(out/'folds.npz')]=base.digest(out/'folds.npz')
    base.write_json(out/'protocol.json',p)
    print('FROZEN',out,'cells',len(folds),'folds',p['fold_cells'],flush=True)


def protocol(out):
    p=json.loads((Path(out)/'protocol.json').read_text())
    for name,h in p['source_sha256'].items():
        if base.digest(base.ROOT/name)!=h:raise ValueError('Frozen source changed: '+name)
    for name,h in p['input_sha256'].items():
        if base.digest(name)!=h:raise ValueError('Frozen input changed: '+name)
    return p


def predict(model,x,device,batch=64):
    parts={'gene':[],'drug':[]}
    with torch.no_grad():
        for begin in range(0,len(x),batch):
            gene,drug=model(torch.as_tensor(np.ascontiguousarray(x[begin:begin+batch]),device=device))
            parts['gene'].append(gene.cpu().numpy());parts['drug'].append(drug.cpu().numpy())
    return {name:np.concatenate(values) for name,values in parts.items()}


def evaluate(a,indices,pred,stats):
    results={}
    for name in ('gene','drug'):
        results[name]=assay_metrics(a[name][indices],pred[name],stats[name+'_mean'],
                                   stats[name+'_scale'],stats[name+'_eligible'])
    use=a['matched_drug_outputs']
    results['matched_drug']=assay_metrics(a['drug'][indices][:,use],pred['drug'][:,use],
                                        stats['drug_mean'][use],stats['drug_scale'][use],stats['drug_eligible'][use])
    return results


def verify_result(dest,protocol_hash):
    row=json.loads((dest/'result.json').read_text())
    if row['protocol_sha256']!=protocol_hash:raise ValueError('Run protocol differs')
    for name,h in row['files'].items():
        if base.digest(dest/name)!=h:raise ValueError('Run artifact changed: '+str(dest/name))
    return row


def run(out,device,kind=None,stage='cv'):
    out=Path(out);p=protocol(out);a=load_arrays()
    protocol_hash=base.digest(out/'protocol.json')
    with np.load(out/'folds.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(a['cells'],z['cells']);folds=z['folds'].copy()
    if kind is not None and kind not in p['kinds']:raise ValueError('Unknown functional arm')
    kinds=[kind] if kind is not None else p['kinds']
    if stage=='full':
        assessment=json.loads((out/'ASSESSMENT.json').read_text())
        if assessment['protocol_sha256']!=protocol_hash or not assessment['external_functional_gate']:
            raise ValueError('External functional gate must pass before full-data fitting')
        for arm in p['kinds']:
            for fold in range(p['fold_count']):verify_result(out/'runs'/arm/str(fold),protocol_hash)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    for arm in kinds:
        for fold in (range(p['fold_count']) if stage=='cv' else ['full']):
            dest=out/'runs'/arm/str(fold)
            if (dest/'result.json').exists():
                verify_result(dest,protocol_hash);continue
            if stage=='cv':
                train=np.flatnonzero(folds!=fold);val=np.flatnonzero(folds==fold)
                if np.intersect1d(a['cells'][train],a['cells'][val]).size:raise ValueError('Cell leakage')
                seed=p['seed']+1009*fold
            else:
                train=val=np.arange(len(folds));seed=p['seed']
            model,stats,trace,seconds,permutation=fit(a,train,arm,p,device,seed)
            pred=predict(model,a['x'][val],device)
            metrics=evaluate(a,val,pred,stats)
            dest.mkdir(parents=True,exist_ok=True)
            torch.save(model.state_dict(),dest/'model.pt')
            loaded=SharedFunctionalModel(stats,p['rank'],arm=='random_encoder').to(device)
            loaded.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True),strict=True)
            loaded.eval()
            replay=predict(loaded,a['x'][val][::-1],device,batch=37)
            errors={}
            for name in ('gene','drug'):
                np.testing.assert_allclose(replay[name][::-1],pred[name],atol=1e-4,rtol=1e-4)
                errors[name]=float(np.max(np.abs(replay[name][::-1]-pred[name])))
            np.savez_compressed(dest/'predictions.npz',indices=val,cells=a['cells'][val],**pred)
            np.savez_compressed(dest/'fit_indices.npz',train=train,val=val,expression_train=train[permutation])
            base.write_json(dest/'trace.json',trace)
            row=dict(kind=arm,fold=fold,stage=stage,seed=int(seed),protocol_sha256=protocol_hash,
                     train_cells=len(train),evaluation_cells=len(val),
                     assessment_use='held-out cells' if stage=='cv' else 'in-sample replay only; excluded from external gate',
                     seconds=seconds,metrics=metrics,replay_max_abs=errors,
                     shuffled_training_cells=int((permutation!=np.arange(len(train))).sum()),
                     gene_eligible=int(stats['gene_eligible'].sum()),drug_eligible=int(stats['drug_eligible'].sum()),
                     files={name:base.digest(dest/name) for name in ('model.pt','predictions.npz','fit_indices.npz','trace.json')})
            base.write_json(dest/'result.json',row)
            print('DONE',arm,fold,round(seconds,2),json.dumps({n:round(m['normalized_mse'],6) for n,m in metrics.items()}),flush=True)
            del model,loaded


def assess_metrics(frame,p):
    threshold=p['gate']['mean_mse_advantage'];minimum=p['gate']['min_positive_folds']
    checks=[]
    for assay in p['gate']['baseline_assays']:
        joint=frame[(frame.kind=='joint')&(frame.assay==assay)].sort_values('fold')
        if len(joint)!=p['fold_count']:raise ValueError('Incomplete joint folds')
        delta=(joint.training_mean_baseline_mse-joint.normalized_mse).to_numpy()
        checks.append(dict(assay=assay,control='training_mean',mean_mse_advantage=float(delta.mean()),
                           positive_folds=int((delta>threshold).sum()),passed=bool(delta.mean()>threshold and (delta>threshold).sum()>=minimum)))
    for assay,controls in p['gate']['control_assays'].items():
        joint=frame[(frame.kind=='joint')&(frame.assay==assay)].sort_values('fold')
        for control in controls:
            other=frame[(frame.kind==control)&(frame.assay==assay)].sort_values('fold')
            if len(other)!=p['fold_count']:raise ValueError('Incomplete control folds')
            np.testing.assert_array_equal(joint.fold.to_numpy(),other.fold.to_numpy())
            delta=other.normalized_mse.to_numpy()-joint.normalized_mse.to_numpy()
            checks.append(dict(assay=assay,control=control,mean_mse_advantage=float(delta.mean()),
                               positive_folds=int((delta>threshold).sum()),passed=bool(delta.mean()>threshold and (delta>threshold).sum()>=minimum)))
    return bool(all(row['passed'] for row in checks)),checks


def report(out):
    out=Path(out);p=protocol(out);a=load_arrays();protocol_hash=base.digest(out/'protocol.json')
    with np.load(out/'folds.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['cells'],a['cells']);folds=z['folds'].copy()
    records=[];results=[];max_error=0.
    for kind in p['kinds']:
        observed=[]
        for fold in range(p['fold_count']):
            dest=out/'runs'/kind/str(fold);row=verify_result(dest,protocol_hash);results.append(row)
            with np.load(dest/'fit_indices.npz',allow_pickle=False) as z:
                train=z['train'].copy();val=z['val'].copy();permuted=z['expression_train'].copy()
            np.testing.assert_array_equal(train,np.flatnonzero(folds!=fold))
            np.testing.assert_array_equal(val,np.flatnonzero(folds==fold))
            if np.intersect1d(train,val).size or set(train)|set(val)!=set(range(len(a['cells']))):
                raise ValueError('Invalid fitting/evaluation partition')
            if set(permuted)!=set(train):raise ValueError('Invalid expression permutation')
            np.testing.assert_array_equal(a['lineages'][train],a['lineages'][permuted])
            observed.extend(val.tolist())
            stats=normalization(a,train)
            with np.load(dest/'predictions.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(val,z['indices']);np.testing.assert_array_equal(a['cells'][val],z['cells'])
                pred={name:z[name].copy() for name in ('gene','drug')}
            metrics=evaluate(a,val,pred,stats)
            for assay,values in metrics.items():
                for metric,value in values.items():
                    np.testing.assert_allclose(value,row['metrics'][assay][metric],atol=1e-9,rtol=1e-9)
                records.append(dict(kind=kind,fold=fold,assay=assay,**values))
            max_error=max(max_error,*row['replay_max_abs'].values())
        np.testing.assert_array_equal(np.sort(observed),np.arange(len(a['cells'])))
    frame=pd.DataFrame(records);frame.to_csv(out/'metrics.csv',index=False)
    metrics=['normalized_mse','training_mean_baseline_mse','relative_mse_reduction','raw_mse','macro_pearson']
    means=frame.groupby(['kind','assay'])[metrics].mean().reset_index();means.to_csv(out/'summary.csv',index=False)
    passed,checks=assess_metrics(frame,p)
    assessment=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=protocol_hash,
                    external_functional_gate=passed,checks=checks,completed_cell_cv_fits=len(results),
                    full_fits_authorized_by_protocol=passed,
                    association_validation_completed=False,validated_new_association_techniques=0,
                    interpretation='Descriptive prespecified within-cohort held-out-cell checks. No significance claim, no independent biological cohort, no RNA intervention interpretation.',
                    decision='Proceed to frozen full-data fits and a separate downstream transfer protocol.' if passed else 'Stop this joint functional-transfer hypothesis; do not expand downstream association tuning under the same claim.')
    base.write_json(out/'ASSESSMENT.json',assessment)
    base.write_json(out/'CV_COMPLETION.json',dict(protocol_sha256=protocol_hash,runs=len(results),
                    each_cell_predicted_once_per_kind=True,source_input_artifact_hashes_verified=True,
                    metrics_recomputed_from_predictions=True,all_heldout_predictions_reloaded_reordered_and_rebatched=True,
                    max_prediction_replay_error=max_error,association_labels_read=False))
    full=[]
    for kind in p['kinds']:
        dest=out/'runs'/kind/'full'
        if (dest/'result.json').exists():full.append(verify_result(dest,protocol_hash))
    if len(full)==len(p['kinds']):
        if not passed:raise ValueError('Full models exist despite failed gate')
        base.write_json(out/'FULL_COMPLETION.json',dict(protocol_sha256=protocol_hash,runs=len(full),
                        excluded_from_external_gate=True,files_verified=True,
                        max_prediction_replay_error=max(max(r['replay_max_abs'].values()) for r in full)))
    print(means.to_string(index=False),flush=True)
    print(json.dumps(assessment,ensure_ascii=False,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','run','report'])
    ap.add_argument('--out-dir',default=str(OUT));ap.add_argument('--device',default='cuda:1')
    ap.add_argument('--kind',choices=KINDS);ap.add_argument('--stage',choices=['cv','full'],default='cv')
    args=ap.parse_args()
    if args.command=='prepare':prepare(args.out_dir)
    elif args.command=='run':run(args.out_dir,args.device,args.kind,args.stage)
    else:report(args.out_dir)


if __name__=='__main__':main()
