"""Frozen compact LinkPFN pilot: prepare, generate, pretrain, tune, select,
evaluate, report, validate. Uses ONLY the previous outer training partitions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, ndcg_score
import torch
from torch.nn import functional as F

from improve.graph_tabm_modules import GraphTabM
from improve.link_pfn_modules import (
    FEATURE_DIM, FEATURE_NAMES, PRIORS, LinkPFN, balanced_indices,
    synthetic_episode, tensor_batch,
)
from improve.link_pfn_data import TASKS, snapshot_inputs, scenario, summary


ARMS = ('scratch', 'random', 'hetero', 'missing', 'tabm')
SOURCE_FILES = ('link_pfn.py','link_pfn_data.py','link_pfn_modules.py','test_link_pfn.py','graph_tabm_modules.py')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_json(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def source_hashes():
    return {n:digest(Path(__file__).with_name(n)) for n in SOURCE_FILES}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)


def metrics(y,score,pairs):
    y,score=np.asarray(y),np.asarray(score)
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite prediction')
    rows=[]
    for d in np.unique(pairs[:,1]):
        keep=pairs[:,1]==d
        yy,ss=y[keep],score[keep]
        if len(np.unique(yy))!=2:
            continue
        order=np.argsort(-ss,kind='stable')[:min(10,len(ss))]
        rows.append(dict(drug=int(d),ap=float(average_precision_score(yy,ss)),
            auc=float(roc_auc_score(yy,ss)),recall10=float(yy[order].sum()/yy.sum()),
            ndcg10=float(ndcg_score(yy[None],ss[None],k=min(10,len(ss)))),
            positives=int(yy.sum()),rows=len(yy)))
    if not rows:
        raise ValueError('No eligible drug in fixed partition')
    result=dict(auc=float(roc_auc_score(y,score)),ap=float(average_precision_score(y,score)),macro_drugs=len(rows))
    for k in ('ap','auc','recall10','ndcg10'):
        result['macro_'+k]=float(np.mean([r[k] for r in rows]))
    result['criterion']=.5*result['macro_ap']+.25*(result['auc']+result['ap'])
    return result,rows


def choose_checkpoint(trace,window):
    if trace[0]['step']!=0:
        raise ValueError('Initial checkpoint is required')
    choices=[(trace[0]['criterion'],0)]
    trained=trace[1:]
    for i in range(window-1,len(trained)):
        choices.append((float(np.mean([r['criterion'] for r in trained[i-window+1:i+1]])),trained[i]['step']))
    return sorted(choices,key=lambda v:(-v[0],v[1]))[0][::-1]


def prepare(args):
    out=Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    out.mkdir(parents=True,exist_ok=True)
    smoke=bool(args.smoke)
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),smoke=smoke,
        reference_dir=str(Path(args.reference_dir).resolve()),tasks=list(TASKS),arms=list(ARMS),
        priors=list(PRIORS),seeds=[666] if smoke else [666,0,1],split_seed=2026091201,
        generation_seed=83000000,probe_seed=93000000,
        synthetic_tasks=96 if smoke else 4096,synthetic_batch=8,
        context_size=48 if smoke else 128,synthetic_queries=24 if smoke else 64,
        synthetic_probe_tasks=3 if smoke else 16,
        pretrain_lr=3e-4,weight_decay=1e-4,
        model=dict(width=64,depth=2,heads=4,dropout=.1),
        tabm=dict(width=128,depth=3,members=16,dropout=.1),
        fine_steps=16 if smoke else 300,evaluation_interval=8 if smoke else 20,
        smoothing_window=2 if smoke else 3,query_batch=64 if smoke else 128,
        tabm_batch=1024,prediction_batch=512,context_panels=2,
        configurations={'pfn':[dict(id='c'+str(i),lr=lr) for i,lr in enumerate([3e-4] if smoke else [1e-4,3e-4])],
                        'tabm':[dict(id='c'+str(i),lr=lr) for i,lr in enumerate([1e-3] if smoke else [1e-3,3e-3])]},
        support_budgets=[0,1,3],feature_names=FEATURE_NAMES,source_sha256=source_hashes(),files={},
        split_policy='Only historical OUTER TRAIN labels are read. Combined training-positive counts stratify 216 drug identities into five groups: 0/1/2 fit, 3 select, 4 assess. Assessment is a new grouped DEVELOPMENT check, not new external evidence. Same split for both RNA types.',
        graph_policy='Reconstruct LncDrug and MiDrug from their own outer TRAIN positives only. For each query drug fold, remove ALL its target AND other-RNA associations. Reverse views are transposes. CircRNA relations are excluded from every arm. Chemical/RNA similarities and LncMi side relations remain available.',
        attributes='RNA/drug raw attributes projected to 16 dimensions each; scalers and PCA fit only on fitting endpoints. Query drug attributes do not fit PCA. Same descriptors for every arm.',
        synthetic_policy='Fixed paired synthetic seeds; hetero and missing share latent complete labels, auxiliary graphs, query partitions and observation-count budgets. Missing changes which positive labels are observed, using drug/RNA-biased exposure plus zero-exposure drugs. Context zeros can be hidden positives. Query supervision uses latent truth. Generic random prior uses random attribute functions and independent auxiliary graphs.',
        model_scope='From-scratch compact pair-token PFN with explicit support-graph descriptors and endpoint-relation attention. Not official GraphPFN/TabPFN, not a large pretrained foundation-model reproduction, and not an end-to-end learned graph encoder.',
        selection='For each task/arm, compare two learning rates on seed 666 using a trailing-three mean of .5 macro AP + .25 (AP+AUC), with step zero competing. Freeze learning rate, select checkpoints of seeds 0/1 on the same dev groups; freeze all paths before assessment. No assessment-driven refit or further search.',
        fewshot_policy='Reserve <=3 and <=half of each held drug positive records, exclude ALL reserves from the common query set at every budget; reveal 0/1/3 own-task positives per drug. Other-RNA held-drug labels remain absent. All arms get recomputed support features; PFN context always includes revealed records. No weight updates on assessed drugs.',
        budget_scope='All four PFN arms have identical architecture and real adaptation budget. Three synthetic priors use identical task counts, optimizer updates, and seeds. Scratch has no synthetic pretraining. TabM gets the same feature columns, support information, selection metric and two-recipe search, with architecture-appropriate batch size/lr.',
        primary_rules=dict(mean_macro_min=.005,task_macro_floor=-.005,mean_auc_floor=-.005,mean_ap_floor=-.005,positive_seeds=2),
        no_historical_validation_loaded=True,no_test_evaluation=True,
        limitations=['Grouped assessment remains development on previously available training records.',
          'Only 4096 synthetic tasks per prior in the formal pilot; a null result cannot rule out larger PFNs.',
          'All initialization seeds share a fixed synthetic task bank and the same real drug split.',
          'Real zero labels are unobserved associations, not verified biological negatives.',
          'Synthetic exposure is an imposed prior, not an identified real-world observation mechanism.',
          'No calibrated biological probabilities or causal effects are claimed.',
          'Cold drug refers to association labels; query chemical attributes and fixed side similarities remain available.',
          'The generic-prior control is this compact architecture, not official pretrained TabPFN.'])
    data,provenance=snapshot_inputs(p['reference_dir'],p['split_seed'])
    np.savez_compressed(out/'inputs.npz',**data)
    p['files']['inputs.npz']=digest(out/'inputs.npz')
    p['input_provenance']=provenance
    for task in TASKS:
        src=Path(provenance[task]['archive'])
        provenance[task]['source_sha256']=digest(src)
    provenance['side_graph']['source_sha256']=digest(provenance['side_graph']['path'])
    write_json(out/'PARTITIONS.json',summary(data))
    p['files']['PARTITIONS.json']=digest(out/'PARTITIONS.json')
    (out/'source_snapshot').mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(Path(__file__).with_name(name),out/'source_snapshot'/name)
    p['sha256']=json_digest(p)
    write_json(out/'protocol.json',p)
    print('FROZEN',out,p['sha256'],flush=True)
    print(json.dumps(summary(data),indent=2),flush=True)


def protocol(out):
    out=Path(out)
    p=json.loads((out/'protocol.json').read_text())
    if p['sha256']!=json_digest({k:v for k,v in p.items() if k!='sha256'}) or p['source_sha256']!=source_hashes():
        raise ValueError('Frozen protocol/source changed; create a new experiment')
    for name,sha in p['files'].items():
        if digest(out/name)!=sha:
            raise ValueError('Frozen input changed: '+name)
    return p


def load_npz(path):
    with np.load(path) as archive:
        return {k:archive[k].copy() for k in archive.files}


def save_manifest(folder,p,files,**metadata):
    value=dict(protocol_sha256=p['sha256'],files={name:digest(Path(folder)/name) for name in files},**metadata)
    write_json(Path(folder)/'manifest.json',value)
    return value


def check_manifest(folder,p):
    value=json.loads((Path(folder)/'manifest.json').read_text())
    if value['protocol_sha256']!=p['sha256']:
        raise ValueError('Wrong artifact protocol')
    for name,sha in value['files'].items():
        if digest(Path(folder)/name)!=sha:
            raise ValueError('Artifact changed: '+str(Path(folder)/name))
    return value


def cached_scene(out,p,task,phase,fold=None,budget=0):
    folder=Path(out)/'features'/task/(('fit_'+str(fold)) if phase=='fit' else phase+'_k'+str(budget))
    if (folder/'manifest.json').exists():
        check_manifest(folder,p)
        return load_npz(folder/'scene.npz')
    if phase=='assess':
        decision(out,'DECISIONS.json',p)
    folder.mkdir(parents=True,exist_ok=True)
    data=load_npz(Path(out)/'inputs.npz')
    s=scenario(data,task,phase,fold,budget,full_candidates=phase=='assess')
    np.savez_compressed(folder/'scene.npz',**s)
    save_manifest(folder,p,['scene.npz'],task=task,phase=phase,fold=fold,budget=budget,
                  context_rows=len(s['cy']),query_rows=len(s['qy']),association_counts=s['association_counts'].tolist())
    return s


def generate(args):
    out,p=Path(args.out_dir),protocol(args.out_dir)
    for prior in ([args.arm] if args.arm else p['priors']):
        if prior not in p['priors']:
            raise ValueError(prior)
        folder=out/'synthetic'/prior
        if (folder/'manifest.json').exists():
            check_manifest(folder,p)
            continue
        folder.mkdir(parents=True,exist_ok=True)
        values,metas={},[]
        started=time.monotonic()
        for i in range(p['synthetic_tasks']):
            ep,meta=synthetic_episode(p['generation_seed']+i,prior,p['context_size'],p['synthetic_queries'])
            for key,value in ep.items():
                values.setdefault(key,[]).append(value)
            metas.append(meta)
            if (i+1)%512==0:
                print('GENERATE',prior,i+1,'seconds',round(time.monotonic()-started,1),flush=True)
        arrays={k:np.stack(v) for k,v in values.items()}
        np.savez(folder/'episodes.npz',**arrays)
        write_json(folder/'episodes.json',metas)
        save_manifest(folder,p,['episodes.npz','episodes.json'],tasks=len(metas),
            query_sha256=hashlib.sha256(arrays['qp'].tobytes()+arrays['qy'].tobytes()).hexdigest(),
            false_context_zeros=sum(m['context_false_zero'] for m in metas),seconds=time.monotonic()-started)
        print('GENERATED',prior,len(metas),flush=True)


def model_for(p,arm,seed,device):
    set_seed(seed)
    return (GraphTabM(FEATURE_DIM,**p['tabm']) if arm=='tabm' else LinkPFN(**p['model'])).to(device)


def state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def synthetic_probes(p):
    result={}
    for name,prior,shift,fp in [('hetero','hetero',False,0.),('missing','missing',False,0.),
                                ('shift','missing',True,0.),('false_positive','missing',False,.05)]:
        episodes=[synthetic_episode(p['probe_seed']+i,prior,p['context_size'],p['synthetic_queries'],shift,fp)[0]
                  for i in range(p['synthetic_probe_tasks'])]
        result[name]={k:np.stack([e[k] for e in episodes]) for k in episodes[0]}
    return result


@torch.no_grad()
def evaluate_probes(model,probes,device):
    model.eval()
    rows,scores=[],{}
    for name,a in probes.items():
        predictions=[]
        for start in range(0,len(a['qy']),4):
            b=tensor_batch(**{k:a[k][start:start+4] for k in ('cx','cy','cp','qx','qp')},device=device)
            predictions.append(model(**b).cpu().numpy())
        score=np.concatenate(predictions)
        scores[name]=score
        auc=[roc_auc_score(y,s) for y,s in zip(a['qy'],score)]
        ap=[average_precision_score(y,s) for y,s in zip(a['qy'],score)]
        rows.append(dict(distribution=name,auc=float(np.mean(auc)),ap=float(np.mean(ap))))
    return rows,scores


def pretrain(args):
    out,p=Path(args.out_dir),protocol(args.out_dir)
    for prior in ([args.arm] if args.arm else p['priors']):
        if prior not in p['priors']:
            raise ValueError(prior)
        check_manifest(out/'synthetic'/prior,p)
        a=load_npz(out/'synthetic'/prior/'episodes.npz')
        probes=synthetic_probes(p)
        for seed in ([args.seed] if args.seed is not None else p['seeds']):
            folder=out/'pretrain'/prior/('s'+str(seed))
            if (folder/'manifest.json').exists():
                check_manifest(folder,p)
                continue
            folder.mkdir(parents=True,exist_ok=True)
            model=model_for(p,prior,seed,args.device)
            opt=torch.optim.AdamW(model.parameters(),lr=p['pretrain_lr'],weight_decay=p['weight_decay'])
            steps=p['synthetic_tasks']//p['synthetic_batch']
            checkpoints=sorted(set([max(1,steps//8),max(1,steps//2),steps]))
            trace,curves=[],[]
            started=time.monotonic()
            for step in range(1,steps+1):
                start=(step-1)*p['synthetic_batch']
                sl=slice(start,start+p['synthetic_batch'])
                batch=tensor_batch(**{k:a[k][sl] for k in ('cx','cy','cp','qx','qp')},device=args.device)
                model.train()
                target=torch.as_tensor(a['qy'][sl],device=args.device)
                opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(model(**batch),target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Synthetic loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                lr=p['pretrain_lr']*(.1+.45*(1+np.cos(np.pi*step/steps)))
                for group in opt.param_groups:
                    group['lr']=lr
                trace.append(dict(step=step,tasks=step*p['synthetic_batch'],loss=float(loss.detach()),lr=lr))
                if step in checkpoints:
                    rows,scores=evaluate_probes(model,probes,args.device)
                    curves.extend(dict(step=step,tasks=step*p['synthetic_batch'],**r) for r in rows)
                    torch.save(state(model),folder/('step_'+str(step)+'.pt'))
                    np.savez_compressed(folder/('probe_'+str(step)+'.npz'),**scores)
                    print('PRETRAIN',prior,seed,step,'loss',round(trace[-1]['loss'],5),'seconds',round(time.monotonic()-started,1),flush=True)
            torch.save(state(model),folder/'checkpoint.pt')
            pd.DataFrame(trace).to_csv(folder/'trace.csv',index=False)
            pd.DataFrame(curves).to_csv(folder/'curves.csv',index=False)
            files=['checkpoint.pt','trace.csv','curves.csv']+[stem+str(step)+suffix for step in checkpoints for stem,suffix in [('step_','.pt'),('probe_','.npz')]]
            save_manifest(folder,p,files,prior=prior,seed=seed,tasks=p['synthetic_tasks'],steps=steps,
                params=sum(v.numel() for v in model.parameters()),seconds=time.monotonic()-started,
                synthetic_manifest_sha256=digest(out/'synthetic'/prior/'manifest.json'))
            del model


def feature_command(args):
    out,p=Path(args.out_dir),protocol(args.out_dir)
    for task in ([args.task] if args.task else TASKS):
        for fold in (0,1,2):
            cached_scene(out,p,task,'fit',fold)
        cached_scene(out,p,task,'dev')
        print('FEATURES_READY',task,flush=True)


@torch.no_grad()
def predict(model,s,p,arm,device,scaler=None,full=False,intervention=None):
    model.eval()
    qx,qp=(s['all_x'],s['all_p']) if full else (s['qx'],s['qp'])
    if arm=='tabm':
        x=np.clip((qx-scaler['mean'])/scaler['scale'],-8,8)
        result=[]
        for start in range(0,len(x),p['prediction_batch']):
            logits=model(torch.as_tensor(x[start:start+p['prediction_batch']],dtype=torch.float32,device=device))
            probability=torch.sigmoid(logits).mean(1).clamp(1e-6,1-1e-6)
            result.append(torch.logit(probability).cpu().numpy())
        return np.concatenate(result)
    panels=[]
    for panel in range(p['context_panels']):
        ids=balanced_indices(s['cy'],p['context_size'],np.random.RandomState(27181+panel),s.get('priority'))
        results=[]
        for start in range(0,len(qx),p['prediction_batch']):
            batch=tensor_batch(s['cx'][ids],s['cy'][ids],s['cp'][ids],
                qx[start:start+p['prediction_batch']],qp[start:start+p['prediction_batch']],device)
            results.append(model(**batch,intervention=intervention)[0].cpu().numpy())
        panels.append(np.concatenate(results))
    return np.mean(panels,axis=0)


def configurations(p,arm):
    return p['configurations']['tabm' if arm=='tabm' else 'pfn']


def result_folder(out,stage,task,arm,cfg,seed):
    return Path(out)/stage/task/arm/cfg/('s'+str(seed))


def check_result(folder,p):
    check_manifest(folder,p)
    result=json.loads((Path(folder)/'result.json').read_text())
    if result['protocol_sha256']!=p['sha256']:
        raise ValueError('Result protocol mismatch')
    return result


def fit_model(out,p,task,arm,cfg,seed,device,stage):
    folder=result_folder(out,stage,task,arm,cfg['id'],seed)
    if (folder/'manifest.json').exists():
        return check_result(folder,p)
    folder.mkdir(parents=True,exist_ok=True)
    folds=[cached_scene(out,p,task,'fit',f) for f in (0,1,2)]
    dev=cached_scene(out,p,task,'dev')
    model=model_for(p,arm,seed,device)
    prior_hash=None
    if arm in PRIORS:
        pre=out/'pretrain'/arm/('s'+str(seed))
        check_manifest(pre,p)
        model.load_state_dict(torch.load(pre/'checkpoint.pt',map_location=device,weights_only=False))
        prior_hash=digest(pre/'checkpoint.pt')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=p['weight_decay'])
    rng=np.random.RandomState(seed+91973)
    scaler=None
    if arm=='tabm':
        raw=np.concatenate([s['qx'] for s in folds])
        yy=np.concatenate([s['qy'] for s in folds])
        scaler=dict(mean=raw.mean(0).astype(np.float32),scale=np.maximum(raw.std(0),.1).astype(np.float32))
        xx=torch.as_tensor(np.clip((raw-scaler['mean'])/scaler['scale'],-8,8),device=device)
        ty=torch.as_tensor(yy,device=device)
    started=time.monotonic()
    initial=predict(model,dev,p,arm,device,scaler)
    m,_=metrics(dev['qy'],initial,dev['qp'])
    trace=[dict(step=0,loss=None,**m)]
    selected,selection=0,m['criterion']
    selected_state,selected_scores=state(model),initial.copy()
    for step in range(1,p['fine_steps']+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        if arm=='tabm':
            idx=torch.as_tensor(rng.choice(len(yy),min(p['tabm_batch'],len(yy)),replace=False),device=device)
            logits=model(xx[idx])
            loss=F.binary_cross_entropy_with_logits(logits,ty[idx,None].expand_as(logits))
        else:
            s=folds[(step-1)%3]
            c=balanced_indices(s['cy'],p['context_size'],rng)
            q=rng.choice(len(s['qy']),min(p['query_batch'],len(s['qy'])),replace=False)
            batch=tensor_batch(s['cx'][c],s['cy'][c],s['cp'][c],s['qx'][q],s['qp'][q],device)
            loss=F.binary_cross_entropy_with_logits(model(**batch)[0],torch.as_tensor(s['qy'][q],device=device))
        if not torch.isfinite(loss):
            raise FloatingPointError('Real adaptation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        opt.step()
        if step%p['evaluation_interval']==0 or step==p['fine_steps']:
            score=predict(model,dev,p,arm,device,scaler)
            m,_=metrics(dev['qy'],score,dev['qp'])
            trace.append(dict(step=step,loss=float(loss.detach()),**m))
            chosen,value=choose_checkpoint(trace,p['smoothing_window'])
            if chosen!=selected:
                if chosen!=step:
                    raise AssertionError('Checkpoint rule drift')
                selected,selection=chosen,value
                selected_state,selected_scores=state(model),score.copy()
        if step==1 or step%100==0 or step==p['fine_steps']:
            print('ADAPT',task,arm,cfg['id'],seed,step,'loss',round(float(loss.detach()),5),'seconds',round(time.monotonic()-started,1),flush=True)
    model.load_state_dict(selected_state)
    replay=predict(model,dev,p,arm,device,scaler)
    np.testing.assert_allclose(replay,selected_scores,atol=3e-5,rtol=3e-5)
    m,drug_rows=metrics(dev['qy'],replay,dev['qp'])
    chosen,value=choose_checkpoint(trace,p['smoothing_window'])
    if chosen!=selected or abs(value-selection)>1e-10:
        raise AssertionError('Selected checkpoint inconsistent')
    result=dict(task=task,arm=arm,config=cfg,seed=seed,stage=stage,selected_step=selected,
        trained_steps=p['fine_steps'],selection_score=selection,protocol_sha256=p['sha256'],
        pretrained_checkpoint_sha256=prior_hash,params=sum(v.numel() for v in model.parameters()),
        elapsed_seconds=time.monotonic()-started,**m)
    write_json(folder/'result.json',result)
    torch.save(selected_state,folder/'checkpoint.pt')
    np.savez_compressed(folder/'dev_scores.npz',scores=replay,y=dev['qy'],pairs=dev['qp'])
    pd.DataFrame(trace).to_csv(folder/'trace.csv',index=False)
    pd.DataFrame(drug_rows).to_csv(folder/'per_drug.csv',index=False)
    files=['result.json','checkpoint.pt','dev_scores.npz','trace.csv','per_drug.csv']
    if scaler is not None:
        np.savez_compressed(folder/'scaler.npz',**scaler)
        files.append('scaler.npz')
    save_manifest(folder,p,files,stage=stage,task=task,arm=arm,seed=seed)
    print('ADAPTED',task,arm,seed,'selected',selected,'dev macro AP',round(m['macro_ap'],6),flush=True)
    return result


def decision(out,name,p):
    d=json.loads((Path(out)/name).read_text())
    if d['protocol_sha256']!=p['sha256'] or d['sha256']!=json_digest({k:v for k,v in d.items() if k!='sha256'}):
        raise ValueError('Decision changed')
    for name,sha in d['files'].items():
        if digest(Path(out)/name)!=sha:
            raise ValueError('Decision input changed')
    return d


def tuning_jobs(out,p,stage):
    chosen=decision(out,'PARAMETERS.json',p)['chosen'] if stage=='confirm' else None
    for task in TASKS:
        for arm in ARMS:
            configs=[chosen[task][arm]] if stage=='confirm' else configurations(p,arm)
            seeds=p['seeds'][1:] if stage=='confirm' else [p['seeds'][0]]
            for cfg in configs:
                for seed in seeds:
                    yield task,arm,cfg,seed


def tune(args):
    out,p=Path(args.out_dir),protocol(args.out_dir)
    for task,arm,cfg,seed in tuning_jobs(out,p,args.stage):
        if (args.task and args.task!=task) or (args.arm and args.arm!=arm) or (args.seed is not None and args.seed!=seed):
            continue
        fit_model(out,p,task,arm,cfg,seed,args.device,args.stage)


def select(args):
    out,p=Path(args.out_dir),protocol(args.out_dir)
    name='PARAMETERS.json' if args.stage=='screen' else 'DECISIONS.json'
    if (out/name).exists():
        print('DECISION_EXISTS',name,decision(out,name,p)['sha256'],flush=True)
        return
    value=dict(protocol_sha256=p['sha256'],files={},chosen={})
    if args.stage=='screen':
        for task in TASKS:
            value['chosen'][task]={}
            for arm in ARMS:
                rows=[]
                for cfg in configurations(p,arm):
                    folder=result_folder(out,'screen',task,arm,cfg['id'],p['seeds'][0])
                    r=check_result(folder,p)
                    rows.append(r)
                    rel=str((folder/'result.json').relative_to(out))
                    value['files'][rel]=digest(folder/'result.json')
                chosen=sorted(rows,key=lambda r:(-r['selection_score'],r['config']['id']))[0]
                value['chosen'][task][arm]=chosen['config']
    else:
        parameters=decision(out,'PARAMETERS.json',p)['chosen']
        for task in TASKS:
            value['chosen'][task]={}
            for arm in ARMS:
                cfg=parameters[task][arm]
                value['chosen'][task][arm]={}
                for i,seed in enumerate(p['seeds']):
                    folder=result_folder(out,'screen' if i==0 else 'confirm',task,arm,cfg['id'],seed)
                    r=check_result(folder,p)
                    rel=str((folder/'result.json').relative_to(out))
                    value['files'][rel]=digest(folder/'result.json')
                    value['chosen'][task][arm][str(seed)]=dict(folder=str(folder.relative_to(out)),
                        selected_step=r['selected_step'],checkpoint_sha256=digest(folder/'checkpoint.pt'))
    value['sha256']=json_digest(value)
    write_json(out/name,value)
    print('FROZEN_DECISION',name,value['sha256'],flush=True)


def recall_full(s,scores,k=20):
    values=[]
    for drug in np.unique(s['qp'][s['qy']==1,1]):
        true=s['qp'][(s['qp'][:,1]==drug)&(s['qy']==1),0]
        idx=np.flatnonzero(s['all_p'][:,1]==drug)
        top=idx[np.argsort(-scores[idx],kind='stable')[:k]]
        values.append(float(np.isin(true,s['all_p'][top,0]).mean()))
    return float(np.mean(values))


def evaluate(args):
    out,p=Path(args.out_dir),protocol(args.out_dir)
    chosen=decision(out,'DECISIONS.json',p)['chosen']
    for task in ([args.task] if args.task else TASKS):
        scenes={k:cached_scene(out,p,task,'assess',budget=k) for k in p['support_budgets']}
        for arm in ([args.arm] if args.arm else ARMS):
            for seed in ([args.seed] if args.seed is not None else p['seeds']):
                checkpoint=chosen[task][arm][str(seed)]
                fit=out/checkpoint['folder']
                check_result(fit,p)
                if digest(fit/'checkpoint.pt')!=checkpoint['checkpoint_sha256']:
                    raise ValueError('Chosen checkpoint changed')
                model=model_for(p,arm,seed,args.device)
                model.load_state_dict(torch.load(fit/'checkpoint.pt',map_location=args.device,weights_only=False))
                scaler=load_npz(fit/'scaler.npz') if arm=='tabm' else None
                if arm in PRIORS:
                    zero_folder=out/'zero_shot'/task/arm/('s'+str(seed))
                    if (zero_folder/'manifest.json').exists():
                        check_result(zero_folder,p)
                    else:
                        zero_folder.mkdir(parents=True,exist_ok=True)
                        pretrained=out/'pretrain'/arm/('s'+str(seed))
                        frozen=model_for(p,arm,seed,args.device)
                        frozen.load_state_dict(torch.load(pretrained/'checkpoint.pt',map_location=args.device,weights_only=False))
                        zz=predict(frozen,scenes[0],p,arm,args.device)
                        zm,_=metrics(scenes[0]['qy'],zz,scenes[0]['qp'])
                        write_json(zero_folder/'result.json',dict(protocol_sha256=p['sha256'],task=task,arm=arm,seed=seed,
                            pretrained_checkpoint_sha256=digest(pretrained/'checkpoint.pt'),**zm))
                        np.savez_compressed(zero_folder/'scores.npz',scores=zz,y=scenes[0]['qy'],pairs=scenes[0]['qp'])
                        save_manifest(zero_folder,p,['result.json','scores.npz'])
                        del frozen
                for budget,s in scenes.items():
                    folder=out/'assessment'/task/arm/('s'+str(seed))/('k'+str(budget))
                    if (folder/'manifest.json').exists():
                        check_result(folder,p)
                        continue
                    folder.mkdir(parents=True,exist_ok=True)
                    score=predict(model,s,p,arm,args.device,scaler)
                    all_score=predict(model,s,p,arm,args.device,scaler,full=True)
                    m,rows=metrics(s['qy'],score,s['qp'])
                    m['full_recall20']=recall_full(s,all_score)
                    interventions={}
                    saved={}
                    if arm=='missing' and budget==0:
                        for name in ('hide_context_labels','shuffle_context_labels','no_relational_bias'):
                            ss=predict(model,s,p,arm,args.device,intervention=name)
                            interventions[name]=metrics(s['qy'],ss,s['qp'])[0]
                            saved[name]=ss
                    result=dict(protocol_sha256=p['sha256'],task=task,arm=arm,seed=seed,budget=budget,
                        checkpoint=checkpoint,query_rows=len(s['qy']),query_positives=int(s['qy'].sum()),
                        revealed_supports=len(s['priority']),interventions=interventions,**m)
                    write_json(folder/'result.json',result)
                    np.savez_compressed(folder/'scores.npz',scores=score,y=s['qy'],pairs=s['qp'],
                        full_scores=all_score,full_pairs=s['all_p'],**saved)
                    pd.DataFrame(rows).to_csv(folder/'per_drug.csv',index=False)
                    save_manifest(folder,p,['result.json','scores.npz','per_drug.csv'])
                    print('ASSESS',task,arm,seed,budget,'AP',round(m['ap'],6),'macro',round(m['macro_ap'],6),flush=True)
                del model


def report(args):
    out,p=Path(args.out_dir),protocol(args.out_dir)
    decision(out,'DECISIONS.json',p)
    rows=[]
    for task in TASKS:
        for arm in ARMS:
            for seed in p['seeds']:
                for budget in p['support_budgets']:
                    folder=out/'assessment'/task/arm/('s'+str(seed))/('k'+str(budget))
                    r=check_result(folder,p)
                    rows.append({k:v for k,v in r.items() if k not in ('checkpoint','interventions')})
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'assessment.csv',index=False)
    names=['auc','ap','macro_ap','macro_auc','macro_recall10','macro_ndcg10','full_recall20']
    means=frame.groupby(['task','budget','arm'],sort=False)[names].agg(['mean','std'])
    means.columns=['_'.join(c) for c in means.columns]
    means.reset_index().to_csv(out/'summary.csv',index=False)
    contrasts=[]
    for control in ('scratch','random','hetero','tabm'):
        for task in TASKS:
            a=frame[(frame.task==task)&(frame.arm=='missing')&(frame.budget==0)].set_index('seed')
            b=frame[(frame.task==task)&(frame.arm==control)&(frame.budget==0)].set_index('seed')
            for seed in p['seeds']:
                contrasts.append(dict(task=task,control=control,seed=seed,**{n:float(a.loc[seed,n]-b.loc[seed,n]) for n in names}))
    delta=pd.DataFrame(contrasts)
    delta.to_csv(out/'paired_deltas.csv',index=False)
    screens=[]
    rule=p['primary_rules']
    for control in ('scratch','random','hetero','tabm'):
        a=delta[delta.control==control]
        by_seed=a.groupby('seed')[names].mean()
        task_macro=a.groupby('task').macro_ap.mean()
        minimum=0. if control in ('random','hetero') else rule['mean_macro_min']
        passed=bool(a.macro_ap.mean()>minimum and task_macro.min()>=rule['task_macro_floor'] and
                    a.auc.mean()>=rule['mean_auc_floor'] and a.ap.mean()>=rule['mean_ap_floor'] and
                    (by_seed.macro_ap>0).sum()>=min(rule['positive_seeds'],len(p['seeds'])))
        screens.append(dict(control=control,passed=passed,delta_macro_ap=float(a.macro_ap.mean()),
            worst_task_delta_macro_ap=float(task_macro.min()),delta_auc=float(a.auc.mean()),delta_ap=float(a.ap.mean()),
            positive_seeds=int((by_seed.macro_ap>0).sum())))
    pd.DataFrame(screens).to_csv(out/'screens.csv',index=False)
    curves=[]
    for prior in PRIORS:
        for seed in p['seeds']:
            folder=out/'pretrain'/prior/('s'+str(seed))
            check_manifest(folder,p)
            df=pd.read_csv(folder/'curves.csv')
            df['prior'],df['seed']=prior,seed
            curves.append(df)
    pd.concat(curves,ignore_index=True).to_csv(out/'synthetic_curves.csv',index=False)
    zero_rows=[check_result(out/'zero_shot'/task/arm/('s'+str(seed)),p)
               for task in TASKS for arm in PRIORS for seed in p['seeds']]
    pd.DataFrame(zero_rows).to_csv(out/'zero_shot.csv',index=False)
    txt=['# 缺失机制链路 PFN：紧凑原型验证','',
        '**短流程检查，不能用于性能结论。**' if p['smoke'] else
        ('**主候选通过预设初筛，仍需独立数据验证。**' if all(v['passed'] for v in screens) else '**当前主候选未通过全部预设筛选。**'),'',
        '仅使用历史外层训练部分，按药物分组为拟合、选参和评估；两种 RNA 的关联标签同步隔离。以下是新分组的开发验证，不是原随机配对验证，也不是独立外部测试。','',
        '| 任务 | 支持数上限 | 模型 | AUC | AP | 药物宏 AP | 全候选 Recall@20 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for task in TASKS:
        for budget in p['support_budgets']:
            for arm in ARMS:
                g=frame[(frame.task==task)&(frame.budget==budget)&(frame.arm==arm)]
                txt.append('| %s | %d | %s | %.6f | %.6f | %.6f | %.6f |'%(task,budget,arm,g.auc.mean(),g.ap.mean(),g.macro_ap.mean(),g.full_recall20.mean()))
    txt+=['','## 零支持主筛选','','| 相对对照 | Δ 宏 AP（两任务均值） | 最弱任务 Δ 宏 AP | 正向种子 | 通过 |','|---|---:|---:|---:|---|']
    for s in screens:
        txt.append('| %s | %+.6f | %+.6f | %d/%d | %s |'%(s['control'],s['delta_macro_ap'],s['worst_task_delta_macro_ap'],s['positive_seeds'],len(p['seeds']),'是' if s['passed'] else '否'))
    txt+=['','## 不更新真实任务权重的上下文迁移','',
          '下表只载入合成预训练权重，但上下文仍包含拟合药物的真实已观测标签；评估药物保持零关联支持。它不是完全不使用真实标签的推理。',
          '','| 任务 | 先验 | AUC | AP | 药物宏 AP |','|---|---|---:|---:|---:|']
    zz=pd.DataFrame(zero_rows)
    for (task,arm),g in zz.groupby(['task','arm'],sort=False):
        txt.append('| %s | %s | %.6f | %.6f | %.6f |'%(task,arm,g.auc.mean(),g.ap.mean(),g.macro_ap.mean()))
    txt+=['','## 解释边界','',
        '- `random/hetero/missing` 使用同一配对上下文架构、相同合成任务数和真实适配预算；`scratch` 为无合成预训练对照。`tabm` 使用完全相同的 59 维配对输入及支持图。',
        '- 合成查询监督来自完整潜在标签；支持上下文来自可观测关联，零标签可含漏记正例。真实阶段沿用基准的未标记零标签，没有识别真实生物学阴性。',
        '- 模型在固定异构结构描述符上做配对上下文推理；不是官方 GraphPFN/TabPFN，也没有复制百万级合成任务预训练。',
        '- 每种药物的全部保留支持行在所有预算下均排除于共同查询集；支持预算为上限，有些药物没有足够记录。1/3 支持是少样本评估，只有 0 支持属于本协议的关联冷药物。',
        '- 同一初始化跨任务使用同一个合成预训练检查点；三种初始化共享固定合成任务库及真实划分，不能替代分布与数据源重复。',
        '- 宏 AP 只覆盖两类标签都存在的评估药物，数量见 PARTITIONS.json。全候选召回只衡量已知留出正例恢复，不能视作实验命中率。',
        '- 标签隐藏/置换干预固定图特征，只诊断显式上下文标签通道；不代表删除所有支持证据或因果解释。',
        '- 历史验证和测试数组没有载入；旧图中的其他任务全量关联与 circRNA 关联均未用于本轮。',
        '',f'正式配置每个先验使用 {p["synthetic_tasks"]} 个合成任务；模型宽度 64、两层，真实适配最多 {p["fine_steps"]} 步。预训练曲线见 synthetic_curves.csv，逐种子评估见 assessment.csv。','',
        '复现与方法说明见 [LINK_PFN.md](../../LINK_PFN.md)。','']
    (out/'REPORT.md').write_text('\n'.join(txt))
    print('REPORT',out,'passed',all(v['passed'] for v in screens),flush=True)


def validate(args):
    out,p=Path(args.out_dir),protocol(args.out_dir)
    chosen=decision(out,'DECISIONS.json',p)['chosen']
    data=load_npz(out/'inputs.npz')
    features_checked,runs,replays,zero_replays=0,0,[],[]
    tasks=[args.task] if args.task else TASKS
    for task in tasks:
        for phase,fold,budget in [('fit',f,0) for f in (0,1,2)]+[('dev',None,0)]+[('assess',None,k) for k in p['support_budgets']]:
            saved=cached_scene(out,p,task,phase,fold,budget)
            fresh=scenario(data,task,phase,fold,budget,full_candidates=phase=='assess')
            for name in saved:
                np.testing.assert_allclose(saved[name],fresh[name],atol=2e-6,rtol=2e-6)
            features_checked+=1
        for stage in ('screen','confirm'):
            for t,arm,cfg,seed in tuning_jobs(out,p,stage):
                if t!=task:
                    continue
                folder=result_folder(out,stage,task,arm,cfg['id'],seed)
                result=check_result(folder,p)
                step,value=choose_checkpoint(pd.read_csv(folder/'trace.csv').to_dict('records'),p['smoothing_window'])
                assert step==result['selected_step'] and np.isclose(value,result['selection_score'],atol=1e-10)
                model=model_for(p,arm,seed,args.device)
                model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=args.device,weights_only=False))
                scaler=load_npz(folder/'scaler.npz') if arm=='tabm' else None
                s=cached_scene(out,p,task,'dev')
                score=predict(model,s,p,arm,args.device,scaler)
                a=load_npz(folder/'dev_scores.npz')
                np.testing.assert_allclose(score,a['scores'],atol=3e-5,rtol=3e-5)
                for k,v in metrics(s['qy'],score,s['qp'])[0].items():
                    assert np.isclose(v,result[k],atol=2e-5)
                runs+=1
                del model
        for arm in ARMS:
            for seed in p['seeds']:
                spec=chosen[task][arm][str(seed)]
                fit=out/spec['folder']
                assert digest(fit/'checkpoint.pt')==spec['checkpoint_sha256']
                model=model_for(p,arm,seed,args.device)
                model.load_state_dict(torch.load(fit/'checkpoint.pt',map_location=args.device,weights_only=False))
                scaler=load_npz(fit/'scaler.npz') if arm=='tabm' else None
                if arm in PRIORS:
                    frozen=model_for(p,arm,seed,args.device)
                    pre=out/'pretrain'/arm/('s'+str(seed))
                    frozen.load_state_dict(torch.load(pre/'checkpoint.pt',map_location=args.device,weights_only=False))
                    s=cached_scene(out,p,task,'assess',budget=0)
                    zero=out/'zero_shot'/task/arm/('s'+str(seed))
                    check_result(zero,p)
                    zz=predict(frozen,s,p,arm,args.device)
                    saved_z=load_npz(zero/'scores.npz')
                    np.testing.assert_allclose(zz,saved_z['scores'],atol=3e-5,rtol=3e-5)
                    zero_replays.append(dict(task=task,arm=arm,seed=seed))
                    del frozen
                for budget in p['support_budgets']:
                    s=cached_scene(out,p,task,'assess',budget=budget)
                    folder=out/'assessment'/task/arm/('s'+str(seed))/('k'+str(budget))
                    result=check_result(folder,p)
                    saved=load_npz(folder/'scores.npz')
                    score=predict(model,s,p,arm,args.device,scaler)
                    full=predict(model,s,p,arm,args.device,scaler,full=True)
                    np.testing.assert_allclose(score,saved['scores'],atol=3e-5,rtol=3e-5)
                    np.testing.assert_allclose(full,saved['full_scores'],atol=3e-5,rtol=3e-5)
                    m=metrics(s['qy'],score,s['qp'])[0]
                    m['full_recall20']=recall_full(s,full)
                    for key,value in m.items():
                        assert np.isclose(value,result[key],atol=2e-5),key
                    for intervention in result['interventions']:
                        pred=predict(model,s,p,arm,args.device,intervention=intervention)
                        np.testing.assert_allclose(pred,saved[intervention],atol=3e-5,rtol=3e-5)
                    replays.append(dict(task=task,arm=arm,seed=seed,budget=budget,max_abs=float(np.max(np.abs(score-saved['scores'])))))
                del model
        print('VALIDATED',task,flush=True)
    # Synthetic artifact integrity and paired latent-query equality.
    synth={prior:check_manifest(out/'synthetic'/prior,p) for prior in PRIORS}
    assert synth['hetero']['query_sha256']==synth['missing']['query_sha256']
    synthetic_replays=[]
    probes=synthetic_probes(p)
    for prior in PRIORS:
        for seed in p['seeds']:
            folder=out/'pretrain'/prior/('s'+str(seed))
            check_manifest(folder,p)
            curves=pd.read_csv(folder/'curves.csv')
            model=model_for(p,prior,seed,args.device)
            for step in sorted(curves.step.unique()):
                model.load_state_dict(torch.load(folder/('step_'+str(step)+'.pt'),map_location=args.device,weights_only=False))
                rows,score=evaluate_probes(model,probes,args.device)
                saved=load_npz(folder/('probe_'+str(step)+'.npz'))
                for row in rows:
                    name=row['distribution']
                    np.testing.assert_allclose(score[name],saved[name],atol=3e-5,rtol=3e-5)
                    stored=curves[(curves.step==step)&(curves.distribution==name)].iloc[0]
                    assert np.isclose(row['auc'],stored.auc,atol=2e-5) and np.isclose(row['ap'],stored.ap,atol=2e-5)
                synthetic_replays.append(dict(prior=prior,seed=seed,step=int(step)))
            del model
    value=dict(passed=True,protocol_sha256=p['sha256'],smoke=p['smoke'],features_recomputed=features_checked,
        adaptation_checkpoints_replayed=runs,assessment_replays=replays,synthetic_pairing_passed=True,
        zero_shot_replays=zero_replays,synthetic_checkpoint_replays=synthetic_replays,
        no_historical_validation_loaded=True,no_test_evaluation=True)
    write_json(out/('VALIDATION_'+args.task+'.json' if args.task else 'VALIDATION.json'),value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','generate','pretrain','features','tune','select','evaluate','report','validate'):
        q=sub.add_parser(command)
        q.add_argument('--out-dir',required=True)
        q.add_argument('--device',default='cuda:1')
        q.add_argument('--task',choices=TASKS)
        q.add_argument('--arm',choices=ARMS)
        q.add_argument('--seed',type=int)
        if command=='prepare':
            q.add_argument('--reference-dir',default='improve/results/backbone_replacement_20260910')
            q.add_argument('--smoke',action='store_true')
        if command in ('tune','select'):
            q.add_argument('--stage',choices=('screen','confirm'),required=True)
    args=parser.parse_args()
    dispatch=dict(prepare=prepare,generate=generate,pretrain=pretrain,features=feature_command,
                  tune=tune,select=select,evaluate=evaluate,report=report,validate=validate)
    dispatch[args.command](args)


if __name__=='__main__':
    main()
