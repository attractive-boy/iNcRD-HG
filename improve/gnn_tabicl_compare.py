"""Compare evidence additions in structural HGT, a matched MLP and TabICLv2.

Only the historical inner development partition is accessible. Training
checkpoints and settings are frozen before any new prediction is evaluated.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import gc
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
from sklearn.preprocessing import QuantileTransformer

from improve import tabular_foundation as base


DEFAULT_OUT='improve/results/gnn_tabicl_compare_20260913_logits'
REFERENCE='improve/results/tabicl_pathway_context_20260913'
EXPANDED='improve/results/tabicl_functional_20260913'
GNN_CONFIG='improve/results/ragt_backbone_tuned_20260911'
VIEWS=('graph36','tiered62','expanded62')
ARCHITECTURES=('mlp','sahgt')
RNA_TYPES={'MiDrug':'miRNA','LncDrug':'lncRNA'}
OWN=('gnn_tabicl_compare.py','gnn_tabicl_compare_modules.py','test_gnn_tabicl_compare.py','test_ragt_backbone.py',
     'ragt_modules.py','pair_exchange_modules.py','backbone_replacement.py',
     'network_feature_ablation.py','graph_tabm_modules.py','tabular_foundation.py',
     'sampling_bias_audit.py','tabicl_context_program.py','tabicl_context_program_modules.py',
     'tabicl_functional.py','tabicl_functional_modules.py','tabicl_graph_context.py','tabicl_pathway.py')


def sources():
    return {name:base.digest(Path(__file__).with_name(name)) for name in OWN}


def normalized_extra(train,query):
    transform=QuantileTransformer(n_quantiles=min(256,len(train)),output_distribution='normal',
        subsample=100000,random_state=20260910)
    x=transform.fit_transform(train).astype(np.float32)
    q=transform.transform(query).astype(np.float32)
    return x,q,transform.quantiles_,transform.references_


def feature_conditions(a,old_train,old_val,new_train,new_val):
    """Zero padding fixes network capacity; every normalizer uses support only."""
    n,m=len(a['y_train']),len(a['y_val'])
    if a['x_train'].shape!=(n,36) or a['x_val'].shape!=(m,36):
        raise ValueError('Expected exactly 36 frozen graph columns')
    if any(x.shape!=(size,26) for x,size in ((old_train,n),(old_val,m),(new_train,n),(new_val,m))):
        raise ValueError('Expected exactly 26 aligned evidence columns')
    result=dict(graph36_train=np.concatenate([a['x_train'],np.zeros((n,26),np.float32)],axis=1),
                graph36_val=np.concatenate([a['x_val'],np.zeros((m,26),np.float32)],axis=1))
    for name,tr,va in (('tiered62',old_train,old_val),('expanded62',new_train,new_val)):
        x,q,quantiles,refs=normalized_extra(tr,va)
        result[name+'_train']=np.concatenate([a['x_train'],x],axis=1)
        result[name+'_val']=np.concatenate([a['x_val'],q],axis=1)
        result[name+'_raw_extra_train']=tr
        result[name+'_raw_extra_val']=va
        result[name+'_quantiles']=quantiles
        result[name+'_references']=refs
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite comparison inputs')
    return result


def copy_reference_scores(a,task,gp,rp,fp):
    from improve import tabicl_graph_context as graph
    from improve import tabicl_pathway as operations
    from improve import tabicl_functional as functional
    scores={};artifacts={}
    tp=base.read_json(Path(gp['parent_dir'])/'protocol.json')
    for seed in (666,0,1):
        dest=Path(gp['parent_dir'])/gp['baseline']['inner'][task][str(seed)]
        base.checked_result(dest,tp)
        scores['graph36_s'+str(seed)]=graph.baseline_score(gp,'inner',task,seed,a)
        for path in (dest/'scores.npz',dest/'result.json'):
            artifacts[str(path.resolve())]=base.digest(path)
        for view,root,protocol,variant in (('tiered62',REFERENCE,rp,'tiered'),('expanded62',EXPANDED,fp,'expanded')):
            dest=operations.directory(root,'inner',task,variant,seed,'full')
            result=operations.checked_result(dest,protocol)
            if result['context_rows']!=len(a['y_train']) or result['input_columns']!=62:
                raise ValueError('Reference did not use the same complete support and feature budget')
            with np.load(dest/'scores.npz',allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['y_val'],a['y_val'])
                np.testing.assert_array_equal(archive['pairs_val'],a['pairs_val'])
                scores[view+'_s'+str(seed)]=archive['views'][0].copy()
            for path in (dest/'scores.npz',dest/'result.json'):
                artifacts[str(path.resolve())]=base.digest(path)
        accepted_score,_=functional.reference_scores(fp,task,'inner',seed,a)
        scores['accepted_s'+str(seed)]=accepted_score
        path=Path(REFERENCE)/f'predictions/inner/{task}/s{seed}/scores.npz'
        artifacts[str(path.resolve())]=base.digest(path)
    return scores,artifacts


def prepare(args):
    from improve import tabicl_context_program as accepted
    from improve import tabicl_context_program_modules as old_features
    from improve import tabicl_functional as functional
    from improve import tabicl_functional_modules as new_features
    from improve import tabicl_graph_context as graph
    out=Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new comparison directory')
    rp=accepted.protocol(REFERENCE);fp=functional.protocol(EXPANDED);gp=graph.protocol(rp['parent_dir'])
    config_root=Path(GNN_CONFIG).resolve()
    history=base.check_seal(base.read_json(config_root/'protocol.json'))
    parameters=base.check_seal(base.read_json(config_root/'PARAMETERS.json'))
    refit=base.check_seal(base.read_json(config_root/'REFIT.json'))
    for decision in (parameters,refit):
        if decision['protocol_sha256']!=history['sha256']:
            raise ValueError('Historical GNN decision protocol mismatch')
        for path,digest in decision['results_sha256'].items():
            if base.digest(config_root/path)!=digest:
                raise ValueError('Historical inner-selected GNN setting changed')
    graph_root=Path(gp['graph_reference_dir'])
    graph_protocol=base.check_seal(base.read_json(graph_root/'protocol.json'))
    old=old_features.ContextPrograms.load(rp['inputs_dir'])
    new=new_features.FunctionalPrograms.load(fp['inputs_dir'],fp['drug_inputs_dir'],Path(EXPANDED)/'drug_gene_rewired.npz')
    out.mkdir(parents=True);files={};references={};configs={};checkpoints={}
    for task in ('MiDrug','LncDrug'):
        cfg=parameters['chosen'][task]['sahgt']
        if cfg!=refit['chosen'][task]['sahgt']['config']:
            raise ValueError('Historical GNN configuration drift')
        configs[task]=cfg
        checkpoints[task]={'historical':2,'long':3} if args.smoke else dict(
            historical=refit['chosen'][task]['sahgt']['epochs'],long=history['epochs'][task])
        a=graph.features(rp['parent_dir'],gp,task,'inner')
        extras=[old.transform(task,a['pairs_train'],'tiered'),old.transform(task,a['pairs_val'],'tiered'),
                new.transform(task,a['pairs_train'],'expanded'),new.transform(task,a['pairs_val'],'expanded')]
        values=feature_conditions(a,*extras)
        values.update({key:a[key] for key in ('y_train','y_val','pairs_train','pairs_val','fold_ids','raw_train','raw_val','x_train','x_val')})
        path=out/'inputs'/task;path.mkdir(parents=True)
        np.savez_compressed(path/'features.npz',**values)
        scores,origin=copy_reference_scores(a,task,gp,rp,fp);references.update(origin)
        np.savez_compressed(path/'tabicl_reference.npz',**scores,y_val=a['y_val'],pairs_val=a['pairs_val'])
        for name in ('features.npz','tabicl_reference.npz'):
            files[str((path/name).relative_to(out))]=base.digest(path/name)
        for name in ('inner.npz','train_graph.pth'):
            path=graph_root/'inputs'/task/name
            expected=graph_protocol['files'][str(path.relative_to(graph_root))]
            if base.digest(path)!=expected:
                raise ValueError('Historical GNN input changed')
            references[str(path.resolve())]=expected
    for root,names in ((config_root,('protocol.json','PARAMETERS.json','REFIT.json')),
                       (Path(REFERENCE),('protocol.json','SELECTION.json','CANDIDATE.json','VALIDATION_inner.json')),
                       (Path(EXPANDED),('protocol.json','VALIDATION_inner.json')),
                       (graph_root,('protocol.json',)),
                       (Path(rp['inputs_dir']),('INPUTS.json',)),(Path(fp['drug_inputs_dir']),('INPUTS.json',))):
        for name in names:
            references[str((root/name).resolve())]=base.digest(root/name)
    p=base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),smoke=args.smoke,
        tasks=['MiDrug','LncDrug'],seeds=[666] if args.smoke else [666,0,1],
        architectures=list(ARCHITECTURES),views=list(VIEWS),configs=configs,checkpoints=checkpoints,
        files=files,references=references,source_sha256=sources(),
        graph_reference_dir=str(graph_root.resolve()),graph_reference_protocol_sha256=graph_protocol['sha256'],
        graph_source_sha256=graph_protocol['source_sha256'],config_reference_dir=str(config_root),
        reference_dir=str(Path(REFERENCE).resolve()),functional_dir=str(Path(EXPANDED).resolve()),
        query_fraction=.5,phase='inner',allowed_phases=['inner'],parameter_input_columns=62,
        input_policy='graph36 uses the same frozen 36 cross-fitted graph columns and 26 zero columns. tiered62/expanded62 append the identical raw evidence columns used by the corresponding TabICL full-context models. MLP/HGT share support-only quantile preprocessing; TabICL keeps its frozen preprocessing.',
        graph_policy='Reuse the original PCA/RNA, drug and circRNA node inputs and all 13 relations. Restrict target and reverse edges to training positives, then remove all sampled query positives in both directions at every update. New evidence enters the pair decoder, not new graph nodes.',
        training='Reuse the latest historical inner-selected SA-HGT settings. Same width, pair-head initialization, 62-column parameter shape, optimizer, training queries and epochs across input conditions. Matched MLP uses the identical pair head and optimizer. Evaluate only two preregistered epochs; no new checkpoint or hyperparameter selection.',
        primary='Historical epoch checkpoint: paired within-model changes graph36 -> tiered62 and tiered62 -> expanded62. Longer fixed training budget is a sensitivity analysis; report both without selecting a winner across checkpoints.',
        comparator='Use verified unblended full-support TabICL models (32 pretrained members), not the accepted mixture, as the main learner comparator. Report the accepted graph/tiered/context recipe separately as a deployment reference.',
        score_policy='Evaluate raw logits for the trainable models to avoid rank collapse from saturated float32 sigmoid. Frozen TabICL predictions are probabilities. Only rank-based AUC/AP and macro/weighted AP are compared; no probability calibration comparison.',
        selection='No candidate promotion, no alpha search, and no outer/test access in this comparison.',
        replay_atol=5e-5,replay_rtol=5e-5,query_invariance_atol=5e-5,query_invariance_rtol=5e-5,
        no_test_evaluation=True,
        limitations=['Historical inner data and the previous GNN settings have been used in development; this is not independent generalization evidence.',
            'Training seeds for GNN/MLP and inference seeds for TabICL have different meanings and are not independent data splits.',
            'HGT has raw graph/node access; TabICL/MLP use compressed graph pair features. Added evidence is identical, but this is a pipeline comparison, not a pure architecture isolation.',
            'Frozen pretrained TabICL has 32 members; GNN/MLP are trained from scratch. Pretraining and total compute are not matched.',
            'Results apply to the specified structural HGT adaptation, not all GNN architectures or direct gene-node message passing.']))
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name),out/'source_snapshot'/name)
    base.write_json(out/'protocol.json',p)
    print('COMPARISON_FROZEN',out,p['sha256'],'checkpoints',checkpoints,flush=True)


def protocol(out):
    out=Path(out);p=base.check_seal(base.read_json(out/'protocol.json'))
    if p['source_sha256']!=sources():
        raise ValueError('Frozen comparison implementation changed')
    for name,expected in p['graph_source_sha256'].items():
        if base.digest(Path(__file__).with_name(name))!=expected:
            raise ValueError('Frozen graph loader changed')
    for name,expected in p['source_sha256'].items():
        if base.digest(out/'source_snapshot'/name)!=expected:
            raise ValueError('Comparison source snapshot changed: '+name)
    for name,expected in p['files'].items():
        if base.digest(out/name)!=expected:
            raise ValueError('Frozen comparison input changed: '+name)
    for name,expected in p['references'].items():
        if base.digest(name)!=expected:
            raise ValueError('Comparison reference changed: '+name)
    return p


def load_features(out,p,task,phase='inner'):
    if phase!='inner' or task not in p['tasks']:
        raise ValueError('This comparison permits inner data only; outer/test access prohibited')
    with np.load(Path(out)/'inputs'/task/'features.npz',allow_pickle=False) as archive:
        return {name:archive[name].copy() for name in archive.files}


def load_graph(p,task,a,device):
    import torch
    from improve import backbone_replacement as graph
    if graph.protocol(p['graph_reference_dir'])['sha256']!=p['graph_reference_protocol_sha256']:
        raise ValueError('Graph input protocol changed')
    original,data,weights=graph.stage_inputs(p['graph_reference_dir'],task,'inner')
    for name in ('pairs_train','pairs_val','y_train','y_val'):
        np.testing.assert_array_equal(original[name],a[name])
    if any(not bool(torch.all(w==1)) for w in weights.values()):
        raise ValueError('Expected the frozen unit-weight relation graph')
    return ({nt:x.to(device) for nt,x in data.x_dict.items()},
            {key:value.to(device) for key,value in data.edge_index_dict.items()},
            {key:value.to(device) for key,value in weights.items()})


def checked_checkpoint(folder,p,expected):
    result=base.read_json(Path(folder)/'result.json')
    if result['protocol_sha256']!=p['sha256'] or not result['replay_passed']:
        raise ValueError('Unverified comparison checkpoint')
    if any(result.get(key)!=value for key,value in expected.items()):
        raise ValueError('Comparison checkpoint identity mismatch')
    for name,digest in result['artifacts'].items():
        if base.digest(Path(folder)/name)!=digest:
            raise ValueError('Saved comparison artifact changed')
    return result


def run(args):
    import pandas as pd
    import torch
    import torch_geometric
    from improve import gnn_tabicl_compare_modules as models
    out=Path(args.out_dir);p=protocol(out)
    if args.task not in p['tasks']:
        raise ValueError('Choose a registered task')
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    a=load_features(out,p,args.task);x_dict,edges,weights=load_graph(p,args.task,a,args.device)
    cfg=p['configs'][args.task];rna=RNA_TYPES[args.task]
    y=torch.from_numpy(a['y_train']).float().to(args.device)
    pairs=torch.from_numpy(a['pairs_train']).to(args.device)
    query_pairs=torch.from_numpy(a['pairs_val']).to(args.device)
    phases=p['checkpoints'][args.task];maximum=max(phases.values())
    architectures=[args.architecture] if args.architecture else p['architectures']
    views=[args.view] if args.view else p['views']
    seeds=[args.seed] if args.seed is not None else p['seeds']
    if any(arch not in p['architectures'] for arch in architectures) or any(v not in p['views'] for v in views) or any(s not in p['seeds'] for s in seeds):
        raise ValueError('Unregistered comparison job')
    for architecture in architectures:
        for view in views:
            x=torch.from_numpy(a[view+'_train']).to(args.device)
            q=torch.from_numpy(a[view+'_val']).to(args.device)
            for seed in seeds:
                dest=out/'fits'/args.task/architecture/view/('s'+str(seed));dest.mkdir(parents=True,exist_ok=True)
                with (dest/'.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if (dest/'COMPLETE.json').exists():
                        for phase,epoch in phases.items():
                            checked_checkpoint(dest/phase,p,dict(task=args.task,architecture=architecture,
                                view=view,seed=seed,budget=phase,epoch=epoch))
                        continue
                    model=models.make_model(architecture,cfg,x_dict,edges,rna,seed,args.device)
                    initial=models.state_digest(model)
                    initial_head=models.state_digest(model.feature_mlp)
                    query_digest=hashlib.sha256()
                    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
                    trace=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats(args.device)
                    for epoch in range(1,maximum+1):
                        idx,visible=models.training_context(a['pairs_train'],a['y_train'],edges,weights,
                            args.task,rna,x_dict['drug'].shape[0],p['query_fraction'],seed,epoch)
                        query_digest.update(idx.astype('<i8').tobytes())
                        index=torch.as_tensor(idx,device=args.device)
                        model.train();optimizer.zero_grad(set_to_none=True)
                        logits=model(x_dict,visible,pairs[:,index],x[index])
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[index])
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('Nonfinite comparison loss')
                        loss.backward()
                        if not all(bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.grad is not None):
                            raise ValueError('Nonfinite comparison gradient')
                        optimizer.step()
                        trace.append(dict(epoch=epoch,loss=float(loss.detach()),queries=len(idx)))
                        for phase,chosen_epoch in phases.items():
                            if epoch!=chosen_epoch:
                                continue
                            folder=dest/phase;folder.mkdir(exist_ok=True)
                            score=models.predict(model,x_dict,edges,query_pairs,q)
                            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                            torch.save(state,folder/'checkpoint.pt')
                            np.savez_compressed(folder/'scores.npz',score=score,y_val=a['y_val'],pairs_val=a['pairs_val'])
                            # Construct/reload under forked RNG: checkpoint verification must not
                            # change subsequent training dropout or CUDA random state.
                            with torch.random.fork_rng(devices=[torch.device(args.device).index]):
                                replay_model=models.make_model(architecture,cfg,x_dict,edges,rna,seed,args.device)
                                replay_model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=args.device,weights_only=True),strict=True)
                                replay=models.predict(replay_model,x_dict,edges,query_pairs,q)
                                np.testing.assert_allclose(replay,score,atol=p['replay_atol'],rtol=p['replay_rtol'])
                                checks={}
                                if p['smoke']:
                                    reverse=models.predict(replay_model,x_dict,edges,query_pairs.flip(1),q.flip(0))[::-1]
                                    parts=np.array_split(np.arange(len(q)),3)
                                    split=np.concatenate([models.predict(replay_model,x_dict,edges,query_pairs[:,part],q[part]) for part in parts])
                                    for name,prediction in (('reverse',reverse),('split',split)):
                                        np.testing.assert_allclose(prediction,score,atol=p['query_invariance_atol'],rtol=p['query_invariance_rtol'])
                                        checks[name]=float(np.max(np.abs(prediction-score)))
                                contribution=models.graph_contribution(replay_model,x_dict,edges,query_pairs,q)
                                del replay_model
                            result=dict(protocol_sha256=p['sha256'],task=args.task,architecture=architecture,view=view,
                                seed=seed,budget=phase,epoch=epoch,parameter_count=sum(v.numel() for v in model.parameters()),
                                initial_state_sha256=initial,initial_head_sha256=initial_head,
                                training_queries_sha256=query_digest.hexdigest(),checkpoint_selection_used_query_labels=False,
                                replay_passed=True,replay_max_abs=float(np.max(np.abs(replay-score))),smoke_checks=checks,
                                elapsed_seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device),
                                environment=dict(torch=torch.__version__,torch_geometric=torch_geometric.__version__,numpy=np.__version__),
                                **contribution,artifacts={name:base.digest(folder/name) for name in ('checkpoint.pt','scores.npz')})
                            base.write_json(folder/'result.json',result)
                            print('CHECKPOINT',args.task,architecture,view,seed,phase,epoch,'replay',result['replay_max_abs'],flush=True)
                        if epoch==1 or epoch%50==0 or epoch==maximum:
                            print('TRAIN',args.task,architecture,view,seed,epoch,'loss',round(trace[-1]['loss'],6),'seconds',round(time.monotonic()-started,1),flush=True)
                    pd.DataFrame(trace).to_csv(dest/'training_trace.csv',index=False)
                    base.write_json(dest/'COMPLETE.json',dict(protocol_sha256=p['sha256'],epochs=maximum,
                        trace_sha256=base.digest(dest/'training_trace.csv'),checkpoints={phase:base.digest(dest/phase/'result.json') for phase in phases}))
                    del model,optimizer;gc.collect();torch.cuda.empty_cache()


def report(args):
    import pandas as pd
    from improve.sampling_bias_audit import matched_weights,weighted_metrics
    out=Path(args.out_dir);p=protocol(out)
    rows=[];diagnostics=[];artifacts={};identities={};heads={};schedules={};params={};jobs=0
    for task in p['tasks']:
        a=load_features(out,p,task)
        matched,_=matched_weights(a['y_val'],a['pairs_val'][1])
        for architecture in p['architectures']:
            for view in p['views']:
                for seed in p['seeds']:
                    dest=out/'fits'/task/architecture/view/('s'+str(seed))
                    completion=base.read_json(dest/'COMPLETE.json')
                    if completion['protocol_sha256']!=p['sha256'] or base.digest(dest/'training_trace.csv')!=completion['trace_sha256']:
                        raise ValueError('Incomplete or changed training run')
                    for budget,epoch in p['checkpoints'][task].items():
                        folder=dest/budget
                        result=checked_checkpoint(folder,p,dict(task=task,architecture=architecture,
                            view=view,seed=seed,budget=budget,epoch=epoch));jobs+=1
                        if base.digest(folder/'result.json')!=completion['checkpoints'][budget]:
                            raise ValueError('Training checkpoint changed after completion')
                        key=(task,architecture,seed)
                        identities.setdefault(key,set()).add(result['initial_state_sha256'])
                        heads.setdefault((task,seed),set()).add(result['initial_head_sha256'])
                        schedules.setdefault((task,seed,budget),set()).add(result['training_queries_sha256'])
                        params.setdefault((task,architecture),set()).add(result['parameter_count'])
                        with np.load(folder/'scores.npz',allow_pickle=False) as archive:
                            np.testing.assert_array_equal(archive['y_val'],a['y_val'])
                            np.testing.assert_array_equal(archive['pairs_val'],a['pairs_val'])
                            score=archive['score'].copy()
                        if p['smoke'] and set(result['smoke_checks'])!={'reverse','split'}:
                            raise ValueError('Missing smoke query invariance checks')
                        rows.append(dict(task=task,learner=architecture,view=view,seed=seed,budget=budget,
                            epoch=result['epoch'],parameter_count=result['parameter_count'],
                            graph_gate=result['graph_gate'],graph_to_pair_logit_std_ratio=result['graph_to_pair_logit_std_ratio'],
                            **base.metrics(a['y_val'],score,a['pairs_val'])))
                        for name,weight in (('drug_matched',matched),('same_subset_unweighted',(matched>0).astype(float))):
                            diagnostics.append(dict(task=task,learner=architecture,view=view,seed=seed,budget=budget,
                                evaluation=name,**weighted_metrics(a['y_val'],score,weight)))
                        artifacts[str((folder/'result.json').relative_to(out))]=base.digest(folder/'result.json')
        if p['smoke']:
            continue
        with np.load(out/'inputs'/task/'tabicl_reference.npz',allow_pickle=False) as archive:
            for view in (*p['views'],'accepted'):
                for seed in p['seeds']:
                    score=archive[view+'_s'+str(seed)]
                    learner='tabicl_accepted' if view=='accepted' else 'tabicl_full'
                    rows.append(dict(task=task,learner=learner,view=view,seed=seed,budget='pretrained',epoch=0,
                                     parameter_count=None,graph_gate=None,graph_to_pair_logit_std_ratio=None,
                                     **base.metrics(a['y_val'],score,a['pairs_val'])))
                    for name,weight in (('drug_matched',matched),('same_subset_unweighted',(matched>0).astype(float))):
                        diagnostics.append(dict(task=task,learner=learner,view=view,seed=seed,budget='pretrained',evaluation=name,
                            **weighted_metrics(a['y_val'],score,weight)))
    if any(len(values)!=1 for values in identities.values()) or any(len(values)!=1 for values in params.values()):
        raise ValueError('Input conditions did not share initialization and model capacity')
    if any(len(values)!=1 for values in heads.values()) or any(len(values)!=1 for values in schedules.values()):
        raise ValueError('Learners did not share pair-head initialization and training query schedule')
    if p['smoke']:
        base.write_json(out/'SMOKE_VALIDATION.json',dict(protocol_sha256=p['sha256'],passed=True,checkpoints=jobs,
            same_initialization=True,same_capacity=True,same_pair_head=True,same_training_queries=True,
            query_invariance=True,performance_evidence=False))
        print('SMOKE_VERIFIED',jobs,flush=True)
        return
    table=pd.DataFrame(rows);diag=pd.DataFrame(diagnostics)
    table.to_csv(out/'metrics.csv',index=False);diag.to_csv(out/'diagnostics.csv',index=False)
    metrics=list(base.METRICS)
    summary=table.groupby(['task','learner','budget','view'],sort=False)[metrics].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    differences=[]
    for (task,learner,budget),group in table[table.learner!='tabicl_accepted'].groupby(['task','learner','budget'],sort=False):
        for before,after in (('graph36','tiered62'),('tiered62','expanded62')):
            left=group[group.view==before].set_index('seed');right=group[group.view==after].set_index('seed')
            if set(left.index)!=set(p['seeds']) or set(right.index)!=set(p['seeds']):
                raise ValueError('Incomplete paired input comparison')
            for metric in metrics:
                delta=right.loc[p['seeds'],metric]-left.loc[p['seeds'],metric]
                differences.append(dict(task=task,learner=learner,budget=budget,before=before,after=after,metric=metric,
                    mean_delta=float(delta.mean()),positive_seeds=int((delta>0).sum()),nonnegative_seeds=int((delta>=0).sum()),
                    minimum_delta=float(delta.min()),maximum_delta=float(delta.max())))
    differences=pd.DataFrame(differences);differences.to_csv(out/'paired_differences.csv',index=False)
    lines=['**同一批新增证据在结构增强 HGT、MLP 和 TabICLv2 中的对照。**','',
        '仅评价历史内部开发集；配置和两个检查点事先冻结，没有本轮验证选轮次或选超参数。',
        '输入条件：graph36；tiered62（旧证据）；expanded62（新增靶点证据）。GNN/MLP 的输入层始终为 62 列，graph36 的后 26 列置零；同任务、同架构、同种子在三档输入下参数量和完整初始化一致。',
        'HGT 保留原始节点特征和关系图，新证据进入配对解码器。MLP 使用完全相同的配对分支。每步从支持集中抽样查询并双向移除查询正边，结构配对特征仍使用已冻结的五折交叉拟合。',
        'TabICL 主对照是 32 成员、全支持上下文的单一原始预测器；已接受的图／证据混合模型单独列为参考。各模型新增证据相同，但原始图访问、预训练和总算力不同，不能将结果归因于架构本身。','',
        'GNN/MLP 使用原始 logit 计算排序指标，避免高置信输出经过 float32 sigmoid 后饱和并丢失排序；TabICL 沿用已验证概率。本轮不比较概率校准。','',
        '|任务|模型|预算|输入|AUC|AP|药物宏 AP|','|---|---|---|---|---:|---:|---:|']
    for _,r in summary.iterrows():
        lines.append(f'|{r.task}|{r.learner}|{r.budget}|{r.view}|{r.auc_mean:.6f}|{r.ap_mean:.6f}|{r.macro_drug_ap_mean:.6f}|')
    lines+=['','**配对变化（均值；推理／初始化种子不代表独立划分）。**','',
        '|任务|模型|预算|信息变化|ΔAUC|正种子|ΔAP|正种子|','|---|---|---|---|---:|---:|---:|---:|']
    for key,group in differences.groupby(['task','learner','budget','before','after'],sort=False):
        task,learner,budget,before,after=key
        auc=group[group.metric=='auc'].iloc[0];ap=group[group.metric=='ap'].iloc[0]
        lines.append(f'|{task}|{learner}|{budget}|{before} → {after}|{auc.mean_delta:+.6f}|{auc.positive_seeds}/3|{ap.mean_delta:+.6f}|{ap.positive_seeds}/3|')
    lines+=['',
        'historical 检查点：LncDrug 24 步，MiDrug 74 步；long 为 150／450 步。两者分别报告，不择优混用。每一步是相同抽样比例的支持集查询更新。',
        '药物匹配诊断见 diagnostics.csv；保持保留子集正例比例，同时给出同子集未加权对照，不能直接与全局 AP 比较。',
        '完整模型状态、恢复预测、训练损失、图分支门控和相对输出幅度均归档。更高的 AUC 与更高的 AP 不一定同时出现，不能只选有利指标。',
        '没有替换当前已接受模型，没有读取外层或原始测试集。该实验只针对指定 SA-HGT 的配对证据融合，不覆盖直接增加 gene/pathway 节点或其他 GNN 结构。',
        '历史内层已经参与既往配置选择；这些是开发对照结果，不能声称独立泛化或统计显著性。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for name in ('metrics.csv','diagnostics.csv','summary.csv','paired_differences.csv','REPORT.md'):
        artifacts[name]=base.digest(out/name)
    base.write_json(out/'VALIDATION.json',dict(protocol_sha256=p['sha256'],passed=True,checkpoints=jobs,
        same_initialization=True,same_capacity=True,same_pair_head=True,same_training_queries=True,
        no_new_validation_selection=True,no_outer_arrays=True,
        no_original_test_evaluation=True,baseline_preserved=True,artifacts=artifacts))
    print('COMPARISON_REPORTED',out,'verified_checkpoints',jobs,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--out-dir',default=DEFAULT_OUT)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--task',choices=('MiDrug','LncDrug'))
    parser.add_argument('--architecture',choices=ARCHITECTURES)
    parser.add_argument('--view',choices=VIEWS)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();dict(prepare=prepare,run=run,report=report)[args.command](args)


if __name__=='__main__':
    main()
