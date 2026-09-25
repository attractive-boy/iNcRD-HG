"""Add CatBoost and randomly initialized TabM to the frozen HGT comparison.

Only predictor pretraining is excluded. All learners retain the same existing
side information, including the shared ChemBERTa-derived drug inputs.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import gc
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from improve import gnn_tabicl_compare as previous
from improve import tabular_foundation as base


DEFAULT_OUT='improve/results/nonpretrained_tabular_compare_20260913'
PARENT='improve/results/gnn_tabicl_compare_20260913_logits'
HISTORY='improve/results/graph_tabm_20260910'
DEPENDENCIES='/data1/licheng/incrd-hg/.external/nonpretrained_tabular_20260913'
OWN=tuple(dict.fromkeys((*previous.OWN,'nonpretrained_tabular_compare.py',
    'nonpretrained_tabular_modules.py','test_nonpretrained_tabular.py','test_graph_tabm.py')))


def sources():
    return {name:base.digest(Path(__file__).with_name(name)) for name in OWN}


def verify_dependencies(root):
    root=Path(root);p=base.check_seal(base.read_json(root/'DEPENDENCIES.json'))
    for name,digest in p['files'].items():
        if base.digest(root/name)!=digest:
            raise ValueError('Isolated dependency changed: '+name)
    return p


def prepare(args):
    out=Path(args.out_dir).resolve();parent=Path(PARENT).resolve();history=Path(HISTORY).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    pp=previous.protocol(parent)
    validation=base.check_seal(base.read_json(parent/'ROUND_VALIDATION.json'))
    if not validation['passed'] or validation['protocol_sha256']!=pp['sha256']:
        raise ValueError('Need the verified full HGT comparison')
    for name,digest in validation['artifacts'].items():
        if base.digest(parent/name)!=digest:
            raise ValueError('Parent comparison changed')
    hp=base.check_seal(base.read_json(history/'protocol.json'))
    parameters=base.check_seal(base.read_json(history/'PARAMETERS.json'))
    refit=base.check_seal(base.read_json(history/'REFIT.json'))
    references={}
    for decision in (parameters,refit):
        if decision['protocol_sha256']!=hp['sha256']:
            raise ValueError('Historical TabM selection mismatch')
        for name,digest in decision['results_sha256'].items():
            if 'outer' in Path(name).parts or base.digest(history/name)!=digest:
                raise ValueError('Invalid historical inner decision reference')
            references[str(history/name)]=digest
    for name,expected in hp['source_sha256'].items():
        if base.digest(Path(__file__).with_name(name))!=expected:
            raise ValueError('Historical TabM implementation changed')
    dp=verify_dependencies(DEPENDENCIES)
    out.mkdir(parents=True);files={};configs={};checkpoints={}
    for task in pp['tasks']:
        cfg=parameters['chosen'][task]['tabm']
        if cfg!=refit['chosen'][task]['tabm']['config']:
            raise ValueError('TabM configuration drift')
        configs[task]=cfg
        checkpoints[task]=dict(matched_hgt_steps=1,historical=2,long=3) if args.smoke else dict(
            matched_hgt_steps=pp['checkpoints'][task]['historical'],
            historical=refit['chosen'][task]['tabm']['epochs'],long=hp['epochs'][task])
        path=out/'inputs'/task;path.mkdir(parents=True)
        source=parent/'inputs'/task/'features.npz'
        if base.digest(source)!=pp['files'][str(source.relative_to(parent))]:
            raise ValueError('Shared feature cache changed')
        shutil.copy2(source,path/'features.npz')
        files[str((path/'features.npz').relative_to(out))]=base.digest(path/'features.npz')
        references[str(source)]=base.digest(source)
    for name in ('metrics.csv','diagnostics.csv'):
        table=pd.read_csv(parent/name)
        table=table[table.learner.isin(['mlp','sahgt'])].copy()
        table.to_csv(out/('reference_'+name),index=False)
        files['reference_'+name]=base.digest(out/('reference_'+name))
    for root,names in ((parent,('protocol.json','ROUND_VALIDATION.json','VALIDATION.json','metrics.csv','diagnostics.csv')),
                       (history,('protocol.json','PARAMETERS.json','REFIT.json')),
                       (Path(DEPENDENCIES),('DEPENDENCIES.json',))):
        for name in names:
            references[str((root/name).resolve())]=base.digest(root/name)
    tree_checkpoints=dict(trees300=4,trees1000=8) if args.smoke else dict(trees300=300,trees1000=1000)
    tree_config=dict(iterations=max(tree_checkpoints.values()),learning_rate=.03,depth=6,
        l2_leaf_reg=3.,loss_function='Logloss',bootstrap_type='Bayesian',bagging_temperature=1.,
        random_strength=1.,border_count=128,boosting_type='Plain',task_type='CPU',thread_count=4,
        allow_writing_files=False,use_best_model=False,verbose=False)
    p=base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),smoke=args.smoke,
        tasks=pp['tasks'],seeds=[666] if args.smoke else pp['seeds'],views=pp['views'],
        learners=['tabm','catboost'],tabm_configs=configs,tabm_checkpoints=checkpoints,
        tree_config=tree_config,tree_checkpoints=tree_checkpoints,batch_size=hp['batch_size'],
        files=files,references=references,source_sha256=sources(),
        parent_dir=str(parent),parent_protocol_sha256=pp['sha256'],dependency_root=DEPENDENCIES,
        dependency_dir=str(Path(DEPENDENCIES)/'packages'),dependency_sha256=dp['sha256'],
        phase='inner',no_outer_arrays=True,no_original_test_evaluation=True,pretrained_predictor=False,
        feature_policy='Identical fixed pairs, labels and three evidence conditions as the HGT/MLP comparison. TabM uses the same support-fitted quantile inputs; CatBoost uses the corresponding raw 36 graph columns and raw 26 evidence columns. graph36 has 26 zero columns; every input shape is 62. Existing drug embeddings remain common side information.',
        tabm_policy='Reuse the historical 128-wide, three-block, 16-member TabM implementation and inner-selected task learning rates. Every update accumulates full-support independent-member BCE over batches of 1024, as in the historical TabM experiment. All views share initialization, row order, optimizer and checkpoints. No new early stopping or validation selection.',
        tree_policy='One prespecified CPU CatBoost recipe from an unfitted model, without init_model or eval_set. Save the 300-tree prefix and full 1000-tree model; no validation-selected tree count or search.',
        primary_budgets=dict(sahgt='historical',mlp='historical',tabm='historical',catboost='trees300'),
        primary='Compare each learner at its prespecified primary budget and report all other checkpoints separately. TabM matched_hgt_steps matches optimizer-update counts only; HGT samples half the support while TabM uses all support and 16 members, so compute is not matched.',
        score_policy='Use stable log odds of the mean of TabM member probabilities, preserving the historical aggregation without saturated sigmoid ties. CatBoost, HGT and MLP use raw binary logits. Compare rank metrics only.',
        replay_atol=5e-5,replay_rtol=5e-5,
        limitations=['Predictors are trained from scratch, but common upstream drug features still derive from ChemBERTa.',
            'Historical inner development data have been reused; seeds are not independent data partitions.',
            'Learning schedules, total computation and model capacities differ by learner; this is a fixed pipeline comparison.',
            'The project TabM implementation is a compact adaptation, not an official benchmark reproduction.',
            'TabM settings come from historical inner selection, while CatBoost uses a fixed standard recipe without a new search.',
            'No promotion or change of the accepted model follows automatically from this comparison.']))
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name),out/'source_snapshot'/name)
    base.write_json(out/'protocol.json',p)
    print('NONPRETRAINED_FROZEN',out,p['sha256'],flush=True)


def protocol(out):
    out=Path(out);p=base.check_seal(base.read_json(out/'protocol.json'))
    if p['source_sha256']!=sources():
        raise ValueError('Frozen implementation changed')
    for name,digest in p['source_sha256'].items():
        if base.digest(out/'source_snapshot'/name)!=digest:
            raise ValueError('Source snapshot changed')
    for name,digest in p['files'].items():
        if base.digest(out/name)!=digest:
            raise ValueError('Input changed: '+name)
    for name,digest in p['references'].items():
        if base.digest(name)!=digest:
            raise ValueError('Reference changed: '+name)
    if verify_dependencies(p['dependency_root'])['sha256']!=p['dependency_sha256']:
        raise ValueError('Dependency manifest changed')
    return p


def load_inputs(out,p,task,phase='inner'):
    return previous.load_features(out,p,task,phase)


def checkpoints(p,learner,task):
    return p['tabm_checkpoints'][task] if learner=='tabm' else p['tree_checkpoints']


def checked(folder,p,identity):
    result=base.read_json(Path(folder)/'result.json')
    if result['protocol_sha256']!=p['sha256'] or not result['replay_passed']:
        raise ValueError('Unverified predictor checkpoint')
    if any(result.get(k)!=v for k,v in identity.items()):
        raise ValueError('Checkpoint identity mismatch')
    for name,digest in result['artifacts'].items():
        if base.digest(Path(folder)/name)!=digest:
            raise ValueError('Checkpoint artifact changed')
    return result


def save_result(folder,p,identity,a,score,replay,state_file,checks,details):
    np.testing.assert_allclose(replay,score,atol=p['replay_atol'],rtol=p['replay_rtol'])
    np.savez_compressed(folder/'scores.npz',score=score,y_val=a['y_val'],pairs_val=a['pairs_val'])
    result=dict(identity,protocol_sha256=p['sha256'],replay_passed=True,
        replay_max_abs=float(np.max(np.abs(replay-score))),smoke_checks=checks,
        pretrained_predictor=False,query_labels_used_in_fit=False,query_labels_used_for_selection=False,
        **details,artifacts={name:base.digest(folder/name) for name in (state_file,'scores.npz')})
    base.write_json(folder/'result.json',result)
    print('CHECKPOINT',identity,'replay',result['replay_max_abs'],flush=True)


def invariance(p,predict,x,score):
    if not p['smoke']:
        return {}
    reverse=predict(x.flip(0) if hasattr(x,'flip') else x[::-1])[::-1]
    split=np.concatenate([predict(x[part]) for part in np.array_split(np.arange(len(x)),3)])
    result={}
    for name,value in (('reverse',reverse),('split',split)):
        np.testing.assert_allclose(value,score,atol=p['replay_atol'],rtol=p['replay_rtol'])
        result[name]=float(np.max(np.abs(value-score)))
    return result


def run(args):
    import torch
    from improve import nonpretrained_tabular_modules as models
    out=Path(args.out_dir);p=protocol(out)
    if args.task not in p['tasks'] or args.learner not in p['learners']:
        raise ValueError('Choose a registered task and learner')
    torch.set_num_threads(4)
    if args.learner=='tabm':
        torch.cuda.set_device(args.device)
    a=load_inputs(out,p,args.task)
    budgets=checkpoints(p,args.learner,args.task)
    for view in p['views']:
        for seed in p['seeds']:
            dest=out/'fits'/args.task/args.learner/view/('s'+str(seed));dest.mkdir(parents=True,exist_ok=True)
            identity=dict(task=args.task,learner=args.learner,view=view,seed=seed)
            with (dest/'.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if (dest/'COMPLETE.json').exists():
                    for budget,steps in budgets.items():
                        checked(dest/budget,p,dict(identity,budget=budget,iterations=steps))
                    continue
                started=time.monotonic();trace=[]
                if args.learner=='tabm':
                    cfg=p['tabm_configs'][args.task]
                    model=models.make_tabm(cfg,seed,args.device)
                    initial=models.state_digest(model)
                    x=torch.from_numpy(a[view+'_train']).to(args.device)
                    q=torch.from_numpy(a[view+'_val']).to(args.device)
                    y=torch.from_numpy(a['y_train']).float().to(args.device)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
                    for step in range(1,max(budgets.values())+1):
                        loss=models.train_tabm_epoch(model,optimizer,x,y,p['batch_size'])
                        trace.append(dict(step=step,training_loss=loss))
                        for budget,chosen in budgets.items():
                            if step!=chosen:
                                continue
                            folder=dest/budget;folder.mkdir(exist_ok=True)
                            score=models.predict_tabm(model,q,p['batch_size'])
                            torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},folder/'model.pt')
                            with torch.random.fork_rng(devices=[torch.device(args.device).index]):
                                restored=models.make_tabm(cfg,seed,args.device)
                                restored.load_state_dict(torch.load(folder/'model.pt',map_location=args.device,weights_only=True),strict=True)
                                predict=lambda value:models.predict_tabm(restored,value,p['batch_size'])
                                replay=predict(q);checks=invariance(p,predict,q,score)
                                del restored
                            save_result(folder,p,dict(identity,budget=budget,iterations=step),a,score,replay,'model.pt',checks,
                                dict(initial_state_sha256=initial,parameter_count=sum(v.numel() for v in model.parameters()),
                                    training_rows_per_update=len(y),elapsed_seconds=time.monotonic()-started,
                                    torch_version=torch.__version__,unit='optimizer_updates'))
                        if step==1 or step%100==0 or step==max(budgets.values()):
                            print('TRAIN',args.task,'tabm',view,seed,step,round(loss,6),round(time.monotonic()-started,1),flush=True)
                    del optimizer,model,x,q,y;gc.collect();torch.cuda.empty_cache()
                else:
                    x,q=models.tree_inputs(a,view)
                    model=models.fresh_catboost(p['tree_config'],seed,p['dependency_dir'])
                    models.fit_catboost(model,x,a['y_train'])
                    if model.tree_count_!=max(budgets.values()):
                        raise ValueError('Unexpected CatBoost training budget')
                    trace=[dict(step=i+1,training_loss=float(loss)) for i,loss in enumerate(model.get_evals_result()['learn']['Logloss'])]
                    for budget,steps in budgets.items():
                        folder=dest/budget;folder.mkdir(exist_ok=True)
                        prefix=model.copy();prefix.shrink(ntree_end=steps)
                        score=models.predict_catboost(prefix,q)
                        prefix.save_model(str(folder/'model.cbm'))
                        restored=models.catboost_class(p['dependency_dir'])()
                        restored.load_model(str(folder/'model.cbm'))
                        if restored.tree_count_!=steps:
                            raise ValueError('Wrong restored tree count')
                        predict=lambda value:models.predict_catboost(restored,value)
                        replay=predict(q);checks=invariance(p,predict,q,score)
                        save_result(folder,p,dict(identity,budget=budget,iterations=steps),a,score,replay,'model.cbm',checks,
                            dict(initial_state_sha256=None,parameter_count=None,started_unfitted=True,
                                training_rows=len(x),catboost_version='1.2.8',fitted_parameters=prefix.get_all_params(),
                                elapsed_seconds=time.monotonic()-started,unit='trees'))
                        del prefix,restored
                    del model;gc.collect()
                pd.DataFrame(trace).to_csv(dest/'training_trace.csv',index=False)
                base.write_json(dest/'COMPLETE.json',dict(protocol_sha256=p['sha256'],
                    trace_sha256=base.digest(dest/'training_trace.csv'),
                    checkpoints={name:base.digest(dest/name/'result.json') for name in budgets}))


def report(args):
    from improve.sampling_bias_audit import matched_weights,weighted_metrics
    out=Path(args.out_dir);p=protocol(out)
    rows=[];diagnostics=[];artifacts={};initializations={};capacities={};count=0
    for task in p['tasks']:
        a=load_inputs(out,p,task);weights,_=matched_weights(a['y_val'],a['pairs_val'][1])
        for learner in p['learners']:
            for view in p['views']:
                for seed in p['seeds']:
                    dest=out/'fits'/task/learner/view/('s'+str(seed))
                    complete=base.read_json(dest/'COMPLETE.json')
                    if complete['protocol_sha256']!=p['sha256'] or base.digest(dest/'training_trace.csv')!=complete['trace_sha256']:
                        raise ValueError('Changed or incomplete training trace')
                    for budget,steps in checkpoints(p,learner,task).items():
                        identity=dict(task=task,learner=learner,view=view,seed=seed,budget=budget,iterations=steps)
                        folder=dest/budget;r=checked(folder,p,identity);count+=1
                        if base.digest(folder/'result.json')!=complete['checkpoints'][budget]:
                            raise ValueError('Changed completed checkpoint')
                        if p['smoke'] and set(r['smoke_checks'])!={'reverse','split'}:
                            raise ValueError('Missing query invariance check')
                        if learner=='tabm':
                            initializations.setdefault((task,seed),set()).add(r['initial_state_sha256'])
                            capacities.setdefault(task,set()).add(r['parameter_count'])
                        elif not r['started_unfitted']:
                            raise ValueError('CatBoost did not start unfitted')
                        with np.load(folder/'scores.npz',allow_pickle=False) as archive:
                            np.testing.assert_array_equal(archive['pairs_val'],a['pairs_val'])
                            np.testing.assert_array_equal(archive['y_val'],a['y_val'])
                            score=archive['score'].copy()
                        rows.append(dict(identity,unit=r['unit'],parameter_count=r['parameter_count'],
                                         **base.metrics(a['y_val'],score,a['pairs_val'])))
                        for name,weight in (('drug_matched',weights),('same_subset_unweighted',(weights>0).astype(float))):
                            diagnostics.append(dict(identity,evaluation=name,**weighted_metrics(a['y_val'],score,weight)))
                        artifacts[str((folder/'result.json').relative_to(out))]=base.digest(folder/'result.json')
    if any(len(v)!=1 for v in initializations.values()) or any(len(v)!=1 for v in capacities.values()):
        raise ValueError('TabM initializations/capacities differ across input conditions')
    if p['smoke']:
        base.write_json(out/'SMOKE_VALIDATION.json',dict(protocol_sha256=p['sha256'],passed=True,checkpoints=count,
            same_initialization=True,same_capacity=True,query_invariance=True,performance_evidence=False))
        print('SMOKE_VERIFIED',count,flush=True);return
    reference=pd.read_csv(out/'reference_metrics.csv');reference['iterations']=reference['epoch']
    reference['unit']='optimizer_updates'
    table=pd.concat([reference,pd.DataFrame(rows)],ignore_index=True)
    diag=pd.concat([pd.read_csv(out/'reference_diagnostics.csv'),pd.DataFrame(diagnostics)],ignore_index=True)
    table.to_csv(out/'metrics.csv',index=False);diag.to_csv(out/'diagnostics.csv',index=False)
    metrics=list(base.METRICS)
    summary=table.groupby(['task','learner','budget','view'],sort=False)[metrics].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    differences=[]
    for (task,learner,budget),group in table.groupby(['task','learner','budget'],sort=False):
        for before,after in (('graph36','tiered62'),('tiered62','expanded62')):
            first=group[group['view']==before].set_index('seed');second=group[group['view']==after].set_index('seed')
            if set(first.index)!=set(p['seeds']) or set(second.index)!=set(p['seeds']):
                raise ValueError('Incomplete paired comparison')
            for metric in metrics:
                delta=second.loc[p['seeds'],metric]-first.loc[p['seeds'],metric]
                differences.append(dict(task=task,learner=learner,budget=budget,before=before,after=after,
                    metric=metric,mean_delta=float(delta.mean()),positive_seeds=int((delta>0).sum()),
                    minimum_delta=float(delta.min()),maximum_delta=float(delta.max())))
    pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
    primary=summary[summary.apply(lambda row:row['budget']==p['primary_budgets'][row['learner']],axis=1)]
    primary.to_csv(out/'primary_summary.csv',index=False)
    lines=['**不使用预训练预测器：CatBoost、TabM、HGT 和配对 MLP 的同证据对照。**','',
        '三档输入沿用同一批样本、标签和新增证据。所有预测器从头拟合，已有 ChemBERTa 药物特征作为共有输入保留。本轮没有 TabICL/TabPFN 预测器。',
        '仅使用历史内部开发集。三个种子并非独立数据划分，各算法的完整容量和计算量并未匹配；结果适用于所列配置。','',
        '**预先指定的主要预算。** HGT/配对 MLP 为 24/74 步；TabM 为历史内部选定的 141/441 步（LncDrug/MiDrug），每步累积完整训练集梯度；CatBoost 为 300 棵树。没有用本轮分数挑选预算。','',
        '|任务|模型|输入|AUC|AP|逐药物平均 AP|','|---|---|---|---:|---:|---:|']
    for r in primary.to_dict('records'):
        lines.append('|{task}|{learner}|{view}|{auc_mean:.6f}|{ap_mean:.6f}|{macro_drug_ap_mean:.6f}|'.format(**r))
    lines+=['','**所有预先保存的预算。** TabM 的 matched_hgt_steps 只匹配更新次数：HGT 每次使用一半训练配对，TabM 使用全量配对及 16 个内部成员。long 为 150/450 步，CatBoost 另报告 1000 棵树。','',
        '|任务|模型|预算|输入|AUC|AP|逐药物平均 AP|','|---|---|---|---|---:|---:|---:|']
    for r in summary.to_dict('records'):
        lines.append('|{task}|{learner}|{budget}|{view}|{auc_mean:.6f}|{ap_mean:.6f}|{macro_drug_ap_mean:.6f}|'.format(**r))
    lines+=['','TabM 为项目现有的紧凑实现；不声称复现官方基准。配对输入的图特征使用训练集内部交叉拟合，所有新增特征与上一轮相同。',
        'TabM 用成员平均概率的稳定 log-odds 进行排序，CatBoost/HGT/MLP 用原始 logit，避免概率饱和产生并列分数。',
        '药物频率匹配及同子集未加权结果在 diagnostics.csv；不同子集的 AP 不应直接比较。',
        '原始训练状态均保存并恢复验证。没有本轮超参数搜索、验证集早停、外层／原始测试集评价或模型替换。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for name in ('metrics.csv','diagnostics.csv','summary.csv','primary_summary.csv','paired_differences.csv','REPORT.md'):
        artifacts[name]=base.digest(out/name)
    base.write_json(out/'VALIDATION.json',dict(protocol_sha256=p['sha256'],passed=True,new_checkpoints=count,
        no_predictor_pretraining=True,common_pretrained_drug_inputs_retained=True,
        no_new_validation_selection=True,no_outer_arrays=True,no_original_test_evaluation=True,
        same_tabm_initialization=True,same_tabm_capacity=True,baseline_preserved=True,artifacts=artifacts))
    print('NONPRETRAINED_REPORTED',count,'new checkpoints',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--out-dir',default=DEFAULT_OUT)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--task',choices=('MiDrug','LncDrug'))
    parser.add_argument('--learner',choices=('tabm','catboost'))
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();dict(prepare=prepare,run=run,report=report)[args.command](args)


if __name__=='__main__':
    main()
