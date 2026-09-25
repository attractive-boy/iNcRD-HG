"""Fixed-budget input experiments against the accepted TabICLv2 reference.

Only the existing inner development arrays are available in this runner.
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from improve import tabular_foundation as base
from improve import tabicl_context_program as accepted
from improve import tabicl_context_program_modules as evidence
from improve import tabicl_pathway as operations
from improve.rna_functional_experiment import reference_scores
from improve.sampling_bias_audit import matched_weights, weighted_metrics


DEFAULT_REFERENCE = 'improve/results/tabicl_pathway_context_20260913'
MODULES = {'site':'site_evidence_features','drug':'drug_multimodal_features'}
OWN = ('frozen_feature_validation.py','rna_functional_experiment.py','tabicl_pathway.py',
       'tabicl_context_program.py','tabicl_context_program_modules.py','rna_functional_modules.py')


def module(family):
    return importlib.import_module('improve.'+MODULES[family])


def sources(family):
    names = [*OWN,MODULES[family]+'.py','test_'+MODULES[family]+'.py']
    return {name:base.digest(Path(__file__).with_name(name)) for name in names}


def encoder(p,task):
    m = module(p['family'])
    return m.SiteFeatures(p['representation_dir'],'association') if p['family']=='site' else m.DrugFeatures(p['representation_dir'],task)


def prepare(args):
    out,reference,input_dir = (Path(s).resolve() for s in (args.out_dir,args.reference_dir,args.inputs_dir))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    rp = accepted.protocol(reference)
    choice = accepted.selected(reference,rp)
    candidate = base.read_json(reference/'CANDIDATE.json')
    if rp['smoke'] or not candidate['accepted'] or choice['recipe']['variant']!='tiered' or choice['recipe']['alpha']!=.5:
        raise ValueError('Expected accepted tiered 0.5 reference')
    m = module(args.family)
    m.verify(input_dir)
    variants = list(m.VARIANTS)
    primary = ['evidence','target'] if args.family=='site' else ['m2umol']
    tasks = ['MiDrug'] if args.family=='site' else list(rp['tasks'])
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family=args.family,
        smoke=args.smoke,tasks=tasks,seeds=[666] if args.smoke else [666,0,1],members=2 if args.smoke else 32,
        variants=variants,primary_variants=primary,
        reference_dir=str(reference),reference_protocol_sha256=rp['sha256'],reference_recipe=choice['recipe'],
        reference_files={f:base.digest(reference/f) for f in ('protocol.json','SELECTION.json','CANDIDATE.json','VALIDATION_inner.json','VALIDATION_outer.json')},
        parent_dir=rp['parent_dir'],parent_protocol_sha256=rp['parent_protocol_sha256'],
        input_configs=rp['input_configs'],graph_recipes=rp['graph_recipes'],inputs_dir=rp['inputs_dir'],
        representation_dir=str(input_dir),representation_manifest_sha256=base.digest(input_dir/'INPUTS.json'),
        source_sha256=sources(args.family),parameter_sha256=rp['parameter_sha256'],
        extra_columns=50,input_columns=86,
        representation='Site: original 8 RNA metadata + 2 explicit coverage/gene-count fields + 14 PCs. Drug: 8 shared structure covariates + 16 PCs. 24 appended columns in every candidate.',
        compression='PCA, standardization and whitening fit on unique covered support RNA entities (Q1), RNA-drug pairs (Q2), or drugs (Q3). Query-only entities never fit the projection. Clip PCs to [-10,10]; preserve uncovered rows.',
        composition='0.5 * frozen graph score + 0.5 * extended tiered branch. Same graph context recipe; no coefficient search.',
        controls='Q1: matched generic/flat, count-only and site-vector shuffle within evidence/support membership. Q2: matched flat/evidence, count-only and within-RNA/evidence vector-to-gene shuffle. Q3: descriptor-only, fingerprint, ChemBERTa and support/coverage/size-stratified M2UMol shuffle.',
        performance_gate='MiDrug mean AUC and AP must both exceed reference, with each improving in at least 2/3 seeds. For Q3 also LncDrug AUC/AP must not decrease in ANY seed. No tolerance or rounded comparisons.',
        information_gate='Each primary must exceed all its registered representation controls on mean AP in the covered subset and covered drug-matched subset, with no mean macro drug AP decrease versus flat (sites) or ChemBERTa (drugs). Descriptive, not significance.',
        parameter_updates=0,hyperparameter_searches=0,no_outer_evaluation=True,no_original_test_evaluation=True,
        automatic_promotion=False,site_success_cannot_establish_joint_improvement=True,
        replay_atol=1e-6,replay_rtol=1e-6,query_invariance_atol=5e-5,query_invariance_rtol=1e-5,
        limitations=['Repeatedly used inner development validation; inference seeds are not independent biological replicates.',
          'Unknown associations are not verified negatives. Source-grouped direction is a separate task.',
          'Source selection can be filtered, but public encoder pretraining is not proven source independent.',
          'Binding/MTI evidence does not determine RNA intervention drug-response direction.']))
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name),out/'source_snapshot'/name)
    base.write_json(out/'protocol.json',p)
    print('FROZEN',p['family'],out,p['sha256'],flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out)/'protocol.json'))
    if p['source_sha256'] != sources(p['family']):
        raise ValueError('Frozen implementation changed')
    for name,digest in p['reference_files'].items():
        if base.digest(Path(p['reference_dir'])/name)!=digest:
            raise ValueError('Accepted reference changed')
    if accepted.protocol(p['reference_dir'])['sha256'] != p['reference_protocol_sha256']:
        raise ValueError('Reference protocol mismatch')
    if base.digest(Path(p['representation_dir'])/'INPUTS.json') != p['representation_manifest_sha256']:
        raise ValueError('Representation manifest changed')
    module(p['family']).verify(p['representation_dir'])
    return p


def run(args):
    import torch
    from tabicl import TabICLClassifier
    out = Path(args.out_dir)
    p = protocol(out)
    if args.task not in p['tasks']:
        raise ValueError('Unregistered task')
    a,gp = accepted.stage_data(out,p,args.task,'inner')
    tp = base.read_json(Path(gp['parent_dir'])/'protocol.json')
    enc = encoder(p,args.task)
    tiered = evidence.ContextPrograms.load(p['inputs_dir'])
    original_train = tiered.transform(args.task,a['pairs_train'],'tiered')
    original_query = tiered.transform(args.task,a['pairs_val'],'tiered')
    variants = p['variants'] if args.variant is None else args.variant.split(',')
    seeds = p['seeds'] if args.seed is None else [args.seed]
    if any(v not in p['variants'] for v in variants) or any(s not in p['seeds'] for s in seeds):
        raise ValueError('Unregistered variant or seed')
    kinds = operations.kinds(p,args.task)
    if p['smoke']:
        kinds = [k for k in kinds if k in ('full','fold0')]
    torch.cuda.set_device(args.device)
    for variant in variants:
        for kind in kinds:
            keep = np.arange(len(a['y_train'])) if kind=='full' else np.flatnonzero(a['fold_ids']==int(kind[-1]))
            numeric,state,metadata = enc.transform(variant,a['pairs_train'][:,keep])
            train = np.concatenate([original_train,enc.select(numeric,a['pairs_train'],variant)],axis=1)
            query = np.concatenate([original_query,enc.select(numeric,a['pairs_val'],variant)],axis=1)
            x,y,q = operations.job_arrays(p,args.task,a,kind,train,query)
            if x.shape[1] != p['input_columns'] or not np.isfinite(x).all() or not np.isfinite(q).all():
                raise ValueError('Invalid input or changed column budget')
            for seed in seeds:
                dest = operations.directory(out,'inner',args.task,variant,seed,kind)
                if (dest/'result.json').exists():
                    operations.checked_result(dest,p)
                    continue
                start = time.monotonic()
                torch.cuda.reset_peak_memory_stats(args.device)
                clf = base.fit_context(base.estimator(dict(n_estimators=p['members']),seed,tp,args.device),x,y)
                if base.parameter_digest(clf)!=p['parameter_sha256']:
                    raise ValueError('Pretrained parameters changed')
                flat = q.reshape(-1,x.shape[1])
                score = base.query_scores(clf,flat).reshape(q.shape[:2])
                dest.mkdir(parents=True,exist_ok=True)
                clf.save(dest/'fitted.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=False)
                np.savez_compressed(dest/'scores.npz',views=score,y_val=a['y_val'],pairs_val=a['pairs_val'])
                np.savez_compressed(dest/'projection.npz',**state)
                np.savez_compressed(dest/'numeric_table.npz',features=numeric)
                base.write_json(dest/'projection.json',metadata)
                del clf
                gc.collect()
                torch.cuda.empty_cache()
                clf = TabICLClassifier.load(dest/'fitted.pkl',device=args.device)
                replay = base.query_scores(clf,flat).reshape(q.shape[:2])
                np.testing.assert_allclose(replay,score,atol=p['replay_atol'],rtol=p['replay_rtol'])
                if base.parameter_digest(clf)!=p['parameter_sha256']:
                    raise ValueError('Reload changed pretrained parameters')
                invariance = {}
                if p['smoke']:
                    for name,value in dict(reverse=base.query_scores(clf,flat[::-1])[::-1],
                        split=np.concatenate([base.query_scores(clf,part) for part in np.array_split(flat,3)])).items():
                        np.testing.assert_allclose(value,score.ravel(),atol=p['query_invariance_atol'],rtol=p['query_invariance_rtol'])
                        invariance[name] = float(np.max(np.abs(value-score.ravel())))
                result = dict(protocol_sha256=p['sha256'],phase='inner',task=args.task,variant=variant,seed=seed,kind=kind,
                    replay_passed=True,replay_max_abs=float(np.max(np.abs(score-replay))),smoke_invariance=invariance,
                    context_rows=len(y),input_columns=x.shape[1],query_views=len(q),parameter_sha256=p['parameter_sha256'],
                    seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device),
                    artifacts={name:base.digest(dest/name) for name in ('fitted.pkl','scores.npz','projection.npz','projection.json','numeric_table.npz')})
                base.write_json(dest/'result.json',result)
                print('VERIFIED',args.task,variant,seed,kind,'seconds',round(result['seconds'],1),flush=True)
                del clf
                gc.collect()
                torch.cuda.empty_cache()


def masks(p,enc,a):
    n = len(a['y_val'])
    groups = {'all':np.ones(n,bool)}
    if p['family']=='site':
        for name in ['evidence','target']:
            data,_ = enc.table(name,a['pairs_train'][0])
            ids = a['pairs_val'][0]*enc.n_drugs+a['pairs_val'][1] if name=='target' else a['pairs_val'][0]
            groups[name+'_covered'] = np.isin(ids,data['ids'])
            groups[name+'_uncovered'] = ~groups[name+'_covered']
    else:
        groups['m2umol_covered'] = enc.present[a['pairs_val'][1]]
        groups['m2umol_uncovered'] = ~groups['m2umol_covered']
    positive_degree = np.bincount(a['pairs_train'][1,a['y_train']==1],minlength=enc.n_drugs)
    threshold = float(np.median(positive_degree[positive_degree>0]))
    groups['low_frequency_drug'] = positive_degree[a['pairs_val'][1]]<=threshold
    return groups,positive_degree,threshold


def report(args):
    out = Path(args.out_dir)
    p = protocol(out)
    if p['smoke'] or (out/'VALIDATION.json').exists():
        raise ValueError('Only complete, unreported formal experiments may be reported')
    rows,diagnostics,per_drug,coverage,artifacts = [],[],[],{},{}
    for task in p['tasks']:
        a,_ = accepted.stage_data(out,p,task,'inner')
        enc = encoder(p,task)
        groups,degree,threshold = masks(p,enc,a)
        coverage[task] = {name:dict(pairs=int(mask.sum()),positives=int(a['y_val'][mask].sum()),
            rnas=len(np.unique(a['pairs_val'][0,mask])),drugs=len(np.unique(a['pairs_val'][1,mask]))) for name,mask in groups.items()}
        coverage[task]['low_frequency_threshold'] = threshold
        for seed in p['seeds']:
            baseline,graph = reference_scores(p,task,seed,a)
            predictions = dict(baseline=baseline)
            for variant in p['variants']:
                raw,hashes = operations.augmented_score(out,p,a,'inner',task,variant,seed)
                artifacts.update(hashes)
                predictions[variant] = .5*graph+.5*raw
            for variant,score in predictions.items():
                rows.append(dict(task=task,seed=seed,variant=variant,**base.metrics(a['y_val'],score,a['pairs_val'])))
                for name,mask in groups.items():
                    weights = np.zeros(len(score))
                    if mask.any():
                        weights[mask],_ = matched_weights(a['y_val'][mask],a['pairs_val'][1,mask])
                    for suffix,w in [('',mask.astype(float)),('_drug_matched',weights)]:
                        diagnostics.append(dict(task=task,seed=seed,variant=variant,subset=name+suffix,
                            pairs=int(np.sum(w>0)),**weighted_metrics(a['y_val'],score,w)))
                for drug in np.unique(a['pairs_val'][1]):
                    mask = a['pairs_val'][1]==drug
                    per_drug.append(dict(task=task,seed=seed,variant=variant,drug_index=int(drug),train_positive_degree=int(degree[drug]),
                        query_rows=int(mask.sum()),query_positives=int(a['y_val'][mask].sum()),
                        **weighted_metrics(a['y_val'],score,mask.astype(float))))
            dest = out/'predictions'/task/('s'+str(seed))
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'scores.npz',**predictions,y_val=a['y_val'],pairs_val=a['pairs_val'])
            artifacts[str((dest/'scores.npz').relative_to(out))] = base.digest(dest/'scores.npz')
    table,diag = pd.DataFrame(rows),pd.DataFrame(diagnostics)
    table.to_csv(out/'metrics.csv',index=False)
    diag.to_csv(out/'diagnostics.csv',index=False)
    pd.DataFrame(per_drug).to_csv(out/'per_drug.csv',index=False)
    means = table.groupby(['task','variant'])[list(base.METRICS)].mean()
    dm = diag.groupby(['task','variant','subset'])[['auc','ap']].mean()
    controls = {'evidence':['generic','flat','evidence_stats','evidence_shuffled'],
                'target':['flat','evidence','target_stats','target_shuffled'],
                'm2umol':['descriptors','fingerprint','chemberta','m2umol_shuffled']}
    gates = {}
    for primary in p['primary_variants']:
        gates[primary] = {}
        for task in p['tasks']:
            t = table[table.task==task].set_index(['variant','seed'])
            delta = {metric:np.array([t.loc[(primary,s),metric]-t.loc[('baseline',s),metric] for s in p['seeds']]) for metric in ('auc','ap')}
            performance = all(np.all(v>=0) for v in delta.values()) if task=='LncDrug' else all(v.mean()>0 and (v>0).sum()>=2 for v in delta.values())
            comparisons = {subset:{other:float(dm.loc[(task,primary,subset),'ap']-dm.loc[(task,other,subset),'ap']) for other in controls[primary]}
                           for subset in [primary+'_covered',primary+'_covered_drug_matched']}
            macro_ref = 'flat' if p['family']=='site' else 'chemberta'
            macro_delta = float(means.loc[(task,primary),'macro_drug_ap']-means.loc[(task,macro_ref),'macro_drug_ap'])
            information = all(v>0 for section in comparisons.values() for v in section.values()) and macro_delta>=0
            gates[primary][task] = dict(performance_gate=bool(performance),per_seed_delta={k:v.tolist() for k,v in delta.items()},
                information_gate=bool(information),information_ap_deltas=comparisons,macro_ap_delta=macro_delta)
    outcome = dict(gates=gates,coverage=coverage,accepted_reference_replaced=False,historical_outer_evaluated=False,
        original_test_evaluated=False,source_holdout_evaluated=False,
        joint_performance_tested=p['family']=='drug',
        joint_performance_passed=(all(g['performance_gate'] for g in gates['m2umol'].values()) if p['family']=='drug' else None))
    base.write_json(out/'ASSESSMENT.json',outcome)
    for name in ('metrics.csv','diagnostics.csv','per_drug.csv','ASSESSMENT.json'):
        artifacts[name] = base.digest(out/name)
    base.write_json(out/'VALIDATION.json',dict(protocol_sha256=p['sha256'],passed=True,
        fitted_states=len([name for name in artifacts if name.endswith('/result.json')]),artifacts=artifacts,**outcome))
    lines = ['固定预算输入验证已完成：每个分支 86 列，32 个冻结 TabICLv2 推理成员，种子 666/0/1。',
        '', '沿用支持样本、查询样本和图上下文配方。新分支固定与图分支等权；没有调参或融合系数搜索。',
        '', '|任务|分支|AUC|AP|药物宏 AP|ΔAP（百分点）|','|---|---|---:|---:|---:|---:|']
    for task in p['tasks']:
        for variant in ['baseline',*p['variants']]:
            r = means.loc[(task,variant)]
            lines.append('|%s|%s|%.9f|%.9f|%.9f|%+.6f|'%(task,variant,r.auc,r.ap,r.macro_drug_ap,100*(r.ap-means.loc[(task,'baseline'),'ap'])))
    lines += ['', '主比较及完整精度门槛见 ASSESSMENT.json。覆盖、长尾和药物频率匹配结果见 diagnostics.csv；逐药物结果见 per_drug.csv。',
        '', '重载预测逐任务验证；输入顺序/分批不变性由另存的 smoke 运行检查。推理种子不是独立生物重复。',
        '', '未打开历史外层或原始关联测试数组；未评分来源保留集；未替换接受模型。miRNA 位点分支不能证明 lncRNA 任务改进。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(means.to_string(),flush=True)
    print('GATES',gates,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','report'])
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--family',choices=list(MODULES),default='site')
    parser.add_argument('--reference-dir',default=DEFAULT_REFERENCE)
    parser.add_argument('--inputs-dir',default='improve/results/site_evidence_inputs_20260914')
    parser.add_argument('--task',choices=['MiDrug','LncDrug'])
    parser.add_argument('--variant')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    globals()[args.command](args)


if __name__=='__main__':
    main()
