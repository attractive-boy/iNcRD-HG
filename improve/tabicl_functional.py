"""Test new binding evidence against the accepted tiered TabICLv2 predictor."""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse

from improve import tabular_foundation as base
from improve import tabicl_graph_context as graph
from improve import tabicl_context_program as accepted
from improve import tabicl_pathway as operations
from improve import tabicl_functional_modules as feature
from improve.sampling_bias_audit import matched_weights, weighted_metrics


DEFAULT_OUT='improve/results/tabicl_functional_20260913'
DEFAULT_REFERENCE='improve/results/tabicl_pathway_context_20260913'
DEFAULT_DRUG_INPUTS='/data1/licheng/incrd-hg/data_external/functional_priority_20260913/chembl'
OWN=('tabicl_functional.py','tabicl_functional_modules.py','functional_evidence_data.py',
     'test_tabicl_functional.py','test_functional_evidence_data.py','sampling_bias_audit.py')


def sources():
    return {name:base.digest(Path(__file__).with_name(name)) for name in OWN}


def prepare(args):
    out,reference,drug_inputs=map(lambda s:Path(s).resolve(),(args.out_dir,args.reference_dir,args.drug_inputs_dir))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    rp=accepted.protocol(reference)
    candidate=base.read_json(reference/'CANDIDATE.json')
    if rp['smoke'] or not candidate['accepted'] or candidate['protocol_sha256']!=rp['sha256']:
        raise ValueError('An accepted formal reference is required')
    choice=accepted.selected(reference,rp)
    if choice['recipe']!=candidate['selected_inner_recipe'] or choice['recipe']['variant']!='tiered':
        raise ValueError('Expected the latest accepted tiered reference')
    if not base.read_json(reference/'VALIDATION_outer.json')['eligible']:
        raise ValueError('The reference did not pass its outer gate')
    inputs=Path(rp['inputs_dir'])
    feature.verify_files(inputs);manifest=feature.verify_files(drug_inputs)
    if manifest['reference_manifest_sha256']!=base.digest(inputs/'INPUTS.json'):
        raise ValueError('Mismatched reference evidence')
    original=sparse.load_npz(inputs/'drug_gene.npz')
    expanded=sparse.load_npz(drug_inputs/'drug_gene_expanded.npz')
    rewired,control=feature.rewire_additions(original,expanded)
    if control['accepted_swaps']<control['target_swaps'] or control['fraction_added_edges_changed']<.5:
        raise ValueError('Insufficient control mixing; inspect before freezing an experiment')
    out.mkdir(parents=True)
    sparse.save_npz(out/'drug_gene_rewired.npz',rewired)
    base.write_json(out/'CONTROL.json',control)
    reference_files={name:base.digest(reference/name) for name in
        ('protocol.json','CANDIDATE.json','SELECTION.json','VALIDATION_inner.json','VALIDATION_outer.json')}
    p=base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),smoke=args.smoke,
        tasks=rp['tasks'],seeds=[666] if args.smoke else [666,0,1],members=2 if args.smoke else 32,
        variants=list(feature.VARIANTS),recipes=feature.recipes(),
        feature_names={v:feature.names(v) for v in feature.VARIANTS},
        reference_dir=str(reference),reference_files=reference_files,reference_protocol_sha256=rp['sha256'],
        reference_recipe=choice['recipe'],parent_dir=rp['parent_dir'],
        parent_protocol_sha256=rp['parent_protocol_sha256'],input_configs=rp['input_configs'],
        graph_recipes=rp['graph_recipes'],inputs_dir=str(inputs),
        input_manifest_sha256=base.digest(inputs/'INPUTS.json'),drug_inputs_dir=str(drug_inputs),
        drug_input_manifest_sha256=base.digest(drug_inputs/'INPUTS.json'),
        control_files={name:base.digest(out/name) for name in ('CONTROL.json','drug_gene_rewired.npz')},
        source_sha256=sources(),parameter_sha256=rp['parameter_sha256'],
        mechanisms={
            'expanded':'Replace only the drug-gene matrix with old+high-confidence human single-protein binding edges; exactly reuse all 26 accepted tiered feature definitions.',
            'target_local':'Same evidence, graph features and 26-column budget. Keep the first 14 columns and each RNA tier presence/count. Replace the tier gene/program cosines with mean and top-3 mean RNA-program match to individual drug target pathway profiles.',
            'rewired_controls':'Retain every old edge, randomize only added edges by 20 accepted bipartite swaps per edge. Preserve drug and gene degrees exactly. Use the same control matrix for both tasks and both encodings; controls cannot be selected.'},
        composition='B=(1-f)*G+f*T is the accepted baseline. V=(1-f)*G+f*E. Score=(1-alpha)*B+alpha*V. f=.5 is frozen from the reference; one new variant/alpha is shared by both tasks. Graph mass stays fixed.',
        parameter_updates=0,no_test_evaluation=True,
        gate='LncDrug AUC and AP must each be >= accepted baseline in every inference seed, with zero negative tolerance. MiDrug mean AUC and AP must both improve, each in at least 2/3 seeds.',
        selection='One inner selection: maximize MiDrug mean (AUC+AP)/2 among eligible candidates, then lower alpha, then recipe ID. If none qualify, preserve baseline and prohibit all outer array access.',
        diagnostic_metrics='Report global AUC/AP, macro drug AP, matched-drug metrics with the same-subset unweighted control, and old/new/missing drug-target coverage strata. These diagnostics cannot change selection.',
        replay_atol=1e-6,replay_rtol=1e-6,query_invariance_atol=5e-5,query_invariance_rtol=1e-5,
        limitations=['Historical inner/outer partitions have been repeatedly used for development; three seeds are inference randomness, not independent datasets.',
                    'Binding evidence does not identify inhibition, activation, resistance direction or cell-specific effects.',
                    'Target pathway matching is an annotation-derived prior, not experimentally measured pathway activity or causal validation.',
                    'Swaps preserve degrees and missingness, but one finite-mixing control is not a uniform null distribution.',
                    'Auxiliary biological sources are not time-disjoint from the original association database.']))
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name),out/'source_snapshot'/name)
    shutil.copy2(drug_inputs/'COVERAGE.json',out/'INPUT_COVERAGE.json')
    base.write_json(out/'protocol.json',p)
    print('PROTOCOL_FROZEN',out,p['sha256'],'control_changed',control['fraction_added_edges_changed'],flush=True)


def protocol(out):
    out=Path(out);p=base.check_seal(base.read_json(out/'protocol.json'))
    if p['source_sha256']!=sources():
        raise ValueError('Frozen implementation changed')
    for name,expected in p['reference_files'].items():
        if base.digest(Path(p['reference_dir'])/name)!=expected:
            raise ValueError('Accepted reference changed: '+name)
    for key,expected in (('inputs_dir',p['input_manifest_sha256']),('drug_inputs_dir',p['drug_input_manifest_sha256'])):
        if base.digest(Path(p[key])/'INPUTS.json')!=expected:
            raise ValueError('Frozen evidence manifest changed')
    for name,expected in p['control_files'].items():
        if base.digest(out/name)!=expected:
            raise ValueError('Frozen control changed')
    if accepted.protocol(p['reference_dir'])['sha256']!=p['reference_protocol_sha256']:
        raise ValueError('Accepted reference protocol changed')
    return p


def selected(out,p):
    out=Path(out);s=base.check_seal(base.read_json(out/'SELECTION.json'))
    if (s['protocol_sha256']!=p['sha256'] or p['smoke'] or not s['eligible']
        or s['recipe'] not in p['recipes'] or s['recipe']['variant'] not in feature.CANDIDATES):
        raise ValueError('No eligible formal inner candidate; outer access prohibited')
    for name,expected in s['artifacts'].items():
        if base.digest(out/name)!=expected:
            raise ValueError('Inner artifact changed after selection')
        if name.endswith('/result.json'):
            operations.checked_result((out/name).parent,p)
    return s


def stage_data(out,p,task,phase):
    if phase not in ('inner','outer') or task not in p['tasks']:
        raise ValueError('Unregistered task/phase')
    if phase=='outer':
        selected(out,p)
    # Reuse exactly the reference's support/query recipe and smoke subsampling.
    return accepted.stage_data(out,p,task,phase)


def reference_scores(p,task,phase,seed,a):
    directory=Path(p['reference_dir'])
    validation=base.read_json(directory/('VALIDATION_'+phase+'.json'))
    path=f'predictions/{phase}/{task}/s{seed}/scores.npz'
    if (validation['protocol_sha256']!=p['reference_protocol_sha256']
        or base.digest(directory/path)!=validation['artifacts'].get(path)):
        raise ValueError('Unverified accepted reference prediction')
    with np.load(directory/path,allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['y_val'],a['y_val'])
        np.testing.assert_array_equal(archive['pairs_val'],a['pairs_val'])
        return archive[p['reference_recipe']['id']].copy(),archive['baseline'].copy()


def run(args):
    import torch
    from tabicl import TabICLClassifier
    out=Path(args.out_dir);p=protocol(out)
    if args.phase=='outer':
        variant=selected(out,p)['recipe']['variant']
        variants=[variant,variant+'_rewired']
    else:
        variants=p['variants']
    if args.variant:
        if args.variant not in variants:
            raise ValueError('Variant not registered for this phase')
        variants=[args.variant]
    a,gp=stage_data(out,p,args.task,args.phase)
    tp=base.read_json(Path(gp['parent_dir'])/'protocol.json')
    encoder=feature.FunctionalPrograms.load(p['inputs_dir'],p['drug_inputs_dir'],out/'drug_gene_rewired.npz')
    seeds=p['seeds'] if args.seed is None else [args.seed]
    if any(s not in p['seeds'] for s in seeds):
        raise ValueError('Unregistered inference seed')
    torch.cuda.set_device(args.device)
    for variant in variants:
        extra_train=encoder.transform(args.task,a['pairs_train'],variant)
        extra_val=encoder.transform(args.task,a['pairs_val'],variant)
        for seed in seeds:
            for kind in operations.kinds(p,args.task):
                dest=operations.directory(out,args.phase,args.task,variant,seed,kind)
                if (dest/'result.json').exists():
                    operations.checked_result(dest,p)
                    continue
                start=time.monotonic()
                x,y,q=operations.job_arrays(p,args.task,a,kind,extra_train,extra_val)
                torch.cuda.reset_peak_memory_stats(args.device)
                clf=base.fit_context(base.estimator(dict(n_estimators=p['members']),seed,tp,args.device),x,y)
                if base.parameter_digest(clf)!=p['parameter_sha256']:
                    raise ValueError('Frozen neural weights changed')
                flat=q.reshape(-1,x.shape[1]);score=base.query_scores(clf,flat).reshape(q.shape[:2])
                if base.parameter_digest(clf)!=p['parameter_sha256']:
                    raise ValueError('Inference changed neural weights')
                dest.mkdir(parents=True,exist_ok=True)
                clf.save(dest/'fitted.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=False)
                np.savez_compressed(dest/'scores.npz',views=score,y_val=a['y_val'],pairs_val=a['pairs_val'])
                del clf;gc.collect();torch.cuda.empty_cache()
                clf=TabICLClassifier.load(dest/'fitted.pkl',device=args.device)
                if base.parameter_digest(clf)!=p['parameter_sha256']:
                    raise ValueError('Reload changed frozen neural weights')
                replay=base.query_scores(clf,flat).reshape(q.shape[:2])
                np.testing.assert_allclose(replay,score,atol=p['replay_atol'],rtol=p['replay_rtol'])
                invariance={}
                if p['smoke']:
                    checks=dict(reverse=base.query_scores(clf,flat[::-1])[::-1],
                        split=np.concatenate([base.query_scores(clf,part) for part in np.array_split(flat,3)]))
                    for name,prediction in checks.items():
                        np.testing.assert_allclose(prediction,score.ravel(),atol=p['query_invariance_atol'],rtol=p['query_invariance_rtol'])
                        invariance[name]=float(np.max(np.abs(prediction-score.ravel())))
                result=dict(protocol_sha256=p['sha256'],phase=args.phase,task=args.task,variant=variant,
                    seed=seed,kind=kind,context_rows=len(y),input_columns=x.shape[1],query_views=len(q),
                    members=p['members'],parameter_sha256=p['parameter_sha256'],replay_passed=True,
                    replay_max_abs=float(np.max(np.abs(replay-score))),smoke_invariance=invariance,
                    seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device),
                    artifacts={name:base.digest(dest/name) for name in ('fitted.pkl','scores.npz')})
                base.write_json(dest/'result.json',result)
                print('VERIFIED',args.phase,args.task,variant,seed,kind,'seconds',round(result['seconds'],1),flush=True)
                del clf;gc.collect();torch.cuda.empty_cache()


def diagnostic_views(y,pairs,encoder):
    weights,matching=matched_weights(y,pairs[1])
    old=np.asarray(encoder.original.sum(1)).ravel()>0
    new=np.asarray(encoder.expanded.sum(1)).ravel()>0
    views=dict(drug_matched=weights,drug_eligible_unweighted=(weights>0).astype(float),
        previously_covered_drug=old[pairs[1]].astype(float),
        newly_covered_drug=(new & ~old)[pairs[1]].astype(float),
        still_uncovered_drug=(~new)[pairs[1]].astype(float))
    return views,matching


def report(args):
    import pandas as pd
    out=Path(args.out_dir);p=protocol(out)
    if p['smoke']:
        raise ValueError('Smoke predictions are not performance evidence')
    if (out/('VALIDATION_'+args.phase+'.json')).exists():
        raise ValueError('This phase already has a frozen report; do not reselect')
    chosen=selected(out,p) if args.phase=='outer' else None
    if chosen:
        configs=[p['recipes'][0],chosen['recipe']]+[c for c in p['recipes']
            if c['variant']==chosen['recipe']['variant']+'_rewired' and c['alpha']==chosen['recipe']['alpha']]
    else:
        configs=p['recipes']
    rows,diagnostics,artifacts,coverage=[],[],{},{}
    encoder=feature.FunctionalPrograms.load(p['inputs_dir'],p['drug_inputs_dir'],out/'drug_gene_rewired.npz')
    for task in p['tasks']:
        a,_=stage_data(out,p,task,args.phase)
        views,coverage[task]=diagnostic_views(a['y_val'],a['pairs_val'],encoder)
        for seed in p['seeds']:
            baseline,graph_baseline=reference_scores(p,task,args.phase,seed,a)
            variants={}
            for variant in sorted({c['variant'] for c in configs if c['variant']!='baseline'}):
                variants[variant],hashes=operations.augmented_score(out,p,a,args.phase,task,variant,seed)
                artifacts.update(hashes)
            predictions={}
            for config in configs:
                score=(baseline if config['variant']=='baseline' else feature.compose(baseline,graph_baseline,
                    variants[config['variant']],p['reference_recipe']['alpha'],config['alpha']))
                predictions[config['id']]=score
                rows.append(dict(task=task,seed=seed,recipe=config['id'],variant=config['variant'],
                    alpha=config['alpha'],role=config['role'],**base.metrics(a['y_val'],score,a['pairs_val'])))
                for view,weight in views.items():
                    diagnostics.append(dict(task=task,seed=seed,recipe=config['id'],evaluation=view,
                        retained_pairs=int((weight>0).sum()),positive_pairs=int(a['y_val'][weight>0].sum()),
                        **weighted_metrics(a['y_val'],score,weight)))
            dest=out/'predictions'/args.phase/task/('s'+str(seed));dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'scores.npz',**predictions,y_val=a['y_val'],pairs_val=a['pairs_val'])
            artifacts[str((dest/'scores.npz').relative_to(out))]=base.digest(dest/'scores.npz')
    table=pd.DataFrame(rows);diag=pd.DataFrame(diagnostics)
    table.to_csv(out/(args.phase+'_metrics.csv'),index=False)
    diag.to_csv(out/(args.phase+'_diagnostics.csv'),index=False)
    summary=table.groupby(['task','recipe'],sort=False)[list(base.METRICS)].mean().reset_index()
    for metric in base.METRICS:
        refs=summary[summary.recipe=='baseline'].set_index('task')[metric].to_dict()
        summary['delta_'+metric]=summary[metric]-summary['task'].map(refs)
    summary.to_csv(out/(args.phase+'_summary.csv'),index=False)
    assessments=[feature.assess(rows,c,p['seeds']) for c in configs if c['variant']!='baseline']
    base.write_json(out/(args.phase+'_ASSESSMENTS.json'),assessments)
    base.write_json(out/(args.phase+'_MATCHING.json'),coverage)
    for name in (args.phase+'_metrics.csv',args.phase+'_diagnostics.csv',args.phase+'_summary.csv',
                 args.phase+'_ASSESSMENTS.json',args.phase+'_MATCHING.json'):
        artifacts[name]=base.digest(out/name)
    if args.phase=='inner':
        eligible=[r for r in assessments if r['eligible']]
        best=min(eligible,key=lambda r:(-(r['metrics']['MiDrug']['auc']['mean']+r['metrics']['MiDrug']['ap']['mean'])/2,
                                       r['recipe']['alpha'],r['recipe']['id'])) if eligible else None
        selection=base.seal(dict(protocol_sha256=p['sha256'],eligible=bool(best),
            recipe=best['recipe'] if best else p['recipes'][0],artifacts=artifacts,
            created_at_utc=datetime.now(timezone.utc).isoformat()))
        base.write_json(out/'SELECTION.json',selection)
    else:
        selection=chosen
    actual=next((r for r in assessments if r['recipe']['id']==selection['recipe']['id']),None)
    passed=bool(actual and actual['eligible'])
    base.write_json(out/('VALIDATION_'+args.phase+'.json'),dict(protocol_sha256=p['sha256'],phase=args.phase,
        all_states_replayed=True,source_unchanged=True,no_test_evaluation=True,
        eligible=passed,selection=selection['recipe'],artifacts=artifacts))
    state=('accepted_on_historical_validation' if passed else 'rejected_on_historical_validation') if args.phase=='outer' else (
           'selected_pending_outer' if passed else 'rejected_inner_outer_prohibited')
    base.write_json(out/'CANDIDATE.json',dict(protocol_sha256=p['sha256'],accepted=args.phase=='outer' and passed,
        status=state,phase=args.phase,selected_inner_recipe=selection['recipe'],reference_dir=p['reference_dir'],
        reference_recipe=p['reference_recipe'],same_new_rule_for_both_tasks=True,task_specific_bypass=False,
        new_feature_count=26,neural_parameter_updates=0,no_original_test_evaluation=True,
        outer_evaluated=args.phase=='outer',limitations=p['limitations']))
    lines=['**新增药物结合证据与靶点局部通路匹配：受控实验。**','',
        f"阶段：{args.phase}；参考模型：{p['reference_recipe']['id']}；状态：{state}。",
        '三种推理种子 666/0/1，每个 32 成员；同一新增规则和强度用于两项任务。',
        '保留已接受图分支权重，只替换分层证据分支；所有分支均为 36＋26 个输入特征。',
        'expanded 只扩充数据；target_local 更换分层交互汇总；rewired 为保持双方度数的新增边错配对照，不参加选择。','',
        '|任务|方案|AUC|AP|药物宏 AP|ΔAUC|ΔAP|Δ宏 AP|','|---|---|---:|---:|---:|---:|---:|---:|']
    for _,r in summary.iterrows():
        lines.append(f"|{r.task}|{r.recipe}|{r.auc:.9f}|{r.ap:.9f}|{r.macro_drug_ap:.9f}|{r.delta_auc:+.9f}|{r.delta_ap:+.9f}|{r.delta_macro_drug_ap:+.9f}|")
    lines+=['',f"固定内部选择：{selection['recipe']['id']}；本阶段通过验收：{passed}。",
        'LncDrug 每种子 AUC/AP 均不得下降；MiDrug 两项均值提高且每项至少 2/3 种子为正。',
        '按药物匹配及同集合未加权对照、覆盖分组见 diagnostics.csv；匹配后 AP 仅在同一评价分布下比较。',
        '这里的靶点局部匹配仍来自通路注释，不是结合位点模型或实验测得的功能响应。',
        '历史划分重复用于开发；三个推理种子不是三个独立数据划分。未评价原始测试集。']
    (out/('REPORT_'+args.phase+'.md')).write_text('\n'.join(lines)+'\n')
    print('REPORT',args.phase,'status',state,'selection',selection['recipe'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--out-dir',default=DEFAULT_OUT)
    parser.add_argument('--reference-dir',default=DEFAULT_REFERENCE)
    parser.add_argument('--drug-inputs-dir',default=DEFAULT_DRUG_INPUTS)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--phase',choices=('inner','outer'),default='inner')
    parser.add_argument('--task',choices=('MiDrug','LncDrug'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--variant',choices=feature.VARIANTS)
    args=parser.parse_args()
    dict(prepare=prepare,run=run,report=report)[args.command](args)


if __name__=='__main__':
    main()
