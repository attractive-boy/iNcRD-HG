"""Evaluate the shared evidence/context pathway adapter against graph TabICLv2."""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
import time

import numpy as np

from improve import tabular_foundation as base
from improve import tabicl_graph_context as graph
from improve import tabicl_graph_context_modules as graph_kernels
from improve import tabicl_pathway as operations
from improve import tabicl_context_program_modules as feature


DEFAULT_OUT = 'improve/results/tabicl_pathway_context_20260913'
DEFAULT_INPUTS = 'improve/results/tabicl_pathway_context_inputs_20260913'
DEFAULT_PARENT = 'improve/results/tabicl_graph_context_20260912'
OWN = ('tabicl_context_program.py', 'tabicl_context_program_modules.py',
       'pathway_context_data.py', 'test_tabicl_context_program.py',
       'tabicl_pathway.py', 'tabicl_pathway_modules.py')


def sources():
    return {name:base.digest(Path(__file__).with_name(name)) for name in OWN}


def prepare(args):
    out, inputs, parent = [Path(x).resolve() for x in (args.out_dir, args.inputs_dir, args.parent_dir)]
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    gp = graph.protocol(parent)
    choice = graph.selection(parent, gp)
    if gp['smoke'] or not base.read_json(parent/'VALIDATION.json')['passed']:
        raise ValueError('Need a verified formal graph-context baseline')
    manifest = base.read_json(inputs/'INPUTS.json')
    for name, expected in manifest['files'].items():
        if base.digest(inputs/name) != expected:
            raise ValueError('Frozen context input changed: '+name)
    if manifest['labels_read']:
        raise ValueError('External feature construction used labels')
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        tasks=['MiDrug','LncDrug'], seeds=[666] if args.smoke else [666,0,1], members=2 if args.smoke else 32,
        variants=list(feature.VARIANTS), recipes=feature.recipes(),
        feature_names={v:feature.names(v) for v in feature.VARIANTS},
        parent_dir=str(parent), parent_protocol_sha256=gp['sha256'],
        parent_selection_sha256=base.digest(parent/'SELECTION.json'), input_configs=gp['input_configs'],
        graph_recipes={t:choice['chosen'][t]['recipe'] for t in gp['tasks']},
        inputs_dir=str(inputs), input_manifest_sha256=base.digest(inputs/'INPUTS.json'),
        source_sha256=sources(), parameter_sha256=gp['parameter_sha256'],
        mechanisms={
            'annotation':'Same 14 gene/pathway summary features as the previous direct adapter, now with uploaded functional/weak evidence and existing lower-support ENCORI observations.',
            'tiered':'Use fixed shared maximum-per-edge confidence {functional:1,binding:.25,association:.1}; append 4 separate overlap/program features per evidence tier (26 total).',
            'context':'Append 14 rank-expression/pathway features from matched DepMap cells, including within-lineage centered associations and agreement with curated programs (40 total).',
            'context_permuted':'Negative control: keep the same 26 annotation features and 14 context columns, but permute whole RNA context blocks with seed 20260913. Never eligible for selection.'},
        input_policy='Keep the 36 frozen raw/quantile graph columns and graph context recipes. Append label-free evidence/context columns; TabICL preprocessing is fitted only on support rows.',
        composition='(1-alpha)*historical graph baseline + alpha*augmented graph model. One variant and alpha for both tasks; no task-identity bypass or learned per-task new coefficient.',
        parameter_updates=0, no_test_evaluation=True,
        gate='LncDrug AUC and AP must each be >= baseline in every inference seed, with zero negative tolerance. MiDrug seed means must both improve and each metric must improve in at least 2/3 seeds. Macro drug AP reported separately.',
        selection='Among eligible candidates maximize mean MiDrug (AUC+AP)/2, then prefer lower alpha and recipe ID. If none qualify, freeze baseline and do not load outer arrays. Select once on inner; no outer reselection.',
        outer_control='If context is selected, also evaluate its matched-alpha context_permuted control on outer, for interpretation only; it cannot change selection.',
        replay_atol=1e-6, replay_rtol=1e-6, query_invariance_atol=5e-5, query_invariance_rtol=1e-5,
        limitations=['Historical inner/outer partitions were repeatedly used in development; inference seeds are not independent splits.',
                    'This changes the feature/context adapter while keeping the pretrained neural backbone frozen.',
                    'RNA/target expression associations are not demonstrated causal regulation or experimentally measured pathway activities.',
                    'Lower-support ENCORI coverage is incomplete for NEAT1/MALAT1; all actual retrieved evidence is recorded.',
                    'The original RNA-expression release is unknown; 25Q2 mRNA is aligned by cell ID but may differ in processing.',
                    'Same-column-count permuted context controls expression alignment; annotation feature expansions are not fully isolated causal ablations.']))
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name), out/'source_snapshot'/name)
    base.write_json(out/'protocol.json', p)
    shutil.copy2(inputs/'COVERAGE.json', out/'INPUT_COVERAGE.json')
    print('PROTOCOL_FROZEN', out, p['sha256'], flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out)/'protocol.json'))
    if p['source_sha256'] != sources():
        raise ValueError('Frozen source changed; use a new trial')
    if base.digest(Path(p['inputs_dir'])/'INPUTS.json') != p['input_manifest_sha256']:
        raise ValueError('Frozen input manifest changed')
    if base.digest(Path(p['parent_dir'])/'SELECTION.json') != p['parent_selection_sha256']:
        raise ValueError('Reference graph choice changed')
    return p


def selected(out, p):
    selection = base.check_seal(base.read_json(Path(out)/'SELECTION.json'))
    if selection['protocol_sha256'] != p['sha256'] or p['smoke'] or not selection['eligible']:
        raise ValueError('No eligible formal inner result; outer access prohibited')
    for name, expected in selection['artifacts'].items():
        if base.digest(Path(out)/name) != expected:
            raise ValueError('Inner artifact changed after selection')
    return selection


def stage_data(out, p, task, phase):
    if phase not in ('inner','outer') or task not in p['tasks']:
        raise ValueError('Unregistered task/phase')
    if phase == 'outer':
        selected(out, p)
    gp = graph.protocol(p['parent_dir'])
    if gp['sha256'] != p['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    a = graph.features(p['parent_dir'], gp, task, phase)
    if p['smoke']:
        keep = np.sort(np.concatenate([np.flatnonzero(a['fold_ids']==fold)[
            base.balanced_subset(a['y_train'][a['fold_ids']==fold],32,20260913+fold)] for fold in range(5)]))
        for prefix in ('x','raw','y'):
            a[prefix+'_train'] = a[prefix+'_train'][keep]
        a['pairs_train'], a['fold_ids'] = a['pairs_train'][:,keep], a['fold_ids'][keep]
        keep = base.balanced_subset(a['y_val'],40,20260913)
        for prefix in ('x','raw','y'):
            a[prefix+'_val'] = a[prefix+'_val'][keep]
        a['pairs_val'] = a['pairs_val'][:,keep]
        for prefix in ('x','raw'):
            a[prefix+'_query_views'] = a[prefix+'_query_views'][:,keep]
    base.validate_arrays(a)
    return a, gp


def run(args):
    import torch
    from tabicl import TabICLClassifier
    out = Path(args.out_dir)
    p = protocol(out)
    if args.phase == 'outer':
        variant = selected(out,p)['recipe']['variant']
        variants = [variant] + (['context_permuted'] if variant=='context' else [])
    else:
        variants = p['variants']
    a, gp = stage_data(out,p,args.task,args.phase)
    tp = base.read_json(Path(gp['parent_dir'])/'protocol.json')
    encoder = feature.ContextPrograms.load(p['inputs_dir'])
    seeds = p['seeds'] if args.seed is None else [args.seed]
    if any(seed not in p['seeds'] for seed in seeds):
        raise ValueError('Unregistered seed')
    torch.cuda.set_device(args.device)
    for variant in variants:
        extra_train = encoder.transform(args.task,a['pairs_train'],variant)
        extra_val = encoder.transform(args.task,a['pairs_val'],variant)
        for seed in seeds:
            for kind in operations.kinds(p,args.task):
                dest = operations.directory(out,args.phase,args.task,variant,seed,kind)
                if (dest/'result.json').exists():
                    operations.checked_result(dest,p)
                    continue
                start = time.monotonic()
                x,y,q = operations.job_arrays(p,args.task,a,kind,extra_train,extra_val)
                torch.cuda.reset_peak_memory_stats(args.device)
                clf = base.fit_context(base.estimator(dict(n_estimators=p['members']),seed,tp,args.device),x,y)
                if base.parameter_digest(clf) != p['parameter_sha256']:
                    raise ValueError('Official neural weights changed')
                flat = q.reshape(-1,x.shape[1])
                score = base.query_scores(clf,flat).reshape(q.shape[:2])
                if base.parameter_digest(clf) != p['parameter_sha256']:
                    raise ValueError('Inference modified neural weights')
                dest.mkdir(parents=True,exist_ok=True)
                clf.save(dest/'fitted.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=False)
                np.savez_compressed(dest/'scores.npz',views=score,y_val=a['y_val'],pairs_val=a['pairs_val'])
                del clf
                gc.collect()
                torch.cuda.empty_cache()
                clf = TabICLClassifier.load(dest/'fitted.pkl',device=args.device)
                if base.parameter_digest(clf) != p['parameter_sha256']:
                    raise ValueError('Reload changed neural weights')
                replay = base.query_scores(clf,flat).reshape(q.shape[:2])
                np.testing.assert_allclose(replay,score,atol=p['replay_atol'],rtol=p['replay_rtol'])
                invariance = {}
                if p['smoke']:
                    preds = dict(reverse=base.query_scores(clf,flat[::-1])[::-1],
                                 split=np.concatenate([base.query_scores(clf,part) for part in np.array_split(flat,3)]))
                    for name,pred in preds.items():
                        np.testing.assert_allclose(pred,score.ravel(),atol=p['query_invariance_atol'],rtol=p['query_invariance_rtol'])
                        invariance[name] = float(np.max(np.abs(pred-score.ravel())))
                result = dict(protocol_sha256=p['sha256'],phase=args.phase,task=args.task,variant=variant,
                    seed=seed,kind=kind,context_rows=len(y),input_columns=x.shape[1],query_views=len(q),
                    members=p['members'],parameter_sha256=p['parameter_sha256'],replay_passed=True,
                    replay_max_abs=float(np.max(np.abs(replay-score))),smoke_invariance=invariance,
                    seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device),
                    artifacts={name:base.digest(dest/name) for name in ('fitted.pkl','scores.npz')})
                base.write_json(dest/'result.json',result)
                print('VERIFIED',args.phase,args.task,variant,seed,kind,'seconds',round(result['seconds'],1),flush=True)
                del clf
                gc.collect()
                torch.cuda.empty_cache()


def report(args):
    import pandas as pd
    out = Path(args.out_dir)
    p = protocol(out)
    if p['smoke']:
        raise ValueError('Smoke output is not performance evidence')
    if args.phase == 'outer':
        chosen = selected(out,p)
        configs = [p['recipes'][0], chosen['recipe']]
        if chosen['recipe']['variant']=='context':
            configs += [c for c in p['recipes'] if c['variant']=='context_permuted' and c['alpha']==chosen['recipe']['alpha']]
    else:
        configs = p['recipes']
    rows, artifacts, coverage = [], {}, {}
    encoder = feature.ContextPrograms.load(p['inputs_dir'])
    for task in p['tasks']:
        a, gp = stage_data(out,p,task,args.phase)
        ann = encoder.transform(task,a['pairs_val'],'annotation')
        context = encoder.transform(task,a['pairs_val'],'context')
        coverage[task] = dict(pairs=len(ann), both_gene_annotated=int(((ann[:,0]>0)&(ann[:,1]>0)).sum()),
                             both_pathway_annotated=int(((ann[:,8]>0)&(ann[:,9]>0)).sum()),
                             variable_rna_expression=int((context[:,-14]>0).sum()),
                             rna_expression_and_drug_targets=int(((context[:,-14]>0)&(context[:,-13]>0)).sum()))
        for seed in p['seeds']:
            old_full, old_components, _ = graph.components(Path(p['parent_dir']),gp,args.phase,task,seed,a)
            baseline = graph_kernels.combine(old_full,old_components,p['graph_recipes'][task])
            variants = {}
            for variant in sorted({c['variant'] for c in configs if c['variant']!='baseline'}):
                variants[variant], hashes = operations.augmented_score(out,p,a,args.phase,task,variant,seed)
                artifacts.update(hashes)
            predictions = {}
            for config in configs:
                prediction = (baseline if config['variant']=='baseline' else
                              (1-config['alpha'])*baseline+config['alpha']*variants[config['variant']])
                predictions[config['id']] = prediction
                rows.append(dict(task=task,seed=seed,recipe=config['id'],variant=config['variant'],
                                 alpha=config['alpha'],role=config.get('role','baseline'),
                                 **base.metrics(a['y_val'],prediction,a['pairs_val'])))
            dest = out/'predictions'/args.phase/task/('s'+str(seed))
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'scores.npz',**predictions,y_val=a['y_val'],pairs_val=a['pairs_val'])
            artifacts[str((dest/'scores.npz').relative_to(out))] = base.digest(dest/'scores.npz')
    table = pd.DataFrame(rows)
    table.to_csv(out/(args.phase+'_metrics.csv'),index=False)
    summary = table.groupby(['task','recipe'],sort=False)[list(base.METRICS)].mean().reset_index()
    for metric in base.METRICS:
        refs = summary[summary.recipe=='baseline'].set_index('task')[metric].to_dict()
        summary['delta_'+metric] = summary[metric]-summary['task'].map(refs)
    summary.to_csv(out/(args.phase+'_summary.csv'),index=False)
    assessments = [feature.assess(rows,config,p['seeds']) for config in configs if config['variant']!='baseline']
    base.write_json(out/(args.phase+'_ASSESSMENTS.json'),assessments)
    base.write_json(out/(args.phase+'_PAIR_COVERAGE.json'),coverage)
    for name in (args.phase+'_metrics.csv',args.phase+'_summary.csv',args.phase+'_ASSESSMENTS.json',args.phase+'_PAIR_COVERAGE.json'):
        artifacts[name] = base.digest(out/name)
    if args.phase == 'inner':
        eligible = [item for item in assessments if item['eligible']]
        best = min(eligible,key=lambda item:(-(item['metrics']['MiDrug']['auc']['mean']+item['metrics']['MiDrug']['ap']['mean'])/2,
                                             item['recipe']['alpha'],item['recipe']['id'])) if eligible else None
        selection = base.seal(dict(protocol_sha256=p['sha256'],eligible=bool(best),
            recipe=best['recipe'] if best else p['recipes'][0],artifacts=artifacts,
            created_at_utc=datetime.now(timezone.utc).isoformat()))
        if (out/'SELECTION.json').exists():
            previous = base.read_json(out/'SELECTION.json')
            if previous['recipe'] != selection['recipe'] or previous['eligible'] != selection['eligible']:
                raise ValueError('Recomputed selection differs')
        else:
            base.write_json(out/'SELECTION.json',selection)
    else:
        selection = chosen
    lines = ['# 共享证据与表达通路适配器','',f'阶段：{args.phase}；三种子 666/0/1，32 成员。',
             '同一个新增分支与融合系数用于 MiDrug 和 LncDrug；神经主干冻结。',
             'context_permuted 为等维度表达错配对照，不参加选择。','',
             '|任务|方案|AUC|AP|按药物宏 AP|ΔAUC|ΔAP|Δ宏 AP|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for _,row in summary.iterrows():
        lines.append(f"|{row['task']}|{row['recipe']}|{row['auc']:.9f}|{row['ap']:.9f}|{row['macro_drug_ap']:.9f}|{row['delta_auc']:+.9f}|{row['delta_ap']:+.9f}|{row['delta_macro_drug_ap']:+.9f}|")
    lines += ['',f"内部固定选择：{selection['recipe']['id']}；内部 eligible={selection['eligible']}。",'',
              'LncDrug 每个种子的 AUC、AP 均不下降，无负容差；MiDrug 两项均值提高且每项至少 2/3 种子为正。',
              '历史划分重复参与开发，推理种子不是独立重复；没有访问原始测试集。',
              '已补 LncTarD/miRTarBase 与 DepMap 25Q2；覆盖及证据类型见 INPUT_COVERAGE.json。',
              '表达通路分数与相关性不是实验验证的通路活性或因果调控。']
    (out/('REPORT_'+args.phase+'.md')).write_text('\n'.join(lines)+'\n')
    actual = next((item for item in assessments if item['recipe']['id']==selection['recipe']['id']),None)
    base.write_json(out/('VALIDATION_'+args.phase+'.json'),dict(protocol_sha256=p['sha256'],phase=args.phase,
        all_states_replayed=True,source_unchanged=True,no_test_evaluation=True,
        eligible=bool(actual and actual['eligible']),selection=selection['recipe'],artifacts=artifacts))
    print('REPORT',args.phase,'selected',selection['recipe'],'actual_eligible',bool(actual and actual['eligible']),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--out-dir',default=DEFAULT_OUT)
    parser.add_argument('--inputs-dir',default=DEFAULT_INPUTS)
    parser.add_argument('--parent-dir',default=DEFAULT_PARENT)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--phase',choices=('inner','outer'),default='inner')
    parser.add_argument('--task',choices=('MiDrug','LncDrug'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int)
    args = parser.parse_args()
    if args.command=='prepare':
        prepare(args)
    elif args.command=='run':
        run(args)
    else:
        report(args)


if __name__=='__main__':
    main()
