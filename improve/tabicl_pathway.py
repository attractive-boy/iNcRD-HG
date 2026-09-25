"""Evaluate shared RNA–gene–pathway pair features with frozen TabICLv2.

prepare -> smoke/run -> inner report and freeze -> eligible outer only.
Both tasks use the same variant and probability blend coefficient.
"""
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
from improve import tabicl_pathway_modules as feature


DEFAULT_OUT = 'improve/results/tabicl_pathway_20260913'
DEFAULT_INPUTS = 'improve/results/tabicl_pathway_inputs_20260913'
DEFAULT_PARENT = 'improve/results/tabicl_graph_context_20260912'
OWN = ('tabicl_pathway.py','tabicl_pathway_modules.py','test_tabicl_pathway.py','rna_pathway_data.py')


def sources():
    return {name:base.digest(Path(__file__).with_name(name)) for name in OWN}


def prepare(args):
    out, inputs, parent = map(lambda x:Path(x).resolve(), (args.out_dir,args.inputs_dir,args.parent_dir))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    gp = graph.protocol(parent)
    chosen = graph.selection(parent,gp)
    validation = base.read_json(parent/'VALIDATION.json')
    if gp['smoke'] or not validation['passed']:
        raise ValueError('Need a verified formal graph-context baseline')
    manifest = base.read_json(inputs/'INPUTS.json')
    for name, expected in manifest['files'].items():
        if base.digest(inputs/name) != expected:
            raise ValueError('Input evidence changed: '+name)
    queries = base.read_json(inputs/'ENTITY_QUERIES.json')
    if queries['failures']:
        raise ValueError('Unfinished external RNA queries cannot be treated as missing annotations')
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        tasks=['MiDrug','LncDrug'], seeds=[666] if args.smoke else [666,0,1],
        members=2 if args.smoke else 32, recipes=feature.recipes(), variants=list(feature.VARIANTS),
        feature_names={v:feature.names(v) for v in feature.VARIANTS},
        parent_dir=str(parent), parent_protocol_sha256=gp['sha256'],
        parent_selection_sha256=base.digest(parent/'SELECTION.json'),
        input_configs=gp['input_configs'], graph_recipes={t:chosen['chosen'][t]['recipe'] for t in gp['tasks']},
        inputs_dir=str(inputs), input_manifest_sha256=base.digest(inputs/'INPUTS.json'),
        query_manifest_sha256=base.digest(inputs/'ENTITY_QUERIES.json'),
        source_sha256=sources(), parameter_sha256=gp['parameter_sha256'],
        mechanisms={
            'direct':'Append 14 shared gene-overlap and normalized pathway-program features to the frozen 36 graph features.',
            'cross_rna':'Append 8 additional neighbor-program features. Apply the same normalized lncRNA–miRNA propagation to both RNA types; keep indirect associations separate.'},
        program_projection='Binary direct gene relations. Downweight genes by sqrt(number of annotated pathways) and pathways by sqrt(number of genes); L2 normalize program vectors. No RNA–drug labels used in annotation features.',
        input_policy='Preserve old 36-column raw/quantile baseline inputs. Append raw annotation features with identical meaning in both tasks. Official TabICL preprocessing fits only on the selected context rows.',
        composition='For each variant, retain the frozen task-specific graph-context composition. Then (1-alpha)*historical_graph_baseline + alpha*augmented_graph_model; a single alpha and variant apply to both tasks.',
        parameter_updates=0, no_test_evaluation=True,
        gate='LncDrug seed-mean AUC and AP must each be >= historical graph-context baseline, with no negative tolerance. MiDrug means must both improve, each with at least 2/3 positive paired seed differences. Macro drug AP is reported separately.',
        selection='Among eligible shared recipes maximize mean MiDrug (AUC+AP)/2; exact ties prefer lower alpha then recipe ID. If none qualify select baseline and stop before outer arrays. Freeze once; no outer reselection.',
        replay_atol=1e-6, replay_rtol=1e-6, query_invariance_atol=5e-5, query_invariance_rtol=1e-5,
        limitations=['Historical inner/outer validation has repeatedly participated in development; inference seeds are not independent data splits.',
                    'ENCORI bindings and cross-RNA neighborhood programs do not establish functional regulation or pathway activity.',
                    'miRTarBase/LncTarD were inaccessible; RNA and drug target/pathway coverage remains incomplete.',
                    'Current annotations are transductive entity attributes, not a temporally isolated resource snapshot.',
                    'Additional features change TabICL feature permutations; this first information-increment trial is not a controlled attribution of each biological feature.']))
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name),out/'source_snapshot'/name)
    base.write_json(out/'protocol.json',p)
    shutil.copy2(inputs/'COVERAGE.json',out/'INPUT_COVERAGE.json')
    print('PROTOCOL_FROZEN',out,p['sha256'],flush=True)


def protocol(out):
    p=base.check_seal(base.read_json(Path(out)/'protocol.json'))
    if p['source_sha256'] != sources():
        raise ValueError('Frozen source changed; start a new trial')
    for name,key in [('INPUTS.json','input_manifest_sha256'),('ENTITY_QUERIES.json','query_manifest_sha256')]:
        if base.digest(Path(p['inputs_dir'])/name) != p[key]:
            raise ValueError('Frozen input manifest changed')
    if base.digest(Path(p['parent_dir'])/'SELECTION.json') != p['parent_selection_sha256']:
        raise ValueError('Graph-context reference choice changed')
    return p


def selected(out,p):
    s=base.check_seal(base.read_json(Path(out)/'SELECTION.json'))
    if s['protocol_sha256'] != p['sha256'] or not s['eligible'] or p['smoke']:
        raise ValueError('No eligible formal inner result; outer loading prohibited')
    for name, expected in s['artifacts'].items():
        if base.digest(Path(out)/name) != expected:
            raise ValueError('Inner artifact changed after selection')
    return s


def stage_data(out,p,task,phase):
    if phase not in ('inner','outer') or task not in p['tasks']:
        raise ValueError('Unregistered task/phase')
    if phase == 'outer':
        selected(out,p)
    gp=graph.protocol(p['parent_dir'])
    if gp['sha256'] != p['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    a=graph.features(p['parent_dir'],gp,task,phase)
    if p['smoke']:
        keep=np.sort(np.concatenate([np.flatnonzero(a['fold_ids']==fold)[
            base.balanced_subset(a['y_train'][a['fold_ids']==fold],32,20260913+fold)] for fold in range(5)]))
        for prefix in ('x','raw','y'):
            a[prefix+'_train']=a[prefix+'_train'][keep]
        a['pairs_train'],a['fold_ids']=a['pairs_train'][:,keep],a['fold_ids'][keep]
        keep=base.balanced_subset(a['y_val'],40,20260913)
        for prefix in ('x','raw','y'):
            a[prefix+'_val']=a[prefix+'_val'][keep]
        a['pairs_val']=a['pairs_val'][:,keep]
        for prefix in ('x','raw'):
            a[prefix+'_query_views']=a[prefix+'_query_views'][:,keep]
    base.validate_arrays(a)
    return a,gp


def kinds(p,task):
    arm=p['graph_recipes'][task]['arm']
    return ['full'] if arm in ('frozen','query_bag') else ['full']+['fold'+str(i) for i in range(5)]


def job_arrays(p,task,a,kind,extra_train,extra_val):
    prefix='raw' if p['input_configs'][task]['input_view']=='raw' else 'x'
    x,full,views=a[prefix+'_train'],a[prefix+'_val'],a[prefix+'_query_views']
    arm=p['graph_recipes'][task]['arm']
    if kind=='full':
        keep=np.arange(len(x))
        q=np.concatenate([full[None],views]) if arm=='query_bag' else full[None]
    else:
        fold=int(kind[-1])
        keep=np.flatnonzero(a['fold_ids']==fold)
        q=full[None] if arm=='fold_full' else views[fold:fold+1]
    x=np.concatenate([x[keep],extra_train[keep]],axis=1).astype(np.float32)
    q=np.concatenate([q,np.broadcast_to(extra_val,(len(q),*extra_val.shape))],axis=2).astype(np.float32)
    return x,a['y_train'][keep],q


def directory(out,phase,task,variant,seed,kind):
    return Path(out)/phase/task/variant/('s'+str(seed))/kind


def checked_result(dest,p):
    r=base.read_json(Path(dest)/'result.json')
    if r['protocol_sha256'] != p['sha256'] or not r['replay_passed']:
        raise ValueError('Unverified prediction result')
    for name, expected in r['artifacts'].items():
        if base.digest(Path(dest)/name) != expected:
            raise ValueError('Saved prediction or model changed')
    return r


def run(args):
    import torch
    from tabicl import TabICLClassifier
    out=Path(args.out_dir)
    p=protocol(out)
    variants=[selected(out,p)['recipe']['variant']] if args.phase=='outer' else p['variants']
    a,gp=stage_data(out,p,args.task,args.phase)
    tp=base.read_json(Path(gp['parent_dir'])/'protocol.json')
    encoder=feature.ProgramPairs.load(p['inputs_dir'])
    seeds=p['seeds'] if args.seed is None else [args.seed]
    if any(s not in p['seeds'] for s in seeds):
        raise ValueError('Unregistered seed')
    torch.cuda.set_device(args.device)
    for variant in variants:
        extra_train=encoder.transform(args.task,a['pairs_train'],variant)
        extra_val=encoder.transform(args.task,a['pairs_val'],variant)
        for seed in seeds:
            for kind in kinds(p,args.task):
                dest=directory(out,args.phase,args.task,variant,seed,kind)
                if (dest/'result.json').exists():
                    checked_result(dest,p)
                    continue
                start=time.monotonic()
                x,y,q=job_arrays(p,args.task,a,kind,extra_train,extra_val)
                torch.cuda.reset_peak_memory_stats(args.device)
                clf=base.fit_context(base.estimator(dict(n_estimators=p['members']),seed,tp,args.device),x,y)
                if base.parameter_digest(clf) != p['parameter_sha256']:
                    raise ValueError('Official neural weights changed')
                flat=q.reshape(-1,x.shape[1])
                score=base.query_scores(clf,flat).reshape(q.shape[:2])
                if base.parameter_digest(clf) != p['parameter_sha256']:
                    raise ValueError('Inference modified neural weights')
                dest.mkdir(parents=True,exist_ok=True)
                clf.save(dest/'fitted.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=False)
                np.savez_compressed(dest/'scores.npz',views=score,y_val=a['y_val'],pairs_val=a['pairs_val'])
                del clf
                gc.collect()
                torch.cuda.empty_cache()
                clf=TabICLClassifier.load(dest/'fitted.pkl',device=args.device)
                if base.parameter_digest(clf) != p['parameter_sha256']:
                    raise ValueError('Reload changed neural weights')
                replay=base.query_scores(clf,flat).reshape(q.shape[:2])
                np.testing.assert_allclose(replay,score,atol=p['replay_atol'],rtol=p['replay_rtol'])
                invariance={}
                if p['smoke']:
                    reverse=base.query_scores(clf,flat[::-1])[::-1]
                    split=np.concatenate([base.query_scores(clf,part) for part in np.array_split(flat,3)])
                    for name,pred in [('reverse',reverse),('split',split)]:
                        np.testing.assert_allclose(pred,score.ravel(),atol=p['query_invariance_atol'],rtol=p['query_invariance_rtol'])
                        invariance[name]=float(np.max(np.abs(pred-score.ravel())))
                r=dict(protocol_sha256=p['sha256'],phase=args.phase,task=args.task,variant=variant,
                       seed=seed,kind=kind,context_rows=len(y),input_columns=x.shape[1],query_views=len(q),
                       members=p['members'],parameter_sha256=p['parameter_sha256'],replay_passed=True,
                       replay_max_abs=float(np.max(np.abs(replay-score))),smoke_invariance=invariance,
                       seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device),
                       artifacts={name:base.digest(dest/name) for name in ['fitted.pkl','scores.npz']})
                base.write_json(dest/'result.json',r)
                print('VERIFIED',args.phase,args.task,variant,seed,kind,'seconds',round(r['seconds'],1),flush=True)
                del clf
                gc.collect()
                torch.cuda.empty_cache()


def augmented_score(out,p,a,phase,task,variant,seed):
    scores={}
    artifacts={}
    for kind in kinds(p,task):
        dest=directory(out,phase,task,variant,seed,kind)
        r=checked_result(dest,p)
        if (r['phase'],r['task'],r['variant'],r['seed'],r['kind']) != (phase,task,variant,seed,kind):
            raise ValueError('Wrong prediction identity')
        with np.load(dest/'scores.npz',allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['y_val'],a['y_val'])
            np.testing.assert_array_equal(archive['pairs_val'],a['pairs_val'])
            scores[kind]=archive['views'].copy()
        artifacts[str((dest/'result.json').relative_to(out))]=base.digest(dest/'result.json')
    recipe=p['graph_recipes'][task]
    full=scores['full'][0]
    if recipe['arm']=='frozen':
        return full,artifacts
    extra=(scores['full'][1:].mean(0) if recipe['arm']=='query_bag' else
           np.stack([scores['fold'+str(i)][0] for i in range(5)]).mean(0))
    return (1-recipe['weight'])*full+recipe['weight']*extra,artifacts


def report(args):
    import pandas as pd
    out=Path(args.out_dir)
    p=protocol(out)
    if p['smoke']:
        raise ValueError('Smoke predictions are not performance evidence')
    if args.phase=='outer':
        chosen=selected(out,p)
        configs=[p['recipes'][0],chosen['recipe']]
    else:
        configs=p['recipes']
    rows,artifacts=[],{}
    for task in p['tasks']:
        a,gp=stage_data(out,p,task,args.phase)
        for seed in p['seeds']:
            old_full,old_components,_=graph.components(Path(p['parent_dir']),gp,args.phase,task,seed,a)
            baseline=graph_kernels.combine(old_full,old_components,p['graph_recipes'][task])
            variants={}
            for variant in sorted({c['variant'] for c in configs if c['variant']!='baseline'}):
                variants[variant],hashes=augmented_score(out,p,a,args.phase,task,variant,seed)
                artifacts.update(hashes)
            predictions={}
            for cfg in configs:
                prediction=(baseline if cfg['variant']=='baseline' else
                            (1-cfg['alpha'])*baseline+cfg['alpha']*variants[cfg['variant']])
                predictions[cfg['id']]=prediction
                rows.append(dict(task=task,seed=seed,recipe=cfg['id'],variant=cfg['variant'],alpha=cfg['alpha'],
                                 **base.metrics(a['y_val'],prediction,a['pairs_val'])))
            dest=out/'predictions'/args.phase/task/('s'+str(seed))
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'scores.npz',**predictions,y_val=a['y_val'],pairs_val=a['pairs_val'])
            artifacts[str((dest/'scores.npz').relative_to(out))]=base.digest(dest/'scores.npz')
    table=pd.DataFrame(rows)
    table.to_csv(out/(args.phase+'_metrics.csv'),index=False)
    summary=table.groupby(['task','recipe'],sort=False)[list(base.METRICS)].mean().reset_index()
    for metric in base.METRICS:
        refs=summary[summary.recipe=='baseline'].set_index('task')[metric].to_dict()
        summary['delta_'+metric]=summary[metric]-summary['task'].map(refs)
    summary.to_csv(out/(args.phase+'_summary.csv'),index=False)
    assessments=[feature.assess(rows,cfg,p['seeds']) for cfg in configs if cfg['variant']!='baseline']
    base.write_json(out/(args.phase+'_ASSESSMENTS.json'),assessments)
    for file in (args.phase+'_metrics.csv',args.phase+'_summary.csv',args.phase+'_ASSESSMENTS.json'):
        artifacts[file]=base.digest(out/file)
    if args.phase=='inner':
        eligible=[item for item in assessments if item['eligible']]
        choice=(min(eligible,key=lambda item:(
                    -(item['metrics']['MiDrug']['auc']['mean']+item['metrics']['MiDrug']['ap']['mean'])/2,
                    item['recipe']['alpha'],item['recipe']['id'])) if eligible else None)
        selection=base.seal(dict(protocol_sha256=p['sha256'],eligible=bool(choice),
            recipe=choice['recipe'] if choice else p['recipes'][0],artifacts=artifacts,
            created_at_utc=datetime.now(timezone.utc).isoformat()))
        if (out/'SELECTION.json').exists():
            previous=base.read_json(out/'SELECTION.json')
            if previous['recipe'] != selection['recipe'] or previous['eligible'] != selection['eligible']:
                raise ValueError('Recomputed selection differs')
        else:
            base.write_json(out/'SELECTION.json',selection)
    else:
        selection=chosen
    lines=['# RNA–gene–pathway TabICLv2 增量实验','',
           f'阶段：{args.phase}。三个推理种子 666/0/1，32 成员；历史图上下文为基线。',
           '两任务共用新增特征定义和同一个融合系数。所有新增上下文状态保存后均重载重放。','',
           '|任务|方案|AUC|AP|按药物宏 AP|ΔAUC|ΔAP|Δ宏 AP|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for _,r in summary.iterrows():
        lines.append(f"|{r['task']}|{r['recipe']}|{r['auc']:.9f}|{r['ap']:.9f}|{r['macro_drug_ap']:.9f}|{r['delta_auc']:+.9f}|{r['delta_ap']:+.9f}|{r['delta_macro_drug_ap']:+.9f}|")
    lines += ['',f"内部冻结选择：{selection['recipe']['id']}；eligible={selection['eligible']}。",'',
              'LncDrug 平均 AUC、AP 逐项不下降为硬约束，无负容差；MiDrug 两项均提高且各至少 2/3 种子为正。按药物宏 AP 单列。',
              '历史划分已反复用于开发，推理种子不是独立数据重复。本实验没有读取原始测试集。',
              '数据为结合/相互作用及间接邻域通路关联，不代表已验证调控方向或细胞系通路活性。',
              'miRTarBase/LncTarD 未取得；真实覆盖率见 INPUT_COVERAGE.json。']
    (out/('REPORT_'+args.phase+'.md')).write_text('\n'.join(lines)+'\n')
    base.write_json(out/('VALIDATION_'+args.phase+'.json'),dict(protocol_sha256=p['sha256'],
                   phase=args.phase,all_states_replayed=True,source_unchanged=True,no_test_evaluation=True,
                   eligible=bool(assessments and any(x['eligible'] for x in assessments)),
                   selection=selection['recipe'],artifacts=artifacts))
    print('REPORT',args.phase,'selected',selection['recipe'],'eligible',selection['eligible'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--out-dir',default=DEFAULT_OUT)
    parser.add_argument('--inputs-dir',default=DEFAULT_INPUTS)
    parser.add_argument('--parent-dir',default=DEFAULT_PARENT)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--phase',choices=('inner','outer'),default='inner')
    parser.add_argument('--task',choices=('MiDrug','LncDrug'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='run':run(args)
    else:report(args)


if __name__=='__main__':
    main()
