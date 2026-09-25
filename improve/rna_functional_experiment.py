"""Frozen-budget RNA representation comparisons against accepted TabICLv2."""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from improve import tabular_foundation as base
from improve import tabicl_context_program as accepted
from improve import tabicl_context_program_modules as evidence
from improve import tabicl_pathway as operations
from improve import rna_functional_modules as features
from improve import rna_functional_inputs as inputs
from improve.sampling_bias_audit import matched_weights, weighted_metrics


DEFAULT_OUT = 'improve/results/rna_functional_experiment_20260914'
DEFAULT_REFERENCE = 'improve/results/tabicl_pathway_context_20260913'
OWN = ('rna_functional_experiment.py','rna_functional_modules.py','rna_functional_inputs.py',
       'test_rna_functional_modules.py','test_rna_functional_inputs.py')


def sources():
    return {name:base.digest(Path(__file__).with_name(name)) for name in OWN}


def prepare(args):
    out,reference,input_dir = (Path(s).resolve() for s in (args.out_dir,args.reference_dir,args.inputs_dir))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    rp = accepted.protocol(reference)
    choice = accepted.selected(reference,rp)
    candidate = base.read_json(reference/'CANDIDATE.json')
    if rp['smoke'] or not candidate['accepted'] or choice['recipe']['variant']!='tiered' or choice['recipe']['alpha']!=.5:
        raise ValueError('Expected the accepted tiered 0.5 reference')
    inputs.verify(input_dir/'assembled')
    encoder = features.Representations.load(input_dir)
    dictionary = base.read_json(Path(rp['inputs_dir'])/'dictionary.json')
    for task,table in encoder.tables.items():
        np.testing.assert_array_equal(table['names'],dictionary['rna_names'][task])
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),smoke=args.smoke,
        tasks=rp['tasks'],seeds=[666] if args.smoke else [666,0,1],members=2 if args.smoke else 32,
        variants=['functional_moments','functional_shuffled'] if args.smoke else list(features.VARIANTS),
        reference_dir=str(reference),reference_protocol_sha256=rp['sha256'],reference_recipe=choice['recipe'],
        reference_files={f:base.digest(reference/f) for f in ('protocol.json','SELECTION.json','CANDIDATE.json','VALIDATION_inner.json','VALIDATION_outer.json')},
        parent_dir=rp['parent_dir'],parent_protocol_sha256=rp['parent_protocol_sha256'],
        input_configs=rp['input_configs'],graph_recipes=rp['graph_recipes'],inputs_dir=rp['inputs_dir'],
        representation_dir=str(input_dir),representation_manifest_sha256=base.digest(input_dir/'assembled/VALIDATION.json'),
        source_sha256=sources(),parameter_sha256=rp['parameter_sha256'],
        representation_dimension=features.DIMENSION,extra_columns=26+8+features.DIMENSION,
        input_columns=36+26+8+features.DIMENSION,
        compression='Per fitted support context, standardize and PCA on unique covered support RNAs only; whiten with variance floor 1e-10, clip to [-10,10], pad to 16D. Never fit on query-only RNA or labels.',
        control='Both encoders use matched local inputs and coverage. Functional mean and mean+SD use the exact same local units. Shuffle preserves RNA coverage, support membership, and train-derived length/GC/unit-count tertile strata; simple metadata stays attached.',
        composition='Every new score is 0.5*frozen graph baseline + 0.5*extended tiered branch, retaining the accepted graph recipes. No blend-strength search. Reference remains 0.5*graph+0.5*original tiered branch.',
        parameter_updates=0,hyperparameter_searches=0,no_outer_evaluation=True,no_original_test_evaluation=True,
        promotion='This is a development information experiment. Never replace the accepted model automatically or score historical outer arrays.',
        performance_gate='Functional moments: LncDrug AUC and AP >= reference in every seed; MiDrug mean AUC and AP > reference, each improving in >=2/3 seeds.',
        information_gate='In BOTH tasks, functional moments must beat generic and shuffled on mean AP for covered RNA pairs AND on covered drug-matched AP. A descriptive gate only, not significance or source generalization.',
        replay_atol=1e-6,replay_rtol=1e-6,query_invariance_atol=5e-5,query_invariance_rtol=1e-5,
        limitations=['Historical inner data have been repeatedly used for development. Seeds are inference randomness, not independent biological replicates.',
            'Unknown association pairs are not experimentally verified negatives.',
            'Auxiliary site provenance is retained but this historical association experiment is not source-disjoint.',
            'miRBind is a binding-site model, not its transferred repression model. PARNET uses a historical eCLIP checkpoint.',
            'RNA-level pooling does not yet model a particular drug target or cell-dependent mechanism.']))
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name),out/'source_snapshot'/name)
    base.write_json(out/'protocol.json',p)
    print('PROTOCOL_FROZEN',out,p['sha256'],flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out)/'protocol.json'))
    if p['source_sha256']!=sources():
        raise ValueError('Frozen experiment implementation changed')
    for name,digest in p['reference_files'].items():
        if base.digest(Path(p['reference_dir'])/name)!=digest:
            raise ValueError('Accepted reference changed')
    if accepted.protocol(p['reference_dir'])['sha256']!=p['reference_protocol_sha256']:
        raise ValueError('Reference protocol mismatch')
    if base.digest(Path(p['representation_dir'])/'assembled/VALIDATION.json')!=p['representation_manifest_sha256']:
        raise ValueError('Representation manifest changed')
    return p


def stage(out,p,task):
    return accepted.stage_data(out,p,task,'inner')


def reference_scores(p,task,seed,arrays):
    directory = Path(p['reference_dir'])
    v = base.read_json(directory/'VALIDATION_inner.json')
    path = 'predictions/inner/'+task+'/s'+str(seed)+'/scores.npz'
    if base.digest(directory/path)!=v['artifacts'].get(path):
        raise ValueError('Unverified reference predictions')
    with np.load(directory/path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['pairs_val'],arrays['pairs_val'])
        np.testing.assert_array_equal(z['y_val'],arrays['y_val'])
        return z[p['reference_recipe']['id']],z['baseline']


def run(args):
    import torch
    from tabicl import TabICLClassifier
    out = Path(args.out_dir)
    p = protocol(out)
    a,gp = stage(out,p,args.task)
    tp = base.read_json(Path(gp['parent_dir'])/'protocol.json')
    encoder = features.Representations.load(p['representation_dir'])
    tiered = evidence.ContextPrograms.load(p['inputs_dir'])
    original_train = tiered.transform(args.task,a['pairs_train'],'tiered')
    original_query = tiered.transform(args.task,a['pairs_val'],'tiered')
    variants = p['variants'] if args.variant is None else [args.variant]
    seeds = p['seeds'] if args.seed is None else [args.seed]
    if any(v not in p['variants'] for v in variants) or any(s not in p['seeds'] for s in seeds):
        raise ValueError('Unregistered variant or inference seed')
    kinds = operations.kinds(p,args.task)
    if p['smoke']:
        kinds = [k for k in kinds if k in ('full','fold0')]
    torch.cuda.set_device(args.device)
    for variant in variants:
        for kind in kinds:
            keep = np.arange(len(a['y_train'])) if kind=='full' else np.flatnonzero(a['fold_ids']==int(kind[-1]))
            extra,state,metadata = encoder.transform(args.task,variant,a['pairs_train'][0,keep],p['representation_dimension'])
            train = np.concatenate([original_train,extra[a['pairs_train'][0]]],axis=1)
            query = np.concatenate([original_query,extra[a['pairs_val'][0]]],axis=1)
            x,y,q = operations.job_arrays(p,args.task,a,kind,train,query)
            if x.shape[1]!=p['input_columns']:
                raise ValueError('Ablation column budget changed')
            for seed in seeds:
                dest = operations.directory(out,'inner',args.task,variant,seed,kind)
                if (dest/'result.json').exists():
                    operations.checked_result(dest,p)
                    continue
                start = time.monotonic()
                torch.cuda.reset_peak_memory_stats(args.device)
                clf = base.fit_context(base.estimator(dict(n_estimators=p['members']),seed,tp,args.device),x,y)
                if base.parameter_digest(clf)!=p['parameter_sha256']:
                    raise ValueError('TabICL weights changed')
                flat = q.reshape(-1,x.shape[1])
                score = base.query_scores(clf,flat).reshape(q.shape[:2])
                dest.mkdir(parents=True,exist_ok=True)
                clf.save(dest/'fitted.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=False)
                np.savez_compressed(dest/'scores.npz',views=score,y_val=a['y_val'],pairs_val=a['pairs_val'])
                np.savez_compressed(dest/'projection.npz',**state)
                base.write_json(dest/'projection.json',metadata)
                del clf
                gc.collect()
                torch.cuda.empty_cache()
                clf = TabICLClassifier.load(dest/'fitted.pkl',device=args.device)
                replay = base.query_scores(clf,flat).reshape(q.shape[:2])
                np.testing.assert_allclose(replay,score,atol=p['replay_atol'],rtol=p['replay_rtol'])
                if base.parameter_digest(clf)!=p['parameter_sha256']:
                    raise ValueError('TabICL inference/reload changed pretrained parameters')
                checks = {}
                if p['smoke']:
                    for name,value in dict(reverse=base.query_scores(clf,flat[::-1])[::-1],
                            split=np.concatenate([base.query_scores(clf,part) for part in np.array_split(flat,3)])).items():
                        np.testing.assert_allclose(value,score.ravel(),atol=p['query_invariance_atol'],rtol=p['query_invariance_rtol'])
                        checks[name] = float(np.max(np.abs(value-score.ravel())))
                result = dict(protocol_sha256=p['sha256'],phase='inner',task=args.task,variant=variant,seed=seed,kind=kind,
                    replay_passed=True,replay_max_abs=float(np.max(np.abs(score-replay))),smoke_invariance=checks,
                    context_rows=len(y),input_columns=x.shape[1],query_views=len(q),parameter_sha256=p['parameter_sha256'],
                    seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device),
                    artifacts={name:base.digest(dest/name) for name in ('fitted.pkl','scores.npz','projection.npz','projection.json')})
                base.write_json(dest/'result.json',result)
                print('VERIFIED_RNA',args.task,variant,seed,kind,'seconds',round(result['seconds'],1),flush=True)
                del clf
                gc.collect()
                torch.cuda.empty_cache()


def diagnostic_masks(a,table):
    pairs,y = a['pairs_val'],a['y_val']
    covered = table['present'][pairs[0]]
    masks = dict(covered=covered.astype(float),uncovered=(~covered).astype(float))
    for name,mask in [('all',np.ones(len(y),bool)),('covered',covered)]:
        weights = np.zeros(len(y))
        weights[mask],_ = matched_weights(y[mask],pairs[1,mask])
        masks[name+'_drug_matched'] = weights
    degree = np.bincount(a['pairs_train'][1,a['y_train']==1],minlength=int(pairs[1].max())+1)
    threshold = float(np.median(degree[degree>0]))
    masks['low_frequency_drug'] = (degree[pairs[1]]<=threshold).astype(float)
    return masks,dict(covered_query_pairs=int(covered.sum()),covered_query_positives=int(y[covered].sum()),
        covered_query_rnas=len(np.unique(pairs[0,covered])),total_query_pairs=len(y),
        low_frequency_drug_degree_threshold=threshold)


def report(args):
    out = Path(args.out_dir)
    p = protocol(out)
    if p['smoke'] or (out/'VALIDATION.json').exists():
        raise ValueError('Only a complete, unreported formal experiment may be reported')
    encoder = features.Representations.load(p['representation_dir'])
    rows,diagnostics,coverage,artifacts = [],[],{},{}
    for task in p['tasks']:
        a,_ = stage(out,p,task)
        masks,coverage[task] = diagnostic_masks(a,encoder.tables[task])
        for seed in p['seeds']:
            baseline,graph = reference_scores(p,task,seed,a)
            predictions = dict(baseline=baseline)
            for variant in p['variants']:
                raw,hashes = operations.augmented_score(out,p,a,'inner',task,variant,seed)
                artifacts.update(hashes)
                predictions[variant] = .5*graph+.5*raw
            for name,score in predictions.items():
                rows.append(dict(task=task,seed=seed,variant=name,**base.metrics(a['y_val'],score,a['pairs_val'])))
                for group,weight in masks.items():
                    diagnostics.append(dict(task=task,seed=seed,variant=name,subset=group,
                        pairs=int((weight>0).sum()),positives=int(a['y_val'][weight>0].sum()),
                        **weighted_metrics(a['y_val'],score,weight)))
            dest = out/'predictions'/task/('s'+str(seed))
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'scores.npz',**predictions,y_val=a['y_val'],pairs_val=a['pairs_val'])
            artifacts[str((dest/'scores.npz').relative_to(out))] = base.digest(dest/'scores.npz')
    table,diag = pd.DataFrame(rows),pd.DataFrame(diagnostics)
    table.to_csv(out/'metrics.csv',index=False)
    diag.to_csv(out/'diagnostics.csv',index=False)
    means = table.groupby(['task','variant'])[list(base.METRICS)].mean()
    dm = diag.groupby(['task','variant','subset'])[['auc','ap']].mean()
    gates = {}
    for task in p['tasks']:
        t = table[table.task==task].set_index(['variant','seed'])
        delta = {metric:np.array([t.loc[('functional_moments',s),metric]-t.loc[('baseline',s),metric] for s in p['seeds']])
                 for metric in ('auc','ap')}
        performance = all(np.all(v>=0) for v in delta.values()) if task=='LncDrug' else all(v.mean()>0 and (v>0).sum()>=2 for v in delta.values())
        comparisons = {subset:{other:float(dm.loc[(task,'functional_moments',subset),'ap']-dm.loc[(task,other,subset),'ap'])
            for other in ('generic','functional_shuffled')} for subset in ('covered','covered_drug_matched')}
        gates[task] = dict(performance_gate=bool(performance),per_seed_delta={k:v.tolist() for k,v in delta.items()},
            information_ap_deltas=comparisons,information_gate=all(v>0 for comparisons_ in comparisons.values() for v in comparisons_.values()))
    outcome = dict(task_gates=gates,performance_gate=all(g['performance_gate'] for g in gates.values()),
        information_gate=all(g['information_gate'] for g in gates.values()),accepted_reference_replaced=False,
        historical_outer_evaluated=False,original_test_evaluated=False,coverage=coverage)
    base.write_json(out/'ASSESSMENT.json',outcome)
    for name in ('metrics.csv','diagnostics.csv','ASSESSMENT.json'):
        artifacts[name] = base.digest(out/name)
    base.write_json(out/'VALIDATION.json',dict(protocol_sha256=p['sha256'],passed=True,
        fitted_states=len([name for name in artifacts if name.endswith('/result.json')]),artifacts=artifacts,**outcome))
    lines = ['固定 TabICLv2 的 RNA 功能表示对照已完成。各新分支均为 86 列：36 个图特征、26 个原分层证据、8 个简单统计、16 个 RNA 表示。',
        '', '同一支持/查询样本、32 个推理成员、种子 666/0/1；每个支持上下文单独在其 RNA 上拟合压缩。保留当前 0.5 图分支权重，无混合系数搜索。',
        '', '|任务|表示|AUC|AP|药物宏 AP|ΔAP（百分点）|', '|---|---|---:|---:|---:|---:|']
    for task in p['tasks']:
        for variant in ['baseline',*p['variants']]:
            r = means.loc[(task,variant)]
            delta = 100*(r.ap-means.loc[(task,'baseline'),'ap'])
            lines.append('|%s|%s|%.9f|%.9f|%.9f|%+.6f|'%(task,variant,r.auc,r.ap,r.macro_drug_ap,delta))
    lines += ['', '共同性能条件通过：'+str(outcome['performance_gate'])+'；功能信息描述性条件通过：'+str(outcome['information_gate'])+'。',
        '', '性能条件和功能语义条件分别判断；两项均不能代替来源泛化或显著性检验。功能均值/均值加标准差使用相同局部输入；打乱保留覆盖、支持实体成员关系和长度/GC/局部单元数量分层。',
        '', '完整覆盖组、药物频率匹配和长尾结果在 diagnostics.csv；逐种子及验收差值在 ASSESSMENT.json。没有评价历史外层或原始测试集，没有替换参考模型。',
        '', '三个种子只反映推理随机性。miRBind 输出不是实测抑制强度，PARNET 是历史 eCLIP 权重；静态 RNA 池化表示也不等于具体药物靶点或细胞状态机制。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(means.to_string(),flush=True)
    print('GATES',outcome,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','report'])
    parser.add_argument('--out-dir',default=DEFAULT_OUT)
    parser.add_argument('--reference-dir',default=DEFAULT_REFERENCE)
    parser.add_argument('--inputs-dir',default=str(inputs.DEFAULT_OUT))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--task',choices=['MiDrug','LncDrug'])
    parser.add_argument('--variant',choices=features.VARIANTS)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    globals()[args.command](args)


if __name__=='__main__':
    main()
