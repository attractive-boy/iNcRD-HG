"""Source-filtered RNA representation controls for observed context direction.

All five adapters have the same linear interaction design and regularization.
No source text, study ID or association graph is used as a predictor.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from improve import tabular_foundation as base
from improve import context_source_eval as context
from improve import rna_functional_inputs as inputs
from improve import rna_functional_modules as features
from improve.sampling_bias_audit import weighted_metrics


DEFAULT_OUT = Path('improve/results/context_functional_20260914')
TASKS = {'MiDir':'MiDrug','LncDir':'LncDrug'}
OWN = ('context_functional_eval.py','test_context_functional_eval.py',
       'rna_functional_modules.py','context_source_eval.py')


def auxiliary_pmids(value):
    # The miRTarBase CSV serializes some integer identifiers as e.g. 23622248.0.
    tokens = re.split(r'[;,|\s]+',str(value))
    normalized = [re.sub(r'^([1-9][0-9]*)\.0$',r'\1',token) for token in tokens]
    return context.pmids(';'.join(normalized))


def source_mask(rows, forbidden):
    keep = np.zeros(len(rows),bool)
    counts = Counter()
    for i,row in enumerate(rows):
        refs = auxiliary_pmids(row['pmids'])
        if not refs:
            counts['missing_source'] += 1
        elif set(refs) & forbidden:
            counts['overlap_with_validation_or_holdout'] += 1
        else:
            keep[i] = True
    counts['retained_sites'] = int(keep.sum())
    return keep,dict(counts)


def filtered_mirna(tables, rows, function, generic, keep):
    result = {key:value.copy() for key,value in tables.items()}
    n = len(result['present'])
    mean,sd,gmean = np.zeros((n,30),np.float32),np.zeros((n,30),np.float32),np.zeros((n,1280),np.float32)
    counts,gc = np.zeros(n,int),np.zeros(n,np.float32)
    indices = np.array([int(r['rna_index']) for r in rows])
    for i in np.unique(indices[keep]):
        ids = np.flatnonzero(keep & (indices==i))
        mean[i],sd[i],gmean[i] = function[ids].mean(0),function[ids].std(0),generic[ids].mean(0)
        counts[i],gc[i] = len(ids),np.mean([inputs.gc_fraction(rows[j]['target_sequence']) for j in ids])
    result.update(present=counts>0,unit_counts=counts,generic=gmean,functional_mean=mean,
                  functional_moments=np.concatenate([mean,sd],axis=1))
    result['simple'][:,5] = counts>0
    result['simple'][:,6] = np.log1p(counts)
    result['simple'][:,7] = gc
    return result


def prepare(args):
    out,source,representation = (Path(s).resolve() for s in (args.out_dir,args.source_dir,args.inputs_dir))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new source-functional experiment directory')
    sp = base.check_seal(base.read_json(source/'protocol.json'))
    manifest = base.read_json(source/'INPUTS.json')
    if manifest['protocol_sha256']!=sp['sha256']:
        raise ValueError('Source split manifest mismatch')
    forbidden = set()
    for task in TASKS:
        file = task+'/records.tsv'
        if base.digest(source/file)!=manifest['files'][file]:
            raise ValueError('Source record table changed')
        # Read source identity and partition only; no holdout labels are loaded.
        records = pd.read_csv(source/file,sep='\t',usecols=['pmid','partition'],dtype=str).fillna('')
        forbidden.update(p for value in records.loc[records.partition!='train','pmid'] for p in context.pmids(value))
    rows = inputs.site_rows(representation)
    keep,filter_stats = source_mask(rows,forbidden)
    encoder = features.Representations.load(representation)
    inputs.verify(representation/'mirbind')
    inputs.verify(representation/'generic_sites')
    with np.load(representation/'mirbind/representations.npz',allow_pickle=False) as z:
        function,site_ids = z['features'],z['site_ids']
    np.testing.assert_array_equal(site_ids,[r['site_id'] for r in rows])
    with np.load(representation/'generic_sites/representations.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(site_ids,z['site_ids'])
        generic = z['sequence_features'][z['site_sequence_indices']].reshape(len(rows),-1)
    encoder.tables['MiDrug'] = filtered_mirna(encoder.tables['MiDrug'],rows,function,generic,keep)
    retained_refs = {p for i,row in enumerate(rows) if keep[i] for p in auxiliary_pmids(row['pmids'])}
    if retained_refs & forbidden:
        raise ValueError('Auxiliary evidence still shares heldout sources')
    out.mkdir(parents=True)
    artifacts,coverage = {},{}
    for task,name in TASKS.items():
        file = 'representations_'+task+'.npz'
        np.savez_compressed(out/file,**encoder.tables[name])
        artifacts[file] = base.digest(out/file)
        coverage[task] = dict(total=len(encoder.tables[name]['present']),covered=int(encoder.tables[name]['present'].sum()),
                              local_units=int(encoder.tables[name]['unit_counts'].sum()))
    base.write_json(out/'SOURCE_FILTER.json',dict(forbidden_pmids=sorted(forbidden),filter_stats=filter_stats,
        retained_site_pmids=sorted(retained_refs),retained_site_pmid_overlap=0,coverage=coverage,
        pretrained_source_disjointness_claimed=False,
        limitation='This excludes shared-PMID auxiliary site selection evidence. Public encoder pretraining provenance is not fully independent or time-disjoint.'))
    artifacts['SOURCE_FILTER.json'] = base.digest(out/'SOURCE_FILTER.json')
    p = base.seal(dict(source_dir=str(source),source_protocol_sha256=sp['sha256'],
        source_input_manifest_sha256=base.digest(source/'INPUTS.json'),representation_dir=str(representation),
        original_representation_sha256=base.digest(representation/'assembled/VALIDATION.json'),
        files=artifacts,source_sha256={name:base.digest(Path(__file__).with_name(name)) for name in OWN},
        variants=list(features.VARIANTS),tasks=TASKS,dimension=16,seed=20260914,
        logistic=sp['logistic'],primary='novel_triple_source_heldout',
        design='Train-fitted one-hot RNA/drug/disease identities plus 8 standardized simple covariates + 16 support-only RNA PCs, plus each of those 24 values interacted with drug and disease one-hot columns. Same matrix width and regularization for all adapters.',
        inference='Fixed logistic model with C=1, liblinear, max_iter=2000. One fit per adapter and task, no hyperparameter search.',
        uncertainty='500 paired source-component bootstrap resamples of fixed validation predictions. Shared source sets for a triple are connected before resampling. Conditional on this split/model; not repeated independent experiments.',
        criteria='Report functional moments minus generic, simple, functional mean and shuffled on the same primary rows. Do not promote a model or change the heldout partition.',
        holdout_labels_loaded=False,holdout_evaluated=False,association_graph_used=False,
        intervention_effect_claimed=False,pretrained_source_disjointness_claimed=False))
    base.write_json(out/'protocol.json',p)
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name),out/'source_snapshot'/name)
    print('CONTEXT_FUNCTIONAL_FROZEN',p['sha256'],filter_stats,coverage,flush=True)


def design(identity,context_ids,numeric):
    return sparse.hstack([identity,sparse.csr_matrix(numeric)] +
        [context_ids.multiply(numeric[:,i,None]) for i in range(numeric.shape[1])],format='csr')


def bootstrap_groups(frame):
    comp = context.Components()
    refs = [json.loads(value) for value in frame.source_groups]
    for group in refs:
        if not group:
            raise ValueError('Missing source group in an evaluation row')
        for value in group:
            comp.union(group[0],value)
    groups = np.array([comp.root(group[0]) for group in refs])
    _,codes = np.unique(groups,return_inverse=True)
    return codes


def bootstrap_delta(y,score,reference,groups,repeats=500,seed=20260914):
    rng = np.random.default_rng(seed)
    count = int(groups.max())+1
    deltas = {'auc':[],'ap':[]}
    for _ in range(repeats):
        mass = np.bincount(rng.integers(0,count,count),minlength=count)[groups]
        a,b = weighted_metrics(y,score,mass),weighted_metrics(y,reference,mass)
        if a['auc'] is None:
            continue
        for metric in deltas:
            deltas[metric].append(a[metric]-b[metric])
    return dict(source_components=count,largest_component_rows=int(np.bincount(groups).max()),
        valid_resamples=len(deltas['auc']),
        intervals={metric:np.quantile(values,[.025,.975]).tolist() if values else None for metric,values in deltas.items()})


def run(args):
    out = Path(args.out_dir)
    if (out/'EVALUATION.json').exists():
        raise ValueError('Completed context-functional evaluation is immutable')
    p = base.check_seal(base.read_json(out/'protocol.json'))
    if p['source_sha256']!={name:base.digest(Path(__file__).with_name(name)) for name in OWN}:
        raise ValueError('Frozen source-functional implementation changed')
    for name,digest in p['files'].items():
        if base.digest(out/name)!=digest:
            raise ValueError('Source-filtered representation changed')
    source = Path(p['source_dir'])
    if base.digest(source/'INPUTS.json')!=p['source_input_manifest_sha256']:
        raise ValueError('Source partition changed')
    manifest = base.read_json(source/'INPUTS.json')
    rows,intervals,artifacts,projection_audit = [],{},{},{}
    for task,name in TASKS.items():
        data = {}
        for part in ('train','validation'):
            file = task+'/'+part+'.tsv'
            if base.digest(source/file)!=manifest['files'][file]:
                raise ValueError('Context partition rows changed')
            data[part] = pd.read_csv(source/file,sep='\t',dtype={'cid':str})
        train,query = data['train'],data['validation']
        with np.load(out/('representations_'+task+'.npz'),allow_pickle=False) as z:
            table = {k:z[k] for k in z.files}
        encoder = features.Representations({name:table})
        lookup = {value:i for i,value in enumerate(table['names'])}
        if not (set(train.ncRNA)|set(query.ncRNA)) <= set(lookup):
            raise ValueError('RNA names do not match representation order')
        train_ids,query_ids = train.ncRNA.map(lookup).to_numpy(int),query.ncRNA.map(lookup).to_numpy(int)
        identity,conditions = DictVectorizer(),DictVectorizer()
        xidentity = identity.fit_transform(train[context.KEYS].to_dict('records'))
        qidentity = identity.transform(query[context.KEYS].to_dict('records'))
        xcondition = conditions.fit_transform(train[['cid','disease']].to_dict('records'))
        qcondition = conditions.transform(query[['cid','disease']].to_dict('records'))
        simple_scaler = StandardScaler().fit(table['simple'][np.unique(train_ids)])
        predictions = context.scores(train,query,base.read_json(source/'protocol.json'))
        projections = {}
        for variant in p['variants']:
            start = time.monotonic()
            numeric,state,meta = encoder.transform(name,variant,train_ids,p['dimension'])
            numeric[:,:8] = simple_scaler.transform(numeric[:,:8])
            x,q = design(xidentity,xcondition,numeric[train_ids]),design(qidentity,qcondition,numeric[query_ids])
            model = LogisticRegression(**p['logistic']).fit(x,train.y)
            score = model.predict_proba(q)[:,list(model.classes_).index(1)]
            dest = out/task/variant
            dest.mkdir(parents=True,exist_ok=True)
            joblib.dump(dict(model=model,identity=identity,conditions=conditions,simple_scaler=simple_scaler),dest/'fitted.joblib')
            np.savez_compressed(dest/'projection.npz',**state)
            np.savez_compressed(dest/'numeric_table.npz',features=numeric,names=table['names'])
            loaded = joblib.load(dest/'fitted.joblib')['model']
            replay = loaded.predict_proba(q[::-1])[:,list(loaded.classes_).index(1)][::-1]
            np.testing.assert_allclose(score,replay,atol=1e-12,rtol=1e-12)
            predictions[variant] = score
            projections[variant] = dict(meta,design_columns=x.shape[1],iterations=model.n_iter_.tolist(),
                seconds=time.monotonic()-start,replay_max_abs=float(np.max(np.abs(score-replay))))
            for file in dest.iterdir():
                artifacts[str(file.relative_to(out))] = base.digest(file)
            print('CONTEXT_VERIFIED',task,variant,'columns',x.shape[1],'seconds',round(time.monotonic()-start,2),flush=True)
        projection_audit[task] = projections
        primary = query.novel_triple.to_numpy(bool)
        frame = query[primary].copy()
        covered = table['present'][query_ids]
        masks = dict(all_source_heldout=np.ones(len(query),bool),novel_triple_source_heldout=primary,
                     covered_novel_triple=primary&covered,uncovered_novel_triple=primary&~covered)
        for variant,score in predictions.items():
            for subset,mask in masks.items():
                rows.append(dict(task=task,variant=variant,subset=subset,**context.metrics(query[mask],score[mask])))
        codes = bootstrap_groups(frame)
        intervals[task] = {other:bootstrap_delta(frame.y.to_numpy(int),predictions['functional_moments'][primary],
            predictions[other][primary],codes) for other in ('simple','generic','functional_mean','functional_shuffled','context_logistic')}
        saved = query[['triple_id','y','novel_triple','source_groups']].copy()
        for variant,score in predictions.items():
            saved[variant] = score
        saved.to_csv(out/task/'validation_predictions.tsv',sep='\t',index=False)
        artifacts[task+'/validation_predictions.tsv'] = base.digest(out/task/'validation_predictions.tsv')
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    base.write_json(out/'BOOTSTRAP.json',intervals)
    base.write_json(out/'PROJECTIONS.json',projection_audit)
    for file in ('metrics.csv','BOOTSTRAP.json','PROJECTIONS.json'):
        artifacts[file] = base.digest(out/file)
    base.write_json(out/'EVALUATION.json',dict(protocol_sha256=p['sha256'],artifacts=artifacts,
        holdout_evaluated=False,auxiliary_shared_pmid_excluded=True,results=rows,bootstrap=intervals))
    selected = pd.DataFrame(rows).query("subset == 'novel_triple_source_heldout'")
    lines = ['来源分组下的功能表示对照已完成。主评价仅含训练侧未出现过的 RNA–drug–disease 三元组，目标仍是数据库方向描述。',
        '', '在两任务统一的 PMID 分组基础上，移除所有与验证/留出来源共享 PMID 的 miRNA 位点后再聚合表示。公共编码器的预训练数据未被证明来源独立。',
        '', '五种 RNA 适配器使用相同线性交互设计：RNA/药物/癌种身份、8 个简单统计、16 个训练侧拟合的 RNA 主成分，以及数值特征与药物/癌种身份的交互。固定 C=1，无参数搜索。',
        '', '|任务|表示|三元组|AUC|耐药描述 AP|敏感描述 AP|药物宏 AP|','|---|---|---:|---:|---:|---:|---:|']
    for r in selected.to_dict('records'):
        values = ['NA' if pd.isna(r[k]) else '%.6f'%r[k] for k in ('auc','ap','sensitive_ap','macro_drug_ap')]
        lines.append('|'+ '|'.join([r['task'],r['variant'],str(r['rows']),*values])+'|')
    lines += ['', 'BOOTSTRAP.json 给出固定验证预测的 500 次来源连通组配对重采样区间；它反映这次划分条件下的来源不确定性，不能当作独立重复实验。',
        '', '来源过滤和覆盖详见 SOURCE_FILTER.json；同维度、PCA 训练实体、打乱比例和重载误差详见 PROJECTIONS.json。未评分来源留出分区，未读取原始关联测试数组，未替换已接受模型。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(selected.to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--out-dir',default=str(DEFAULT_OUT))
    parser.add_argument('--source-dir',default=str(context.DEFAULT_OUT))
    parser.add_argument('--inputs-dir',default=str(inputs.DEFAULT_OUT))
    args = parser.parse_args()
    globals()[args.command](args)


if __name__=='__main__':
    main()
