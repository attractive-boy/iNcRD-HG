"""Source-grouped MiDir validation of site evidence and drug-conditioned pooling."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from improve import tabular_foundation as base
from improve import context_source_eval as context
from improve.context_functional_eval import design, bootstrap_groups, bootstrap_delta
from improve import site_evidence_features as features


DEFAULT_OUT = 'improve/results/site_evidence_source_20260914'
OWN = ('site_evidence_context.py','site_evidence_features.py','test_site_evidence_features.py',
       'context_functional_eval.py','context_source_eval.py','rna_functional_modules.py')


def sources():
    return {name:base.digest(Path(__file__).with_name(name)) for name in OWN}


def prepare(args):
    out,source,inputs = (Path(s).resolve() for s in (args.out_dir,args.source_dir,args.inputs_dir))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new source evaluation directory')
    features.verify(inputs)
    sp = base.check_seal(base.read_json(source/'protocol.json'))
    manifest = base.read_json(source/'INPUTS.json')
    files = {part:manifest['files']['MiDir/'+part+'.tsv'] for part in ('train','validation')}
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),source_dir=str(source),
        source_protocol_sha256=sp['sha256'],source_manifest_sha256=base.digest(source/'INPUTS.json'),
        partitions=files,representation_dir=str(inputs),representation_manifest_sha256=base.digest(inputs/'INPUTS.json'),
        source_sha256=sources(),variants=list(features.VARIANTS),tasks=['MiDir'],logistic=sp['logistic'],
        primary='novel_triple_source_heldout',projection_components=14,numeric_columns=24,seed=20260914,
        design='Train-fit RNA/drug/disease one-hot plus 8 standardized simple RNA covariates, 2 pooled coverage/gene-count fields, 14 support-only PCs; all 24 numeric columns interacted with drug and disease one-hot. Fixed liblinear C=1. Same rows, width and regularization for all 8 adapters.',
        primary_candidates={'evidence':['flat','generic','evidence_stats','evidence_shuffled'],
                            'target':['flat','evidence','target_stats','target_shuffled']},
        source_information_gate='Primary novel triples: candidate AP exceeds each registered control with paired source-component bootstrap 95% lower bound >0, and macro drug AP does not decrease versus flat. Also report AUC and sensitive-description AP. No multiple-testing-adjusted significance claim.',
        bootstrap_repeats=500,hyperparameter_searches=0,
        source_exclusion='Reuse frozen SOURCE_FILTER.json; remove shared-PMID auxiliary sites before any pooling or shuffling. Do not open reserved outcome labels.',
        association_graph_used=False,holdout_labels_loaded=False,holdout_evaluated=False,
        intervention_effect_claimed=False,pretrained_source_disjointness_claimed=False,automatic_promotion=False))
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name),out/'source_snapshot'/name)
    base.write_json(out/'protocol.json',p)
    print('SOURCE_PROTOCOL_FROZEN',p['sha256'],flush=True)


def run(args):
    out = Path(args.out_dir)
    if (out/'EVALUATION.json').exists():
        raise ValueError('Completed source evaluation is immutable')
    p = base.check_seal(base.read_json(out/'protocol.json'))
    if p['source_sha256']!=sources():
        raise ValueError('Frozen source implementation changed')
    if base.digest(Path(p['representation_dir'])/'INPUTS.json')!=p['representation_manifest_sha256']:
        raise ValueError('Representation manifest changed')
    features.verify(p['representation_dir'])
    source = Path(p['source_dir'])
    if base.digest(source/'INPUTS.json')!=p['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    if base.check_seal(base.read_json(source/'protocol.json'))['sha256']!=p['source_protocol_sha256']:
        raise ValueError('Source protocol changed')
    parts = {}
    for part in ('train','validation'):
        file = source/'MiDir'/(part+'.tsv')
        if base.digest(file)!=p['partitions'][part]:
            raise ValueError('Source partition changed: '+part)
        parts[part] = pd.read_csv(file,sep='\t',dtype={'cid':str})
    train,query = parts['train'],parts['validation']
    enc = features.SiteFeatures(p['representation_dir'],'source')
    rna_lookup = {n:i for i,n in enumerate(enc.sites['names'])}
    drug_lookup = {cid:i for i,cid in enumerate(enc.sites['cids'])}
    def pairs(frame):
        if not set(frame.ncRNA)<=set(rna_lookup) or not set(frame.cid)<=set(drug_lookup):
            raise ValueError('Unknown source RNA or drug identity')
        return np.stack([frame.ncRNA.map(rna_lookup).to_numpy(int),frame.cid.map(drug_lookup).to_numpy(int)])
    support,query_pairs = pairs(train),pairs(query)
    identity,conditions = DictVectorizer(),DictVectorizer()
    xidentity = identity.fit_transform(train[context.KEYS].to_dict('records'))
    qidentity = identity.transform(query[context.KEYS].to_dict('records'))
    xconditions = conditions.fit_transform(train[['cid','disease']].to_dict('records'))
    qconditions = conditions.transform(query[['cid','disease']].to_dict('records'))
    scaler = StandardScaler().fit(enc.sites['simple'][np.unique(support[0])])
    predictions = context.scores(train,query,base.read_json(source/'protocol.json'))
    artifacts,projections = {},{}
    for variant in p['variants']:
        start = time.monotonic()
        numeric,state,meta = enc.transform(variant,support)
        numeric[:,:8] = scaler.transform(numeric[:,:8])
        x = design(xidentity,xconditions,enc.select(numeric,support,variant))
        q = design(qidentity,qconditions,enc.select(numeric,query_pairs,variant))
        model = LogisticRegression(**p['logistic']).fit(x,train.y)
        score = model.predict_proba(q)[:,list(model.classes_).index(1)]
        dest = out/'MiDir'/variant
        dest.mkdir(parents=True,exist_ok=True)
        joblib.dump(dict(model=model,identity=identity,conditions=conditions,simple_scaler=scaler),dest/'fitted.joblib')
        np.savez_compressed(dest/'projection.npz',**state)
        np.savez_compressed(dest/'numeric_table.npz',features=numeric)
        loaded = joblib.load(dest/'fitted.joblib')['model']
        reverse = loaded.predict_proba(q[::-1])[:,list(loaded.classes_).index(1)][::-1]
        split = np.concatenate([loaded.predict_proba(q[ids])[:,list(loaded.classes_).index(1)]
                               for ids in np.array_split(np.arange(q.shape[0]),3)])
        np.testing.assert_allclose(score,reverse,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(score,split,atol=1e-12,rtol=1e-12)
        predictions[variant] = score
        projections[variant] = dict(meta,design_columns=x.shape[1],iterations=model.n_iter_.tolist(),
            seconds=time.monotonic()-start,replay_max_abs=float(np.max(np.abs(score-reverse))),
            split_max_abs=float(np.max(np.abs(score-split))))
        for file in dest.iterdir():
            artifacts[str(file.relative_to(out))] = base.digest(file)
        print('SOURCE_VERIFIED',variant,'columns',x.shape[1],flush=True)
    primary = query.novel_triple.to_numpy(bool)
    masks = dict(all_source_heldout=np.ones(len(query),bool),novel_triple_source_heldout=primary)
    for name in ('evidence','target'):
        pack,_ = enc.table(name,support[0])
        ids = query_pairs[0]*enc.n_drugs+query_pairs[1] if name=='target' else query_pairs[0]
        covered = np.isin(ids,pack['ids'])
        masks[name+'_covered_novel'] = primary & covered
        masks[name+'_uncovered_novel'] = primary & ~covered
    train_positive_degree = train.loc[train.y==1,'cid'].value_counts()
    threshold = float(np.median(train_positive_degree.to_numpy()))
    masks['low_frequency_drug_novel'] = primary & (query.cid.map(train_positive_degree).fillna(0).to_numpy()<=threshold)
    rows,drug_rows = [],[]
    for variant,score in predictions.items():
        for subset,mask in masks.items():
            rows.append(dict(task='MiDir',variant=variant,subset=subset,**context.metrics(query[mask],score[mask])))
        for cid in sorted(set(query.loc[primary,'cid'])):
            mask = primary & (query.cid.to_numpy()==cid)
            drug_rows.append(dict(variant=variant,cid=cid,train_positive_degree=int(train_positive_degree.get(cid,0)),
                **context.metrics(query[mask],score[mask])))
    frame = query[primary].copy()
    codes = bootstrap_groups(frame)
    intervals,gates = {},{}
    for candidate,controls in p['primary_candidates'].items():
        intervals[candidate] = {control:bootstrap_delta(frame.y.to_numpy(int),predictions[candidate][primary],
            predictions[control][primary],codes,repeats=p['bootstrap_repeats'],seed=p['seed']) for control in controls}
        current = context.metrics(frame,predictions[candidate][primary])
        flat = context.metrics(frame,predictions['flat'][primary])
        delta = {control:current['ap']-context.metrics(frame,predictions[control][primary])['ap'] for control in controls}
        gates[candidate] = dict(primary_ap_deltas=delta,macro_ap_delta_vs_flat=current['macro_drug_ap']-flat['macro_drug_ap'],
            source_information_gate=bool(all(delta[c]>0 and intervals[candidate][c]['intervals']['ap'][0]>0 for c in controls)
                and current['macro_drug_ap']>=flat['macro_drug_ap']))
    saved = query[['ncRNA','cid','disease','triple_id','y','novel_triple','source_groups']].copy()
    for variant,score in predictions.items():
        saved[variant] = score
    saved.to_csv(out/'validation_predictions.tsv',sep='\t',index=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(drug_rows).to_csv(out/'per_drug.csv',index=False)
    base.write_json(out/'BOOTSTRAP.json',intervals)
    base.write_json(out/'PROJECTIONS.json',projections)
    base.write_json(out/'ASSESSMENT.json',dict(gates=gates,low_frequency_threshold=threshold,
        coverage={name:int(mask.sum()) for name,mask in masks.items()},holdout_evaluated=False,accepted_reference_replaced=False))
    for file in ('validation_predictions.tsv','metrics.csv','per_drug.csv','BOOTSTRAP.json','PROJECTIONS.json','ASSESSMENT.json'):
        artifacts[file] = base.digest(out/file)
    base.write_json(out/'EVALUATION.json',dict(protocol_sha256=p['sha256'],artifacts=artifacts,gates=gates,
        auxiliary_shared_pmid_excluded=True,holdout_labels_loaded=False,holdout_evaluated=False,
        association_graph_used=False,intervention_effect_claimed=False,accepted_reference_replaced=False))
    selected = pd.DataFrame(rows).query("subset == 'novel_triple_source_heldout'")
    lines = ['MiDir 来源分组验证完成。主评价保留训练中未出现的 RNA–drug–disease 三元组；目标仍是数据库方向描述。',
        '', '八个适配器使用相同 24 个数值列预算和固定线性交互模型。所有辅助位点先按既有 PMID 排除表过滤；投影只拟合训练实体/配对。',
        '', '|表示|三元组|AUC|耐药描述 AP|敏感描述 AP|药物宏 AP|','|---|---:|---:|---:|---:|---:|']
    for row in selected.to_dict('records'):
        lines.append('|%s|%d|%.9f|%.9f|%.9f|%.9f|'%(row['variant'],row['rows'],row['auc'],row['ap'],row['sensitive_ap'],row['macro_drug_ap']))
    lines += ['', '主比较、覆盖和长尾阈值见 ASSESSMENT.json；500 次来源连通组配对重采样区间见 BOOTSTRAP.json。区间条件于当前划分/模型，不是独立重复实验。',
        '', '没有读取或评分来源保留集；没有使用关联图或来源/结论文本预测；没有宣称 RNA 干预药敏效应或预训练来源独立性。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(selected.to_string(index=False),flush=True)
    print('SOURCE_GATES',gates,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--out-dir',default=DEFAULT_OUT)
    parser.add_argument('--source-dir',default=str(context.DEFAULT_OUT))
    parser.add_argument('--inputs-dir',default=str(features.DEFAULT_OUT))
    args = parser.parse_args()
    globals()[args.command](args)
