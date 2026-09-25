"""Record-first, PMID-disjoint development evaluation of observed Effect labels.

This predicts database resistance/sensitivity descriptions. No record is
promoted to a verified RNA intervention label by a keyword or assay name.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from improve import tabular_foundation as base


ROOT = Path('/data1/licheng/incrd-hg/incrd-run')
DEFAULT_OUT = Path('improve/results/context_source_20260914')
TASKS = {'MiDir': 'miRNA', 'LncDir': 'lncRNA'}
PARTS = ('train', 'validation', 'holdout')
KEYS = ['ncRNA', 'cid', 'disease']


class Components:
    def __init__(self):
        self.parent = {}

    def root(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, first, second):
        a, b = self.root(first), self.root(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def pmids(value):
    tokens = [x.strip() for x in re.split(r'[;,|\s]+', str(value)) if x.strip()]
    if not tokens:
        return []
    if any(not re.fullmatch(r'[1-9][0-9]*', x) for x in tokens):
        raise ValueError('Unrecognized PMID value: ' + repr(value))
    return sorted(set(tokens))


def source_components(records):
    """Multi-PMID records and the same source across both tasks stay together."""
    comp = Components()
    for row in records:
        refs = pmids(row['pmid'])
        for ref in refs:
            comp.union(refs[0], ref)
    return {r['record_uid']: ('pmid:' + comp.root(pmids(r['pmid'])[0]) if pmids(r['pmid']) else '')
            for r in records}


def assign_groups(sizes, seed=20260914, fractions=(.6, .2, .2)):
    """Largest-first allocation using sizes only, never direction labels."""
    if len(sizes) < len(fractions):
        raise ValueError('Too few independent source groups')
    target = np.array(fractions, float) * sum(sizes.values())
    if not np.isclose(sum(fractions), 1.) or (target <= 0).any():
        raise ValueError('Invalid split fractions')
    rng = np.random.default_rng(seed)
    ties = dict(zip(sorted(sizes), rng.random(len(sizes))))
    loads, result = np.zeros(len(target)), {}
    for group in sorted(sizes, key=lambda g: (-sizes[g], ties[g], g)):
        size = sizes[group]
        cost = ((loads + size)**2 - loads**2) / target
        part = int(np.argmin(cost))
        result[group] = PARTS[part]
        loads[part] += size
    return result


def semantic_flag(row):
    """Triage metadata; explicitly not intervention ground truth."""
    support = row['support_text'].lower()
    if row['pmid'] == '24145123' and row['detection_method'].lower() == 'microarray':
        return 'known_expression_comparison_24145123'
    if row['throughput'] == 'high' and row['detection_method'].lower() in ('microarray', 'rna-seq', 'nanostring'):
        return 'expression_screen_requires_record_review'
    manipulation = re.search(r'overexpress|knock.?down|transfect|mimic|silenc|ectopic|anti.?mir', support)
    response = re.search(r'sensitiv|resistan|ic50|viability|cytotoxic|apoptos', support)
    if manipulation and response:
        return 'intervention_candidate_unverified'
    return 'unresolved_requires_record_review'


def load_records(task, raw, raw_digest):
    legacy = pd.read_csv(ROOT/'11.direction'/(task+'_records.csv'), dtype=str).fillna('')
    selected = raw[(raw.ncRNA_Type == TASKS[task]) & raw.ncRNA_Name.str.upper().isin(set(legacy.ncRNA)) &
        raw.CID.str.replace(r'\.0$', '', regex=True).isin(set(legacy.cid)) &
        raw.Effect.str.lower().isin(['resistant', 'sensitive'])].copy()
    if len(selected) != len(legacy):
        raise ValueError('Raw-to-legacy record count mismatch')
    expected = {
        'ncRNA': selected.ncRNA_Name.str.strip().str.upper(),
        'cid': selected.CID.str.replace(r'\.0$', '', regex=True),
        'effect': selected.Effect.str.strip().str.lower(),
        'pmid': selected.PMID.str.strip(),
        'condition': selected.Condition.str.strip(),
        'throughput': selected.Throughput.str.strip().str.lower(),
        'phenotype': selected.Phenotype.str.strip(),
    }
    for column, values in expected.items():
        np.testing.assert_array_equal(legacy[column], values.to_numpy(), err_msg='Record identity mismatch: '+column)
    frame = legacy.copy()
    frame['task'] = task
    frame['record'] = np.arange(len(frame))
    frame['raw_table_row'] = selected.index.to_numpy() + 2
    frame['record_uid'] = ['ncrnadrug:'+raw_digest[:16]+':'+str(i) for i in frame.raw_table_row]
    for new, old in [('detection_method','Detection_Method'),('support_text','Support'),('reference_title','Reference')]:
        frame[new] = selected[old].to_numpy()
    frame['semantic_status'] = [semantic_flag(r) for r in frame.to_dict('records')]
    frame['intervention_verified'] = False
    frame['label_semantics'] = 'database_observed_effect_not_intervention_ground_truth'
    context = pd.read_csv(ROOT/'13.context'/(task+'_context.csv'), dtype=str).fillna('')
    context['record'] = context.record.astype(int)
    if not context.record.isin(frame.record).all():
        raise ValueError('Unknown original record in context expansion')
    for column in ('ncRNA','cid','effect'):
        np.testing.assert_array_equal(context[column], frame.set_index('record').loc[context.record,column])
    return frame, context[['record','kind','cell_line','disease']]


def aggregate(frame):
    """Consensus is computed inside a partition, after source assignment."""
    mapped = frame[frame.disease != ''].drop_duplicates(['record_uid','disease'])
    rows, conflicts = [], []
    for key, group in mapped.groupby(KEYS, sort=True):
        row = dict(zip(KEYS, key))
        row['triple_id'] = base.json_digest(list(key))[:24]
        for column, name in [('record_uid','record_uids'),('source_group','source_groups'),
                             ('throughput','throughputs'),('semantic_status','semantic_statuses')]:
            row[name] = json.dumps(sorted(set(group[column])), ensure_ascii=False)
        row['pmids'] = json.dumps(sorted({p for value in group.pmid for p in pmids(value)}))
        row['n_original_records'] = group.record_uid.nunique()
        row['n_pmids'] = len(json.loads(row['pmids']))
        if group.effect.nunique() != 1:
            conflicts.append(dict(row, effects='resistant;sensitive'))
        else:
            row['effect'] = group.effect.iloc[0]
            row['y'] = int(row['effect'] == 'resistant')
            rows.append(row)
    columns = KEYS + ['triple_id','record_uids','source_groups','throughputs','semantic_statuses',
                     'pmids','n_original_records','n_pmids','effect','y']
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(conflicts)


def identity_set(frame):
    return set(frame.loc[frame.disease!='',KEYS].itertuples(index=False, name=None))


def add_novelty(frame, earlier):
    frame = frame.copy()
    frame['novel_triple'] = [key not in earlier for key in frame[KEYS].itertuples(index=False,name=None)]
    return frame


def build(args):
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new context evaluation directory')
    raw_path = ROOT/'10.circ/raw/DR_Curated.txt'
    paths = [raw_path] + [ROOT/folder/(task+suffix) for task in TASKS for folder,suffix in
            [('11.direction','_records.csv'),('13.context','_context.csv')]]
    p = base.seal(dict(seed=20260914, fractions=dict(zip(PARTS,[.6,.2,.2])),
        sources={str(x):base.digest(x) for x in paths},
        source_sha256=base.digest(Path(__file__)), tasks=list(TASKS),
        grouping='Assign original records by connected PMID sets jointly across both RNA tasks, then expand contexts and form within-partition consensus. Missing PMID records are quarantined.',
        novelty='Validation primary excludes any triple observed in raw mapped training records, including conflicted training triples. Holdout novelty also excludes raw mapped validation identities.',
        semantic_policy='Preserve Detection_Method, Support, source row, Condition, Throughput, PMID and year. All intervention_verified flags remain false; keyword flags are triage only.',
        controls=['global','rna_prior','drug_prior','disease_prior','pair_logistic','context_logistic'],
        logistic=dict(C=1.,solver='liblinear',max_iter=2000,random_state=20260914),
        predictors='RNA identity, drug identity, and optionally disease identity; no PMID, source text, throughput, year, Effect-derived feature or auxiliary graph.',
        primary='AUC/AP of observed resistant description on novel triples in source-disjoint validation; report per-drug macro AP and context macro AUC.',
        holdout_evaluation=False, no_original_association_test_arrays=True))
    out.mkdir(parents=True)
    base.write_json(out/'protocol.json',p)
    raw = pd.read_csv(raw_path,sep='\t',encoding='latin-1',dtype=str).fillna('')
    loaded = {task:load_records(task,raw,p['sources'][str(raw_path)]) for task in TASKS}
    all_records = [r for frame,_ in loaded.values() for r in frame.to_dict('records')]
    groups = source_components(all_records)
    sizes = Counter(g for g in groups.values() if g)
    assignments = assign_groups(sizes,p['seed'])
    base.write_json(out/'source_assignments.json',dict(sizes=dict(sizes),partitions=assignments))
    report, artifacts = {}, {}
    for task,(records,context) in loaded.items():
        directory = out/task
        directory.mkdir()
        records['source_group'] = records.record_uid.map(groups)
        records['partition'] = records.source_group.map(assignments).fillna('quarantine')
        expanded = context.merge(records,on='record',how='left',validate='many_to_one')
        if expanded.groupby('record_uid').partition.nunique().max() != 1:
            raise ValueError('Expanded contexts crossed a record boundary')
        records.to_csv(directory/'records.tsv',sep='\t',index=False)
        expanded.to_csv(directory/'context_records.tsv',sep='\t',index=False)
        datasets, earlier, split_stats = {}, set(), {}
        for part in PARTS:
            source = expanded[expanded.partition==part]
            triples, conflicts = aggregate(source)
            triples = add_novelty(triples, earlier)
            earlier |= identity_set(source)
            triples.to_csv(directory/(part+'.tsv'),sep='\t',index=False)
            conflicts.to_json(directory/(part+'_conflicts.json'),orient='records',force_ascii=False,indent=2)
            datasets[part] = triples
            split_stats[part] = dict(records=int((records.partition==part).sum()),
                pmids=len({p for value in records.loc[records.partition==part,'pmid'] for p in pmids(value)}),
                source_groups=records.loc[records.partition==part,'source_group'].nunique(),
                expanded_context_rows=len(source), mapped_original_records=source.loc[source.disease!='','record_uid'].nunique(),
                consensus_triples=len(triples), conflicting_triples=len(conflicts),
                resistant_triples=int(triples.y.sum()), novel_triples=int(triples.novel_triple.sum()),
                novel_resistant_triples=int(triples.loc[triples.novel_triple,'y'].sum()))
        sets = {part:{p for value in records.loc[records.partition==part,'pmid'] for p in pmids(value)} for part in PARTS}
        overlaps = {a+'__'+b:len(sets[a]&sets[b]) for i,a in enumerate(PARTS) for b in PARTS[i+1:]}
        if any(overlaps.values()):
            raise ValueError('PMID overlap between partitions')
        # Diagnose whether joining all globally repeated triples would create a giant group.
        strict = Components()
        valid = expanded[(expanded.disease!='') & (expanded.source_group!='')]
        for _,g in valid.groupby(KEYS):
            unique = sorted(set(g.source_group))
            for item in unique:
                strict.union(unique[0],item)
        component_sizes = Counter(strict.root(g) for g in records.source_group if g)
        global_clean, global_conflicts = aggregate(expanded[expanded.source_group!=''])
        report[task] = dict(raw_records=len(records), mapped_global_consensus_triples=len(global_clean),
            mapped_global_conflicting_triples=len(global_conflicts), split_stats=split_stats, pmid_overlap=overlaps,
            record_overlap=0, quarantine_records=int((records.partition=='quarantine').sum()),
            semantic_status_counts=records.semantic_status.value_counts().to_dict(), verified_intervention_records=0,
            strict_source_plus_triple_components=len(component_sizes),
            strict_largest_component_records=max(component_sizes.values()),
            strict_largest_component_fraction=max(component_sizes.values())/len(records),
            strict_largest_10_components=sorted(component_sizes.values(),reverse=True)[:10])
        for path in directory.iterdir():
            artifacts[str(path.relative_to(out))] = base.digest(path)
    base.write_json(out/'DATA_AUDIT.json',dict(protocol_sha256=p['sha256'],tasks=report,
        cross_task_source_groups_jointly_assigned=True, global_source_groups=len(sizes)))
    artifacts['source_assignments.json'] = base.digest(out/'source_assignments.json')
    artifacts['DATA_AUDIT.json'] = base.digest(out/'DATA_AUDIT.json')
    base.write_json(out/'INPUTS.json',dict(protocol_sha256=p['sha256'],files=artifacts))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def scores(train, query, protocol):
    rate = float(train.y.mean())
    result = {'global':np.full(len(query),rate)}
    for column,name in [('ncRNA','rna_prior'),('cid','drug_prior'),('disease','disease_prior')]:
        prior = train.groupby(column).y.mean()
        result[name] = query[column].map(prior).fillna(rate).to_numpy(float)
    for name,columns in [('pair_logistic',['ncRNA','cid']),('context_logistic',KEYS)]:
        vectorizer = DictVectorizer()
        x = vectorizer.fit_transform(train[columns].to_dict('records'))
        q = vectorizer.transform(query[columns].to_dict('records'))
        model = LogisticRegression(**protocol['logistic']).fit(x,train.y)
        result[name] = model.predict_proba(q)[:,list(model.classes_).index(1)]
    return result


def metrics(frame, score):
    y = frame.y.to_numpy(int)
    result = dict(rows=len(y), positives=int(y.sum()), positive_fraction=float(y.mean()) if len(y) else None,
                  auc=None, ap=None, sensitive_ap=None, macro_drug_ap=None, eligible_drugs=0,
                  macro_context_auc=None, eligible_contexts=0)
    if len(set(y)) < 2:
        return result
    result.update(auc=float(roc_auc_score(y,score)),ap=float(average_precision_score(y,score)),
                  sensitive_ap=float(average_precision_score(1-y,1-score)))
    for column,key,count,metric in [('cid','macro_drug_ap','eligible_drugs',average_precision_score),
            ('disease','macro_context_auc','eligible_contexts',roc_auc_score)]:
        values = []
        for _,g in frame.reset_index(drop=True).groupby(column):
            ids = g.index.to_numpy()
            if len(set(y[ids])) == 2:
                values.append(metric(y[ids],score[ids]))
        result[key],result[count] = (float(np.mean(values)) if values else None),len(values)
    return result


def evaluate(args):
    out = Path(args.out_dir)
    if (out/'EVALUATION.json').exists():
        raise ValueError('Evaluation already frozen')
    p = base.check_seal(base.read_json(out/'protocol.json'))
    if p['source_sha256'] != base.digest(Path(__file__)):
        raise ValueError('Frozen evaluation code changed')
    manifest = base.read_json(out/'INPUTS.json')
    if manifest['protocol_sha256'] != p['sha256']:
        raise ValueError('Input protocol mismatch')
    results, files = [], {}
    for task in TASKS:
        data = {}
        for part in ('train','validation'):
            name = task+'/'+part+'.tsv'
            if base.digest(out/name) != manifest['files'][name]:
                raise ValueError('Frozen context rows changed')
            data[part] = pd.read_csv(out/name,sep='\t',dtype={'cid':str})
        train,query = data['train'],data['validation']
        predicted = scores(train,query,p)
        saved = query[['triple_id','y','novel_triple','source_groups']].copy()
        for name,score in predicted.items():
            saved[name] = score
            for subset,mask in [('all_source_heldout',np.ones(len(query),bool)),
                    ('novel_triple_source_heldout',query.novel_triple.to_numpy(bool))]:
                results.append(dict(task=task,subset=subset,model=name,**metrics(query[mask],score[mask])))
        dest = out/task/'validation_predictions.tsv'
        saved.to_csv(dest,sep='\t',index=False)
        files[str(dest.relative_to(out))] = base.digest(dest)
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    files['metrics.csv'] = base.digest(out/'metrics.csv')
    base.write_json(out/'EVALUATION.json',dict(protocol_sha256=p['sha256'],files=files,
        primary_subset='novel_triple_source_heldout',holdout_evaluated=False,results=results))
    lines = ['来源分组的上下文开发评价已完成。预测目标为数据库的 resistant/sensitive 描述，所有记录仍未被确认为 RNA 干预药敏标签。',
        '', '先按原始记录的 PMID 连通组划分，再在每一侧独立汇总三元组；训练与验证的 PMID、原始记录交集均为 0。两个 RNA 任务共用来源分配。按合并记录数目标 20% 预留的来源分区仍未评分，各任务实际比例不同。',
        '', '主评价排除训练侧出现过的全部三元组身份（包括训练侧冲突记录）。所有模型仅输入 RNA/药物/癌种身份，未输入来源、证据文本或测试图。',
        '', '|任务|模型|验证三元组|AUC|耐药描述 AP|敏感描述 AP|药物宏 AP|', '|---|---|---:|---:|---:|---:|---:|']
    for row in results:
        if row['subset']=='novel_triple_source_heldout':
            values = ['NA' if row[k] is None else '%.6f'%row[k] for k in ('auc','ap','sensitive_ap','macro_drug_ap')]
            lines.append('|'+ '|'.join([row['task'],row['model'],str(row['rows']),*values])+'|')
    lines += ['', '药物宏 AP 仅纳入验证中同时含 resistant 和 sensitive 的药物；完整计数见 metrics.csv。这里的两类都是被记录的方向描述，不能当作真实生物阳性/阴性。',
        '', '已按来源分组不等于获得独立新实验，也不验证细胞状态因果作用。疾病映射沿用历史版本；需要继续核对原始 Condition、亲本/耐药衍生株及具体实验单位。',
        '', '原始记录、检测方法、Support 原文、来源行号和保守语义标记保存在各任务 records.tsv；来源审计及全局来源—三元组连通块大小见 DATA_AUDIT.json。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(results).query("subset == 'novel_triple_source_heldout'").to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','evaluate'])
    parser.add_argument('--out-dir',default=str(DEFAULT_OUT))
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
