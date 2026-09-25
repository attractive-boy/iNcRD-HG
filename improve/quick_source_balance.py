"""Rapid source-balanced logistic fits on already frozen RNA representations.

Only the existing source train/validation files are supported. No holdout
loader, RNA encoder training, graph update, or parameter search is provided.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from improve import tabular_foundation as base
from improve import context_source_eval as context
from improve import context_functional_eval as functional
from improve import quick_source_balance_modules as balance


DEFAULT_OUT = 'improve/results/quick_source_balance_20260914'
DEFAULT_REFERENCE = 'improve/results/context_functional_20260914'
TASKS = ('MiDir', 'LncDir')
OWN = ('quick_source_balance.py', 'quick_source_balance_modules.py', 'test_quick_source_balance.py')
DEPENDENCIES = ('tabular_foundation.py', 'context_source_eval.py', 'context_functional_eval.py',
                'sampling_bias_audit.py')


def source_hashes():
    return {name: base.digest(Path(__file__).with_name(name)) for name in (*OWN, *DEPENDENCIES)}


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    rp = base.check_seal(base.read_json(reference / 'protocol.json'))
    evaluation = base.read_json(reference / 'EVALUATION.json')
    if evaluation['protocol_sha256'] != rp['sha256'] or evaluation['holdout_evaluated']:
        raise ValueError('Expected the completed source development experiment')
    source = Path(rp['source_dir'])
    manifest = base.read_json(source / 'INPUTS.json')
    if base.digest(source / 'INPUTS.json') != rp['source_input_manifest_sha256']:
        raise ValueError('Frozen source split changed')
    files = {str(reference / name): base.digest(reference / name) for name in ('protocol.json', 'EVALUATION.json')}
    files[str(source / 'INPUTS.json')] = base.digest(source / 'INPUTS.json')
    for task in TASKS:
        for partition in ('train', 'validation'):
            name = task + '/' + partition + '.tsv'
            if base.digest(source / name) != manifest['files'][name]:
                raise ValueError('Source data changed: ' + name)
            files[str(source / name)] = base.digest(source / name)
        names = [task + '/validation_predictions.tsv']
        names += [task + '/' + representation + '/' + name
                  for representation in ('generic', 'functional_moments')
                  for name in ('numeric_table.npz', 'fitted.joblib')]
        for name in names:
            if base.digest(reference / name) != evaluation['artifacts'][name]:
                raise ValueError('Frozen reference artifact changed: ' + name)
            files[str(reference / name)] = base.digest(reference / name)
    p = base.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), reference_dir=str(reference), source_dir=str(source),
        reference_protocol_sha256=rp['sha256'], source_protocol_sha256=rp['source_protocol_sha256'],
        source_sha256=source_hashes(), input_files=files, tasks=list(TASKS),
        representations=list(balance.REPRESENTATIONS), weightings=list(balance.WEIGHTINGS),
        logistic=rp['logistic'], weight_seed=20260914,
        source_rule='Each k-source triple credits 1/k to each source load N_s; raw sample weight=sum_s 1/(k*N_s). Normalize to mean 1. No validation labels enter weights.',
        class_preservation='source_balanced_preserve_class additionally makes mean weight within each training label class 1, preserving both original class masses.',
        controls='class_prior_only preserves the source-weighted class masses but is uniform within each class; source_weights_shuffled permutes source_balanced weights within each class with a fixed seed.',
        design='Reuse the exact source-filtered, train-fitted 24 numeric covariates and the same identity/context interaction design. Fit logistic C=1 for each weighting, with total sample-weight mass fixed to n_train.',
        primary_representation='functional_moments', primary_weighting='source_balanced_preserve_class',
        primary_evaluation='novel_triple_source_heldout',
        primary_contrast='functional_moments/source_balanced_preserve_class vs functional_moments/uniform; all other contrasts descriptive, no post-result selection',
        descriptive_gate='For the same representation and weighting rule in BOTH tasks, AUC and resistant AP must not decrease relative to the corresponding uniform reference, with at least one strict improvement per task.',
        bootstrap_repeats=500, bootstrap_seed=20260914,
        bootstrap='Paired connected-source-component bootstrap on fixed predictions for the two predefined primary task contrasts only; percentile intervals conditional on this split and fitted models.',
        promotion='Development feasibility comparison only; do not replace association or direction reference models.',
        heldout_labels_loaded=False, source_holdout_evaluated=False, historical_outer_evaluated=False,
        original_association_test_evaluated=False, new_rna_encodings=False,
        label_semantics='Resistant/sensitive database descriptions, not RNA intervention outcomes.',
        limitations=[
            'This source validation partition has already been used in stage A; this is continued development, not an independent confirmation.',
            'Equal source credit can discard real information or upweight small low-quality studies; study size alone is not evidence reliability.',
            'Multiple-source triples mix fractional credit, and class preservation changes exact equal-source mass.',
            'The local auxiliary miRNA sites were filtered in the reference, but public encoder pretraining source independence is not established.',
        ]))
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in (*OWN, *DEPENDENCIES):
        shutil.copy2(Path(__file__).with_name(name), out / 'source_snapshot' / name)
    base.write_json(out / 'protocol.json', p)
    print('SOURCE_BALANCE_FROZEN', p['sha256'], flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out) / 'protocol.json'))
    if source_hashes() != p['source_sha256']:
        raise ValueError('Frozen implementation changed')
    for path, expected in p['input_files'].items():
        if base.digest(path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    return p


def designs(reference, task, representation, train, query):
    identity, conditions = DictVectorizer(), DictVectorizer()
    xi = identity.fit_transform(train[context.KEYS].to_dict('records'))
    qi = identity.transform(query[context.KEYS].to_dict('records'))
    xc = conditions.fit_transform(train[['cid', 'disease']].to_dict('records'))
    qc = conditions.transform(query[['cid', 'disease']].to_dict('records'))
    state = dict(identity=identity, conditions=conditions, numeric_table=None, rna_names=None)
    if representation == 'context_logistic':
        return xi, qi, state
    cached = joblib.load(Path(reference) / task / representation / 'fitted.joblib')
    for name, fitted in (('identity', identity), ('conditions', conditions)):
        np.testing.assert_array_equal(fitted.get_feature_names_out(), cached[name].get_feature_names_out())
    with np.load(Path(reference) / task / representation / 'numeric_table.npz', allow_pickle=False) as archive:
        numeric, names = archive['features'].copy(), archive['names'].copy()
    lookup = {name: i for i, name in enumerate(names)}
    train_ids = train.ncRNA.map(lookup).to_numpy(int)
    query_ids = query.ncRNA.map(lookup).to_numpy(int)
    if not np.isfinite(numeric).all() or numeric.shape[1] != 24:
        raise ValueError('Expected the frozen 24 finite numeric covariates')
    state.update(numeric_table=numeric, rna_names=names)
    return functional.design(xi, xc, numeric[train_ids]), functional.design(qi, qc, numeric[query_ids]), state


def assessment(table):
    primary = table[table.subset == 'novel_triple_source_heldout'].set_index(['task', 'representation', 'weighting'])
    results = {}
    for representation in balance.REPRESENTATIONS:
        for weighting in balance.CANDIDATES:
            tasks = {}
            for task in TASKS:
                score, reference = primary.loc[(task, representation, weighting)], primary.loc[(task, representation, 'uniform')]
                delta = {metric: float(score[metric] - reference[metric]) for metric in ('auc', 'ap', 'sensitive_ap', 'macro_drug_ap')}
                passed = delta['auc'] >= 0 and delta['ap'] >= 0 and max(delta['auc'], delta['ap']) > 0
                tasks[task] = dict(passed=bool(passed), delta=delta)
            results[representation + '/' + weighting] = dict(passed=all(value['passed'] for value in tasks.values()), tasks=tasks)
    return results


def run(args):
    out = Path(args.out_dir)
    if (out / 'COMPLETION.json').exists() or any((out / task).exists() for task in TASKS):
        raise ValueError('Completed or partially run evaluation is immutable')
    started = time.monotonic()
    p = protocol(out)
    rows, audits, artifacts, bootstrap = [], {}, {}, {}
    global_ids = {part: {column: set() for column in ('source_groups', 'record_uids', 'pmids')} for part in ('train', 'validation')}
    fit_seconds = 0.0
    for task in TASKS:
        frames = {part: pd.read_csv(Path(p['source_dir']) / task / (part + '.tsv'), sep='\t', dtype={'cid': str})
                  for part in ('train', 'validation')}
        train, query = frames['train'], frames['validation']
        partition_audit = balance.audit_partitions(train, query)
        for part, frame in frames.items():
            for column in global_ids[part]:
                global_ids[part][column].update(balance.partition_ids(frame, column))
        weights, weight_audit = balance.make_weights(train.source_groups, train.y.to_numpy(), p['weight_seed'])
        target = out / task
        target.mkdir()
        weight_table = train[['triple_id', 'y', 'source_groups']].copy()
        for name, values in weights.items():
            weight_table[name] = values
        weight_table.to_csv(target / 'train_weights.tsv', sep='\t', index=False)
        reference = pd.read_csv(Path(p['reference_dir']) / task / 'validation_predictions.tsv', sep='\t')
        for column in ('triple_id', 'y', 'novel_triple', 'source_groups'):
            np.testing.assert_array_equal(reference[column], query[column])
        predictions = query[['triple_id', 'y', 'novel_triple', 'source_groups']].copy()
        masks = dict(all_source_heldout=np.ones(len(query), bool), novel_triple_source_heldout=query.novel_triple.to_numpy())
        fit_audit = {}
        for representation in balance.REPRESENTATIONS:
            x, q, state = designs(p['reference_dir'], task, representation, train, query)
            for weighting in balance.WEIGHTINGS:
                before = time.monotonic()
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    model = LogisticRegression(**p['logistic']).fit(x, train.y, sample_weight=weights[weighting])
                elapsed = time.monotonic() - before
                fit_seconds += elapsed
                score = model.predict_proba(q)[:, list(model.classes_).index(1)]
                if weighting == 'uniform':
                    np.testing.assert_allclose(score, reference[representation], atol=1e-12, rtol=0)
                dest = target / representation / weighting
                dest.mkdir(parents=True)
                joblib.dump(dict(model=model, **state), dest / 'fitted.joblib')
                loaded = joblib.load(dest / 'fitted.joblib')['model']
                replay = loaded.predict_proba(q[::-1])[:, list(loaded.classes_).index(1)][::-1]
                split = np.concatenate([loaded.predict_proba(q[ids])[:, list(loaded.classes_).index(1)]
                                        for ids in np.array_split(np.arange(len(query)), 5)])
                for value in (replay, split):
                    np.testing.assert_allclose(score, value, atol=1e-12, rtol=0)
                key = representation + '/' + weighting
                fit_audit[key] = dict(seconds=elapsed, iterations=model.n_iter_.tolist(), columns=x.shape[1],
                    replay_max_abs=float(max(np.max(np.abs(score-replay)), np.max(np.abs(score-split)))))
                predictions[key] = score
                for subset, mask in masks.items():
                    rows.append(dict(task=task, representation=representation, weighting=weighting, subset=subset,
                                     **context.metrics(query[mask], score[mask])))
                artifacts[str((dest / 'fitted.joblib').relative_to(out))] = base.digest(dest / 'fitted.joblib')
            print('SOURCE_BALANCE_VERIFIED', task, representation, 'fits', len(balance.WEIGHTINGS), flush=True)
        primary = masks[p['primary_evaluation']]
        prefix = p['primary_representation'] + '/'
        bootstrap[task] = functional.bootstrap_delta(query.loc[primary, 'y'].to_numpy(),
            predictions.loc[primary, prefix + p['primary_weighting']].to_numpy(),
            predictions.loc[primary, prefix + 'uniform'].to_numpy(),
            functional.bootstrap_groups(query[primary]), repeats=p['bootstrap_repeats'], seed=p['bootstrap_seed'])
        predictions.to_csv(target / 'validation_predictions.tsv', sep='\t', index=False)
        for name in ('train_weights.tsv', 'validation_predictions.tsv'):
            artifacts[task + '/' + name] = base.digest(target / name)
        audits[task] = dict(partitions=partition_audit, weights=weight_audit, fits=fit_audit,
                           uniform_reference_reproduced=True, train_rows=len(train), validation_primary_rows=int(primary.sum()))
    global_overlap = {column: len(global_ids['train'][column] & global_ids['validation'][column]) for column in global_ids['train']}
    if any(global_overlap.values()):
        raise ValueError('Cross-task training/validation source overlap')
    audits['joint_partition_intersection'] = global_overlap
    table = pd.DataFrame(rows)
    results = assessment(table)
    primary_key = p['primary_representation'] + '/' + p['primary_weighting']
    outcome = dict(primary_contrast=primary_key, primary_gate_passed=results[primary_key]['passed'],
        comparisons=results, model_fits=len(TASKS)*len(balance.REPRESENTATIONS)*len(balance.WEIGHTINGS),
        logistic_fit_seconds=fit_seconds, total_seconds=time.monotonic()-started,
        accepted_reference_replaced=False, source_holdout_evaluated=False, historical_outer_evaluated=False,
        original_association_test_evaluated=False)
    table.to_csv(out / 'metrics.csv', index=False)
    base.write_json(out / 'ASSESSMENT.json', outcome)
    base.write_json(out / 'AUDIT.json', audits)
    base.write_json(out / 'BOOTSTRAP.json', bootstrap)
    report(out, table, outcome, bootstrap, audits)
    for name in ('metrics.csv', 'ASSESSMENT.json', 'AUDIT.json', 'BOOTSTRAP.json', 'REPORT.md'):
        artifacts[name] = base.digest(out / name)
    protocol(out)
    base.write_json(out / 'COMPLETION.json', dict(passed=True, protocol_sha256=p['sha256'], artifacts=artifacts, **outcome))
    print(table[table.subset == p['primary_evaluation']][['task', 'representation', 'weighting', 'auc', 'ap', 'sensitive_ap']].to_string(index=False), flush=True)
    print('SOURCE_BALANCE_PRIMARY_PASS', outcome['primary_gate_passed'], 'SECONDS', round(outcome['total_seconds'], 2), flush=True)


def report(out, table, outcome, bootstrap, audits):
    primary = table[table.subset == 'novel_triple_source_heldout']
    lines = ['来源加权快速验证已完成。复用已有来源过滤 RNA 表示及训练侧 PCA，只改变线性上下文模型的训练样本权重。',
        '', '3 种表示 × 5 种权重 × 2 个任务，共 30 次固定 C=1 的逻辑回归拟合。主比较提前指定为功能均值＋SD 表示上的“来源平衡且保留类别总权重”对均匀权重。',
        '', '|任务|表示|权重|AUC|耐药描述 AP|敏感描述 AP|药物宏 AP|', '|---|---|---|---:|---:|---:|---:|']
    for row in primary.to_dict('records'):
        lines.append('|' + '|'.join([row['task'], row['representation'], row['weighting'],
            *[f"{row[name]:.6f}" for name in ('auc', 'ap', 'sensitive_ap', 'macro_drug_ap')]]) + '|')
    lines += ['', '来源平衡先给多来源三元组分配 1/k 的来源计数，再按各来源总计数倒数赋权。所有方案总权重保持为训练行数；保留类别方案另使各类别权重总量等于原始类别行数。',
        '', '`class_prior_only` 仅改变类别总权重；`source_weights_shuffled` 在类别内打乱来源权重，保留类别权重总量及分布。这两项只作对照。论文大小并不等于证据质量，来源平衡也不验证干预语义。',
        '', '|任务|训练行|来源数|均匀有效样本量|保留类别的来源加权有效样本量|', '|---|---:|---:|---:|---:|']
    for task in TASKS:
        a = audits[task]
        lines.append(f"|{task}|{a['train_rows']}|{a['weights']['training_sources']}|{a['weights']['schemes']['uniform']['effective_sample_size']:.1f}|{a['weights']['schemes']['source_balanced_preserve_class']['effective_sample_size']:.1f}|")
    lines += ['', '|主比较|ΔAUC|Δ耐药 AP|AUC 95% 区间|AP 95% 区间|', '|---|---:|---:|---|---|']
    for task in TASKS:
        result = outcome['comparisons'][outcome['primary_contrast']]['tasks'][task]
        intervals = bootstrap[task]['intervals']
        lines.append(f"|{task}|{result['delta']['auc']:+.6f}|{result['delta']['ap']:+.6f}|{intervals['auc']}|{intervals['ap']}|")
    lines += ['', f"预定主比较的两任务共同描述性条件通过：{outcome['primary_gate_passed']}。完整其他比较见 ASSESSMENT.json；不据此改选主比较。",
        '', '区间为 500 次来源连通组配对重采样，仅条件于当前划分和模型。此验证集已参与前一轮开发，不能称为新独立确认；两个任务分别报告，不把不同正类比例的 AP 混合。',
        '', '来源、PMID、原始记录的训练/验证交集在两任务联合后仍为 0。主验证保留原协议的未见三元组规则。均匀权重重现已有参考预测；30 个保存模型均通过重载、换序和分批核验。',
        '', f"逻辑回归拟合累计 {outcome['logistic_fit_seconds']:.3f} 秒；读取、拟合、核验、指标及重采样总计 {outcome['total_seconds']:.3f} 秒。未运行 RNA 编码器。",
        '', '方向标签仍是数据库 resistant/sensitive 描述，不能当作 RNA 干预药敏。原关联参考未替换；来源留出集、历史外层数组与原始关联测试数组未评价。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--reference-dir', default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.command](args)


if __name__ == '__main__':
    main()
