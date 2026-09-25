"""Predeclare and evaluate a small CPU-only reuse of frozen inner predictions.

No historical outer/test or reserved source-holdout array is supported.
prepare seals the rules before run calculates any new validation metric.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score

from improve import tabular_foundation as base
from improve import tabicl_graph_context as graph_context
from improve import quick_score_modules as rules
from improve.sampling_bias_audit import matched_weights, weighted_metrics


DEFAULT_REFERENCE = 'improve/results/tabicl_pathway_context_20260913'
DEFAULT_OUT = 'improve/results/quick_score_20260914'
OWN = ('quick_score_eval.py', 'quick_score_modules.py', 'test_quick_score.py')
DEPENDENCIES = ('tabular_foundation.py', 'tabicl_graph_context.py',
                'tabicl_graph_context_modules.py', 'sampling_bias_audit.py')
TASKS = ('MiDrug', 'LncDrug')
SEEDS = (666, 0, 1)


def source_hashes():
    return {name: base.digest(Path(__file__).with_name(name)) for name in (*OWN, *DEPENDENCIES)}


def inner_arrays(reference_protocol, task):
    if task not in TASKS:
        raise ValueError('Unknown task')
    parent = Path(reference_protocol['parent_dir'])
    gp = graph_context.protocol(parent)
    if gp['sha256'] != reference_protocol['parent_protocol_sha256']:
        raise ValueError('Graph reference changed')
    return graph_context.features(parent, gp, task, 'inner')


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    rp = base.check_seal(base.read_json(reference / 'protocol.json'))
    candidate = base.read_json(reference / 'CANDIDATE.json')
    validation = base.read_json(reference / 'VALIDATION_inner.json')
    recipe = candidate['selected_inner_recipe']
    if rp['smoke'] or not candidate['accepted'] or recipe['id'] != 'tiered_0p5000':
        raise ValueError('Expected the accepted tiered 0.5 reference')
    if candidate['protocol_sha256'] != rp['sha256'] or validation['protocol_sha256'] != rp['sha256']:
        raise ValueError('Reference protocol identity mismatch')
    files = {str(reference / name): base.digest(reference / name)
             for name in ('protocol.json', 'CANDIDATE.json', 'VALIDATION_inner.json')}
    for task in TASKS:
        for seed in SEEDS:
            relative = f'predictions/inner/{task}/s{seed}/scores.npz'
            path = reference / relative
            if base.digest(path) != validation['artifacts'].get(relative):
                raise ValueError('Reference prediction is unverified or changed')
            files[str(path)] = base.digest(path)
    evidence = Path(rp['inputs_dir'])
    manifest = base.read_json(evidence / 'INPUTS.json')
    files[str(evidence / 'INPUTS.json')] = base.digest(evidence / 'INPUTS.json')
    if files[str(evidence / 'INPUTS.json')] != rp['input_manifest_sha256']:
        raise ValueError('Reference evidence manifest changed')
    for name in ['dictionary.json', 'drug_gene.npz',
                 *[task + '_' + layer + '.npz' for task in TASKS for layer in ('functional', 'binding', 'association')]]:
        path = evidence / name
        if base.digest(path) != manifest['files'][name]:
            raise ValueError('Evidence changed: ' + name)
        files[str(path)] = base.digest(path)
    p = base.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), reference_dir=str(reference),
        reference_protocol_sha256=rp['sha256'], reference_recipe=recipe['id'],
        tasks=list(TASKS), seeds=list(SEEDS), candidates=list(rules.CANDIDATES), controls=list(rules.CONTROLS),
        source_sha256=source_hashes(), input_files=files,
        recipes={
            'reference': '0.5 * graph + 0.5 * tiered, exactly the already accepted score',
            'logit_pool': 'sigmoid(0.5 * logit(graph) + 0.5 * logit(tiered))',
            'coverage_gate': 'alpha = 0.25 + 0.5 * drug_target_present * max(functional_RNA_present, 0.25*binding_RNA_present, 0.1*association_RNA_present); score = (1-alpha)*graph + alpha*tiered',
            'coverage_gate_shuffled': 'Same gate after independently permuting full RNA/drug evidence-strength vectors using fixed seed 20260914; control only',
            'rna_frequency_correction': 'sigmoid(logit(reference) - 0.25 * (logit(smoothed_support_RNA_label_rate) - logit(support_label_rate)))',
            'drug_frequency_correction': 'Same correction using drug support label rate; within-drug ranking is mathematically unchanged',
        },
        prior_mass=rules.PRIOR_MASS, frequency_correction=rules.CORRECTION, probability_clip=rules.EPSILON,
        same_rule_for_both_tasks=True, new_neural_training=False, new_neural_inference=False,
        model_selection='Report every fixed candidate. No post-result coefficient search, no automatic promotion, no outer evaluation even if a candidate passes.',
        performance_gate='LncDrug AUC and AP >= accepted reference in EVERY inference seed, zero negative tolerance; MiDrug mean AUC and AP > reference and each positive in >= 2/3 seeds. Drug macro AP reported separately.',
        diagnostics='Global metrics, drug/RNA-matched AP, both-class per-drug AP, low-frequency drugs at the training positive-degree median, and annotated/unannotated pair subsets. All methods evaluated on identical rows.',
        uncertainty='Descriptive development screen; no significance claims or bootstrap-selected winner. Seeds are inference randomness on one repeatedly used inner partition.',
        query_independence='No query batch ranks or query-set normalization. Rates use support labels only; coverage uses frozen label-free annotations.',
        source_partition_evaluated=False, historical_outer_evaluated=False, original_test_evaluated=False,
        label_semantics='0 = sampled unobserved association, not verified biological negative; score corrections are not calibrated drug-response probabilities.',
        limitations=[
            'The historical inner partition and related methods have already been inspected in development.',
            'Frequency conflates biological prevalence, database coverage and study effort; removing it need not improve prediction.',
            'Coverage is an evidence-availability heuristic, not a calibrated measure of correctness.',
            'Entity-level shuffle preserves dictionary-level evidence marginals, not exact evaluation-row gate counts or support-degree strata.',
            'The interventions change score combination only and add no new biological information.',
        ]))
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in (*OWN, *DEPENDENCIES):
        shutil.copy2(Path(__file__).with_name(name), out / 'source_snapshot' / name)
    base.write_json(out / 'protocol.json', p)
    print('QUICK_PROTOCOL_FROZEN', p['sha256'], flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out) / 'protocol.json'))
    if p['source_sha256'] != source_hashes():
        raise ValueError('Frozen quick experiment implementation changed')
    for path, expected in p['input_files'].items():
        if base.digest(path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    return p


def evidence_strength(directory, task):
    directory = Path(directory)
    dictionary = base.read_json(directory / 'dictionary.json')
    strength = np.zeros(len(dictionary['rna_names'][task]), dtype=np.float64)
    for layer, weight in (('functional', 1.0), ('binding', 0.25), ('association', 0.1)):
        matrix = sparse.load_npz(directory / (task + '_' + layer + '.npz'))
        strength = np.maximum(strength, weight * (matrix.getnnz(axis=1) > 0))
    drug = (sparse.load_npz(directory / 'drug_gene.npz').getnnz(axis=1) > 0).astype(float)
    return strength, drug


def cached_scores(p, task, seed, a):
    path = Path(p['reference_dir']) / f'predictions/inner/{task}/s{seed}/scores.npz'
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
        np.testing.assert_array_equal(archive['y_val'], a['y_val'])
        return tuple(archive[name].copy() for name in ('baseline', 'tiered_1p0000', p['reference_recipe']))


def gate(rows):
    table = pd.DataFrame(rows).set_index(['task', 'variant', 'seed'])
    result = {}
    for variant in rules.CANDIDATES:
        tasks = {}
        for task in TASKS:
            deltas = {metric: np.array([
                table.loc[(task, variant, seed), metric] - table.loc[(task, 'reference', seed), metric]
                for seed in SEEDS]) for metric in ('auc', 'ap')}
            passed = (all(np.all(delta >= 0) for delta in deltas.values()) if task == 'LncDrug'
                      else all(delta.mean() > 0 and (delta > 0).sum() >= 2 for delta in deltas.values()))
            tasks[task] = dict(passed=bool(passed), deltas={key: value.tolist() for key, value in deltas.items()})
        result[variant] = dict(passed=all(value['passed'] for value in tasks.values()), tasks=tasks)
    return result


def subset_weights(a, adapter):
    pairs, y = a['pairs_val'], a['y_val']
    _, counts, _ = rules.observed_offsets(a['pairs_train'], a['y_train'],
                                         (len(adapter.rna_offset), len(adapter.drug_offset)))
    drug_positive = counts[1]
    threshold = float(np.median(drug_positive[drug_positive > 0]))
    weights = {
        'low_frequency_drug': (drug_positive[pairs[1]] <= threshold).astype(float),
        'pair_evidence_present': (adapter.confidence(pairs) > 0).astype(float),
        'pair_evidence_absent': (adapter.confidence(pairs) == 0).astype(float),
    }
    matched = {}
    for axis, name in enumerate(('rna', 'drug')):
        weights[name + '_matched'], matched[name] = matched_weights(y, pairs[axis])
    return weights, drug_positive, dict(low_frequency_drug_positive_degree_threshold=threshold,
                                      matching=matched)


def run(args):
    out = Path(args.out_dir)
    if (out / 'COMPLETION.json').exists() or (out / 'predictions').exists():
        raise ValueError('Completed or partially run evaluation is immutable; use a new directory')
    started = time.monotonic()
    p = protocol(out)
    rp = base.read_json(Path(p['reference_dir']) / 'protocol.json')
    rows, diagnostics, per_drug, audit, artifacts = [], [], [], {}, {}
    (out / 'predictions').mkdir()
    for task in TASKS:
        a = inner_arrays(rp, task)
        rna_evidence, drug_evidence = evidence_strength(rp['inputs_dir'], task)
        adapter = rules.FrozenScoreAdapter.fit(a['pairs_train'], a['y_train'], rna_evidence, drug_evidence)
        state_path = out / (task + '_adapter.npz')
        adapter.save(state_path)
        loaded = rules.FrozenScoreAdapter.load(state_path)
        artifacts[state_path.name] = base.digest(state_path)
        weights, drug_positive, details = subset_weights(a, adapter)
        y, pairs = a['y_val'], a['pairs_val']
        replay_max, clipped = 0.0, {}
        for seed in SEEDS:
            graph, tiered, reference = cached_scores(p, task, seed, a)
            scores = {}
            for variant in rules.VARIANTS:
                score = adapter.predict(graph, tiered, pairs, variant)
                reversed_score = loaded.predict(graph[::-1], tiered[::-1], pairs[:, ::-1], variant)[::-1]
                chunks = np.array_split(np.arange(len(y)), 7)
                batched = np.concatenate([loaded.predict(graph[i], tiered[i], pairs[:, i], variant) for i in chunks])
                for replay in (reversed_score, batched):
                    np.testing.assert_allclose(score, replay, atol=1e-14, rtol=0)
                    replay_max = max(replay_max, float(np.max(np.abs(score - replay))))
                scores[variant] = score
                row = dict(task=task, seed=seed, variant=variant, **base.metrics(y, score, pairs))
                rows.append(row)
                for subset, weight in weights.items():
                    diagnostics.append(dict(task=task, seed=seed, variant=variant, subset=subset,
                        rows=int((weight > 0).sum()), positives=int(y[weight > 0].sum()),
                        **weighted_metrics(y, score, weight)))
                for drug in np.unique(pairs[1]):
                    mask = pairs[1] == drug
                    if len(np.unique(y[mask])) < 2:
                        continue
                    per_drug.append(dict(task=task, seed=seed, variant=variant, drug_index=int(drug),
                        rows=int(mask.sum()), positives=int(y[mask].sum()), support_positives=int(drug_positive[drug]),
                        ap=float(average_precision_score(y[mask], score[mask]))))
            # Prevent a weaker or numerically different reconstructed comparator.
            np.testing.assert_array_equal(scores['reference'], reference)
            baseline = base.metrics(y, reference, pairs)
            monotone = base.metrics(y, rules.expit(rules.log_odds(reference)), pairs)
            for metric in base.METRICS:
                if baseline[metric] != monotone[metric]:
                    raise ValueError('Probability conversion altered reference ranking')
            # Drug-only offsets must not be sold as within-drug ranking gains.
            drug_corrected = base.metrics(y, scores['drug_frequency_correction'], pairs)
            if baseline['macro_drug_ap'] != drug_corrected['macro_drug_ap']:
                raise ValueError('Drug-only correction changed within-drug rank unexpectedly')
            clipped[str(seed)] = {name: int(((value <= rules.EPSILON) | (value >= 1-rules.EPSILON)).sum())
                                  for name, value in [('graph', graph), ('tiered', tiered), ('reference', reference)]}
            dest = out / 'predictions' / task / ('s' + str(seed))
            dest.mkdir(parents=True)
            np.savez_compressed(dest / 'scores.npz', **scores, y_val=y, pairs_val=pairs)
            artifacts[str((dest / 'scores.npz').relative_to(out))] = base.digest(dest / 'scores.npz')
        audit[task] = dict(support_rows=len(a['y_train']), validation_rows=len(y),
            validation_positives=int(y.sum()), support_validation_pair_intersection=0,
            reference_exactly_reproduced=True, replay_reverse_and_seven_batches_max_abs=replay_max,
            probability_boundary_counts=clipped,
            evidence_gate_alpha_counts={name: {str(float(value)): int(count) for value, count in zip(*np.unique(
                0.25 + 0.5 * adapter.confidence(pairs, shuffled=shuffle), return_counts=True))}
                for name, shuffle in [('actual', False), ('shuffled', True)]}, **details)
        print('QUICK_VERIFIED', task, 'seeds', list(SEEDS), 'variants', len(rules.VARIANTS), flush=True)
    results = gate(rows)
    assessment = dict(candidates=results, eligible_candidates=[name for name, value in results.items() if value['passed']],
        accepted_reference_replaced=False, historical_outer_evaluated=False, original_test_evaluated=False,
        source_holdout_evaluated=False, additional_neural_fits=0, additional_neural_inference=0,
        fixed_candidates=len(rules.CANDIDATES), fixed_controls=len(rules.CONTROLS), elapsed_seconds=time.monotonic()-started)
    table, diag = pd.DataFrame(rows), pd.DataFrame(diagnostics)
    table.to_csv(out / 'metrics.csv', index=False)
    diag.to_csv(out / 'diagnostics.csv', index=False)
    pd.DataFrame(per_drug).to_csv(out / 'per_drug.csv', index=False)
    base.write_json(out / 'ASSESSMENT.json', assessment)
    base.write_json(out / 'AUDIT.json', audit)
    write_report(out, table, diag, assessment, audit)
    for name in ('metrics.csv', 'diagnostics.csv', 'per_drug.csv', 'ASSESSMENT.json', 'AUDIT.json', 'REPORT.md'):
        artifacts[name] = base.digest(out / name)
    # Recheck the original frozen inputs after evaluation, too.
    protocol(out)
    base.write_json(out / 'COMPLETION.json', dict(protocol_sha256=p['sha256'], passed=True,
        artifacts=artifacts, **assessment))
    print(table.groupby(['task', 'variant'])[list(base.METRICS)].mean().to_string(), flush=True)
    print('ELIGIBLE', assessment['eligible_candidates'], 'SECONDS', round(assessment['elapsed_seconds'], 3), flush=True)


def write_report(out, table, diag, assessment, audit):
    means = table.groupby(['task', 'variant'])[list(base.METRICS)].mean()
    dm = diag.groupby(['task', 'variant', 'subset'])[['auc', 'ap']].mean()
    lines = [
        '已完成三类快速验证：logit 融合、证据覆盖门控、训练频率轻度校正。4 个固定候选及 1 个覆盖打乱对照，复用已有预测，无新增神经训练或推理。',
        '', '参考为已接受的 `0.5 × graph + 0.5 × tiered`。种子 666/0/1；仅评价原历史内部开发集。候选公式、强度和验收规则在新指标计算前冻结，无结果后的系数搜索。',
        '', '|任务|方案|AUC|AP|药物宏 AP|ΔAP（百分点）|', '|---|---|---:|---:|---:|---:|',
    ]
    for task in TASKS:
        for variant in rules.VARIANTS:
            row = means.loc[(task, variant)]
            delta = 100 * (row.ap - means.loc[(task, 'reference'), 'ap'])
            lines.append(f'|{task}|{variant}|{row.auc:.9f}|{row.ap:.9f}|{row.macro_drug_ap:.9f}|{delta:+.6f}|')
    lines += ['', '共同门槛：LncDrug 每个种子的 AUC/AP 均不下降；MiDrug 两指标均值提高且各至少 2/3 种子提高。宏 AP 单列。',
              '', '|候选|MiDrug 条件|LncDrug 条件|共同条件|', '|---|---|---|---|']
    for name, result in assessment['candidates'].items():
        lines.append(f"|{name}|{result['tasks']['MiDrug']['passed']}|{result['tasks']['LncDrug']['passed']}|{result['passed']}|")
    lines += ['', '|任务|方案|药物匹配 AP|RNA 匹配 AP|低频药物 AP|', '|---|---|---:|---:|---:|']
    for task in TASKS:
        for variant in rules.VARIANTS:
            values = [dm.loc[(task, variant, subset), 'ap'] for subset in ('drug_matched', 'rna_matched', 'low_frequency_drug')]
            lines.append('|' + task + '|' + variant + '|' + '|'.join(f'{v:.9f}' for v in values) + '|')
    lines += [
        '', '匹配 AP 使用同一批可匹配记录和固定权重，改变了评价分布，不能与全体 AP 直接相减。低频阈值只来自训练正例度数；具体阈值、保留样本和有效样本量见 AUDIT.json。',
        '', '`coverage_gate_shuffled` 只作对照，不可选为改进模型。打乱保留完整实体字典上的覆盖分布，不保证验证样本覆盖数或实体频率分层完全相同；差值不构成覆盖机制的因果证明。',
        '', '药物频率校正对同一药物的所有 RNA 施加同一单调变换，逐药物排名及宏 AP 保持不变；任何全局变化都来自跨药物排序。RNA 频率校正则会改变药物内排序。训练频率既可能含研究偏差，也可能含真实生物学信号。',
        '', f"共同门槛通过候选：{assessment['eligible_candidates']}。当前接受模型未替换。即使通过，这也仅是历史开发集筛查，不是独立泛化确认。",
        '', '保存 2 个适配器状态，每个任务由三个种子共用。6 组种子预测均完成重载、逆序和分 7 批核验；原参考分数逐元素完全复现，输入哈希在运行前后均核对。',
        '', '未访问历史外层数组、原始测试数组或新来源留出集。0 标签仍是采样未观测关联，输出不能解释为干预 RNA 后的药敏概率。',
        '', f"计算耗时 {assessment['elapsed_seconds']:.3f} 秒（含读取、核验与指标计算；不含代码准备）。逐种子完整精度差值见 ASSESSMENT.json；逐药物结果见 per_drug.csv。",
    ]
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
