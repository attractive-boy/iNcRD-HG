"""Score-level evaluation of a frozen tabular-foundation-model transfer.

The reference is the accepted `0.5 * graph_context + 0.5 * tiered_evidence`
TabICLv2 score.  New candidates apply the frozen TabDPT (1.3) in-context
classifier to the evidence-augmented feature banks and combine it with the
reference by a fixed equal logit weight.  All candidates and the acceptance gate
are frozen before any metric is computed; there is no coefficient search after
seeing results.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score

from improve import tabular_foundation as base
from improve import foundation_transfer_features as features


DEFAULT_INPUTS = 'improve/results/foundation_transfer_20260914/inputs'
DEFAULT_SCORES = 'improve/results/foundation_transfer_20260914/tabdpt'
DEFAULT_REFERENCE = 'improve/results/tabicl_pathway_context_20260913'
DEFAULT_OUT = 'improve/results/foundation_transfer_20260914/eval'
OWN = ('foundation_transfer_eval.py', 'foundation_transfer_features.py')
TASKS = ('MiDrug', 'LncDrug')
SEEDS = (666, 0, 1)
BANKS = ('base36', 'tiered62', 'context76')
REFERENCE_BANK = 'tiered62'  # any bank file carries the shared labels/pairs

# Pre-registered score recipes: name -> {source: weight on the logit scale}.
RECIPES = {
    'reference': {'reference': 1.0},
    'tdp_base': {'base36': 1.0},
    'tdp_tiered62': {'tiered62': 1.0},
    'tdp_context76': {'context76': 1.0},
    'ensemble_ref_base36': {'reference': 0.5, 'base36': 0.5},
    'ensemble_ref_tiered62': {'reference': 0.5, 'tiered62': 0.5},
    'ensemble_ref_context76': {'reference': 0.5, 'context76': 0.5},
    'ensemble_ref_tiered62_context76': {'reference': 1 / 3, 'tiered62': 1 / 3, 'context76': 1 / 3},
}
CONTROLS = ('tdp_base', 'ensemble_ref_tiered62', 'ensemble_ref_context76')
INELIGIBLE = ('reference',)
LOGIT_EPSILON = 1e-12


def sources():
    return {name: base.digest(Path(__file__).with_name(name)) for name in OWN}


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return np.log(p) - np.log1p(-p)


def reference_scores(reference, task, seed):
    path = Path(reference) / ('predictions/inner/%s/s%d/scores.npz' % (task, seed))
    with np.load(path, allow_pickle=False) as archive:
        return archive['tiered_0p5000'].copy(), archive['y_val'].copy(), archive['pairs_val'].copy()


def bank_scores(scores, bank, task, seed):
    path = Path(scores) / ('%s_%s_s%d.npz' % (bank, task, seed))
    with np.load(path, allow_pickle=False) as archive:
        return archive['score'].copy(), archive['y_val'].copy(), archive['pairs_val'].copy()


def prepare(args):
    out, inputs, scores, reference = [Path(x).resolve() for x in
                                      (args.out_dir, args.inputs_dir, args.scores_dir, args.reference_dir)]
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    doc = features.verify(inputs)
    rp = base.check_seal(base.read_json(reference / 'protocol.json'))
    candidate = base.read_json(reference / 'CANDIDATE.json')
    if not candidate['accepted'] or candidate['selected_inner_recipe']['id'] != 'tiered_0p5000':
        raise ValueError('Expected the accepted tiered 0.5 reference')
    files = {str(inputs / 'INPUTS.json'): base.digest(inputs / 'INPUTS.json'),
             str(reference / 'protocol.json'): base.digest(reference / 'protocol.json'),
             str(reference / 'CANDIDATE.json'): base.digest(reference / 'CANDIDATE.json')}
    for task in TASKS:
        for seed in SEEDS:
            for bank in BANKS:
                path = scores / ('%s_%s_s%d.npz' % (bank, task, seed))
                if not path.exists():
                    raise ValueError('Missing frozen TabDPT score: ' + str(path))
                files[str(path)] = base.digest(path)
            relative = 'predictions/inner/%s/s%d/scores.npz' % (task, seed)
            files[str(reference / relative)] = base.digest(reference / relative)
    p = base.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        reference_dir=str(reference), reference_protocol_sha256=rp['sha256'],
        reference_recipe='tiered_0p5000',
        inputs_dir=str(inputs), input_manifest_sha256=doc['sha256'], scores_dir=str(scores),
        tasks=list(TASKS), seeds=list(SEEDS), banks=list(BANKS), recipes=RECIPES, controls=list(CONTROLS),
        source_sha256=sources(), input_files=files,
        method='Frozen TabDPT(1.3) in-context classifier on the accepted evidence/context feature banks, combined with the frozen TabICLv2 reference by fixed equal logit weights.',
        neural_parameter_updates=0, new_neural_training=False, new_neural_inference=True,
        gate='LncDrug AUC and AP >= accepted reference in EVERY inference seed, zero negative tolerance; MiDrug mean AUC and AP > reference and each positive in >= 2/3 seeds. Macro drug AP reported separately.',
        model_selection='Every pre-registered fixed candidate is reported. No post-result weight search, no automatic promotion, no outer/test evaluation.',
        uncertainty='Descriptive development screen with paired per-seed deltas; seeds are inference randomness, not independent splits.',
        limitations=[
            'The historical inner partition has been repeatedly used in development.',
            'TabDPT is a public checkpoint; its pretraining prior-data are not controlled here.',
            'The augmented columns are annotation-derived, not measured function.',
            'No outer, original test or source-holdout array is evaluated.']))
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name), out / 'source_snapshot' / name)
    base.write_json(out / 'protocol.json', p)
    print('PROTOCOL_FROZEN', p['sha256'], flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out) / 'protocol.json'))
    if p['source_sha256'] != sources():
        raise ValueError('Frozen evaluation implementation changed')
    for path, expected in p['input_files'].items():
        if base.digest(path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    return p


def gate(metrics):
    table = pd.DataFrame(metrics).set_index(['task', 'recipe', 'seed'])
    present = set(table.index.get_level_values('recipe'))
    result = {}
    for recipe in RECIPES:
        if recipe in INELIGIBLE or recipe not in present:
            continue
        tasks = {}
        for task in TASKS:
            if (task, recipe, SEEDS[0]) not in table.index or (task, 'reference', SEEDS[0]) not in table.index:
                continue
            deltas = {metric: np.array([table.loc[(task, recipe, seed), metric] -
                                        table.loc[(task, 'reference', seed), metric] for seed in SEEDS])
                      for metric in ('auc', 'ap')}
            if task == 'LncDrug':
                passed = all(np.all(delta >= 0) for delta in deltas.values())
            else:
                passed = all(delta.mean() > 0 and (delta > 0).sum() >= 2 for delta in deltas.values())
            tasks[task] = dict(passed=bool(passed), deltas={k: v.tolist() for k, v in deltas.items()})
        result[recipe] = dict(passed=all(v['passed'] for v in tasks.values()), tasks=tasks)
    return result


def run(args):
    out = Path(args.out_dir)
    if (out / 'COMPLETION.json').exists():
        raise ValueError('Completed evaluation is immutable')
    started = time.monotonic()
    p = protocol(out)
    rows, per_drug, artifacts = [], [], {}
    for task in TASKS:
        raw_scores = {}
        for bank in BANKS:
            raw_scores[bank] = {seed: bank_scores(p['scores_dir'], bank, task, seed) for seed in SEEDS}
        for seed in SEEDS:
            y, pairs = raw_scores[REFERENCE_BANK][seed][1], raw_scores[REFERENCE_BANK][seed][2]
            ref, ry, rp = reference_scores(p['reference_dir'], task, seed)
            np.testing.assert_array_equal(y, ry)
            np.testing.assert_array_equal(pairs, rp)
            components = {'reference': ref, **{bank: raw_scores[bank][seed][0] for bank in BANKS}}
            for recipe, weights in RECIPES.items():
                value = expit(sum(weight * logit(components[source]) for source, weight in weights.items()))
                row = dict(task=task, seed=seed, recipe=recipe, **base.metrics(y, value, pairs))
                rows.append(row)
                for drug in np.unique(pairs[1]):
                    mask = pairs[1] == drug
                    if len(np.unique(y[mask])) < 2:
                        continue
                    per_drug.append(dict(task=task, seed=seed, recipe=recipe, drug_index=int(drug),
                                         rows=int(mask.sum()), ap=float(average_precision_score(y[mask], value[mask]))))
    results = gate(rows)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(per_drug).to_csv(out / 'per_drug.csv', index=False)
    assessment = dict(candidates=results,
                      eligible=[name for name, value in results.items() if value['passed']],
                      accepted_reference_replaced=False,
                      historical_outer_evaluated=False, original_test_evaluated=False,
                      source_holdout_evaluated=False, elapsed_seconds=time.monotonic() - started)
    base.write_json(out / 'ASSESSMENT.json', assessment)
    write_report(out, table, assessment, p)
    for name in ('metrics.csv', 'per_drug.csv', 'ASSESSMENT.json', 'REPORT.md'):
        artifacts[name] = base.digest(out / name)
    protocol(out)
    base.write_json(out / 'COMPLETION.json', dict(protocol_sha256=p['sha256'], passed=True,
                                                  artifacts=artifacts, **assessment))
    print(table.groupby(['task', 'recipe'])[list(base.METRICS)].mean().to_string(), flush=True)
    print('ELIGIBLE', assessment['eligible'], flush=True)


def write_report(out, table, assessment, p):
    means = table.groupby(['task', 'recipe'])[list(base.METRICS)].mean()
    lines = [
        '**Frozen tabular-foundation-model transfer onto evidence/context feature banks（2026-09-14）**',
        '',
        '参考为已接受的 `0.5 × graph_context + 0.5 × tiered_evidence` TabICLv2 分数。新候选把官方 TabDPT(1.3) 语境分类器用于证据/表达增广特征库，'
        '再与参考按固定等权 logit 合成。所有候选与验收规则在任何指标计算前冻结，无结果后的权重搜索。',
        '',
        '|任务|方案|AUC|AP|药物宏 AP|ΔAUC|ΔAP|', '|---|---|---:|---:|---:|---:|---:|',
    ]
    for task in TASKS:
        for recipe in RECIPES:
            row = means.loc[(task, recipe)]
            d_auc = row.auc - means.loc[(task, 'reference'), 'auc']
            d_ap = row.ap - means.loc[(task, 'reference'), 'ap']
            lines.append('|%s|%s|%.9f|%.9f|%.9f|%+.7f|%+.7f|' %
                         (task, recipe, row.auc, row.ap, row.macro_drug_ap, d_auc, d_ap))
    lines += ['', '共同门槛：LncDrug 每个种子的 AUC/AP 均不下降；MiDrug 两指标均值提高且各至少 2/3 种子提高。宏 AP 单列。',
              '', '|候选|MiDrug 条件|LncDrug 条件|共同条件|', '|---|---|---|---|']
    for name, result in assessment['candidates'].items():
        lines.append('|%s|%s|%s|%s|' % (name, result['tasks']['MiDrug']['passed'],
                                        result['tasks']['LncDrug']['passed'], result['passed']))
    indexed = table.set_index(['task', 'recipe', 'seed'])
    lines += ['', '**逐种子差值（相对参考，完整精度）**', '',
              '|任务|方案|种子|ΔAUC|ΔAP|', '|---|---|---:|---:|---:|']
    for task in TASKS:
        for recipe in RECIPES:
            if recipe in INELIGIBLE:
                continue
            for seed in SEEDS:
                d_auc = indexed.loc[(task, recipe, seed), 'auc'] - indexed.loc[(task, 'reference', seed), 'auc']
                d_ap = indexed.loc[(task, recipe, seed), 'ap'] - indexed.loc[(task, 'reference', seed), 'ap']
                lines.append('|%s|%s|%d|%+.7f|%+.7f|' % (task, recipe, seed, d_auc, d_ap))
    lines += ['', '共同门槛通过候选：%s。当前接受模型未自动替换；结果仅为历史开发集筛查，不是独立泛化确认。' %
              assessment['eligible'],
              '', '未访问历史外层、原始测试或来源留出数组。0 标签仍是采样未观测关联，不是已验证生物阴性。',
              '', '复现：`python -m improve.foundation_transfer_features` → '
              '`python -m improve.foundation_transfer_runner --task <task> --device <dev>` → '
              '`python -m improve.foundation_transfer_eval prepare|run`。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--inputs-dir', default=DEFAULT_INPUTS)
    parser.add_argument('--scores-dir', default=DEFAULT_SCORES)
    parser.add_argument('--reference-dir', default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.command](args)


if __name__ == '__main__':
    main()
