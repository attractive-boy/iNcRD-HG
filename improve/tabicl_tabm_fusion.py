"""Fuse verified TabICLv2/TabM predictions; select on inner validation only.

prepare -> select -> evaluate -> verify. No model fitting, GPU inference, or test
data loading. Historical outer single-model results are already known.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

from improve import tabular_foundation as base


DEFAULT_OUT = 'improve/results/tabicl_tabm_fusion_20260912'
OWN = ('tabicl_tabm_fusion.py', 'test_tabicl_tabm_fusion.py', 'tabular_foundation.py')
METRICS = base.METRICS
EPS = 1e-7
PANELS = ('tabm', 'tabicl', 'equal_probability', 'selected_fusion')


def sources():
    return {name: base.digest(Path(__file__).with_name(name)) for name in OWN}


def versions():
    return {name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'scikit-learn', 'pandas')}


def recipes():
    # Shared endpoints occur once: 21 probability + 19 interior logit recipes.
    return [dict(id=f'{kind}_{i*5:03d}', kind=kind, tabicl_weight=i/20)
            for kind in ('probability', 'logit')
            for i in (range(21) if kind == 'probability' else range(1, 20))]


def combine(tabicl, tabm, recipe):
    a, b = np.asarray(tabicl, dtype=np.float64), np.asarray(tabm, dtype=np.float64)
    weight, kind = recipe['tabicl_weight'], recipe['kind']
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError('Expected two aligned nonempty probability vectors')
    if not all(np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all() for x in (a, b)):
        raise ValueError('Invalid probabilities')
    if not np.isfinite(weight) or not 0 <= weight <= 1 or kind not in ('probability', 'logit'):
        raise ValueError('Invalid fusion recipe')
    if weight == 0:
        return b.copy()
    if weight == 1:
        return a.copy()
    if kind == 'probability':
        return weight*a + (1-weight)*b
    a, b = np.clip(a, EPS, 1-EPS), np.clip(b, EPS, 1-EPS)
    return expit(weight*(np.log(a)-np.log1p(-a)) + (1-weight)*(np.log(b)-np.log1p(-b)))


def check_alignment(y, pairs, other_y, other_pairs):
    if y.ndim != 1 or pairs.shape != (2, len(y)) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected binary labels and two-row pair identifiers')
    if not np.issubdtype(pairs.dtype, np.integer) or len(set(map(tuple, pairs.T))) != len(y):
        raise ValueError('Pair identifiers must be unique integers')
    if not np.array_equal(y, other_y) or not np.array_equal(pairs, other_pairs):
        raise ValueError('Labels or pair ordering differ')


def prepare(args):
    out, tabicl = Path(args.out_dir).resolve(), Path(args.tabicl_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    tp = base.protocol(tabicl)
    ts = base.selection(tabicl, tp)
    tabm = Path(tp['reference_dir'])
    mp = base.audit_reference(tabm)
    ms = base.check_seal(base.read_json(tabm/'REFIT.json'))
    validations = {model: base.read_json(root/'VALIDATION.json')
                   for model, root in (('tabicl', tabicl), ('tabm', tabm))}
    for model, parent in (('tabicl', tp), ('tabm', mp)):
        v = validations[model]
        replay_key = 'all_estimators_replayed' if model == 'tabicl' else 'all_checkpoints_replayed'
        if parent['smoke'] or not v['passed'] or not v[replay_key] or v['protocol_sha256'] != parent['sha256']:
            raise ValueError('Need complete, verified formal parent predictions')
    if list(tp['seeds']) != [666, 0, 1] or list(mp['refit_seeds']) != list(tp['seeds']):
        raise ValueError('Expected matching three-seed panels')
    files, inputs, features = {}, {}, {}

    def remember(path, expected=None):
        path = Path(path).resolve()
        actual = base.digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Parent artifact changed: '+str(path))
        files[str(path)] = actual

    for root, names in ((tabicl, ('protocol.json', 'SELECTION.json', 'VALIDATION.json')),
                        (tabm, ('protocol.json', 'PARAMETERS.json', 'REFIT.json', 'VALIDATION.json', 'ANCHOR_VALIDATION.json'))):
        for name in names:
            remember(root/name)
    for parent in (tp, mp):
        for name, expected in parent['source_sha256'].items():
            remember(Path(__file__).with_name(name), expected)
    verified_paths = {model: {r['path'] for r in v['replays']} for model, v in validations.items()}
    for phase in ('inner', 'outer'):
        inputs[phase], features[phase] = {}, {}
        for task in tp['tasks']:
            feature_dir = tabm/'features'/phase/task
            manifest = base.read_json(feature_dir/'manifest.json')
            remember(feature_dir/'manifest.json')
            for name, expected in manifest['files'].items():
                remember(feature_dir/name, expected)
            features[phase][task] = str(feature_dir/'arrays.npz')
            inputs[phase][task] = {model: {} for model in ('tabicl', 'tabm')}
            for seed in tp['seeds']:
                stage = 'outer' if phase == 'outer' else 'screen' if seed == 666 else 'confirm'
                configs = {'tabicl': ts['chosen'][task]['config'], 'tabm': ms['chosen'][task]['tabm']['config']}
                paths = {'tabicl': tabicl/phase/task/configs['tabicl']['id']/f's{seed}',
                         'tabm': tabm/stage/task/'tabm'/configs['tabm']['id']/f's{seed}'}
                if phase == 'outer' and paths['tabm'] != tabm/tp['baseline'][task][str(seed)]:
                    raise ValueError('TabM baseline changed from the TabICL comparison')
                for model, dest in paths.items():
                    root, parent = (tabicl, tp) if model == 'tabicl' else (tabm, mp)
                    r = base.read_json(dest/'result.json')
                    expected_phase = phase if model == 'tabicl' else stage
                    if (r['task'], r['seed'], r['config'], r['protocol_sha256']) != (task, seed, configs[model], parent['sha256']):
                        raise ValueError('Wrong source model identity')
                    if r.get('phase', r.get('stage')) != expected_phase or r['feature_manifest_sha256'] != files[str((feature_dir/'manifest.json').resolve())]:
                        raise ValueError('Source phase or feature cache differs')
                    if str(dest.relative_to(root)) not in verified_paths[model]:
                        raise ValueError('Source prediction was not replayed in the parent experiment')
                    remember(dest/'result.json')
                    remember(dest/'scores.npz', r['artifacts_sha256']['scores.npz'])
                    inputs[phase][task][model][str(seed)] = str(dest)
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), tasks=tp['tasks'], seeds=tp['seeds'],
        recipes=recipes(), source_sha256=sources(), versions=versions(), reference_files=files,
        inputs=inputs, features=features, tabicl_dir=str(tabicl), tabm_dir=str(tabm), logit_clip=EPS,
        selection='Per task, maximize mean inner (AUC+AP)/2 over all three seed labels. Exact ties prefer an endpoint, then probability blending, then more TabICL weight. No outer-based selection.',
        panels=list(PANELS), primary='Selected fusion must improve mean AUC/AP over both constituents on both tasks, with at least 2/3 positive seed-label differences in every cell.',
        counts=dict(parent_predictions=24, inner_candidate_evaluations=240, outer_panel_evaluations=24, new_model_fits=0, new_model_inferences=0),
        no_test_evaluation=True, historical_outer_single_model_results_already_seen=True,
        limitations=['Historical inner and outer validation have been reused in development.',
                    'TabM inner checkpoints and both model configurations already used inner labels for selection; these are held-out predictions, not a fresh cross-fitting layer.',
                    'Seed labels align bookkeeping, not common randomness or independent data splits. All TabICL predictions share one pretrained checkpoint.',
                    'No calibrated-probability or statistical-significance claim. Model fits and GPU inference are reused, not repeated.']))
    out.mkdir(parents=True, exist_ok=True)
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name), out/'source_snapshot'/name)
    base.write_json(out/'protocol.json', p)
    base.write_json(out/'ENVIRONMENT.json', dict(executable=sys.executable, python=sys.version, versions=versions()))
    print('FROZEN', out, flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out)/'protocol.json'))
    if p['source_sha256'] != sources() or p['versions'] != versions():
        raise ValueError('Frozen fusion source/environment changed')
    for path, expected in p['reference_files'].items():
        if base.digest(path) != expected:
            raise ValueError('Frozen source prediction or provenance changed: '+path)
    return p


def selection(out, p):
    value = base.check_seal(base.read_json(Path(out)/'SELECTION.json'))
    if value['protocol_sha256'] != p['sha256'] or set(value['chosen']) != set(p['tasks']):
        raise ValueError('Selection belongs to another experiment')
    for name, expected in value['artifacts_sha256'].items():
        if base.digest(Path(out)/name) != expected:
            raise ValueError('Frozen inner results changed')
    if any(v['recipe'] not in p['recipes'] for v in value['chosen'].values()):
        raise ValueError('Unregistered fusion recipe')
    return value


def load_phase(out, p, phase, task):
    if phase not in ('inner', 'outer') or task not in p['tasks']:
        raise ValueError('Unknown task or phase')
    if phase == 'outer':
        selection(out, p)  # Must precede any outer np.load, including feature labels.
    with np.load(p['features'][phase][task], allow_pickle=False) as z:
        if any('test' in key.lower() for key in z.files):
            raise ValueError('Test arrays are forbidden')
        y, pairs = z['y_val'].copy(), z['pairs_val'].copy()
        yt, pt = z['y_train'].copy(), z['pairs_train'].copy()
    check_alignment(y, pairs, y, pairs)
    check_alignment(yt, pt, yt, pt)
    if set(map(tuple, pairs.T)) & set(map(tuple, pt.T)):
        raise ValueError('Fitting/evaluation pairs overlap')
    result = {}
    for seed in p['seeds']:
        result[seed] = {}
        for model in ('tabicl', 'tabm'):
            dest = Path(p['inputs'][phase][task][model][str(seed)])
            with np.load(dest/'scores.npz', allow_pickle=False) as z:
                if set(z.files) != {'val', 'y_val', 'pairs_val'}:
                    raise ValueError('Expected validation-only prediction archive')
                check_alignment(y, pairs, z['y_val'], z['pairs_val'])
                score = z['val'].astype(np.float64)
            combine(score, score, dict(kind='probability', tabicl_weight=1.))
            r = base.read_json(dest/'result.json')
            for name, value in base.metrics(y, score, pairs).items():
                if value is not None:
                    np.testing.assert_allclose(r[name], value, atol=1e-12, rtol=0)
            result[seed][model] = score
    return y, pairs, result


def choose(rows, candidates, seeds):
    summaries = []
    if len(rows) != len(candidates)*len(seeds):
        raise ValueError('Incomplete candidate-by-seed grid')
    for cfg in candidates:
        group = [r for r in rows if r['recipe_id'] == cfg['id']]
        if len(group) != len(seeds) or {r['seed'] for r in group} != set(seeds):
            raise ValueError('Missing or duplicate candidate seed')
        values = {key: float(np.mean([r[key] for r in group])) for key in (*METRICS, 'criterion')}
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Nonfinite selection metrics')
        summaries.append(dict(recipe=cfg, **values))
    return min(summaries, key=lambda r: (-r['criterion'], 0 if r['recipe']['tabicl_weight'] in (0, 1) else 1,
               0 if r['recipe']['kind'] == 'probability' else 1, -r['recipe']['tabicl_weight']))


def inner_results(out, p):
    rows, chosen = [], {}
    for task in p['tasks']:
        y, pairs, scores = load_phase(out, p, 'inner', task)
        group = []
        for cfg in p['recipes']:
            for seed in p['seeds']:
                score = combine(scores[seed]['tabicl'], scores[seed]['tabm'], cfg)
                group.append(dict(task=task, recipe_id=cfg['id'], kind=cfg['kind'], tabicl_weight=cfg['tabicl_weight'],
                                  seed=seed, **base.metrics(y, score, pairs)))
        chosen[task] = choose(group, p['recipes'], p['seeds'])
        rows.extend(group)
    return rows, chosen


def select(out):
    p = protocol(out)
    if (out/'SELECTION.json').exists():
        selection(out, p)
        return
    rows, chosen = inner_results(out, p)
    pd.DataFrame(rows).to_csv(out/'inner_results.csv', index=False)
    pd.DataFrame(rows).groupby(['task', 'recipe_id', 'kind', 'tabicl_weight'])[list(METRICS)+['criterion']].mean().to_csv(out/'inner_summary.csv')
    base.write_json(out/'SELECTION.json', base.seal(dict(protocol_sha256=p['sha256'], chosen=chosen,
        created_at_utc=datetime.now(timezone.utc).isoformat(), artifacts_sha256={name:base.digest(out/name) for name in ('inner_results.csv', 'inner_summary.csv')})))
    print('INNER_SELECTION', chosen, flush=True)


def panel_recipes(chosen):
    return dict(tabm=dict(kind='probability', tabicl_weight=0.), tabicl=dict(kind='probability', tabicl_weight=1.),
                equal_probability=dict(kind='probability', tabicl_weight=.5), selected_fusion=chosen['recipe'])


def evaluate(out):
    p = protocol(out)
    selected = selection(out, p)
    if (out/'EVALUATION.json').exists():
        raise ValueError('Outer evaluation already exists; use verify')
    rows, artifacts = [], {}
    for task in p['tasks']:
        y, pairs, scores = load_phase(out, p, 'outer', task)
        for seed in p['seeds']:
            predictions = {name: combine(scores[seed]['tabicl'], scores[seed]['tabm'], cfg)
                           for name, cfg in panel_recipes(selected['chosen'][task]).items()}
            dest = out/'outer'/task/f's{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/'scores.npz', y_val=y, pairs_val=pairs, **predictions)
            artifacts[str((dest/'scores.npz').relative_to(out))] = base.digest(dest/'scores.npz')
            for method, prediction in predictions.items():
                rows.append(dict(task=task, seed=seed, method=method, **base.metrics(y, prediction, pairs)))
    pd.DataFrame(rows).to_csv(out/'outer_results.csv', index=False)
    artifacts['outer_results.csv'] = base.digest(out/'outer_results.csv')
    base.write_json(out/'EVALUATION.json', base.seal(dict(protocol_sha256=p['sha256'], selection_sha256=selected['sha256'],
        created_at_utc=datetime.now(timezone.utc).isoformat(), artifacts_sha256=artifacts)))
    print('OUTER_EVALUATION_SAVED', len(rows), 'score panels', flush=True)


def compare(frame, p):
    gates, deltas = {}, []
    for method in ('equal_probability', 'selected_fusion'):
        gates[method] = {}
        for baseline in ('tabm', 'tabicl'):
            gates[method][baseline] = {}
            for task in p['tasks']:
                old = frame[(frame.task == task) & (frame.method == baseline)].set_index('seed')
                new = frame[(frame.task == task) & (frame.method == method)].set_index('seed')
                if len(old) != 3 or len(new) != 3 or set(old.index) != set(p['seeds']) or set(new.index) != set(p['seeds']):
                    raise ValueError('Incomplete outer panel')
                d = new[list(METRICS)]-old[list(METRICS)]
                means = {key:float(d[key].mean()) for key in METRICS}
                positive = {key:int((d[key] > 0).sum()) for key in ('auc', 'ap')}
                gates[method][baseline][task] = dict(mean_deltas=means, positive_seed_labels=positive,
                    passed=all(means[key] > 0 and positive[key] >= 2 for key in ('auc', 'ap')))
                for seed, row in d.iterrows():
                    deltas.append(dict(task=task, method=method, baseline=baseline, seed=int(seed), **row.to_dict()))
    joint = {method:all(v['passed'] for tasks in refs.values() for v in tasks.values()) for method, refs in gates.items()}
    return gates, joint, deltas


def verify(out):
    p = protocol(out)
    selected = selection(out, p)
    evaluation = base.check_seal(base.read_json(out/'EVALUATION.json'))
    if (evaluation['protocol_sha256'], evaluation['selection_sha256']) != (p['sha256'], selected['sha256']):
        raise ValueError('Outer evaluation used another selection')
    for name, expected in evaluation['artifacts_sha256'].items():
        if base.digest(out/name) != expected:
            raise ValueError('Saved fusion output changed')
    inner_rows, chosen = inner_results(out, p)
    if chosen != selected['chosen']:
        raise ValueError('Cannot reproduce inner fusion selection')
    pd.testing.assert_frame_equal(pd.read_csv(out/'inner_results.csv'), pd.DataFrame(inner_rows), check_dtype=False, atol=1e-12, rtol=0)
    rows, replayed = [], []
    for task in p['tasks']:
        y, pairs, scores = load_phase(out, p, 'outer', task)
        for seed in p['seeds']:
            path = out/'outer'/task/f's{seed}'/'scores.npz'
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(PANELS) | {'y_val', 'pairs_val'}:
                    raise ValueError('Unexpected saved fusion arrays')
                check_alignment(y, pairs, saved['y_val'], saved['pairs_val'])
                for method, cfg in panel_recipes(chosen[task]).items():
                    score = combine(scores[seed]['tabicl'], scores[seed]['tabm'], cfg)
                    np.testing.assert_array_equal(score, saved[method])
                    replayed.append(dict(task=task, seed=seed, method=method, max_abs=float(np.max(np.abs(score-saved[method])))))
                    rows.append(dict(task=task, seed=seed, method=method, **base.metrics(y, score, pairs)))
    frame = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(pd.read_csv(out/'outer_results.csv'), frame, check_dtype=False, atol=1e-12, rtol=0)
    summary = frame.groupby(['task', 'method'])[list(METRICS)].agg(['mean', 'std'])
    summary.to_csv(out/'summary.csv')
    gates, joint, deltas = compare(frame, p)
    pd.DataFrame(deltas).to_csv(out/'paired_deltas.csv', index=False)
    labels = dict(tabm='TabM', tabicl='TabICLv2', equal_probability='概率 50:50', selected_fusion='内部选定融合')
    lines = ['# TabICLv2 + TabM 分数融合', '',
        '沿用历史外层验证，以下为 seed 666/0/1 的均值。权重仅在内部验证选择；此前单模型外层结果已经查看。', '',
        '| 任务 | 方案 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for method in PANELS:
            r = summary.loc[(task, method)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f |' % (task, labels[method], *(r[(key, 'mean')] for key in METRICS)))
    lines += ['', '## 内部冻结的融合配置', '',
              '| 任务 | 方式 | TabICLv2 权重 | TabM 权重 | 内部 AUC | 内部 AP |', '|---|---|---:|---:|---:|---:|']
    for task, value in chosen.items():
        cfg = value['recipe']
        lines.append('| %s | %s | %.2f | %.2f | %.6f | %.6f |' %
                     (task, cfg['kind'], cfg['tabicl_weight'], 1-cfg['tabicl_weight'], value['auc'], value['ap']))
    lines += ['', '概率融合为 p = w·p_TabICL + (1−w)·p_TabM；logit 融合为 sigmoid(w·logit(p_TabICL) + (1−w)·logit(p_TabM))，端点概率在计算 logit 前截断至 [1e−7, 1−1e−7]。', '',
        '内部比较 0.05 步长的两种融合，共 40 个不重复配置（含两个单模型端点）；每个任务共用三个种子均值选择的一套方式和权重，AUC/AP 不分别选权重。', '',
        '## 相对 TabICLv2 的外层变化', '', '| 任务 | 方案 | ΔAUC | 正向种子标签 | ΔAP | 正向种子标签 | Δ宏 AP |', '|---|---|---:|---:|---:|---:|---:|']
    for method in ('equal_probability', 'selected_fusion'):
        for task, value in gates[method]['tabicl'].items():
            m, c = value['mean_deltas'], value['positive_seed_labels']
            lines.append('| %s | %s | %+.6f | %d/3 | %+.6f | %d/3 | %+.6f |' %
                         (task, labels[method], m['auc'], c['auc'], m['ap'], c['ap'], m['macro_drug_ap']))
    lines += ['', '## 核验与解释边界', '',
        '- 两任务均超过两个单模型的预设判定：' + str(joint) + '。均值 AUC/AP 都需提高，每项至少 2/3 个种子标签的差值为正；50:50 对照和内部选定融合分开判定。',
        '- 复用 24 份已由前序实验重放的单模型预测；本轮没有新增模型训练或 GPU 推理。内部 240 次候选评分全部复算，外层 6 份结果文件的 24 列预测（含单模型对照）重算完全一致。',
        '- 标签与 RNA–drug 配对顺序逐行核对，拟合/验证配对不重叠；输入、选择、源码和结果均校验哈希。选择在本轮首次解读外层数组前冻结。',
        '- TabM 内部检查点和两模型配置此前也使用了内部验证进行选择；这不是新增的交叉拟合层，不能将内部成绩当作无偏泛化估计。',
        '- 外层历史验证已多次参与开发，单模型结果在本轮之前已知。本轮没有读取测试数组；收益仍需独立数据验证。',
        '- 三个种子标签不代表三个独立划分或独立 TabICL 预训练模型，同号配对只用于记账。融合不是新的预训练模型，也不自动保证概率校准。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    base.write_json(out/'OUTCOME.json', dict(protocol_sha256=p['sha256'], gates=gates, joint=joint, scope_complete=True))
    base.write_json(out/'VALIDATION.json', dict(passed=True, protocol_sha256=p['sha256'], selection_replayed=True,
        input_predictions_aligned=True, no_test_evaluation=True, new_model_fits=0, new_model_inferences=0,
        inner_candidate_evaluations=len(inner_rows), outer_score_columns_replayed=len(replayed), replays=replayed))
    print(summary.to_string(), flush=True)
    print('VERIFIED', len(replayed), 'score columns; joint outcome:', joint, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'select', 'evaluate', 'verify'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--tabicl-dir', default=base.DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        {'select': select, 'evaluate': evaluate, 'verify': verify}[args.command](Path(args.out_dir).resolve())


if __name__ == '__main__':
    main()
