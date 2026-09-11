"""Evaluate the frozen MiDrug path smoothing mechanisms on LncDrug."""
import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from improve.attention_mechanisms import save_csv
from improve.drug_condition_attention import file_hash, make_base, metrics, write_json
from improve.edge_path_attention import configure, predict, run_path
from improve.smooth_path_attention import ARMS, ATTENTION_ARMS, SETTINGS, SmoothPathModel, train_one
from improve.smooth_path_attention import source_hashes as smooth_source_hashes
from improve.verify_edge_path_attention import require, state_hash, check_metrics, verify_attention


def source_hashes():
    result = smooth_source_hashes()
    result[Path(__file__).name] = file_hash(__file__)
    return result


def jobs(p):
    for i, seed in enumerate(p['seeds']):
        # Preserve the LncDrug device assignment of the original two-task run.
        group = len(p['seeds']) + i
        for arm in p['arms']:
            for repeat in range(2 if arm == 'base' or p['smoke'] else 1):
                yield group, 'LncDrug', seed, arm, repeat


def initialise(args):
    root = Path(args.out_dir)
    require(not (root / 'protocol.json').exists(), 'Plan already exists')
    parent_path = Path(args.input_protocol)
    parent = json.loads(parent_path.read_text())
    parent_validation = parent_path.with_name('VALIDATION.json')
    validation = json.loads(parent_validation.read_text())
    require(validation['passed'] and validation['protocol_sha256'] == file_hash(parent_path), 'Parent is not verified')
    mi_path = Path(args.mi_protocol)
    mi = json.loads(mi_path.read_text())
    require(mi['settings'] == json.loads(json.dumps(SETTINGS)), 'MiDrug attention settings differ')
    require(mi['source_hashes'] == smooth_source_hashes(), 'MiDrug source changed')
    p = copy.deepcopy(parent)
    for name in ('configs', 'inputs', 'assets'):
        p[name] = {'LncDrug': p[name]['LncDrug']}
    if args.smoke:
        p['configs']['LncDrug']['epochs'] = 3
    p.update(tasks=['LncDrug'], seeds=[666] if args.smoke else mi['seeds'], arms=list(ARMS),
             primary_arms=list(ATTENTION_ARMS), settings=SETTINGS, smoke=args.smoke,
             parent_protocol=str(parent_path.resolve()), parent_protocol_sha256=file_hash(parent_path),
             parent_validation_sha256=file_hash(parent_validation), mi_protocol=str(mi_path.resolve()),
             mi_protocol_sha256=file_hash(mi_path), source_hashes=source_hashes(),
             contrasts=mi['contrasts'],
             repeat_policy='two identical baselines per seed; smoke repeats every arm twice',
             scope='LncDrug transfer of two frozen attention mechanisms; fixed-split exploratory comparison',
             selection='one checkpoint maximizing validation AUC/AUPR mean; earliest exact tie')
    p['expected_runs'] = len(list(jobs(p)))
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / 'source_snapshot'
    snapshot.mkdir()
    for name in p['source_hashes']:
        (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_json(root / 'protocol.json', p)
    print('INITIALISED', root, 'runs', p['expected_runs'], flush=True)


def load_inputs(p):
    require(p['source_hashes'] == source_hashes(), 'Frozen source changed')
    require(file_hash(p['parent_protocol']) == p['parent_protocol_sha256'], 'Parent protocol changed')
    require(file_hash(p['mi_protocol']) == p['mi_protocol_sha256'], 'MiDrug protocol changed')
    data = {}
    for fold, entry in p['inputs']['LncDrug']['folds'].items():
        require(file_hash(entry['path']) == entry['sha256'], 'Frozen graph changed')
        data[fold] = torch.load(entry['path'], map_location='cpu')
    entry = p['assets']['LncDrug']
    require(file_hash(entry['path']) == entry['sha256'], 'Frozen paths changed')
    return data, torch.load(entry['path'], map_location='cpu')


def run(args):
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    data, assets = load_inputs(p)
    require(0 <= args.shard_index < args.shard_count, 'Invalid shard')
    for group, task, seed, arm, repeat in jobs(p):
        if group % args.shard_count == args.shard_index:
            train_one(root, p, data, assets, task, seed, arm, repeat, args.device)


def summarize(args):
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    data, assets = load_inputs(p)
    config, key = p['configs']['LncDrug'], ('lncRNA', 'LncDrug', 'drug')
    protocol_hash = file_hash(root / 'protocol.json')
    parent_root = Path(p['parent_protocol']).parent
    records, initials, rows, controls, repeats, reloads = {}, {}, [], [], [], []
    max_replay_delta, attention_arrays = 0., 0
    for _, task, seed, arm, repeat in jobs(p):
        directory = run_path(root, task, seed, arm, repeat)
        r = json.loads((directory / 'result.json').read_text())
        signature = dict(task=task, seed=seed, arm=arm, repeat=repeat, protocol_sha256=protocol_hash)
        require(r['signature'] == signature, 'Wrong run signature')
        require(json.loads((directory / 'config.json').read_text()) == signature, 'Wrong run config')
        require(r['trained_epochs'] == config['epochs'] and r['validation_replay_delta'] == 0., 'Incomplete or changed run')
        trace = pd.read_csv(directory / 'trace.csv', float_precision='round_trip')
        require(trace.epoch.tolist() == list(range(1, config['epochs'] + 1)), 'Missing epochs')
        require(np.isfinite(trace.to_numpy()).all(), 'Nonfinite trace')
        np.testing.assert_allclose(trace.criterion, (trace.val_auc + trace.val_aupr) / 2, atol=1e-14, rtol=0)
        require(int(trace.loc[trace.criterion.idxmax(), 'epoch']) == r['selected_epoch'], 'Wrong checkpoint selection')
        require(abs(trace.criterion.max() - (r['metrics']['val']['auc'] + r['metrics']['val']['aupr']) / 2) < 1e-14,
                'Selected metric differs from trace')
        state = torch.load(directory / 'checkpoint.pt', map_location='cpu')
        require(state_hash(state) == r['selected_state_sha256'], 'Checkpoint changed')
        records[seed, arm, repeat] = r
        with np.load(directory / 'predictions.npz') as values:
            initials[seed, arm, repeat] = values['initial_val_logits'].copy()
            for fold in ('val', 'test'):
                np.testing.assert_array_equal(values['y_' + fold], data[fold][key].edge_label.numpy())
                np.testing.assert_array_equal(values['pairs_' + fold], assets['pairs'][fold].numpy())
                check_metrics(values['y_' + fold], values[fold], values['pairs_' + fold], r['metrics'][fold])
                for mode, changed in r['interventions'].items():
                    check_metrics(values['y_' + fold], values[mode + '_' + fold], values['pairs_' + fold], changed[fold])
            attention_arrays += verify_attention(directory, arm, data['train'], assets)
            reference_dirs = []
            if repeat:
                reference_dirs.append(('repeat', run_path(root, task, seed, arm, 0)))
            if not p['smoke'] and arm in ('base', 'path_uniform', 'path_attention'):
                reference_dirs.append(('parent', run_path(parent_root, task, seed, arm, repeat)))
            for kind, ref in reference_dirs:
                previous = json.loads((ref / 'result.json').read_text())
                require(previous['device'] == r['device'], 'Control device differs')
                require(previous['selected_state_sha256'] == r['selected_state_sha256'], 'Control state differs')
                require(file_hash(ref / 'trace.csv') == file_hash(directory / 'trace.csv'), 'Control trace differs')
                with np.load(ref / 'predictions.npz') as reference:
                    require(set(reference.files) == set(values.files), 'Control prediction keys differ')
                    for name in values.files:
                        np.testing.assert_array_equal(values[name], reference[name])
                (repeats if kind == 'repeat' else controls).append(dict(seed=seed, arm=arm, repeat=repeat, exact=True))
            if not repeat and seed == p['seeds'][0]:
                base = make_base(data['train'], config, seed, 'cpu')
                model = SmoothPathModel(base, assets, config, arm, p['width'])
                model.load_state_dict(state)
                for fold in ('val', 'test'):
                    for mode in (None, *r['interventions']):
                        scores = predict(model, data['train'], fold, mode)
                        name = fold if mode is None else mode + '_' + fold
                        np.testing.assert_allclose(scores, values[name], rtol=1e-4, atol=1e-3)
                        actual = metrics(values['y_' + fold], scores, values['pairs_' + fold], by_drug=False)
                        expected = r['metrics'][fold] if mode is None else r['interventions'][mode][fold]
                        delta = max(abs(actual[m] - expected[m]) for m in ('auc', 'aupr'))
                        require(delta < 1e-4, 'CPU replay metrics differ')
                        max_replay_delta = max(max_replay_delta, delta)
                reloads.append(dict(seed=seed, arm=arm))
        if not repeat:
            row = dict(task=task, seed=seed, arm=arm, selected_epoch=r['selected_epoch'])
            for fold, values in r['metrics'].items():
                row.update({fold + '_' + name: value for name, value in values.items()})
            rows.append(row)
        del state
    require(len(records) == p['expected_runs'], 'Missing runs')
    for seed in p['seeds']:
        require(len({records[seed, arm, 0]['initial_core_sha256'] for arm in ARMS}) == 1, 'Different initial core weights')
        for arm in ARMS:
            np.testing.assert_array_equal(initials[seed, arm, 0], initials[seed, 'base', 0])
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'per_seed.csv', index=False)
    summary = frame.groupby('arm', sort=False).agg(n=('seed', 'count'), val_auc=('val_auc', 'mean'),
        val_aupr=('val_aupr', 'mean'), test_auc=('test_auc', 'mean'), test_aupr=('test_aupr', 'mean'),
        per_drug_auc=('test_per_drug_auc', 'mean'), per_drug_aupr=('test_per_drug_aupr', 'mean')).reset_index()
    summary.to_csv(root / 'summary.csv', index=False)
    comparisons, interventions = [], []
    for arm, reference in p['contrasts']:
        row = dict(task='LncDrug', arm=arm, reference=reference, n=len(p['seeds']))
        for fold, metric in itertools.product(('val', 'test'), ('auc', 'aupr', 'per_drug_auc', 'per_drug_aupr')):
            delta = np.array([records[s, arm, 0]['metrics'][fold][metric] - records[s, reference, 0]['metrics'][fold][metric]
                              for s in p['seeds']])
            name = fold + '_' + metric
            row.update({name + '_delta': float(delta.mean()), name + '_wins': int((delta > 0).sum()),
                        name + '_std': float(delta.std(ddof=1)) if len(delta) > 1 else 0.})
        comparisons.append(row)
    for arm, mode in itertools.product(ATTENTION_ARMS, ('uniform', 'branch_off')):
        row = dict(arm=arm, intervention=mode, n=len(p['seeds']))
        for fold, metric in itertools.product(('val', 'test'), ('auc', 'aupr')):
            delta = [records[s, arm, 0]['interventions'][mode][fold][metric] - records[s, arm, 0]['metrics'][fold][metric]
                     for s in p['seeds']]
            row[fold + '_' + metric + '_delta'] = float(np.mean(delta))
            row[fold + '_' + metric + '_decreases'] = sum(d < 0 for d in delta)
        interventions.append(row)
    save_csv(root / 'comparisons.csv', comparisons)
    save_csv(root / 'interventions.csv', interventions)
    write_json(root / 'CONTROL_REPLAY.json', dict(parent_controls=controls, within_run_repeats=repeats))
    write_json(root / 'VALIDATION.json', dict(passed=True, runs_verified=len(records), exact_parent_controls=len(controls),
        exact_repeats=len(repeats), cpu_replays=reloads, maximum_cpu_gpu_metric_difference=max_replay_delta,
        attention_arrays_verified=attention_arrays, protocol_sha256=protocol_hash, source_hashes=source_hashes()))
    winner = summary.loc[((summary.val_auc + summary.val_aupr) / 2).idxmax(), 'arm']
    lines = ['# 冻结平滑机制在 LncDrug 上的验证', '',
        f"完成 {len(records)}/{p['expected_runs']} 次训练；按验证集均值选择 `{winner}`。",
        f'{len(controls)} 个共同对照完全复现上一轮，{len(repeats)} 组严格重复一致。', '',
        '| 方案 | Val AUC | Val AUPR | Test AUC | Test AUPR |', '|---|---:|---:|---:|---:|']
    for row in summary.to_dict('records'):
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f |' % tuple(row[n] for n in ('arm','val_auc','val_aupr','test_auc','test_aupr')))
    lines.extend(['', '## 同一机制的两个任务对照', '',
        '| 任务 | 方案 | 对照 | Δ Test AUC | AUC 提升种子 | Δ Test AUPR | AUPR 提升种子 |',
        '|---|---|---|---:|---:|---:|---:|'])
    joint = pd.DataFrame(comparisons)
    if not p['smoke']:
        mi = pd.read_csv(Path(p['mi_protocol']).with_name('comparisons.csv'))
        mi.insert(0, 'task', 'MiDrug')
        joint = pd.concat([mi, joint], ignore_index=True)
        joint.to_csv(root / 'joint_comparisons.csv', index=False)
    for row in joint[joint.arm.isin(['temperature4','shrink25']) & joint.reference.isin(['base','path_uniform'])].to_dict('records'):
        lines.append('| %s | %s | %s | %+.6f | %d/%d | %+.6f | %d/%d |' %
            (row['task'],row['arm'],row['reference'],row['test_auc_delta'],row['test_auc_wins'],row['n'],
             row['test_aupr_delta'],row['test_aupr_wins'],row['n']))
    lines.extend(['', '温度 4、收缩比例 0.25 均直接沿用 MiDrug，未针对 LncDrug 搜索。',
        'MiDrug 与 LncDrug 分别沿用各自原模型宽度、聚合与训练日程。仍是固定划分上的探索，未验证新的数据划分。', ''])
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--out-dir', required=True)
    init.add_argument('--input-protocol', default='improve/results/edge_path_attention_20260908/protocol.json')
    init.add_argument('--mi-protocol', default='improve/results/smooth_path_attention_20260908/protocol.json')
    init.add_argument('--smoke', action='store_true')
    worker = sub.add_parser('run')
    worker.add_argument('--out-dir', required=True)
    worker.add_argument('--device', default='cuda:0')
    worker.add_argument('--shard-index', type=int, default=0)
    worker.add_argument('--shard-count', type=int, default=1)
    summary = sub.add_parser('summarize')
    summary.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    configure()
    {'init': initialise, 'run': run, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
