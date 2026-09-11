"""Validate and summarize the complementary attention experiment."""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from improve.attention_mechanisms import save_csv
from improve.drug_condition_attention import file_hash, make_base, metrics, write_json
from improve.complementary_attention import jobs, load_inputs, make_model, source_hashes
from improve.complementary_attention_modules import ARMS, FUSIONS
from improve.edge_path_attention import configure, predict, run_path
from improve.verify_edge_path_attention import require, state_hash, check_metrics


def verify_routing(data, paths, routes, cfg, task):
    by_rna, by_drug = defaultdict(set), defaultdict(set)
    for rna, drug in data['train'][(cfg['rna'], task, 'drug')].edge_index.t().tolist():
        by_rna[rna].add(drug)
        by_drug[drug].add(rna)
    checked = 0
    for fold, pairs in paths['pairs'].items():
        counts = np.stack([pack['counts'].numpy() for pack in paths['packs'][fold]], -1)
        degrees = np.array([[len(by_rna[rna] - {drug}), len(by_drug[drug] - {rna})]
                            for rna, drug in pairs.t().tolist()], dtype=np.float32)
        expected = np.concatenate([np.log1p(degrees), np.log1p(counts), (counts > 0).astype(np.float32)], -1)
        np.testing.assert_allclose(routes[fold]['features'].numpy(), expected, rtol=1e-6, atol=1e-7)
        mask = routes[fold]['available'].numpy()
        require(mask.shape == (len(counts), 3) and mask[:, :2].all(), 'Wrong expert availability')
        np.testing.assert_array_equal(mask[:, 2], (counts > 0).any(1))
        checked += len(counts)
    return checked


def verify_attention(directory, task, seed, arm, data, paths, routes, cfg):
    diagnostics, arrays_checked = [], 0
    expected = set()
    if arm == 'channel_attention' or (arm in FUSIONS and arm != 'fusion_gate_simple'):
        expected.update('channel_' + nt for nt in (cfg['rna'], 'drug'))
    if arm == 'path_attention' or arm in FUSIONS:
        expected.update('path_' + str(i) for i in range(len(paths['families'])))
    if arm in FUSIONS:
        expected.add('router')
    for fold in ('val', 'test'):
        file = directory / ('attention_' + fold + '.npz')
        if not expected:
            require(not file.exists(), 'Unexpected attention artifact')
            continue
        with np.load(file) as arrays:
            require(set(arrays.files) == expected, 'Missing or extra attention arrays')
            for name in arrays.files:
                weights = arrays[name]
                require(np.isfinite(weights).all() and (weights >= 0).all(), 'Invalid attention weights')
                if name.startswith('channel_'):
                    nt = name[len('channel_'):]
                    require(weights.shape == (data['train'][nt].num_nodes, cfg['hidden']), 'Wrong channel shape')
                    require((weights <= 2.).all(), 'Channel gain out of range')
                elif name.startswith('path_'):
                    i = int(name.split('_')[-1])
                    valid = paths['packs'][fold][i]['valid'].numpy()
                    require(weights.shape == valid.shape and (weights[~valid] == 0).all(), 'Invalid path mask')
                    np.testing.assert_allclose(weights.sum(1), valid.any(1).astype(float), atol=2e-6)
                    if arm == 'fusion_gate_simple':
                        uniform = valid / np.maximum(valid.sum(1, keepdims=True), 1)
                        np.testing.assert_allclose(weights, uniform, rtol=1e-6, atol=1e-7)
                else:
                    valid = routes[fold]['available'].numpy().copy()
                    if arm == 'fusion_no_null':
                        valid[:, 0] = False
                    require(weights.shape == valid.shape and (weights[~valid] == 0).all(), 'Invalid router mask')
                    np.testing.assert_allclose(weights.sum(1), 1., atol=2e-6)
                    if arm == 'fusion_fixed':
                        np.testing.assert_allclose(weights, valid / valid.sum(1, keepdims=True), atol=1e-7)
                    for present in (False, True):
                        take = routes[fold]['available'][:, 2].numpy() == present
                        if not take.any():
                            continue
                        selected = weights[take]
                        diagnostics.append(dict(task=task, seed=seed, arm=arm, fold=fold,
                            has_path=present, pairs=int(take.sum()), null_weight=float(selected[:, 0].mean()),
                            channel_weight=float(selected[:, 1].mean()), path_weight=float(selected[:, 2].mean()),
                            null_weight_std=float(selected[:, 0].std()), channel_weight_std=float(selected[:, 1].std()),
                            path_weight_std=float(selected[:, 2].std()), null_dominant_fraction=float((selected.argmax(1) == 0).mean())))
                arrays_checked += 1
    return arrays_checked, diagnostics


def compare_reference(directory, reference, state, result, arrays, path_control=False):
    previous = json.loads((reference / 'result.json').read_text())
    require(previous['device'] == result['device'], 'Different reference device')
    previous_state = torch.load(reference / 'checkpoint.pt', map_location='cpu')
    current_state = {('branch.' + name[5:] if path_control and name.startswith('path.') else name): value
                     for name, value in state.items()}
    require(state_hash(current_state) == state_hash(previous_state), 'Common control state differs')
    require(file_hash(directory / 'trace.csv') == file_hash(reference / 'trace.csv'), 'Common control trace differs')
    with np.load(reference / 'predictions.npz') as earlier:
        for name in arrays.files:
            mapped = name
            if path_control and name.startswith('base_only_'):
                mapped = name.replace('base_only_', 'branch_off_', 1)
            if path_control and name.startswith('path_uniform_'):
                mapped = name.replace('path_uniform_', 'uniform_', 1)
            require(mapped in earlier.files, 'Reference prediction missing')
            np.testing.assert_array_equal(arrays[name], earlier[mapped])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    configure()
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    data, paths, routes = load_inputs(p)
    protocol_hash = file_hash(root / 'protocol.json')
    records, initial, rows, controls, repeats, reloads, routing = {}, {}, [], [], [], [], []
    attention_count, covariate_rows, max_replay_delta = 0, 0, 0.
    for task in p['tasks']:
        covariate_rows += verify_routing(data[task], paths[task], routes[task], p['configs'][task], task)
        reference = p['references'][task]
        require(file_hash(Path(reference['root']) / 'protocol.json') == reference['protocol_sha256'], 'Reference protocol changed')
        require(file_hash(Path(reference['root']) / 'VALIDATION.json') == reference['validation_sha256'], 'Reference verification changed')
    for _, task, seed, arm, repeat in jobs(p):
        cfg = p['configs'][task]
        key = (cfg['rna'], task, 'drug')
        directory = run_path(root, task, seed, arm, repeat)
        result = json.loads((directory / 'result.json').read_text())
        signature = dict(task=task, seed=seed, arm=arm, repeat=repeat, protocol_sha256=protocol_hash)
        require(result['signature'] == signature, 'Run signature differs')
        require(json.loads((directory / 'config.json').read_text()) == signature, 'Run config differs')
        require(result['trained_epochs'] == cfg['epochs'] and result['validation_replay_delta'] == 0., 'Incomplete or changed run')
        trace = pd.read_csv(directory / 'trace.csv', float_precision='round_trip')
        require(trace.epoch.tolist() == list(range(1, cfg['epochs'] + 1)), 'Missing epochs')
        require(np.isfinite(trace.to_numpy()).all(), 'Nonfinite trace')
        np.testing.assert_allclose(trace.criterion, (trace.val_auc + trace.val_aupr) / 2, atol=1e-14, rtol=0)
        require(int(trace.loc[trace.criterion.idxmax(), 'epoch']) == result['selected_epoch'], 'Wrong checkpoint selection')
        require(abs(trace.criterion.max() - sum(result['metrics']['val'][m] for m in ('auc','aupr')) / 2) < 1e-14,
                'Selected metric differs from trace')
        state = torch.load(directory / 'checkpoint.pt', map_location='cpu')
        require(state_hash(state) == result['selected_state_sha256'], 'Checkpoint changed')
        records[task, seed, arm, repeat] = result
        with np.load(directory / 'predictions.npz') as arrays:
            initial[task, seed, arm, repeat] = arrays['initial_val_logits'].copy()
            for fold in ('val', 'test'):
                np.testing.assert_array_equal(arrays['y_' + fold], data[task][fold][key].edge_label.numpy())
                np.testing.assert_array_equal(arrays['pairs_' + fold], paths[task]['pairs'][fold].numpy())
                check_metrics(arrays['y_' + fold], arrays[fold], arrays['pairs_' + fold], result['metrics'][fold])
                for mode, changed in result['interventions'].items():
                    check_metrics(arrays['y_' + fold], arrays[mode + '_' + fold], arrays['pairs_' + fold], changed[fold])
            count, diagnostics = verify_attention(directory, task, seed, arm, data[task], paths[task], routes[task], cfg)
            attention_count += count
            if not repeat:
                routing.extend(diagnostics)
            if repeat:
                compare_reference(directory, run_path(root, task, seed, arm, 0), state, result, arrays)
                repeats.append(dict(task=task, seed=seed, arm=arm, exact=True))
            if not p['smoke'] and arm in ('base', 'path_attention'):
                ref_arm = 'temperature4' if arm == 'path_attention' else 'base'
                reference = run_path(Path(p['references'][task]['root']), task, seed, ref_arm, repeat)
                compare_reference(directory, reference, state, result, arrays, path_control=arm == 'path_attention')
                controls.append(dict(task=task, seed=seed, arm=arm, repeat=repeat, exact=True))
            if not repeat and seed == p['seeds'][0]:
                base = make_base(data[task]['train'], cfg, seed, 'cpu')
                model = make_model(base, paths[task], routes[task], cfg, arm, seed, p)
                model.load_state_dict(state)
                for fold in ('val', 'test'):
                    for mode in (None, *result['interventions']):
                        scores = predict(model, data[task]['train'], fold, mode)
                        name = fold if mode is None else mode + '_' + fold
                        np.testing.assert_allclose(scores, arrays[name], rtol=2e-4, atol=1e-3)
                        actual = metrics(arrays['y_' + fold], scores, arrays['pairs_' + fold], by_drug=False)
                        expected = result['metrics'][fold] if mode is None else result['interventions'][mode][fold]
                        difference = max(abs(actual[m] - expected[m]) for m in ('auc','aupr'))
                        require(difference < 1e-4, 'CPU replay metric differs')
                        max_replay_delta = max(max_replay_delta, difference)
                reloads.append(dict(task=task, seed=seed, arm=arm))
                del model, base
        if not repeat:
            row = dict(task=task, seed=seed, arm=arm, selected_epoch=result['selected_epoch'],
                       trainable_parameters=result['trainable_parameters'])
            for fold, values in result['metrics'].items():
                row.update({fold + '_' + name: value for name, value in values.items()})
            rows.append(row)
        del state
        print('VERIFIED', task, seed, arm, repeat, flush=True)
    require(len(records) == p['expected_runs'], 'Incomplete experiment')
    for task, seed in itertools.product(p['tasks'], p['seeds']):
        require(len({records[task, seed, arm, 0]['initial_core_sha256'] for arm in ARMS}) == 1, 'Base initializations differ')
        for expert in ('channel','path','router'):
            hashes = {records[task, seed, arm, 0]['initial_experts_sha256'][expert] for arm in ARMS
                      if expert in records[task, seed, arm, 0]['initial_experts_sha256']}
            require(len(hashes) == 1, 'Expert initializations differ: ' + expert)
        for arm in ARMS:
            np.testing.assert_array_equal(initial[task, seed, arm, 0], initial[task, seed, 'base', 0])
    summarize(root, p, rows, records, routing)
    write_json(root / 'CONTROL_REPLAY.json', dict(parent_controls=controls, repeated_runs=repeats))
    write_json(root / 'VALIDATION.json', dict(passed=True, runs_verified=len(records),
        exact_parent_controls=len(controls), exact_repeats=len(repeats),
        independent_covariate_rows=covariate_rows, attention_arrays_verified=attention_count,
        matched_initial_core_and_experts=True, initial_predictions_equal=True,
        cpu_replays=reloads, maximum_cpu_gpu_metric_difference=max_replay_delta,
        protocol_sha256=protocol_hash, verifier_sha256=file_hash(__file__)))
    print('ALL VERIFIED', len(records), 'runs', flush=True)


def summarize(root, p, rows, records, routing):
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'per_seed.csv', index=False)
    summary = frame.groupby(['task','arm'], sort=False).agg(n=('seed','count'), val_auc=('val_auc','mean'),
        val_aupr=('val_aupr','mean'), test_auc=('test_auc','mean'), test_aupr=('test_aupr','mean'),
        per_drug_auc=('test_per_drug_auc','mean'), per_drug_aupr=('test_per_drug_aupr','mean')).reset_index()
    summary.to_csv(root / 'summary.csv', index=False)
    comparisons, intervention_rows = [], []
    for task, (arm, reference) in itertools.product(p['tasks'], p['contrasts']):
        row = dict(task=task, arm=arm, reference=reference, n=len(p['seeds']))
        for fold, metric in itertools.product(('val','test'), ('auc','aupr','per_drug_auc','per_drug_aupr')):
            delta = np.array([records[task,s,arm,0]['metrics'][fold][metric] - records[task,s,reference,0]['metrics'][fold][metric]
                              for s in p['seeds']])
            name = fold + '_' + metric
            row.update({name+'_delta': float(delta.mean()), name+'_wins': int((delta>0).sum()),
                        name+'_std': float(delta.std(ddof=1)) if len(delta)>1 else 0.})
        comparisons.append(row)
    for task, arm in itertools.product(p['tasks'], p['arms']):
        for mode in records[task,p['seeds'][0],arm,0]['interventions']:
            row = dict(task=task, arm=arm, intervention=mode, n=len(p['seeds']))
            for fold, metric in itertools.product(('val','test'), ('auc','aupr','per_drug_auc','per_drug_aupr')):
                delta = [records[task,s,arm,0]['interventions'][mode][fold][metric] - records[task,s,arm,0]['metrics'][fold][metric]
                         for s in p['seeds']]
                row[fold+'_'+metric+'_delta'] = float(np.mean(delta))
                row[fold+'_'+metric+'_decreases'] = sum(d<0 for d in delta)
            intervention_rows.append(row)
    save_csv(root / 'comparisons.csv', comparisons)
    save_csv(root / 'interventions.csv', intervention_rows)
    route_frame = pd.DataFrame(routing)
    route_frame.to_csv(root / 'routing_per_seed.csv', index=False)
    route_summary = route_frame.groupby(['task','arm','fold','has_path'], sort=False)[
        ['null_weight','channel_weight','path_weight','null_weight_std','channel_weight_std','path_weight_std','null_dominant_fraction']].mean().reset_index()
    route_summary.to_csv(root / 'routing_summary.csv', index=False)
    lines = ['# 通道与路径的互补注意力实验', '',
        f"完成 {p['expected_runs']} 次正式训练。" if not p['smoke'] else '本报告为短跑正确性检查，不作性能结论。', '',
        '| 任务 | 方案 | Val AUC | Val AUPR | Test AUC | Test AUPR |',
        '|---|---|---:|---:|---:|---:|']
    for row in summary.to_dict('records'):
        lines.append('| %s | %s | %.6f | %.6f | %.6f | %.6f |' % tuple(row[n] for n in ('task','arm','val_auc','val_aupr','test_auc','test_aupr')))
    lines.extend(['', '## 主要候选的配对对照', '',
        '| 任务 | 对照 | Δ Test AUC | AUC 提升种子 | Δ Test AUPR | AUPR 提升种子 |',
        '|---|---|---:|---:|---:|---:|'])
    for row in comparisons:
        if row['arm'] == 'fusion_gate':
            lines.append('| %s | %s | %+.6f | %d/%d | %+.6f | %d/%d |' %
                (row['task'],row['reference'],row['test_auc_delta'],row['test_auc_wins'],row['n'],row['test_aupr_delta'],row['test_aupr_wins'],row['n']))
    lines.extend(['', '## 门控实际分配', '',
        '| 任务 | 划分 | 有路径 | 原模型权重 | 通道权重 | 路径权重 |',
        '|---|---|---|---:|---:|---:|'])
    for row in route_summary[route_summary.arm == 'fusion_gate'].to_dict('records'):
        lines.append('| %s | %s | %s | %.4f | %.4f | %.4f |' %
            (row['task'],row['fold'],row['has_path'],row['null_weight'],row['channel_weight'],row['path_weight']))
    lines.extend(['', '## 验证集规则与解释范围', ''])
    for task in p['tasks']:
        group = summary[summary.task == task]
        selected = group.loc[((group.val_auc + group.val_aupr) / 2).idxmax(),'arm']
        lines.append('- %s：按验证集均值选择 `%s`。' % (task,selected))
    lines.extend(['- 所有方案沿用同一训练图与任务原配置，每次两个指标来自同一验证集选定 checkpoint。',
        '- `path_attention` 是前一轮 T=4 路径方案；`fusion_gate_simple` 使用普通残差 MLP 和均匀路径池化。',
        '- 所有融合组都屏蔽无路径的专家；固定/全局门控因此仍按路径是否可用重新归一化。',
        '- 原模型选项允许拒绝新增修正，但并不保证训练后的模型不会退化。',
        '- 度数统计逐候选排除其自身关联；基础编码器仍保留原协议的训练正例消息传递。',
        '- 本轮是根据既有结果提出的固定划分探索，尚未确认新划分泛化或文献层面的新颖性。', ''])
    (root / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
