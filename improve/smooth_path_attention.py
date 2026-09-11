"""A fixed MiDrug follow-up: temperature and uniform shrinkage of path attention."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from improve.attention_mechanisms import norm, parameter_hash, save_csv
from improve.drug_condition_attention import file_hash, make_base, masked_weights, metrics, set_seed, write_json
from improve.edge_path_attention import configure, predict, run_path, source_hashes as parent_source_hashes
from improve.edge_path_modules import PathBranch
from improve.verify_edge_path_attention import check_metrics, require, state_hash, verify_attention


ARMS = ('base', 'path_uniform', 'path_attention', 'temperature4', 'shrink25')
ATTENTION_ARMS = ('path_attention', 'temperature4', 'shrink25')
SETTINGS = {'path_uniform': (1., 1.), 'path_attention': (1., 1.),
            'temperature4': (4., 1.), 'shrink25': (1., .25)}


def source_hashes():
    result = parent_source_hashes()
    for name in ('smooth_path_attention.py', 'verify_edge_path_attention.py'):
        result[name] = file_hash(Path(__file__).with_name(name))
    return result


def smooth_weights(scores, valid, temperature=1., fraction=1.):
    if temperature <= 0 or not 0 <= fraction <= 1:
        raise ValueError('Invalid temperature or attention fraction')
    if temperature != 1.:
        scores = scores / temperature
    weights = masked_weights(scores, valid)
    if fraction != 1.:
        uniform = valid.to(scores.dtype) / valid.sum(1, keepdim=True).clamp_min(1)
        weights = fraction * weights + (1. - fraction) * uniform
    return weights


class SmoothPathBranch(PathBranch):
    def __init__(self, hidden, rna_type, assets, arm, width):
        mode = 'path_uniform' if arm == 'path_uniform' else 'path_attention'
        super().__init__(hidden, rna_type, assets, mode, width)
        self.temperature, self.fraction = SETTINGS[arm]

    def forward(self, z, pairs, fold, intervention=None):
        r, d = pairs
        if intervention == 'branch_off':
            return z[self.rna_type].new_zeros(r.size(0))
        normalised = {nt: norm(value) for nt, value in z.items()}
        query = self.query_r(normalised[self.rna_type][r]) + self.query_d(normalised['drug'][d])
        parts, self.last_weights = [], {}
        for i, family in enumerate(self.families):
            valid = getattr(self, f'{fold}_{i}_valid')
            counts = getattr(self, f'{fold}_{i}_counts')
            rows = getattr(self, f'{fold}_{i}_rows')
            slots = getattr(self, f'{fold}_{i}_slots')
            ids = getattr(self, f'{fold}_{i}_flat_ids')
            middle = normalised[family['middle']]
            if self.mode == 'path_uniform' or intervention == 'uniform':
                weights = valid.to(query.dtype) / counts[:, None].clamp_min(1)
            else:
                key = self.keys[i](middle)[ids]
                scores = query.new_full(valid.shape, -1e4)
                scores[rows, slots] = (query[rows] * key).sum(-1) / math.sqrt(self.width)
                weights = smooth_weights(scores, valid, self.temperature, self.fraction)
            values = self.values[i](middle)[ids]
            pooled = query.new_zeros((len(r), self.width)).index_add(
                0, rows, weights[rows, slots, None] * values)
            self.last_weights['path_' + str(i)] = weights.detach()
            parts.extend([pooled, counts.log1p()[:, None]])
        return self.head(torch.cat(parts, -1)).view(-1)


class SmoothPathModel(nn.Module):
    def __init__(self, base, assets, config, arm, width):
        super().__init__()
        self.base, self.arm, self.branch = base, arm, None
        for fold, pairs in assets['pairs'].items():
            self.register_buffer('pairs_' + fold, pairs.clone())
        if arm != 'base':
            self.branch = SmoothPathBranch(config['hidden'], config['rna'], assets, arm, width)

    def forward(self, x, edges, fold, intervention=None):
        z = self.base.encoder(x, edges)
        pairs = getattr(self, 'pairs_' + fold)
        result = self.base.decoder(z, pairs)
        if self.branch is not None:
            result = result + self.branch(z, pairs, fold, intervention)
        return result

    def attention(self):
        return ({} if self.branch is None else
                {name: value.cpu().numpy() for name, value in self.branch.last_weights.items()})


def jobs(p):
    for group, seed in enumerate(p['seeds']):
        for arm in p['arms']:
            for repeat in range(2 if arm == 'base' or p['smoke'] else 1):
                yield group, 'MiDrug', seed, arm, repeat


def initialise(args):
    root, parent_path = Path(args.out_dir), Path(args.input_protocol)
    require(not (root / 'protocol.json').exists(), 'Plan exists; use run to resume')
    parent = json.loads(parent_path.read_text())
    validation_path = parent_path.with_name('VALIDATION.json')
    validation = json.loads(validation_path.read_text())
    require(validation['passed'] and validation['protocol_sha256'] == file_hash(parent_path),
            'Parent experiment must pass verification')
    p = {name: copy.deepcopy(parent[name]) for name in
         ('configs', 'inputs', 'assets', 'width', 'alpha', 'branch_rng_offset', 'training_rng_offset')}
    for name in ('configs', 'inputs', 'assets'):
        p[name] = {'MiDrug': p[name]['MiDrug']}
    if args.smoke:
        p['configs']['MiDrug']['epochs'] = 3
    p.update(tasks=['MiDrug'], seeds=[666] if args.smoke else parent['seeds'], arms=list(ARMS),
             smoke=args.smoke, settings=SETTINGS, parent_protocol=str(parent_path.resolve()),
             parent_protocol_sha256=file_hash(parent_path), parent_validation_sha256=file_hash(validation_path),
             source_hashes=source_hashes(), selection='validation AUC/AUPR mean, earliest exact tie',
             deterministic_algorithms=True, cpu_threads=2, cublas_workspace_config=':4096:8',
             optimizer='RMSprop', loss='unweighted BCEWithLogitsLoss',
             motivation='MiDrug path branch helps; concentrated attention has mixed validation/test uniform-intervention effects',
             scope='adaptive fixed-split exploration after inspecting the parent results; test data are not an untouched confirmation set',
             contrasts=[['path_uniform', 'base'], ['path_attention', 'base'], ['path_attention', 'path_uniform'],
                        ['temperature4', 'base'], ['temperature4', 'path_uniform'], ['temperature4', 'path_attention'],
                        ['shrink25', 'base'], ['shrink25', 'path_uniform'], ['shrink25', 'path_attention']])
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
    data = {}
    for fold, entry in p['inputs']['MiDrug']['folds'].items():
        require(file_hash(entry['path']) == entry['sha256'], 'Frozen graph changed')
        data[fold] = torch.load(entry['path'], map_location='cpu')
    asset = p['assets']['MiDrug']
    require(file_hash(asset['path']) == asset['sha256'], 'Frozen assets changed')
    return data, torch.load(asset['path'], map_location='cpu')


def train_one(root, p, data, assets, task, seed, arm, repeat, device):
    directory = run_path(root, task, seed, arm, repeat)
    signature = dict(task=task, seed=seed, arm=arm, repeat=repeat,
                     protocol_sha256=file_hash(root / 'protocol.json'))
    if (directory / 'result.json').exists():
        require(json.loads((directory / 'result.json').read_text())['signature'] == signature, 'Different completed run')
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'config.json', signature)
    config = p['configs'][task]
    key = (config['rna'], task, 'drug')
    train = copy.deepcopy(data['train']).to(device)
    base = make_base(train, config, seed, device)
    initial_core = parameter_hash(base)
    set_seed(seed + p['branch_rng_offset'])
    model = SmoothPathModel(base, assets, config, arm, p['width']).to(device)
    set_seed(seed + p['training_rng_offset'])
    initial = predict(model, train, 'val')
    labels = {fold: graph[key].edge_label.numpy() for fold, graph in data.items()}
    optimizer = torch.optim.RMSprop((v for v in model.parameters() if v.requires_grad),
                                    lr=config['lr'], alpha=p['alpha'])
    best, trace, started = -float('inf'), [], time.monotonic()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        score = model(train.x_dict, train.edge_index_dict, 'train')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(score, train[key].edge_label)
        require(bool(torch.isfinite(loss)), 'Nonfinite loss')
        loss.backward()
        optimizer.step()
        validation = predict(model, train, 'val')
        values = metrics(labels['val'], validation, assets['pairs']['val'].numpy(), by_drug=False)
        criterion = (values['auc'] + values['aupr']) / 2
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), val_auc=values['auc'],
                          val_aupr=values['aupr'], criterion=criterion))
        if criterion > best:
            best, best_epoch = criterion, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0 or epoch == config['epochs']:
            save_csv(directory / 'trace.csv', trace)
            print(task, seed, arm, repeat, epoch, 'val', values, 'seconds', round(time.monotonic()-started, 1), flush=True)
    model.load_state_dict(best_state)
    predictions, results, interventions = {}, {}, {}
    for fold in ('val', 'test'):
        predictions[fold] = predict(model, train, fold)
        results[fold] = metrics(labels[fold], predictions[fold], assets['pairs'][fold].numpy())
        if model.attention():
            np.savez_compressed(directory / ('attention_' + fold + '.npz'), **model.attention())
    require((results['val']['auc'] + results['val']['aupr']) / 2 == best, 'Validation replay changed')
    for mode in ('uniform', 'branch_off') if arm in ATTENTION_ARMS else ():
        interventions[mode] = {}
        for fold in ('val', 'test'):
            predictions[mode + '_' + fold] = predict(model, train, fold, mode)
            interventions[mode][fold] = metrics(labels[fold], predictions[mode + '_' + fold], assets['pairs'][fold].numpy())
    np.savez_compressed(directory / 'predictions.npz', initial_val_logits=initial, **predictions,
                        **{'y_' + f: labels[f] for f in ('val', 'test')},
                        **{'pairs_' + f: assets['pairs'][f].numpy() for f in ('val', 'test')})
    torch.save(best_state, directory / 'checkpoint.pt')
    write_json(directory / 'result.json', dict(
        signature=signature, initial_core_sha256=initial_core, selected_state_sha256=parameter_hash(model),
        metrics=results, interventions=interventions, selected_epoch=best_epoch,
        trained_epochs=config['epochs'], validation_replay_delta=0., device=device,
        elapsed_seconds=time.monotonic()-started,
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
    print('RESULT', seed, arm, repeat, json.dumps(results), flush=True)
    del optimizer, model, base, train, best_state
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def run(args):
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    require(0 <= args.shard_index < args.shard_count, 'Invalid shard')
    data, assets = load_inputs(p)
    for group, task, seed, arm, repeat in jobs(p):
        if group % args.shard_count == args.shard_index:
            train_one(root, p, data, assets, task, seed, arm, repeat, args.device)


def summarize(args):
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    data, assets = load_inputs(p)
    require(file_hash(p['parent_protocol']) == p['parent_protocol_sha256'], 'Parent protocol changed')
    parent_root = Path(p['parent_protocol']).parent
    require(file_hash(parent_root / 'VALIDATION.json') == p['parent_validation_sha256'], 'Parent verification changed')
    protocol_hash = file_hash(root / 'protocol.json')
    records, initials, rows, repeated, controls, replays = {}, {}, [], [], [], []
    attention_checked, replay_delta = 0, 0.
    key = ('miRNA', 'MiDrug', 'drug')
    for _, task, seed, arm, repeat in jobs(p):
        directory = run_path(root, task, seed, arm, repeat)
        r = json.loads((directory / 'result.json').read_text())
        signature = dict(task=task, seed=seed, arm=arm, repeat=repeat, protocol_sha256=protocol_hash)
        require(r['signature'] == signature, 'Mixed run signatures')
        require(json.loads((directory / 'config.json').read_text()) == signature, 'Run config differs')
        require(r['trained_epochs'] == p['configs'][task]['epochs'], 'Incomplete training')
        require(r['validation_replay_delta'] == 0., 'Validation replay changed')
        trace = pd.read_csv(directory / 'trace.csv', float_precision='round_trip')
        require(trace.epoch.tolist() == list(range(1, r['trained_epochs'] + 1)), 'Incomplete trace')
        require(np.isfinite(trace.to_numpy()).all(), 'Nonfinite trace')
        np.testing.assert_allclose(trace.criterion, (trace.val_auc + trace.val_aupr) / 2, atol=1e-14, rtol=0)
        require(int(trace.loc[trace.criterion.idxmax(), 'epoch']) == r['selected_epoch'], 'Wrong checkpoint selection')
        require(abs(trace.criterion.max() - (r['metrics']['val']['auc'] + r['metrics']['val']['aupr']) / 2) < 1e-14,
                'Saved metric differs from selected epoch')
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
            attention_checked += verify_attention(directory, arm, data['train'], assets)
            if repeat:
                first = records[seed, arm, 0]
                first_dir = run_path(root, task, seed, arm, 0)
                require(first['device'] == r['device'] and first['selected_state_sha256'] == r['selected_state_sha256'],
                        'Same-device repeated state differs')
                require(file_hash(first_dir / 'trace.csv') == file_hash(directory / 'trace.csv'), 'Repeated trace differs')
                with np.load(first_dir / 'predictions.npz') as reference:
                    require(set(values.files) == set(reference.files), 'Repeated prediction keys differ')
                    for name in values.files:
                        np.testing.assert_array_equal(values[name], reference[name])
                repeated.append(dict(seed=seed, arm=arm, exact=True))
            elif seed == p['seeds'][0]:
                base = make_base(data['train'], p['configs'][task], seed, 'cpu')
                model = SmoothPathModel(base, assets, p['configs'][task], arm, p['width'])
                model.load_state_dict(state)
                for fold in ('val', 'test'):
                    for mode in (None, *r['interventions']):
                        score = predict(model, data['train'], fold, mode)
                        name = fold if mode is None else mode + '_' + fold
                        np.testing.assert_allclose(score, values[name], rtol=1e-4, atol=1e-3)
                        actual = metrics(values['y_' + fold], score, values['pairs_' + fold], by_drug=False)
                        expected = r['metrics'][fold] if mode is None else r['interventions'][mode][fold]
                        delta = max(abs(actual[m] - expected[m]) for m in ('auc', 'aupr'))
                        require(delta < 1e-4, 'CPU/GPU replay metric differs')
                        replay_delta = max(replay_delta, delta)
                replays.append(dict(seed=seed, arm=arm))
            if not p['smoke'] and arm in ('base', 'path_uniform', 'path_attention'):
                parent_dir = run_path(parent_root, task, seed, arm, repeat)
                parent_record = json.loads((parent_dir / 'result.json').read_text())
                same_state = parent_record['selected_state_sha256'] == r['selected_state_sha256']
                same_trace = file_hash(parent_dir / 'trace.csv') == file_hash(directory / 'trace.csv')
                with np.load(parent_dir / 'predictions.npz') as reference:
                    same_predictions = (set(values.files) == set(reference.files) and
                                        all(np.array_equal(values[n], reference[n]) for n in values.files))
                require(same_state and same_trace and same_predictions, 'Common control does not reproduce parent exactly')
                controls.append(dict(seed=seed, arm=arm, repeat=repeat, exact=True))
        if not repeat:
            row = dict(seed=seed, arm=arm, selected_epoch=r['selected_epoch'], trainable_parameters=r['trainable_parameters'])
            for fold, values in r['metrics'].items():
                row.update({fold + '_' + name: value for name, value in values.items()})
            rows.append(row)
        del state
    for seed in p['seeds']:
        require(len({records[seed, arm, 0]['initial_core_sha256'] for arm in ARMS}) == 1, 'Initial base weights differ')
        for arm in ARMS:
            np.testing.assert_array_equal(initials[seed, arm, 0], initials[seed, 'base', 0])
    require(len(records) == p['expected_runs'], 'Missing runs')
    frame = pd.DataFrame(rows)
    summary = frame.groupby('arm', sort=False).agg(
        n=('seed', 'count'), val_auc=('val_auc', 'mean'), val_aupr=('val_aupr', 'mean'),
        test_auc=('test_auc', 'mean'), test_aupr=('test_aupr', 'mean'),
        per_drug_auc=('test_per_drug_auc', 'mean'), per_drug_aupr=('test_per_drug_aupr', 'mean')).reset_index()
    comparisons, interventions = [], []
    for arm, reference in p['contrasts']:
        row = dict(arm=arm, reference=reference, n=len(p['seeds']))
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
    frame.to_csv(root / 'per_seed.csv', index=False)
    summary.to_csv(root / 'summary.csv', index=False)
    save_csv(root / 'comparisons.csv', comparisons)
    save_csv(root / 'interventions.csv', interventions)
    write_json(root / 'CONTROL_REPLAY.json', dict(parent_controls=controls, within_run_repeats=repeated))
    write_json(root / 'VALIDATION.json', dict(
        passed=True, runs_verified=len(records), exact_repeats=len(repeated), exact_parent_controls=len(controls),
        initial_predictions_equal=True, attention_arrays_verified=attention_checked, cpu_replays=replays,
        maximum_cpu_gpu_metric_difference=replay_delta, protocol_sha256=protocol_hash,
        source_hashes=source_hashes(), checks=['frozen input and source hashes', 'checkpoint hashes',
        'validation-only checkpoint selection', 'all saved metrics and interventions', 'path attention constraints',
        'exact repeatability', 'common controls reproduce previous runner', 'first-seed CPU replay of every arm']))
    winner = summary.loc[((summary.val_auc + summary.val_aupr) / 2).idxmax(), 'arm']
    lines = ['# MiDrug 路径注意力平滑实验', '',
             f"完成 {len(records)}/{p['expected_runs']} 次训练；{len(repeated)} 组同条件重复完全一致。",
             f'与上一轮完全一致的对照运行：{len(controls)}；按验证集均值选择 `{winner}`。', '',
             '| 方案 | Val AUC | Val AUPR | Test AUC | Test AUPR | 按药物宏平均 Test AUC |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary.to_dict('records'):
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f | %.6f |' %
                     tuple(row[n] for n in ('arm', 'val_auc', 'val_aupr', 'test_auc', 'test_aupr', 'per_drug_auc')))
    lines.extend(['', '## 配对对照', '', '| 方案 | 对照 | Δ Test AUC | Δ Test AUPR | AUPR 提升种子数 |',
                  '|---|---|---:|---:|---:|'])
    for row in comparisons:
        lines.append('| %s | %s | %+.6f | %+.6f | %d/%d |' %
                     (row['arm'], row['reference'], row['test_auc_delta'], row['test_aupr_delta'],
                      row['test_aupr_wins'], row['n']))
    lines.extend(['', '## 设计与边界', '',
                  '- 温度固定为 4；收缩版为 25% 学习权重 + 75% 等权。没有扫描温度或混合比例。',
                  '- 每个运行仍只用验证集 AUC/AUPR 均值选一个 checkpoint；两个指标共享同一个模型。',
                  '- 新方案是在查看上一轮结果后提出，属于同一划分上的追加探索，测试集不是全新的独立确认集。',
                  '- common-control 核验要求初始设置、完整训练曲线、所选状态和预测完全复现上一轮。',
                  '- 干预结果见 interventions.csv；不能仅凭超过 base 就认定注意力优于等权路径信息。', ''])
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--out-dir', required=True)
    init.add_argument('--input-protocol', default='improve/results/edge_path_attention_20260908/protocol.json')
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
