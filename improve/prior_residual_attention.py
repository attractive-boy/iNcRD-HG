"""Paired two-task experiment for bounded similarity-prior corrections."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import copy
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from improve.attention_mechanisms import parameter_hash, save_csv
from improve.drug_condition_attention import METRICS, file_hash, make_base, metrics, set_seed, write_json
from improve.edge_path_attention import configure, predict, run_path, source_hashes as parent_source_hashes
from improve.prior_residual_modules import ARMS, CONTROLS, NEW_ARMS, PRIMARY, interventions, make_model
from improve.verify_edge_path_attention import require


CONTRASTS = (
    ('edge_fixed', 'base'), ('edge_attributes', 'base'),
    ('prior_bounded', 'base'), ('prior_bounded', 'edge_fixed'),
    ('prior_bounded', 'edge_attributes'), ('prior_bounded', 'prior_free'),
    ('prior_bounded', 'bounded_uniform'), ('prior_bounded', 'prior_shuffled'),
    ('prior_free', 'edge_fixed'), ('bounded_uniform', 'base'),
)


def source_hashes():
    result = parent_source_hashes()
    for name in ('prior_residual_attention.py', 'prior_residual_modules.py',
                 'test_prior_residual_attention.py', 'verify_prior_residual_attention.py'):
        result[name] = file_hash(Path(__file__).with_name(name))
    return result


def jobs(p):
    for group, (task, seed) in enumerate(itertools.product(p['tasks'], p['seeds'])):
        for arm in p['arms']:
            for repeat in range(2 if p['smoke'] or arm == 'base' else 1):
                yield group, task, seed, arm, repeat


def initialise(args):
    root, parent_path = Path(args.out_dir), Path(args.input_protocol)
    require(not (root / 'protocol.json').exists(), 'Protocol already exists; use run to resume')
    parent = json.loads(parent_path.read_text())
    validation_path = parent_path.with_name('VALIDATION.json')
    validation = json.loads(validation_path.read_text())
    require(validation['passed'] and validation['protocol_sha256'] == file_hash(parent_path), 'Parent inputs are not verified')
    require(parent['source_hashes'] == parent_source_hashes(), 'Matched reference source changed')
    p = {k: copy.deepcopy(parent[k]) for k in ('tasks', 'configs', 'inputs', 'assets',
         'matrix_hashes', 'width', 'edge_width', 'alpha', 'branch_rng_offset', 'training_rng_offset')}
    p.update(arms=list(ARMS), primary_arms=list(PRIMARY), contrasts=[list(c) for c in CONTRASTS],
             seeds=[666] if args.smoke else [666, 0, 1, 2, 3], smoke=args.smoke,
             epsilon=.25, parent_protocol=str(parent_path.resolve()),
             parent_protocol_sha256=file_hash(parent_path),
             parent_validation_sha256=file_hash(validation_path), source_hashes=source_hashes(),
             optimizer='RMSprop', loss='unweighted BCEWithLogitsLoss',
             deterministic_algorithms=True, cpu_threads=2, cublas_workspace_config=':4096:8',
             torch_version=torch.__version__,
             selection='one checkpoint maximizing validation AUC/AUPR mean; earliest exact tie',
             selection_arms=[a for a in ARMS if a != 'prior_shuffled'],
             common_core='same initialized baseline, scorer architecture and independently reset training RNG',
             edge_policy='only three non-self similarity relations; incoming total equals original non-self degree; every self loop remains weight one',
             prior='absolute supplied similarity; signed and absolute similarities remain scorer attributes',
             bounded='prior * (1 + 0.25 * tanh(score)), then normalize per destination to original degree',
             bound_after_normalization=[.6, 1 / .6],
             prior_free='prior * exp(0.25 * score), stabilized per destination; same derivative at zero as bounded arm',
             shuffled='shuffle multiplicative priors across undirected pairs; retain original scorer attributes and pair reciprocity',
             evaluation_graph='immutable cleaned training graph in all phases, including existing circRNA extension',
             training_labels='released training positives remain message edges; held-out target edges and reverses excluded',
             repeat_policy='two baseline runs per seed on the same device; smoke repeats every arm twice',
             scope='predefined fixed-split exploratory experiment; epsilon is not tuned; no literature novelty claim')
    if args.smoke:
        for cfg in p['configs'].values():
            cfg['epochs'] = 3
    p['expected_runs'] = len(list(jobs(p)))
    for task in p['tasks']:
        for entry in p['inputs'][task]['folds'].values():
            require(file_hash(entry['path']) == entry['sha256'], 'Frozen input changed')
        entry = p['assets'][task]
        require(file_hash(entry['path']) == entry['sha256'], 'Frozen attributes changed')
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in p['source_hashes']:
        (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_json(root / 'protocol.json', p)
    print('INITIALISED', root, 'runs', p['expected_runs'], flush=True)


def load_inputs(p):
    require(p['source_hashes'] == source_hashes(), 'Source changed after protocol was frozen')
    require(file_hash(p['parent_protocol']) == p['parent_protocol_sha256'], 'Parent protocol changed')
    require(file_hash(Path(p['parent_protocol']).with_name('VALIDATION.json')) == p['parent_validation_sha256'], 'Parent validation changed')
    data, assets = {}, {}
    for task in p['tasks']:
        data[task] = {}
        for fold, entry in p['inputs'][task]['folds'].items():
            require(file_hash(entry['path']) == entry['sha256'], 'Frozen graph changed')
            data[task][fold] = torch.load(entry['path'], map_location='cpu')
        entry = p['assets'][task]
        require(file_hash(entry['path']) == entry['sha256'], 'Frozen attributes changed')
        assets[task] = torch.load(entry['path'], map_location='cpu')
    return data, assets


def train_one(root, p, data, assets, task, seed, arm, repeat, device):
    directory = run_path(root, task, seed, arm, repeat)
    signature = dict(task=task, seed=seed, arm=arm, repeat=repeat,
                     protocol_sha256=file_hash(root / 'protocol.json'))
    result_path = directory / 'result.json'
    if result_path.exists():
        require(json.loads(result_path.read_text())['signature'] == signature, 'Existing run belongs to a different protocol')
        print('SKIP', task, seed, arm, repeat, flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'config.json', signature)
    cfg = p['configs'][task]
    key = (cfg['rna'], task, 'drug')
    train = copy.deepcopy(data['train']).to(device)
    base = make_base(train, cfg, seed, device)
    initial_core = parameter_hash(base)
    set_seed(seed + p['branch_rng_offset'])
    model = make_model(base, data['train'], assets, cfg, arm, seed, p).to(device)
    initial_scorers = {path: parameter_hash(torch.nn.ModuleList([m.source, m.target, m.attribute, m.scorer]))
                       for path, m in model.edge_layers}
    set_seed(seed + p['training_rng_offset'])
    initial = predict(model, train, 'val')
    labels = {fold: graph[key].edge_label.numpy() for fold, graph in data.items()}
    optimizer = torch.optim.RMSprop((v for v in model.parameters() if v.requires_grad),
                                    lr=cfg['lr'], alpha=p['alpha'])
    best, trace, started = -float('inf'), [], time.monotonic()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(train.x_dict, train.edge_index_dict, 'train')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, train[key].edge_label)
        require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        loss.backward()
        optimizer.step()
        val = predict(model, train, 'val')
        values = metrics(labels['val'], val, assets['pairs']['val'].numpy(), by_drug=False)
        criterion = (values['auc'] + values['aupr']) / 2
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), val_auc=values['auc'],
                          val_aupr=values['aupr'], criterion=criterion))
        if criterion > best:
            best, best_epoch = criterion, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0 or epoch == cfg['epochs']:
            save_csv(directory / 'trace.csv', trace)
            print(task, seed, arm, repeat, 'epoch', epoch, 'val', values,
                  'seconds', round(time.monotonic() - started, 1), flush=True)
    model.load_state_dict(best_state)
    predictions, results, changed = {}, {}, {}
    for fold in ('val', 'test'):
        predictions[fold] = predict(model, train, fold)
        results[fold] = metrics(labels[fold], predictions[fold], assets['pairs'][fold].numpy())
        arrays = model.attention()
        if arrays:
            np.savez_compressed(directory / ('attention_' + fold + '.npz'), **arrays)
    require((results['val']['auc'] + results['val']['aupr']) / 2 == best, 'Selected validation prediction changed')
    for mode in interventions(arm):
        changed[mode] = {}
        for fold in ('val', 'test'):
            predictions[mode + '_' + fold] = predict(model, train, fold, mode)
            changed[mode][fold] = metrics(labels[fold], predictions[mode + '_' + fold], assets['pairs'][fold].numpy())
    np.savez_compressed(directory / 'predictions.npz', initial_val_logits=initial, **predictions,
                        **{'y_' + fold: labels[fold] for fold in ('val', 'test')},
                        **{'pairs_' + fold: assets['pairs'][fold].numpy() for fold in ('val', 'test')})
    torch.save(best_state, directory / 'checkpoint.pt')
    write_json(result_path, dict(signature=signature, metrics=results, interventions=changed,
        initial_core_sha256=initial_core, initial_scorer_sha256=initial_scorers,
        selected_state_sha256=parameter_hash(model), selected_epoch=best_epoch,
        trained_epochs=cfg['epochs'], validation_replay_delta=0., device=device,
        elapsed_seconds=time.monotonic() - started,
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
    print('RESULT', task, seed, arm, repeat, json.dumps(results), flush=True)
    del optimizer, model, base, train, best_state
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def run(args):
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    data, assets = load_inputs(p)
    require(0 <= args.shard_index < args.shard_count, 'Invalid worker shard')
    for group, task, seed, arm, repeat in jobs(p):
        if group % args.shard_count == args.shard_index:
            train_one(root, p, data[task], assets[task], task, seed, arm, repeat, args.device)


def summarize(args):
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    require(p['source_hashes'] == source_hashes(), 'Frozen source changed')
    records, rows, repeats = {}, [], []
    for _, task, seed, arm, repeat in jobs(p):
        directory = run_path(root, task, seed, arm, repeat)
        r = json.loads((directory / 'result.json').read_text())
        require(r['signature'] == dict(task=task, seed=seed, arm=arm, repeat=repeat,
                                      protocol_sha256=file_hash(root / 'protocol.json')), 'Mixed protocols')
        records[task, seed, arm, repeat] = r
        if repeat:
            first_dir = run_path(root, task, seed, arm, 0)
            first = records[task, seed, arm, 0]
            with np.load(directory / 'predictions.npz') as a, np.load(first_dir / 'predictions.npz') as b:
                same_predictions = set(a.files) == set(b.files) and all(np.array_equal(a[n], b[n]) for n in a.files)
            entry = dict(task=task, seed=seed, arm=arm, same_device=r['device'] == first['device'],
                         same_state=r['selected_state_sha256'] == first['selected_state_sha256'],
                         same_trace=file_hash(directory / 'trace.csv') == file_hash(first_dir / 'trace.csv'),
                         same_predictions=same_predictions)
            require(all(entry[k] for k in ('same_device', 'same_state', 'same_trace', 'same_predictions')), 'Repeated runs differ')
            repeats.append(entry)
        else:
            row = dict(task=task, seed=seed, arm=arm, selected_epoch=r['selected_epoch'],
                       trainable_parameters=r['trainable_parameters'])
            for fold in ('val', 'test'):
                row.update({fold + '_' + name: value for name, value in r['metrics'][fold].items()})
            rows.append(row)
    aggregates, comparisons, intervention_rows = [], [], []
    for task, arm in itertools.product(p['tasks'], p['arms']):
        group = [r for r in rows if r['task'] == task and r['arm'] == arm]
        row = dict(task=task, arm=arm, n=len(group))
        for fold, metric in itertools.product(('val', 'test'), METRICS):
            key = fold + '_' + metric
            values = [r[key] for r in group]
            row[key + '_mean'] = float(np.mean(values))
            row[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        aggregates.append(row)
    for task, (arm, reference) in itertools.product(p['tasks'], p['contrasts']):
        row = dict(task=task, arm=arm, reference=reference, n=len(p['seeds']))
        for fold, metric in itertools.product(('val', 'test'), METRICS):
            delta = [records[task, s, arm, 0]['metrics'][fold][metric] - records[task, s, reference, 0]['metrics'][fold][metric]
                     for s in p['seeds']]
            prefix = fold + '_' + metric
            row[prefix + '_delta'] = float(np.mean(delta))
            row[prefix + '_std'] = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.
            row[prefix + '_wins'] = sum(d > 0 for d in delta)
        comparisons.append(row)
    for task, arm in itertools.product(p['tasks'], p['arms']):
        for mode in interventions(arm):
            row = dict(task=task, arm=arm, intervention=mode, n=len(p['seeds']))
            for fold, metric in itertools.product(('val', 'test'), METRICS):
                delta = [records[task, s, arm, 0]['interventions'][mode][fold][metric] - records[task, s, arm, 0]['metrics'][fold][metric]
                         for s in p['seeds']]
                row[fold + '_' + metric + '_delta'] = float(np.mean(delta))
                row[fold + '_' + metric + '_decreases'] = sum(d < 0 for d in delta)
            intervention_rows.append(row)
    for name, values in (('per_seed', rows), ('summary', aggregates), ('comparisons', comparisons),
                         ('interventions', intervention_rows), ('repeatability', repeats)):
        save_csv(root / (name + '.csv'), values)
    selected = {task: max((r for r in aggregates if r['task'] == task and r['arm'] in p['selection_arms']),
                         key=lambda r: (r['val_auc_mean'] + r['val_aupr_mean']) / 2)['arm'] for task in p['tasks']}
    write_json(root / 'selection.json', dict(selection_arms=p['selection_arms'], selected=selected,
               rule='mean validation AUC/AUPR across predefined initialization seeds; test not used'))
    report = ['# 相似度先验约束的边权修正', '',
              '完成 %d/%d 次训练，%d 组同设备重复完全一致。' % (len(records), p['expected_runs'], len(repeats)), '',
              '| 任务 | 方案 | Val AUC | Val AUPR | Test AUC | Test AUPR | 按药物 Test AUC |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        report.append('| %s | %s | %.6f | %.6f | %.6f | %.6f | %.6f |' %
                      (r['task'], r['arm'], r['val_auc_mean'], r['val_aupr_mean'], r['test_auc_mean'],
                       r['test_aupr_mean'], r['test_per_drug_auc_mean']))
    report.extend(['', '## 主方案的配对对照', '',
                   '| 任务 | 对照 | Δ Test AUC | 提升种子 | Δ Test AUPR | 提升种子 |',
                   '|---|---|---:|---:|---:|---:|'])
    for r in comparisons:
        if r['arm'] == 'prior_bounded':
            report.append('| %s | %s | %+.6f | %d/%d | %+.6f | %d/%d |' %
                          (r['task'], r['reference'], r['test_auc_delta'], r['test_auc_wins'], r['n'],
                           r['test_aupr_delta'], r['test_aupr_wins'], r['n']))
    report.extend(['', '## 验证集选择与解释范围', ''])
    report.extend('- %s：按预设验证集规则选择 `%s`。' % (t, a) for t, a in selected.items())
    report.extend([
        '- 主假设为 `prior_bounded`；其余新增组用于区分先验、有限修正和普通可学习加权的作用。',
        '- 25% 是归一化前的乘子幅度；归一化后相对固定先验的权重比在 [0.6, 1.6667] 内。',
        '- `prior_free` 使用 exp(0.25 × score)，与主方案在零点具有相同一阶导数，但没有小幅修正约束。',
        '- `bounded_uniform` 只移除乘法先验；`prior_shuffled` 只打乱乘法先验，两者保留真实边属性输入。',
        '- 原模型、固定边权与自由属性注意力复用已有实现，独立训练并核对历史相同配置的结果。',
        '- 当前图含已有 circRNA 扩展；只改变 lncRNA、miRNA、drug 三类相似边的权重。',
        '- 所有阶段均使用清理后的训练图；原协议训练正例继续参与消息传递。',
        '- 同一个 checkpoint 同时产生 AUC/AUPR；只用验证集选择 epoch，未搜索 epsilon。',
        '- 固定划分上的初始化种子不能代替新划分验证；测试集已在历史探索中被反复观察。',
        '- 本轮只检验固定相似度先验的有限修正，没有加入表达重采样稳定性或癌种上下文。',
        '- 核验结果见 VALIDATION.json；完整种子差值和干预见 CSV 文件。', ''])
    (root / 'REPORT.md').write_text('\n'.join(report))
    print('\n'.join(report), flush=True)


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
