"""Deterministic edge-attribute/path attention screen with repeated baselines."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from improve.attention_mechanisms import parameter_hash, save_csv
from improve.drug_condition_attention import METRICS, file_hash, make_base, metrics, set_seed, write_json
from improve.edge_path_modules import ARMS, PRIMARY, SIMILARITIES, EdgePathModel, build_assets


CONTRASTS = (
    ('edge_fixed', 'base'), ('edge_blind', 'base'),
    ('edge_attributes', 'base'), ('edge_attributes', 'edge_fixed'),
    ('edge_attributes', 'edge_blind'), ('edge_attributes', 'edge_shuffled'),
    ('path_count', 'base'), ('path_uniform', 'path_count'),
    ('path_attention', 'base'), ('path_attention', 'path_count'),
    ('path_attention', 'path_uniform'),
)


def source_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ('edge_path_attention.py', 'edge_path_modules.py', 'propagation_attention.py',
             'attention_mechanisms.py', 'drug_condition_attention.py', 'model.py')}


def jobs(protocol):
    for group, (task, seed) in enumerate(itertools.product(protocol['tasks'], protocol['seeds'])):
        for arm in ARMS:
            repeats = 2 if arm == 'base' or protocol['smoke'] else 1
            for repeat in range(repeats):
                yield group, task, seed, arm, repeat


def run_path(root, task, seed, arm, repeat=0):
    return root / task / ('seed_' + str(seed)) / arm / ('repeat_' + str(repeat))


def configure():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def initialise(args):
    root = Path(args.out_dir)
    if (root / 'protocol.json').exists():
        raise ValueError('Plan exists; use run to resume')
    root.mkdir(parents=True, exist_ok=True)
    parent_path = Path(args.input_protocol)
    parent = json.loads(parent_path.read_text())
    matrices = {rel: pd.read_csv(Path(args.net_dir) / name, header=None).values
                for rel, name in SIMILARITIES.items()}
    configs, asset_entries, audits = copy.deepcopy(parent['configs']), {}, {}
    for task, cfg in configs.items():
        if args.smoke:
            cfg['epochs'] = 3
        data = {}
        for fold, entry in parent['inputs'][task]['folds'].items():
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('Parent input changed')
            data[fold] = torch.load(entry['path'], map_location='cpu')
        key = (cfg['rna'], task, 'drug')
        pairs = {f: g[key].edge_label_index for f, g in data.items()}
        assets = build_assets(data['train'], pairs, matrices, cfg['rna'])
        destination = root / 'inputs' / (task + '_assets.pt')
        destination.parent.mkdir(exist_ok=True)
        torch.save(assets, destination)
        asset_entries[task] = dict(path=str(destination.resolve()), sha256=file_hash(destination))
        audits[task] = dict(
            families=assets['families'],
            path_coverage={fold: [dict(total_instances=int(pack['counts'].sum()),
                                       pairs_with_path=int((pack['counts'] > 0).sum()),
                                       total_pairs=len(pack['counts']),
                                       max_paths=int(pack['counts'].max())) for pack in packs]
                           for fold, packs in assets['packs'].items()},
            attributes={'|'.join(rel): dict(edges=len(a), negative=int((a[:, 0] < 0).sum()))
                        for rel, a in assets['attributes'].items()},
        )
    protocol = dict(
        tasks=list(configs), configs=configs, inputs=parent['inputs'], assets=asset_entries,
        parent_protocol=str(parent_path.resolve()), parent_protocol_sha256=file_hash(parent_path),
        matrix_hashes={name: file_hash(Path(args.net_dir) / name) for name in SIMILARITIES.values()},
        seeds=[666] if args.smoke else [666, 0, 1, 2, 3], arms=list(ARMS), primary_arms=list(PRIMARY),
        contrasts=[list(c) for c in CONTRASTS], smoke=args.smoke,
        width=32, edge_width=16, alpha=.99, optimizer='RMSprop', loss='unweighted BCEWithLogitsLoss',
        deterministic_algorithms=True, cublas_workspace_config=':4096:8', cpu_threads=2,
        core_rng='seed', branch_rng_offset=3000003, training_rng_offset=9000007,
        repeat_policy='two identical runs per baseline seed on the same device; smoke repeats every arm twice',
        comparison_policy='repeat 0 only in primary tables; repeats test reproducibility, not independent seeds',
        selection='one checkpoint maximising mean validation AUC/AUPR; earliest exact tie',
        evaluation_graph='immutable cleaned training graph in all phases',
        edge_policy='signed and absolute attribute similarity; weight non-self similarity edges only; preserve non-self incoming weight sum; retain every existing self loop at weight 1',
        shuffled_control='permute undirected non-self pairs within each relation, preserving reciprocal assignment',
        path_policy='all unique RNA-mediated two-edge instances in training graph; remove same-type r -> r first steps; never use labels or truncate/sample paths',
        scope='fixed-split exploratory screen; no architecture hyperparameter search; new deterministic runner baseline required',
        source_hashes=source_hashes(), torch_version=torch.__version__,
    )
    protocol['expected_runs'] = len(list(jobs(protocol)))
    write_json(root / 'asset_audit.json', audits)
    write_json(root / 'protocol.json', protocol)
    print('INITIALISED', root, 'runs', protocol['expected_runs'], json.dumps(audits), flush=True)


@torch.no_grad()
def predict(model, train, fold, intervention=None):
    model.eval()
    return model(train.x_dict, train.edge_index_dict, fold, intervention).cpu().numpy()


def train_one(root, protocol, data, assets, task, seed, arm, repeat, device):
    directory = run_path(root, task, seed, arm, repeat)
    signature = dict(task=task, seed=seed, arm=arm, repeat=repeat,
                     protocol_sha256=file_hash(root / 'protocol.json'))
    result_path = directory / 'result.json'
    if result_path.exists():
        if json.loads(result_path.read_text())['signature'] != signature:
            raise ValueError('Incompatible existing run')
        print('SKIP', task, seed, arm, repeat, flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'config.json', signature)
    cfg = protocol['configs'][task]
    key = (cfg['rna'], task, 'drug')
    train = copy.deepcopy(data['train']).to(device)
    base = make_base(train, cfg, seed, device)
    core_hash = parameter_hash(base)
    set_seed(seed + protocol['branch_rng_offset'])
    model = EdgePathModel(base, data['train'], assets, cfg, arm, seed,
                          protocol['width'], protocol['edge_width']).to(device)
    # Construction may consume different numbers of random draws. Reset the
    # training RNG to the SAME state for every arm and repeated run.
    set_seed(seed + protocol['training_rng_offset'])
    initial = predict(model, train, 'val')
    labels = {f: g[key].edge_label.cpu().numpy() for f, g in data.items()}
    optimizer = torch.optim.RMSprop((p for p in model.parameters() if p.requires_grad),
                                    lr=cfg['lr'], alpha=protocol['alpha'])
    best, trace, started = -float('inf'), [], time.monotonic()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        score = model(train.x_dict, train.edge_index_dict, 'train')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(score, train[key].edge_label)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
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
        if epoch == 1 or epoch % 50 == 0 or epoch == cfg['epochs']:
            print(task, arm, 'seed', seed, 'repeat', repeat, 'epoch', epoch,
                  'val %.5f/%.5f' % (values['auc'], values['aupr']),
                  'elapsed %.1fs' % (time.monotonic() - started), flush=True)
            save_csv(directory / 'trace.csv', trace)
    model.load_state_dict(best_state)
    selected_hash = parameter_hash(model)
    predictions, results, interventions = {}, {}, {}
    for fold in ('val', 'test'):
        predictions[fold] = predict(model, train, fold)
        results[fold] = metrics(labels[fold], predictions[fold], assets['pairs'][fold].numpy())
        arrays = model.attention()
        if arrays:
            np.savez_compressed(directory / ('attention_' + fold + '.npz'), **arrays)
    replay_delta = (results['val']['auc'] + results['val']['aupr']) / 2 - best
    if replay_delta != 0.:
        raise ValueError('Deterministic validation replay changed')
    modes = (('uniform', 'no_attributes') if arm == 'edge_attributes' else
             ('uniform', 'branch_off') if arm == 'path_attention' else ())
    for mode in modes:
        interventions[mode] = {}
        for fold in ('val', 'test'):
            name = mode + '_' + fold
            predictions[name] = predict(model, train, fold, mode)
            interventions[mode][fold] = metrics(labels[fold], predictions[name], assets['pairs'][fold].numpy())
    np.savez_compressed(directory / 'predictions.npz', initial_val_logits=initial, **predictions,
                        **{'y_' + f: labels[f] for f in ('val', 'test')},
                        **{'pairs_' + f: assets['pairs'][f].numpy() for f in ('val', 'test')})
    torch.save(best_state, directory / 'checkpoint.pt')
    record = dict(signature=signature, initial_core_sha256=core_hash,
                  selected_state_sha256=selected_hash, metrics=results, interventions=interventions,
                  selected_epoch=best_epoch, trained_epochs=cfg['epochs'], validation_replay_delta=replay_delta,
                  elapsed_seconds=time.monotonic() - started, device=device,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=sum(p.numel() for p in model.parameters()))
    write_json(result_path, record)
    print('RESULT', task, seed, arm, repeat, json.dumps(results), flush=True)
    del optimizer, model, base, train, best_state
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def run(args):
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    if p['source_hashes'] != source_hashes():
        raise ValueError('Source changed after freezing the plan')
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard')
    inputs, assets = {}, {}
    for task, audit in p['inputs'].items():
        inputs[task] = {}
        for fold, entry in audit['folds'].items():
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('Frozen graph changed')
            inputs[task][fold] = torch.load(entry['path'], map_location='cpu')
        entry = p['assets'][task]
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('Frozen assets changed')
        assets[task] = torch.load(entry['path'], map_location='cpu')
    # All arms and repeats for a task/seed stay on one device.
    for group, task, seed, arm, repeat in jobs(p):
        if group % args.shard_count == args.shard_index:
            train_one(root, p, inputs[task], assets[task], task, seed, arm, repeat, args.device)


def summarize(args):
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    rows, repeated, records = [], [], {}
    for _, task, seed, arm, repeat in jobs(p):
        directory = run_path(root, task, seed, arm, repeat)
        r = json.loads((directory / 'result.json').read_text())
        if r['signature'] != dict(task=task, seed=seed, arm=arm, repeat=repeat,
                                  protocol_sha256=file_hash(root / 'protocol.json')):
            raise ValueError('Mixed protocols')
        records[task, seed, arm, repeat] = r
        if repeat:
            first_dir = run_path(root, task, seed, arm, 0)
            first = records[task, seed, arm, 0]
            same_state = r['selected_state_sha256'] == first['selected_state_sha256']
            same_trace = file_hash(directory / 'trace.csv') == file_hash(first_dir / 'trace.csv')
            with np.load(directory / 'predictions.npz') as a, np.load(first_dir / 'predictions.npz') as b:
                same_predictions = all(np.array_equal(a[name], b[name]) for name in a.files)
            repeated.append(dict(task=task, seed=seed, arm=arm, same_device=r['device'] == first['device'],
                                 same_state=same_state, same_trace=same_trace, same_predictions=same_predictions,
                                 test_auc_delta=r['metrics']['test']['auc'] - first['metrics']['test']['auc'],
                                 test_aupr_delta=r['metrics']['test']['aupr'] - first['metrics']['test']['aupr']))
        else:
            row = dict(task=task, seed=seed, arm=arm, selected_epoch=r['selected_epoch'],
                       trainable_parameters=r['trainable_parameters'])
            for fold, scores in r['metrics'].items():
                row.update({fold + '_' + k: v for k, v in scores.items()})
            rows.append(row)
    for task, seed in itertools.product(p['tasks'], p['seeds']):
        if len({records[task, seed, a, 0]['initial_core_sha256'] for a in ARMS}) != 1:
            raise ValueError('Initial core weights do not match')
    save_csv(root / 'repeatability.csv', repeated)
    exact = all(all(r[k] for k in ('same_device', 'same_state', 'same_trace', 'same_predictions')) for r in repeated)
    write_json(root / 'repeatability.json', dict(exact=exact, comparisons=repeated,
                                                scope='same runner, RNG settings, inputs, device and seed'))
    if not exact:
        raise ValueError('Repeated runs differ despite deterministic configuration; inspect repeatability.csv')
    aggregates, comparisons, interventions = [], [], []
    for task, arm in itertools.product(p['tasks'], ARMS):
        group = [r for r in rows if r['task'] == task and r['arm'] == arm]
        agg = dict(task=task, arm=arm, n=len(group))
        for fold, metric in itertools.product(('val', 'test'), METRICS):
            key = fold + '_' + metric
            values = [r[key] for r in group]
            agg[key + '_mean'] = float(np.mean(values))
            agg[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        aggregates.append(agg)
    for task, (arm, reference) in itertools.product(p['tasks'], CONTRASTS):
        row = dict(task=task, arm=arm, reference=reference, n=len(p['seeds']))
        for fold, metric in itertools.product(('val', 'test'), METRICS):
            deltas = [records[task, s, arm, 0]['metrics'][fold][metric] -
                      records[task, s, reference, 0]['metrics'][fold][metric] for s in p['seeds']]
            row[fold + '_' + metric + '_delta'] = float(np.mean(deltas))
            row[fold + '_' + metric + '_wins'] = sum(d > 0 for d in deltas)
        comparisons.append(row)
    for task, arm in itertools.product(p['tasks'], PRIMARY):
        modes = records[task, p['seeds'][0], arm, 0]['interventions']
        for mode in modes:
            row = dict(task=task, arm=arm, intervention=mode, n=len(p['seeds']))
            for fold, metric in itertools.product(('val', 'test'), METRICS):
                deltas = [records[task, s, arm, 0]['interventions'][mode][fold][metric] -
                          records[task, s, arm, 0]['metrics'][fold][metric] for s in p['seeds']]
                row[fold + '_' + metric + '_delta'] = float(np.mean(deltas))
                row[fold + '_' + metric + '_decreases'] = sum(d < 0 for d in deltas)
            interventions.append(row)
    for name, data in (('per_seed', rows), ('summary', aggregates), ('comparisons', comparisons),
                       ('interventions', interventions)):
        save_csv(root / (name + '.csv'), data)
    report = ['# 边属性与路径实例注意力', '',
              f"完成 {len(records)}/{p['expected_runs']} 次训练；{len(repeated)} 组重复运行的完整曲线、预测和选定模型参数完全一致。", '',
              '主表仅统计 repeat_0，重复基线不作为额外独立种子。所有组使用同一确定性训练入口与清理后的训练图。',
              '原模型、模块初始化及训练随机数分别固定；仅以验证集 AUC/AUPR 均值选择 checkpoint。', '',
              '| Task | Arm | Val AUC | Val AUPR | Test AUC | Test AUPR | Per-drug test AUC |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        report.append('| %s | %s | %.5f | %.5f | %.5f | %.5f | %.5f |' %
                      (r['task'], r['arm'], r['val_auc_mean'], r['val_aupr_mean'],
                       r['test_auc_mean'], r['test_aupr_mean'], r['test_per_drug_auc_mean']))
    report.extend(['', '## 配对对照', '',
                   '| Task | Comparison | Δ test AUC | AUC wins | Δ test AUPR | AUPR wins |',
                   '|---|---|---:|---:|---:|---:|'])
    for r in comparisons:
        report.append('| %s | %s − %s | %+.5f | %d/%d | %+.5f | %d/%d |' %
                      (r['task'], r['arm'], r['reference'], r['test_auc_delta'], r['test_auc_wins'],
                       r['n'], r['test_aupr_delta'], r['test_aupr_wins'], r['n']))
    report.extend(['', '## 同一模型的干预', '',
                   '| Task | Arm | Intervention | Δ test AUC | Δ test AUPR | AUPR decreases |',
                   '|---|---|---|---:|---:|---:|'])
    for r in interventions:
        report.append('| %s | %s | %s | %+.5f | %+.5f | %d/%d |' %
                      (r['task'], r['arm'], r['intervention'], r['test_auc_delta'], r['test_aupr_delta'],
                       r['test_aupr_decreases'], r['n']))
    report.extend(['', '## 验证集筛选', ''])
    for task in p['tasks']:
        candidates = [r for r in aggregates if r['task'] == task and r['arm'] in ('base',) + PRIMARY]
        best = max(candidates, key=lambda r: (r['val_auc_mean'] + r['val_aupr_mean']) / 2)
        report.append('- %s：在基线和两个主要候选中选择 `%s`。' % (task, best['arm']))
    report.extend(['', '## 解释范围', '',
                   '- 固定划分探索；重复运行完全一致不等于跨划分泛化。负结果仅适用于本轮架构与日程。',
                   '- 确定性设置与历史轮次不同，只对本轮配对基线做效应比较。',
                   '- 边属性仅作用于三类相似关系；保留自环权重和非自环入度总权重。',
                   '- 路径枚举不使用标签，去掉 r→r→d 捷径；所有有效中间节点均保留，没有按分数截断。',
                   '- 原训练正例仍参与基础编码器的消息传递；留出目标边不进入训练图。',
                   '- 路径计数/均匀路径融合用于区分新增路径信息与学习加权的作用。',
                   '- 属性打乱对照保留关系内分布及互反配对；无属性组冻结不使用的属性投影。',
                   '- 等权组冻结不用的打分参数，参数数目见 per_seed.csv；不将不用的参数视为匹配的有效容量。',
                   '- 本轮未测试双向共注意力、带符号消息或多视图注意力。', ''])
    (root / 'REPORT.md').write_text('\n'.join(report))
    print('\n'.join(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--out-dir', required=True)
    init.add_argument('--input-protocol', default='improve/results/attention_mechanisms_20260907/protocol.json')
    init.add_argument('--net-dir', default='/data1/licheng/incrd-hg/incrd-run/2.construct_network')
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
