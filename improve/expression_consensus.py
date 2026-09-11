"""Controlled validation-only evaluation of cell-subset reliability.

prepare freezes all inputs before training; run accepts no test arrays; report
recomputes metrics and checks checkpoint selection and equal initialization.
The original GNN architecture and training schedules are retained.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from improve.expression_consensus_modules import (
    complementary_partitions, consistency_loss, edge_reliability,
    expression_views, normalize_incoming, shuffle_edge_reliability,
    shuffle_node_reliability)
from improve.expression_scale_mechanism import load_raw, state_digest, val_only_arrays
from improve.first_round_features import (
    json_digest, load_protocol, sha256, source_hashes, verify_files, write_json)
from improve.network_feature_ablation import ExperimentModel, score
from improve.network_feature_inputs import digest_array
from improve.run import set_seed


TASKS = ('LncDrug', 'MiDrug')
RNA_TYPES = ('lncRNA', 'miRNA')
ARMS = ('base', 'pca', 'abs_edges', 'reliable_edges', 'consistency_uniform',
        'consistency_reliable', 'joint', 'joint_shuffled')
DEFINITIONS = {
    'base': 'original normalized expression and unit edges',
    'pca': 'training-fitted PCA64 plus three zeros; unit edges',
    'abs_edges': 'PCA; supplied absolute similarities; original incoming mass',
    'reliable_edges': 'PCA; absolute similarities multiplied by paired-cell-subset support',
    'consistency_uniform': 'PCA; unit edges; uniform full-to-subset prediction consistency',
    'consistency_reliable': 'PCA; unit edges; node-reliability-weighted consistency',
    'joint': 'PCA; reproducibility-weighted edges and selective consistency',
    'joint_shuffled': 'joint with matched shuffled edge and node reliability; original views',
}
CONTRASTS = (
    ('pca', 'base'), ('abs_edges', 'pca'), ('reliable_edges', 'abs_edges'),
    ('consistency_uniform', 'pca'), ('consistency_reliable', 'consistency_uniform'),
    ('joint', 'pca'), ('joint', 'reliable_edges'),
    ('joint', 'consistency_reliable'), ('joint', 'joint_shuffled'),
)
SIMILARITIES = {
    ('lncRNA', 'LncLnc', 'lncRNA'): 'LncSimMat_ken.csv',
    ('miRNA', 'MiMi', 'miRNA'): 'MiSimMat_ken.csv',
    ('drug', 'DrugDrug', 'drug'): 'drugSimMat.csv',
}
OWN_SOURCES = ('expression_consensus.py', 'expression_consensus_modules.py',
               'expression_scale_mechanism.py', 'test_expression_consensus.py')


def all_sources():
    return dict(source_hashes(), **{name: sha256(Path(__file__).with_name(name)) for name in OWN_SOURCES})


def check_no_test(arrays):
    if any('test' in key for key in arrays):
        raise ValueError('Test arrays are forbidden in this exploratory runner')
    for key in ('pairs_train', 'pairs_val', 'y_train', 'y_val'):
        if key not in arrays:
            raise ValueError('Missing ' + key)


def describe(values):
    x = np.asarray(values, dtype=float)
    return {'min': float(x.min()), 'median': float(np.median(x)), 'max': float(x.max()),
            'mean': float(x.mean()), 'std': float(x.std())}


def prepare(args):
    out, ref = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if (out / 'protocol.json').exists():
        raise ValueError('Protocol already frozen; use run/report or a new directory')
    reference = load_protocol(ref)
    for filename, digest in reference['raw_network_files_sha256'].items():
        if sha256(Path(reference['net_dir']) / filename) != digest:
            raise ValueError('Raw input changed: ' + filename)
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'reference_dir': str(ref), 'reference_protocol_sha256': reference['protocol_sha256'],
        'tasks': list(TASKS), 'arms': list(ARMS), 'definitions': DEFINITIONS,
        'seeds': [666] if args.smoke else [666, 0, 1, 2, 3],
        'smoke': bool(args.smoke), 'training': reference['spec']['task_settings'],
        'cell_partitions': 8, 'partition_seed': 20260909,
        'consistency_lambda': 0.1, 'ramp_fraction': 0.2,
        'selection': 'first strict maximum of mean validation AUC/AUPR; full epoch budget',
        'contrasts_candidate_control': CONTRASTS,
        'primary_candidate': 'joint',
        'screen': 'positive mean paired validation AUC and AUPR on both tasks vs PCA; at least 4/5 criterion deltas positive per task; additionally report joint vs shuffled and component controls',
        'scope': 'fixed-split exploratory validation; initialization seeds do not establish new-split generalization',
        'feature_policy': 'training-node PCA/scalers frozen from reference; existing transductive node attributes allowed; no held-out labels used to construct views or reliability',
        'graph_policy': 'immutable cleaned training graph; existing target training positives remain message edges; similarity topology and self-loops unchanged',
        'edge_policy': 'same Kendall tau-b on complementary cell-line halves; oriented support does not imply signed propagation or causal inhibition; each incoming non-self total equals original degree',
        'no_test_evaluation': True, 'source_sha256': all_sources(), 'files': {}, 'audit': {},
    }
    if args.smoke:
        protocol['training'] = {t: dict(c, train_epochs=2) for t, c in protocol['training'].items()}
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        verify_files(ref, reference, prefix='inputs/' + task + '/')
        source_dir, dest = ref / 'inputs' / task, out / 'inputs' / task
        dest.mkdir(parents=True, exist_ok=True)
        graph = torch.load(source_dir / 'train_graph.pth', map_location='cpu')
        base = val_only_arrays(source_dir / 'base.npz')
        pca = val_only_arrays(source_dir / 'expr_pca64_stats.npz')
        arrays, audit = dict(base), {'nodes': {}, 'edges': {}}
        raw_by_type, partitions_by_type = {}, {}
        with np.load(source_dir / 'expression_transform.npz') as transform:
            for offset, nt in enumerate(RNA_TYPES):
                params = {k[len(nt) + 1:]: transform[k].copy() for k in transform.files if k.startswith(nt + '_')}
                raw = load_raw(reference['net_dir'], nt)
                partitions = complementary_partitions(raw.shape[1], protocol['cell_partitions'], protocol['partition_seed'] + offset)
                full, views, reliability, sensitivity = expression_views(raw, params, partitions)
                np.testing.assert_allclose(full, pca['x_' + nt][:, :64], atol=2e-5, rtol=2e-5)
                zeros = np.zeros((len(raw), 3), dtype=np.float32)
                arrays['pca_x_' + nt] = np.concatenate([pca['x_' + nt][:, :64], zeros], axis=1)
                arrays['views_' + nt] = np.pad(views, ((0, 0), (0, 0), (0, 3)))
                arrays['reliability_' + nt] = reliability
                arrays['shuffled_reliability_' + nt] = shuffle_node_reliability(reliability, params['fit_ids'], protocol['partition_seed'] + offset)
                raw_by_type[nt], partitions_by_type[nt] = raw, partitions
                audit['nodes'][nt] = {
                    'fit_ids_sha256': digest_array(params['fit_ids']), 'fit_nodes': len(params['fit_ids']),
                    'pca_matches_frozen': True, 'raw_sha256': digest_array(raw),
                    'reliability': describe(reliability), 'sensitivity': describe(sensitivity),
                    'view_mean_max_error': float(np.max(np.abs(views.mean(0) - full))),
                    'partition_sha256': [digest_array(ids) for pair in partitions for ids in pair],
                }
        for i, key in enumerate(graph.edge_types):
            edge = graph[key].edge_index.numpy()
            unit = base['weight_' + str(i)]
            np.testing.assert_array_equal(unit, np.ones_like(unit))
            absolute = reliable = shuffled = unit.copy()
            if key in SIMILARITIES:
                matrix = pd.read_csv(Path(reference['net_dir']) / SIMILARITIES[key], header=None).values
                strength = np.abs(matrix[edge[0], edge[1]])
                strength[edge[0] == edge[1]] = 1.0
                absolute = normalize_incoming(edge, strength, graph[key[2]].num_nodes)
                factors = np.ones_like(unit)
                if key[0] in RNA_TYPES:
                    print('PREPARE reliability', task, key[1], flush=True)
                    factors, details = edge_reliability(raw_by_type[key[0]], edge, matrix, partitions_by_type[key[0]])
                    np.savez_compressed(dest / (key[1] + '_reliability.npz'), **details)
                    audit['edges'][key[1]] = {
                        'undirected_pairs': len(details['pairs']), 'factor': describe(details['factor']),
                        'support': describe(details['support']), 'sign_agreement': describe(details['sign_agreement']),
                        'factor_abs_similarity_correlation': float(np.corrcoef(details['factor'], np.abs(details['original']))[0, 1]),
                    }
                reliable = normalize_incoming(edge, strength * factors, graph[key[2]].num_nodes)
                shuffled_factors = shuffle_edge_reliability(edge, matrix, factors, protocol['partition_seed'] + i)
                shuffled = normalize_incoming(edge, strength * shuffled_factors, graph[key[2]].num_nodes)
            arrays['abs_weight_' + str(i)] = absolute
            arrays['reliable_weight_' + str(i)] = reliable
            arrays['shuffled_weight_' + str(i)] = shuffled
        check_no_test(arrays)
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Non-finite prepared input')
        np.savez_compressed(dest / 'arrays.npz', **arrays)
        shutil.copy2(source_dir / 'train_graph.pth', dest / 'train_graph.pth')
        for path in dest.iterdir():
            if path.is_file():
                protocol['files'][str(path.relative_to(out))] = sha256(path)
        protocol['audit'][task] = audit
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in protocol['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    protocol['sha256'] = json_digest(protocol)
    write_json(out / 'protocol.json', protocol)
    print('FROZEN', out, 'runs', len(TASKS) * len(ARMS) * len(protocol['seeds']), flush=True)


def load_experiment(out):
    out = Path(out)
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['sha256'] != json_digest({k: v for k, v in protocol.items() if k != 'sha256'}):
        raise ValueError('Protocol hash changed')
    if protocol['source_sha256'] != all_sources():
        raise ValueError('Frozen source changed; use a new experiment directory')
    for name, digest in protocol['files'].items():
        if sha256(out / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return protocol


def make_model(arrays, graph, task, arm, seed, device):
    check_no_test(arrays)
    if arm not in ARMS:
        raise ValueError(arm)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    graph = graph.clone()
    for nt in graph.node_types:
        name = ('pca_x_' if arm != 'base' and nt in RNA_TYPES else 'x_') + nt
        graph[nt].x = torch.from_numpy(arrays[name])
    graph = graph.to(device)
    prefix = ('abs_' if arm == 'abs_edges' else 'reliable_' if arm in ('reliable_edges', 'joint')
              else 'shuffled_' if arm == 'joint_shuffled' else '')
    weights = {key: torch.from_numpy(arrays[prefix + 'weight_' + str(i)]).to(device)
               for i, key in enumerate(graph.edge_types)}
    pairs = {fold: torch.from_numpy(arrays['pairs_' + fold]).to(device) for fold in ('train', 'val')}
    model = ExperimentModel(graph, task).to(device)
    with torch.no_grad():
        model(graph.x_dict, graph.edge_index_dict, pairs['train'], weights)
    return model, graph, pairs, weights


def train_one(arrays, graph, task, arm, seed, protocol, device):
    start = time.monotonic()
    model, data, pairs, weights = make_model(arrays, graph, task, arm, seed, device)
    initial_hash = state_digest(model)
    settings = protocol['training'][task]
    optimizer = torch.optim.RMSprop(model.parameters(), lr=settings['lr'], alpha=.99)
    labels = torch.from_numpy(arrays['y_train']).to(device)
    has_consistency = arm in ('consistency_uniform', 'consistency_reliable', 'joint', 'joint_shuffled')
    views = {nt: torch.from_numpy(arrays['views_' + nt]).to(device) for nt in RNA_TYPES} if has_consistency else {}
    reliability = None
    if has_consistency and arm != 'consistency_uniform':
        name = ('shuffled_reliability_' if arm == 'joint_shuffled' else 'reliability_') + settings['rna']
        reliability = torch.from_numpy(arrays[name]).to(device)[pairs['train'][0]]
    best, best_state, best_prediction, best_epoch = -np.inf, None, None, None
    trace = []
    for epoch in range(1, settings['train_epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        teacher = model(data.x_dict, data.edge_index_dict, pairs['train'], weights)
        supervised = F.binary_cross_entropy_with_logits(teacher, labels)
        auxiliary = teacher.new_zeros(())
        ramp = min(1., (epoch - 1) / max(1., protocol['ramp_fraction'] * settings['train_epochs']))
        if has_consistency and ramp > 0:
            view_id = (epoch - 1 + seed) % len(views[RNA_TYPES[0]])
            x_view = {nt: views[nt][view_id] if nt in views else x for nt, x in data.x_dict.items()}
            student = model(x_view, data.edge_index_dict, pairs['train'], weights)
            auxiliary = consistency_loss(student, teacher, reliability)
        loss = supervised + protocol['consistency_lambda'] * ramp * auxiliary
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = model(data.x_dict, data.edge_index_dict, pairs['val'], weights).cpu().numpy()
        auc, ap = score(arrays['y_val'], val)
        criterion = (auc + ap) / 2
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), supervised=float(supervised.detach()),
                          consistency=float(auxiliary.detach()), val_auc=auc, val_aupr=ap, criterion=criterion))
        if criterion > best:
            best, best_epoch, best_prediction = criterion, epoch, val.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored = model(data.x_dict, data.edge_index_dict, pairs['val'], weights).cpu().numpy()
    np.testing.assert_allclose(restored, best_prediction, atol=3e-5, rtol=3e-5)
    auc, ap = score(arrays['y_val'], best_prediction)
    return {
        'task': task, 'arm': arm, 'seed': seed, 'val_auc': auc, 'val_aupr': ap,
        'val_criterion': (auc + ap)/2, 'selected_epoch': best_epoch,
        'trained_epochs': settings['train_epochs'], 'initial_state_sha256': initial_hash,
        'params': sum(p.numel() for p in model.parameters()),
        'node_dims': {nt: data[nt].x.shape[1] for nt in data.node_types},
        'restore_max_logit_abs_error': float(np.max(np.abs(restored - best_prediction))),
        'elapsed_seconds': time.monotonic() - start, 'device': device,
        'protocol_sha256': protocol['sha256'],
    }, best_prediction, trace, best_state


def check_result(directory, protocol):
    result = json.loads((directory / 'result.json').read_text())
    if result['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Result protocol differs')
    for name, digest in result['artifacts_sha256'].items():
        if sha256(directory / name) != digest:
            raise ValueError('Output changed: ' + str(directory / name))
    return result


def run(args):
    out = Path(args.out_dir)
    protocol = load_experiment(out)
    for task in [args.task] if args.task else TASKS:
        with (out / (task + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with np.load(out / 'inputs' / task / 'arrays.npz') as z:
                arrays = {k: z[k].copy() for k in z.files}
            graph = torch.load(out / 'inputs' / task / 'train_graph.pth', map_location='cpu')
            for seed in protocol['seeds']:
                for arm in protocol['arms']:
                    directory = out / 'runs' / task / ('s' + str(seed)) / arm
                    directory.mkdir(parents=True, exist_ok=True)
                    if (directory / 'result.json').exists():
                        check_result(directory, protocol)
                        print('SKIP', task, seed, arm, flush=True)
                        continue
                    print('START', task, seed, arm, flush=True)
                    result, val, trace, state = train_one(arrays, graph, task, arm, seed, protocol, args.device)
                    np.savez_compressed(directory / 'scores.npz', val=val, y_val=arrays['y_val'], pairs_val=arrays['pairs_val'])
                    pd.DataFrame(trace).to_csv(directory / 'trace.csv', index=False)
                    torch.save(state, directory / 'checkpoint.pt')
                    result['artifacts_sha256'] = {name: sha256(directory / name) for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                    write_json(directory / 'result.json', result)
                    print('DONE', task, seed, arm, 'val %.6f/%.6f epoch %d %.1fs' % (
                        result['val_auc'], result['val_aupr'], result['selected_epoch'], result['elapsed_seconds']), flush=True)


def report(args):
    out = Path(args.out_dir)
    protocol = load_experiment(out)
    rows, initial = [], {}
    for task in TASKS:
        with np.load(out / 'inputs' / task / 'arrays.npz') as z:
            arrays = {k: z[k].copy() for k in z.files}
        check_no_test(arrays)
        graph = torch.load(out / 'inputs' / task / 'train_graph.pth', map_location='cpu')
        for seed in protocol['seeds']:
            for arm in protocol['arms']:
                directory = out / 'runs' / task / ('s' + str(seed)) / arm
                result = check_result(directory, protocol)
                with np.load(directory / 'scores.npz') as z:
                    if set(z.files) != {'val', 'y_val', 'pairs_val'}:
                        raise ValueError('Unexpected evaluation arrays')
                    for name in ('y_val', 'pairs_val'):
                        np.testing.assert_array_equal(z[name], arrays[name])
                    saved = z['val'].copy()
                auc, ap = score(arrays['y_val'], saved)
                np.testing.assert_allclose([auc, ap], [result['val_auc'], result['val_aupr']], atol=1e-12, rtol=0)
                trace = pd.read_csv(directory / 'trace.csv')
                np.testing.assert_array_equal(trace.epoch, np.arange(1, protocol['training'][task]['train_epochs'] + 1))
                if int(trace.loc[trace.criterion.idxmax(), 'epoch']) != result['selected_epoch']:
                    raise AssertionError('Incorrect checkpoint selection')
                np.testing.assert_allclose(trace.criterion.max(), result['val_criterion'], atol=1e-12, rtol=0)
                key = (task, seed, tuple(result['node_dims'].items()))
                value = (result['initial_state_sha256'], result['params'])
                if key in initial and initial[key] != value:
                    raise AssertionError('Equal-width arms have different initial parameters')
                initial[key] = value
                if args.replay:
                    model, data, pairs, weights = make_model(arrays, graph, task, arm, seed, args.device)
                    if state_digest(model) != result['initial_state_sha256']:
                        raise AssertionError('Initialization cannot be reproduced')
                    model.load_state_dict(torch.load(directory / 'checkpoint.pt', map_location='cpu'))
                    model.eval()
                    with torch.no_grad():
                        replay = model(data.x_dict, data.edge_index_dict, pairs['val'], weights).cpu().numpy()
                    np.testing.assert_allclose(replay, saved, atol=3e-5, rtol=3e-5)
                    result['independent_replay_max_abs'] = float(np.max(np.abs(replay - saved)))
                rows.append(result)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'results.csv', index=False)
    summary = frame.groupby(['task', 'arm'])[['val_auc', 'val_aupr', 'val_criterion']].agg(['mean', 'std'])
    summary.to_csv(out / 'summary.csv')
    deltas = []
    for task in TASKS:
        task_frame = frame[frame.task == task].set_index(['arm', 'seed'])
        for candidate, control in CONTRASTS:
            for metric in ('val_auc', 'val_aupr', 'val_criterion'):
                delta = np.array([task_frame.loc[(candidate, s), metric] - task_frame.loc[(control, s), metric] for s in protocol['seeds']])
                deltas.append(dict(task=task, candidate=candidate, control=control, metric=metric,
                                   mean_delta=float(delta.mean()), std_delta=float(delta.std(ddof=1)) if len(delta) > 1 else 0.,
                                   positive_seeds=int((delta > 0).sum()), seeds=len(delta)))
    pd.DataFrame(deltas).to_csv(out / 'paired_deltas.csv', index=False)
    lines = ['# 跨细胞系子集可靠性：原 GNN 对照实验', '',
             '仅使用验证集探索；不读取测试预测。每个种子的两个指标来自同一验证选定检查点。', '',
             '| 任务 | 方案 | Val AUC 均值 | Val AUPR 均值 |', '|---|---|---:|---:|']
    for task in TASKS:
        for arm in ARMS:
            subset = frame[(frame.task == task) & (frame.arm == arm)]
            lines.append('| %s | %s | %.6f | %.6f |' % (task, arm, subset.val_auc.mean(), subset.val_aupr.mean()))
    lines += ['', '## 预设关键对照', '', '| 任务 | 候选 − 对照 | 指标 | 配对均值差 | 提升种子 |', '|---|---|---|---:|---:|']
    for d in deltas:
        if d['metric'] != 'val_criterion':
            lines.append('| %s | %s − %s | %s | %+.6f | %d/%d |' % (d['task'], d['candidate'], d['control'], d['metric'], d['mean_delta'], d['positive_seeds'], d['seeds']))
    lines += ['', '五个初始化种子不能替代新划分验证。可靠性是细胞系子集敏感度，不是癌种条件化效果或因果证据。',
              '普通一致性约束、绝对相似度和打乱可靠性均保留为对照；所有方案及失败结果完整保留。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    write_json(out / 'VALIDATION.json', {'passed': True, 'completed_runs': len(rows),
        'expected_runs': len(TASKS) * len(ARMS) * len(protocol['seeds']), 'smoke': protocol['smoke'],
        'replayed': bool(args.replay), 'protocol_sha256': protocol['sha256'],
        'checks': ['input/source hashes', 'no test arrays', 'saved prediction metrics', 'epoch budgets',
                   'one validation-selected checkpoint', 'equal-width parameter initialization', 'checkpoint replay' if args.replay else 'training-time restore']})
    print(summary.to_string(), flush=True)
    print('VERIFIED', len(rows), 'runs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'report'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--reference-dir', default='improve/results/first_round_features_20260909')
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    {'prepare': prepare, 'run': run, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
