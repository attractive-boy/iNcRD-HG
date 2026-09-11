"""Diagnosis-grouped expression bags with a shared original GNN.

An adaptive follow-up to the negative cell-subset consistency experiment.
Global association labels supervise a bag; no group-specific label is invented.
"""
import argparse
from collections import Counter
import copy
import fcntl
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

from improve.expression_consensus import (
    RNA_TYPES, TASKS, all_sources, check_no_test, describe,
    load_experiment, make_model)
from improve.expression_scale_mechanism import load_raw, state_digest
from improve.first_round_features import json_digest, load_protocol, sha256, verify_files, write_json
from improve.network_feature_ablation import score


ARMS = ('pca', 'context_mean', 'context_softor', 'random_softor')
CONTRASTS = (('context_mean', 'pca'), ('context_softor', 'pca'),
             ('context_softor', 'context_mean'), ('context_softor', 'random_softor'))


def own_sources():
    return dict(all_sources(), **{name: sha256(Path(__file__).with_name(name))
                for name in ('cell_context_mil.py', 'test_cell_context_mil.py')})


def depmap_diagnoses(path):
    mapping, identifiers, diagnosis = {}, [], None
    with open(path, encoding='utf-8') as source:
        for line in source:
            if line.startswith('//'):
                for identifier in identifiers:
                    if identifier in mapping and mapping[identifier] != diagnosis:
                        raise ValueError('Conflicting diagnoses for ' + identifier)
                    mapping[identifier] = diagnosis
                identifiers, diagnosis = [], None
            elif line.startswith('DR   DepMap; '):
                identifiers.append(line[len('DR   DepMap; '):].strip())
            elif line.startswith('DI   ') and diagnosis is None:
                parts = [x.strip() for x in line[5:].split(';')]
                if len(parts) >= 3:
                    diagnosis = parts[2]
    return mapping


def make_groups(cell_ids, diagnoses, n_named=3, seed=20260910):
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError('Duplicate cell-line identifiers')
    counts = Counter(diagnoses.get(cell) for cell in cell_ids)
    selected = sorted((name for name in counts if name is not None),
                      key=lambda name: (-counts[name], name))[:n_named]
    if len(selected) != n_named or min(counts[name] for name in selected) < 20:
        raise ValueError('Insufficient coverage for named diagnosis contexts')
    assignment = {cell: selected.index(diagnoses[cell]) if diagnoses.get(cell) in selected else n_named
                  for cell in cell_ids}
    canonical = sorted(cell_ids)
    labels = np.array([assignment[cell] for cell in canonical])
    random_labels = np.random.default_rng(seed).permutation(labels)
    random_assignment = dict(zip(canonical, map(int, random_labels)))
    return selected + ['remaining_cell_lines'], assignment, random_assignment


def context_features(raw, parameters, cell_ids, assignment, n_contexts):
    """Project each context into a common PCA basis, then fit its scale on train nodes."""
    raw = np.asarray(raw, dtype=np.float64)
    standard = ((raw - parameters['input_mean']) / parameters['input_scale']
                - parameters['pca_mean'])
    features, audit = [], []
    for context in range(n_contexts):
        ids = np.array([i for i, cell in enumerate(cell_ids) if assignment[cell] == context])
        if not len(ids):
            raise ValueError('Empty context')
        partial = (standard[:, ids] @ parameters['pca_components'][:, ids].T * (raw.shape[1] / len(ids))
                   - parameters['component_mean']) / parameters['component_scale']
        scaler = StandardScaler().fit(partial[parameters['fit_ids']])
        z = scaler.transform(partial).astype(np.float32)
        features.append(np.pad(z, ((0, 0), (0, 3))))
        audit.append({'cells': len(ids), 'fit_nodes': len(parameters['fit_ids']),
                      'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist()})
    return np.stack(features), audit


def pack_context_graph(data, pairs, weights, views, rna_type):
    """Disjoint copies of one heterogeneous graph; all copies share GNN parameters."""
    bags = len(next(iter(views.values()))) + 1
    counts = {nt: data[nt].num_nodes for nt in data.node_types}
    x = {}
    for nt, full in data.x_dict.items():
        blocks = torch.cat([full.unsqueeze(0), views[nt]], dim=0) if nt in views else full.unsqueeze(0).expand(bags, -1, -1)
        x[nt] = blocks.reshape(bags * counts[nt], -1)
    edges, repeated_weights = {}, {}
    for key, index in data.edge_index_dict.items():
        size = index.new_tensor([[counts[key[0]]], [counts[key[2]]]])
        edges[key] = torch.cat([index + bag * size for bag in range(bags)], dim=1)
        repeated_weights[key] = weights[key].repeat(bags)
    indices = {}
    for fold, index in pairs.items():
        size = index.new_tensor([[counts[rna_type]], [counts['drug']]])
        indices[fold] = torch.cat([index + bag * size for bag in range(bags)], dim=1)
    return x, edges, indices, repeated_weights, bags


def pool_context_logits(logits, mode, alpha=.25, temperature=2.):
    if logits.ndim != 2 or logits.shape[0] < 2 or not 0 <= alpha <= 1 or temperature <= 0:
        raise ValueError('Invalid bag logits or pooling settings')
    full, contexts = logits[0], logits[1:]
    if mode == 'mean':
        pooled = contexts.mean(0)
    elif mode == 'softor':
        pooled = temperature * (torch.logsumexp(contexts / temperature, dim=0) - math.log(len(contexts)))
    else:
        raise ValueError(mode)
    return (1 - alpha) * full + alpha * pooled


def prepare(args):
    out, parent_dir = Path(args.out_dir).resolve(), Path(args.parent_dir).resolve()
    if (out / 'protocol.json').exists():
        raise ValueError('Existing frozen protocol; use a new directory')
    parent = load_experiment(parent_dir)
    validation = json.loads((parent_dir / 'VALIDATION.json').read_text())
    if not validation['passed'] or validation['smoke']:
        raise ValueError('Requires verified full first-stage experiment')
    reference_dir = Path(parent['reference_dir'])
    reference = load_protocol(reference_dir)
    net = Path(reference['net_dir'])
    cell_ids = {}
    for nt, filename in (('lncRNA', 'LncExp.csv'), ('miRNA', 'MiExp.csv')):
        ids = pd.read_csv(net / filename, usecols=[0]).iloc[:, 0].tolist()
        cell_ids[nt] = sorted(ids, key=lambda s: int(s.split('-')[-1])) if nt == 'miRNA' else ids
    if set(cell_ids['lncRNA']) != set(cell_ids['miRNA']):
        raise ValueError('RNA types do not share exactly the same cell lines')
    diagnoses = depmap_diagnoses(args.cellosaurus)
    names, assigned, random_assigned = make_groups(cell_ids['lncRNA'], diagnoses)
    protocol = {
        'parent_dir': str(parent_dir), 'parent_protocol_sha256': parent['sha256'],
        'adaptive_followup': 'Designed after first-stage validation rejected the joint consistency candidate; not independent confirmation',
        'source_sha256': own_sources(), 'cellosaurus_sha256': sha256(args.cellosaurus),
        'tasks': list(TASKS), 'arms': list(ARMS), 'primary': 'context_softor',
        'seeds': [666] if args.smoke else [666, 0, 1, 2, 3], 'smoke': bool(args.smoke),
        'training': copy.deepcopy(parent['training']), 'alpha': .25, 'temperature': 2.,
        'named_diagnoses': names, 'context_sizes': [sum(v == i for v in assigned.values()) for i in range(len(names))],
        'mapped_cell_lines': sum(diagnoses.get(cell) is not None for cell in cell_ids['lncRNA']),
        'cell_assignment': assigned, 'random_cell_assignment': random_assigned,
        'cell_ids': cell_ids, 'contrasts': CONTRASTS,
        'selection': parent['selection'], 'evaluation_folds': ['val'],
        'context_selection': 'three most frequent exact Cellosaurus diagnosis terms among the 938 cells, ties alphabetical; all other cells pooled; no labels used',
        'model': 'same original GNN shared across full expression and four context graphs; original edge sets and unit weights; no added parameters',
        'features': 'same training-fitted PCA basis; each partial projection rescaled using training nodes only; three zero padding dimensions',
        'pooling': '.75 full-context logit + .25 context pool; mean or temperature-2 log-mean-exp; global bag labels only',
        'screen': 'both mean validation AUC and AUPR above PCA on both tasks and at least 4/5 paired criterion gains; separately require gains over mean pooling and size-matched randomized contexts for a mechanism claim',
        'limitations': ['no context-specific supervision or causal claims', 'adaptive fixed-validation exploration', 'diagnosis coverage is uneven', 'five initialization seeds are not independent data splits'],
        'files': {}, 'audit': {},
    }
    if args.smoke:
        for config in protocol['training'].values():
            config['train_epochs'] = 2
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        verify_files(reference_dir, reference, prefix='inputs/' + task + '/')
        dest = out / 'inputs' / task
        dest.mkdir(parents=True, exist_ok=True)
        with np.load(parent_dir / 'inputs' / task / 'arrays.npz') as z:
            arrays = {key: z[key].copy() for key in z.files if key.startswith(('x_', 'pca_x_', 'weight_'))
                      or key in ('pairs_train', 'pairs_val', 'y_train', 'y_val')}
        audit = {}
        with np.load(reference_dir / 'inputs' / task / 'expression_transform.npz') as z:
            for nt in RNA_TYPES:
                params = {k[len(nt)+1:]: z[k].copy() for k in z.files if k.startswith(nt + '_')}
                raw = load_raw(net, nt)
                for prefix, assignment in (('context', assigned), ('random', random_assigned)):
                    values, info = context_features(raw, params, cell_ids[nt], assignment, len(names))
                    arrays[prefix + '_x_' + nt] = values
                    audit[prefix + '_' + nt] = info
        check_no_test(arrays)
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Non-finite context input')
        np.savez_compressed(dest / 'arrays.npz', **arrays)
        shutil.copy2(parent_dir / 'inputs' / task / 'train_graph.pth', dest / 'train_graph.pth')
        for file in dest.iterdir():
            protocol['files'][str(file.relative_to(out))] = sha256(file)
        protocol['audit'][task] = audit
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in protocol['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    protocol['sha256'] = json_digest(protocol)
    write_json(out / 'protocol.json', protocol)
    print('FROZEN', out, 'contexts', list(zip(names, protocol['context_sizes'])), flush=True)


def load_config(out):
    out = Path(out)
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['sha256'] != json_digest({k: v for k, v in protocol.items() if k != 'sha256'}):
        raise ValueError('Changed protocol')
    if protocol['source_sha256'] != own_sources():
        raise ValueError('Source changed after freezing')
    for name, digest in protocol['files'].items():
        if sha256(out / name) != digest:
            raise ValueError('Input changed: ' + name)
    return protocol


def build(arrays, graph, task, arm, seed, protocol, device):
    model, data, pairs, weights = make_model(arrays, graph, task, 'pca', seed, device)
    if arm == 'pca':
        def forward(fold):
            return model(data.x_dict, data.edge_index_dict, pairs[fold], weights)
        return model, forward
    prefix = 'random' if arm == 'random_softor' else 'context'
    views = {nt: torch.from_numpy(arrays[prefix + '_x_' + nt]).to(device) for nt in RNA_TYPES}
    x, edges, indices, repeated_weights, bags = pack_context_graph(data, pairs, weights, views, protocol['training'][task]['rna'])
    mode = 'mean' if arm == 'context_mean' else 'softor'
    def forward(fold):
        logits = model(x, edges, indices[fold], repeated_weights).view(bags, -1)
        return pool_context_logits(logits, mode, protocol['alpha'], protocol['temperature'])
    return model, forward


def train(arrays, graph, task, arm, seed, protocol, device):
    started = time.monotonic()
    model, forward = build(arrays, graph, task, arm, seed, protocol, device)
    initial = state_digest(model)
    config = protocol['training'][task]
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'], alpha=.99)
    y = torch.from_numpy(arrays['y_train']).to(device)
    best, state, prediction, chosen = -np.inf, None, None, None
    trace = []
    for epoch in range(1, config['train_epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(forward('train'), y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = forward('val').cpu().numpy()
        auc, ap = score(arrays['y_val'], val)
        criterion = (auc + ap)/2
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), val_auc=auc, val_aupr=ap, criterion=criterion))
        if criterion > best:
            best, chosen, prediction = criterion, epoch, val.copy()
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        restored = forward('val').cpu().numpy()
    np.testing.assert_allclose(restored, prediction, atol=4e-5, rtol=4e-5)
    auc, ap = score(arrays['y_val'], prediction)
    return dict(task=task, arm=arm, seed=seed, val_auc=auc, val_aupr=ap,
        val_criterion=(auc + ap)/2, selected_epoch=chosen, trained_epochs=config['train_epochs'],
        initial_state_sha256=initial, params=sum(p.numel() for p in model.parameters()),
        restore_max_abs=float(np.max(np.abs(restored - prediction))),
        elapsed_seconds=time.monotonic() - started, protocol_sha256=protocol['sha256']), prediction, trace, state


def checked_result(directory, protocol):
    result = json.loads((directory / 'result.json').read_text())
    if result['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Result protocol differs')
    for filename, digest in result['artifacts_sha256'].items():
        if sha256(directory / filename) != digest:
            raise ValueError('Changed output: ' + filename)
    return result


def run(args):
    out = Path(args.out_dir)
    protocol = load_config(out)
    for task in [args.task] if args.task else TASKS:
        with (out / (task + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with np.load(out / 'inputs' / task / 'arrays.npz') as z:
                arrays = {k: z[k].copy() for k in z.files}
            graph = torch.load(out / 'inputs' / task / 'train_graph.pth', map_location='cpu')
            for seed in protocol['seeds']:
                for arm in ARMS:
                    directory = out / 'runs' / task / ('s' + str(seed)) / arm
                    directory.mkdir(parents=True, exist_ok=True)
                    if (directory / 'result.json').exists():
                        checked_result(directory, protocol)
                        print('SKIP', task, seed, arm, flush=True)
                        continue
                    print('START', task, seed, arm, flush=True)
                    result, prediction, trace, state = train(arrays, graph, task, arm, seed, protocol, args.device)
                    np.savez_compressed(directory / 'scores.npz', val=prediction, y_val=arrays['y_val'], pairs_val=arrays['pairs_val'])
                    pd.DataFrame(trace).to_csv(directory / 'trace.csv', index=False)
                    torch.save(state, directory / 'checkpoint.pt')
                    result['artifacts_sha256'] = {name: sha256(directory / name) for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                    write_json(directory / 'result.json', result)
                    print('DONE', task, seed, arm, 'val %.6f/%.6f epoch %d %.1fs' % (
                        result['val_auc'], result['val_aupr'], result['selected_epoch'], result['elapsed_seconds']), flush=True)


def report(args):
    out = Path(args.out_dir)
    protocol = load_config(out)
    rows, initial = [], {}
    for task in TASKS:
        with np.load(out / 'inputs' / task / 'arrays.npz') as z:
            arrays = {k: z[k].copy() for k in z.files}
        check_no_test(arrays)
        graph = torch.load(out / 'inputs' / task / 'train_graph.pth', map_location='cpu')
        for seed in protocol['seeds']:
            for arm in ARMS:
                directory = out / 'runs' / task / ('s' + str(seed)) / arm
                result = checked_result(directory, protocol)
                with np.load(directory / 'scores.npz') as z:
                    if set(z.files) != {'val', 'y_val', 'pairs_val'}:
                        raise ValueError('Unexpected evaluation arrays')
                    for key in ('y_val', 'pairs_val'):
                        np.testing.assert_array_equal(z[key], arrays[key])
                    saved = z['val'].copy()
                auc, ap = score(arrays['y_val'], saved)
                np.testing.assert_allclose([auc, ap], [result['val_auc'], result['val_aupr']], atol=1e-12, rtol=0)
                trace = pd.read_csv(directory / 'trace.csv')
                np.testing.assert_array_equal(trace.epoch, np.arange(1, protocol['training'][task]['train_epochs'] + 1))
                if int(trace.loc[trace.criterion.idxmax(), 'epoch']) != result['selected_epoch']:
                    raise ValueError('Incorrect checkpoint selection')
                key = (task, seed)
                value = (result['initial_state_sha256'], result['params'])
                if key in initial and initial[key] != value:
                    raise ValueError('Initial parameters differ between arms')
                initial[key] = value
                if args.replay:
                    model, forward = build(arrays, graph, task, arm, seed, protocol, args.device)
                    if state_digest(model) != result['initial_state_sha256']:
                        raise ValueError('Initialization differs on replay')
                    model.load_state_dict(torch.load(directory / 'checkpoint.pt', map_location='cpu'))
                    model.eval()
                    with torch.no_grad():
                        replay = forward('val').cpu().numpy()
                    np.testing.assert_allclose(replay, saved, atol=4e-5, rtol=4e-5)
                rows.append(result)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'results.csv', index=False)
    summary = frame.groupby(['task', 'arm'])[['val_auc', 'val_aupr']].agg(['mean', 'std'])
    summary.to_csv(out / 'summary.csv')
    indexed = frame.set_index(['task', 'arm', 'seed'])
    deltas = []
    for task in TASKS:
        for candidate, control in CONTRASTS:
            for metric in ('val_auc', 'val_aupr', 'val_criterion'):
                delta = np.array([indexed.loc[(task, candidate, seed), metric] - indexed.loc[(task, control, seed), metric] for seed in protocol['seeds']])
                deltas.append(dict(task=task, candidate=candidate, control=control, metric=metric,
                    mean_delta=float(delta.mean()), std_delta=float(delta.std(ddof=1)) if len(delta) > 1 else 0.,
                    positive_seeds=int((delta > 0).sum()), seeds=len(delta)))
    pd.DataFrame(deltas).to_csv(out / 'paired_deltas.csv', index=False)
    lines = ['# 来源诊断分组的多实例 GNN', '',
        '第二轮自适应探索，沿用已反复使用的验证划分。没有测试评估或癌种级监督。', '',
        '| 任务 | 方案 | Val AUC | Val AUPR |', '|---|---|---:|---:|']
    for task in TASKS:
        for arm in ARMS:
            s = frame[(frame.task == task) & (frame.arm == arm)]
            lines.append('| %s | %s | %.6f | %.6f |' % (task, arm, s.val_auc.mean(), s.val_aupr.mean()))
    lines += ['', '| 任务 | 候选 − 对照 | 指标 | 配对均值差 | 提升种子 |', '|---|---|---|---:|---:|']
    for d in deltas:
        if d['metric'] != 'val_criterion':
            lines.append('| %s | %s − %s | %s | %+.6f | %d/%d |' % (d['task'], d['candidate'], d['control'], d['metric'], d['mean_delta'], d['positive_seeds'], d['seeds']))
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_json(out / 'VALIDATION.json', dict(passed=True, smoke=protocol['smoke'],
        completed_runs=len(rows), replayed=args.replay, protocol_sha256=protocol['sha256'],
        checks=['frozen inputs/source', 'no test arrays', 'prediction metrics', 'single checkpoint',
                'epoch budgets', 'identical initial parameters', 'checkpoint replay' if args.replay else 'restore']))
    print(summary.to_string())
    print('VERIFIED', len(rows), 'runs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'report'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--parent-dir', default='improve/results/expression_consensus_20260909')
    parser.add_argument('--cellosaurus', default='/data1/licheng/incrd-hg/incrd-run/13.context/cellosaurus.txt')
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    {'prepare': prepare, 'run': run, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
