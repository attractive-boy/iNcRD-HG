"""Controlled channel, sparse-neighbour and layer attention experiments.

Run ``init`` to freeze inputs and the experiment plan, then ``run`` on one or
more disjoint shards, and finally ``summarize``. Existing experiments and source
datasets are never overwritten. All variants start at exactly the same base
prediction for a given seed. Only validation selects the checkpoint.
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve.drug_condition_attention import (
    CONFIG, METRICS, file_hash, make_base, masked_weights, metrics, pair_set,
    rna_neighbour_masks, set_seed, write_json,
)


ARMS = (
    'base', 'channel_mlp', 'channel_attention',
    'neighbour_uniform', 'neighbour_softmax', 'neighbour_randomk', 'neighbour_topk',
    'layer_uniform', 'layer_attention',
)
CONTRASTS = (
    ('channel_mlp', 'base'), ('channel_attention', 'base'),
    ('channel_attention', 'channel_mlp'),
    ('neighbour_uniform', 'base'), ('neighbour_softmax', 'neighbour_uniform'),
    ('neighbour_topk', 'base'), ('neighbour_topk', 'neighbour_softmax'),
    ('neighbour_topk', 'neighbour_randomk'),
    ('layer_uniform', 'base'), ('layer_attention', 'base'),
    ('layer_attention', 'layer_uniform'),
)
PRIMARY = ('channel_attention', 'neighbour_topk', 'layer_attention')


def source_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ('attention_mechanisms.py', 'drug_condition_attention.py', 'model.py')}


def clean_splits(original, key):
    """Give held-out pairs priority; remove overlaps from training and its graph.

    Within-training multiplicity is retained (two MiDrug positive repeats).
    Conflicting labels are rejected, never silently reclassified.
    """
    for data in original.values():
        for nt, x in original['train'].x_dict.items():
            if not torch.equal(x, data[nt].x) or not torch.isfinite(x).all():
                raise ValueError('Inconsistent or nonfinite features')
    labels_by_pair = {}
    for fold, data in original.items():
        ids, y = data[key].edge_label_index, data[key].edge_label
        if set(y.tolist()) != {0., 1.}:
            raise ValueError('Expected binary labels in each fold')
        if fold != 'train' and len(pair_set(ids)) != ids.size(1):
            raise ValueError('Duplicate held-out rows')
        for pair, label in zip(ids.t().tolist(), y.tolist()):
            pair = tuple(pair)
            if pair in labels_by_pair and labels_by_pair[pair] != label:
                raise ValueError('Conflicting labels for pair ' + str(pair))
            labels_by_pair[pair] = label
    selected, removed, occupied = {}, {}, set()
    for fold in ('test', 'val', 'train'):
        ids, y = original[fold][key].edge_label_index, original[fold][key].edge_label
        keep = torch.tensor([tuple(p) not in occupied for p in ids.t().tolist()])
        removed[fold] = [dict(pair=p, label=int(label)) for p, label in
                         zip(ids[:, ~keep].t().tolist(), y[~keep].tolist())]
        selected[fold] = (ids[:, keep].clone(), y[keep].clone())
        occupied.update(pair_set(ids[:, keep]))
    train = copy.deepcopy(original['train'])
    original_pos = pair_set(original['train'][key].edge_label_index[:,
                            original['train'][key].edge_label == 1])
    if pair_set(train[key].edge_index) != original_pos:
        raise ValueError('Unexpected original training graph')
    reverse = ('drug', 'rev_' + key[1], key[0])
    if pair_set(train[reverse].edge_index[[1, 0]]) != original_pos:
        raise ValueError('Inconsistent reverse edges')
    allowed = pair_set(selected['train'][0][:, selected['train'][1] == 1])
    edges = train[key].edge_index
    keep = torch.tensor([tuple(p) in allowed for p in edges.t().tolist()])
    train[key].edge_index = edges[:, keep].clone()
    train[reverse].edge_index = train[key].edge_index.flip(0).clone()
    splits = {}
    for fold, (ids, y) in selected.items():
        splits[fold] = copy.deepcopy(train)
        splits[fold][key].edge_label_index = ids
        splits[fold][key].edge_label = y
    sets = {f: pair_set(g[key].edge_label_index) for f, g in splits.items()}
    for a, b in itertools.combinations(sets, 2):
        if sets[a] & sets[b]:
            raise ValueError('Cross-fold overlap after cleaning')
    for fold in ('val', 'test'):
        if sets[fold] & pair_set(train[key].edge_index):
            raise ValueError('Held-out pair enters message passing')
    return splits, removed


def initialise(args):
    root = Path(args.out_dir)
    if (root / 'protocol.json').exists():
        raise ValueError('Plan already exists; use run to resume: ' + str(root))
    root.mkdir(parents=True, exist_ok=True)
    configs, audits = copy.deepcopy(CONFIG), {}
    if args.smoke:
        for c in configs.values():
            c['epochs'] = 3
    for task, config in configs.items():
        key = (config['rna'], task, 'drug')
        paths = {f: Path(args.run_dir) / '3.heter_data' / (task + '_' + f + '_data.pth')
                 for f in ('train', 'val', 'test')}
        original = {f: torch.load(p, map_location='cpu') for f, p in paths.items()}
        splits, removed = clean_splits(original, key)
        audit = dict(source_sha256={f: file_hash(p) for f, p in paths.items()},
                     removed_cross_fold_rows=removed,
                     cleaning='test > validation > training; training multiplicity retained',
                     node_shapes={nt: list(x.shape) for nt, x in splits['train'].x_dict.items()},
                     training_edges={'|'.join(k): int(e.size(1)) for k, e in
                                     splits['train'].edge_index_dict.items()}, folds={})
        for fold, data in splits.items():
            destination = root / 'inputs' / (task + '_' + fold + '.pth')
            destination.parent.mkdir(exist_ok=True)
            torch.save(data, destination)
            audit['folds'][fold] = dict(path=str(destination.resolve()), sha256=file_hash(destination),
                                       rows=data[key].edge_label.numel(),
                                       positives=int(data[key].edge_label.sum()),
                                       duplicate_rows=data[key].edge_label.numel() -
                                       len(pair_set(data[key].edge_label_index)))
        audits[task] = audit
        write_json(root / (task + '_audit.json'), audit)
    protocol = dict(
        configs=configs, tasks=list(configs), arms=list(ARMS),
        seeds=[666] if args.smoke else args.seeds, width=64, top_k=8,
        optimizer='RMSprop', alpha=0.99, loss='unweighted BCEWithLogitsLoss',
        selection='single checkpoint maximising mean validation AUC/AUPR; earliest exact tie',
        evaluation_graph='cleaned training graph for train, validation and test',
        initialisation='same materialised base per seed; zero-initialised residual output',
        channel_control='same parameter count and bottleneck; additive RMS-scaled MLP versus multiplicative channel gating',
        neighbour_control='full uniform, full softmax, fixed random-k softmax and learned top-k softmax; identical RNA-only edge support',
        layer_control='uniform versus node-dependent softmax over projected input, layer 1 and layer 2',
        residual_scale='detached per-node RMS of final base embedding; floor 1e-6',
        top_k_policy='8 fixed before evaluation, clipped to available degree; no k search',
        inference='exploratory fixed-split screen; seeds describe optimisation variation, not split generalisation',
        source_hashes=source_hashes(), inputs=audits, smoke=args.smoke,
        torch_version=torch.__version__, primary_arms=list(PRIMARY),
        contrasts=[list(c) for c in CONTRASTS],
    )
    protocol['expected_runs'] = len(protocol['tasks']) * len(protocol['seeds']) * len(ARMS)
    write_json(root / 'protocol.json', protocol)
    print('INITIALISED', root, 'expected runs', protocol['expected_runs'], flush=True)


def norm(x):
    return F.layer_norm(x, (x.size(-1),))


def rms(x):
    return x.detach().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)


def zero_output(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def capture_first_layer(encoder, node_types):
    """Expose FX intermediates without recomputing or changing any base layer."""
    graph = copy.deepcopy(encoder.graph)
    first = {}
    for nt in node_types:
        matches = [n for n in graph.nodes if n.op == 'call_function' and
                   n.target == F.leaky_relu and n.name == 'leaky_relu__' + nt]
        if len(matches) != 1:
            raise ValueError('Expected one first-layer activation for ' + nt)
        first[nt] = matches[0]
    output = next(n for n in graph.nodes if n.op == 'output')
    output.args = ((output.args[0], first),)
    graph.lint()
    encoder.graph = graph
    encoder.recompile()


def topk_mask(logits, support, k):
    if k < 1:
        raise ValueError('k must be positive')
    indices = logits.masked_fill(~support, -torch.inf).topk(min(k, logits.size(-1)), -1).indices
    selected = torch.zeros_like(support).scatter_(-1, indices, True)
    return selected & support


class ChannelBranch(nn.Module):
    def __init__(self, hidden, node_types, mode):
        super().__init__()
        self.mode = mode
        self.gates = nn.ModuleDict({nt: nn.Sequential(
            nn.Linear(hidden, max(8, hidden // 4)), nn.LeakyReLU(0.1),
            zero_output(nn.Linear(max(8, hidden // 4), hidden))) for nt in node_types})

    def forward(self, x, first, z):
        out, weights = dict(z), {}
        for nt, gate in self.gates.items():
            correction = torch.tanh(gate(norm(z[nt])))
            if self.mode == 'channel_attention':
                out[nt] = z[nt] * (1 + correction)
                weights['channel_' + nt] = 1 + correction
            else:
                out[nt] = z[nt] + rms(z[nt]) * correction
        return out, weights


class NeighbourBranch(nn.Module):
    def __init__(self, hidden, rna_type, relations, masks, mode, width, k, seed):
        super().__init__()
        self.rna_type, self.mode, self.k = rna_type, mode, k
        self.relations = relations
        self.query = nn.Linear(hidden, width, bias=False)
        self.keys = nn.ModuleList([nn.Linear(hidden, width, bias=False) for _ in masks])
        self.values = nn.ModuleList([nn.Linear(hidden, width, bias=False) for _ in masks])
        self.head = nn.Sequential(nn.Linear(len(masks) * width, width, bias=False),
                                  nn.LeakyReLU(0.1),
                                  zero_output(nn.Linear(width, hidden, bias=False)))
        generator = torch.Generator().manual_seed(seed + 70001)
        for i, mask in enumerate(masks):
            self.register_buffer('support_' + str(i), mask.clone())
            random_mask = topk_mask(torch.rand(mask.shape, generator=generator), mask, k)
            self.register_buffer('random_' + str(i), random_mask)
        if mode == 'neighbour_uniform':
            self.query.requires_grad_(False)
            self.keys.requires_grad_(False)

    def forward(self, x, first, z):
        normalised = {nt: norm(v) for nt, v in z.items()}
        query = self.query(normalised[self.rna_type])
        contexts, distributions = [], {}
        for i, relation in enumerate(self.relations):
            source = normalised[relation[0]]
            support = getattr(self, 'support_' + str(i))
            scores = query @ self.keys[i](source).t() / math.sqrt(query.size(-1))
            if self.mode == 'neighbour_uniform':
                weights = support.to(query.dtype) / support.sum(-1, keepdim=True).clamp_min(1)
            else:
                if self.mode == 'neighbour_topk':
                    support = topk_mask(scores, support, self.k)
                elif self.mode == 'neighbour_randomk':
                    support = getattr(self, 'random_' + str(i))
                weights = masked_weights(scores, support)
            contexts.append(weights @ self.values[i](source))
            distributions['neighbour_' + '|'.join(relation)] = weights
        out = dict(z)
        out[self.rna_type] = z[self.rna_type] + rms(z[self.rna_type]) * self.head(torch.cat(contexts, -1))
        return out, distributions


class LayerBranch(nn.Module):
    def __init__(self, hidden, input_dims, mode, width):
        super().__init__()
        self.mode = mode
        self.projections = nn.ModuleDict({nt: nn.ModuleList(
            [nn.Linear(d, width) for d in (dim, hidden, hidden)]) for nt, dim in input_dims.items()})
        self.scorers = nn.ModuleDict({nt: nn.Sequential(nn.Linear(width, 16), nn.Tanh(),
                                                       nn.Linear(16, 1, bias=False)) for nt in input_dims})
        self.outputs = nn.ModuleDict({nt: zero_output(nn.Linear(width, hidden, bias=False))
                                      for nt in input_dims})
        if mode == 'layer_uniform':
            self.scorers.requires_grad_(False)

    def forward(self, x, first, z):
        out, distributions = dict(z), {}
        for nt, projections in self.projections.items():
            views = torch.stack([projection(norm(value)) for projection, value in
                                 zip(projections, (x[nt], first[nt], z[nt]))], dim=1)
            if self.mode == 'layer_uniform':
                weights = views.new_full(views.shape[:2], 1 / 3)
            else:
                weights = self.scorers[nt](views).squeeze(-1).softmax(-1)
            pooled = (views * weights.unsqueeze(-1)).sum(1)
            out[nt] = z[nt] + rms(z[nt]) * self.outputs[nt](pooled)
            distributions['layer_' + nt] = weights
        return out, distributions


class MechanismModel(nn.Module):
    def __init__(self, base, data, config, arm, width=64, k=8, seed=0):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.base, self.arm, self.branch = base, arm, None
        self.node_types = (config['rna'], 'drug')
        if arm.startswith('channel_'):
            self.branch = ChannelBranch(config['hidden'], self.node_types, arm)
        elif arm.startswith('neighbour_'):
            relations, masks = rna_neighbour_masks(data, config['rna'])
            self.branch = NeighbourBranch(config['hidden'], config['rna'], relations, masks,
                                          arm, width, k, seed)
        elif arm.startswith('layer_'):
            capture_first_layer(base.encoder, data.node_types)
            dims = {nt: data[nt].x.size(1) for nt in self.node_types}
            self.branch = LayerBranch(config['hidden'], dims, arm, width)

    def encode(self, x, edges, details=False):
        encoded = self.base.encoder(x, edges)
        z, first = encoded if self.arm.startswith('layer_') else (encoded, None)
        out, weights = (z, {}) if self.branch is None else self.branch(x, first, z)
        return (out, z, weights) if details else out

    def forward(self, x, edges, pairs):
        return self.base.decoder(self.encode(x, edges), pairs)


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def score(model, train, pairs):
    model.eval()
    return model(train.x_dict, train.edge_index_dict, pairs).cpu().numpy()


def save_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(root, protocol, splits, task, arm, seed, device):
    directory = root / task / ('seed_' + str(seed)) / arm
    signature = dict(task=task, arm=arm, seed=seed, protocol_sha256=file_hash(root / 'protocol.json'))
    result_path = directory / 'result.json'
    if result_path.exists():
        record = json.loads(result_path.read_text())
        if record['signature'] != signature:
            raise ValueError('Incompatible completed run: ' + str(directory))
        print('SKIP', task, arm, seed, flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'config.json', signature)
    config = protocol['configs'][task]
    key = (config['rna'], task, 'drug')
    train = copy.deepcopy(splits['train']).to(device)
    base = make_base(train, config, seed, device)
    initial_hash = parameter_hash(base)
    pairs = {f: g[key].edge_label_index.to(device) for f, g in splits.items()}
    labels = {f: g[key].edge_label.cpu().numpy() for f, g in splits.items()}
    set_seed(seed + 1000003)
    model = MechanismModel(base, splits['train'], config, arm, protocol['width'],
                           protocol['top_k'], seed).to(device)
    with torch.no_grad():
        # CUDA scatter reductions can differ in the final few bits between
        # encoder calls. Compare decodes of the SAME encoding to test exactly
        # the invariant of interest: the added branch is initially identity.
        initial_out, initial_z, _ = model.encode(train.x_dict, train.edge_index_dict, details=True)
        expected = base.decoder(initial_z, pairs['train'][:, :64])
        actual = base.decoder(initial_out, pairs['train'][:, :64])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    optimizer = torch.optim.RMSprop((p for p in model.parameters() if p.requires_grad),
                                    lr=config['lr'], alpha=protocol['alpha'])
    best, best_state, trace = -float('inf'), None, []
    started = time.monotonic()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train.x_dict, train.edge_index_dict, pairs['train'])
        loss = F.binary_cross_entropy_with_logits(prediction, train[key].edge_label)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        optimizer.step()
        validation = score(model, train, pairs['val'])
        val = metrics(labels['val'], validation, pairs['val'].cpu().numpy(), by_drug=False)
        criterion = (val['auc'] + val['aupr']) / 2
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), val_auc=val['auc'],
                          val_aupr=val['aupr'], criterion=criterion))
        if criterion > best:
            best, best_epoch = criterion, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0 or epoch == config['epochs']:
            print(task, arm, 'seed', seed, 'epoch', epoch, 'val %.5f/%.5f' %
                  (val['auc'], val['aupr']), 'elapsed %.1fs' % (time.monotonic() - started), flush=True)
            save_csv(directory / 'trace.csv', trace)
    model.load_state_dict(best_state)
    predictions, results = {}, {}
    for fold in ('val', 'test'):
        predictions[fold] = score(model, train, pairs[fold])
        results[fold] = metrics(labels[fold], predictions[fold], pairs[fold].cpu().numpy())
    replay_delta = (results['val']['auc'] + results['val']['aupr']) / 2 - best
    if abs(replay_delta) > 5e-5:
        raise ValueError('Checkpoint does not reproduce selected validation score')
    with torch.no_grad():
        out, z, weights = model.encode(train.x_dict, train.edge_index_dict, details=True)
        diagnostics = dict(relative_embedding_change={nt: float(
            ((out[nt] - z[nt]).square().mean(-1).sqrt() / rms(z[nt]).squeeze(-1)).mean())
            for nt in model.node_types}, attention={})
        for name, w in weights.items():
            diagnostic = dict(mean=float(w.mean()), std=float(w.std()),
                              node_variation=float(w.std(0).mean()))
            if name.startswith('neighbour_'):
                count = (w > 0).sum(-1)
                diagnostic.update(mean_support=float(count.float().mean()),
                                  max_support=int(count.max()), empty_rows=int((count == 0).sum()))
            elif name.startswith('layer_'):
                diagnostic['mean_layer_weights'] = w.mean(0).cpu().tolist()
            diagnostics['attention'][name] = diagnostic
        if weights:
            np.savez_compressed(directory / 'attention.npz',
                                **{k: w.cpu().numpy() for k, w in weights.items()})
    np.savez_compressed(directory / 'predictions.npz', **predictions,
                        **{'y_' + f: labels[f] for f in predictions},
                        **{'pairs_' + f: pairs[f].cpu().numpy() for f in predictions})
    torch.save(best_state, directory / 'checkpoint.pt')
    record = dict(signature=signature, initial_base_sha256=initial_hash, metrics=results,
                  selected_epoch=best_epoch, trained_epochs=config['epochs'],
                  validation_replay_delta=replay_delta,
                  elapsed_seconds=time.monotonic() - started,
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  diagnostics=diagnostics, device=device)
    write_json(result_path, record)
    print('RESULT', task, arm, seed, json.dumps(results), flush=True)
    del model, optimizer, base, train, best_state
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def run(args):
    root = Path(args.out_dir)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['source_hashes'] != source_hashes():
        raise ValueError('Source changed since the plan was frozen')
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard')
    torch.set_num_threads(args.threads)
    inputs = {}
    for task, audit in protocol['inputs'].items():
        inputs[task] = {}
        for fold, entry in audit['folds'].items():
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('Frozen input changed')
            inputs[task][fold] = torch.load(entry['path'], map_location='cpu')
    jobs = list(itertools.product(protocol['tasks'], protocol['seeds'], protocol['arms']))
    for i, (task, seed, arm) in enumerate(jobs):
        if i % args.shard_count == args.shard_index:
            train_one(root, protocol, inputs[task], task, arm, seed, args.device)


def summarize(args):
    root = Path(args.out_dir)
    protocol = json.loads((root / 'protocol.json').read_text())
    records, rows = {}, []
    for task, seed, arm in itertools.product(protocol['tasks'], protocol['seeds'], protocol['arms']):
        path = root / task / ('seed_' + str(seed)) / arm / 'result.json'
        if not path.exists():
            raise ValueError('Incomplete experiment: ' + str(path))
        record = json.loads(path.read_text())
        if record['signature']['protocol_sha256'] != file_hash(root / 'protocol.json'):
            raise ValueError('Mixed protocols')
        records[task, seed, arm] = record
        row = dict(task=task, seed=seed, arm=arm, selected_epoch=record['selected_epoch'],
                   trained_epochs=record['trained_epochs'], trainable_parameters=record['trainable_parameters'])
        for fold, scores in record['metrics'].items():
            row.update({fold + '_' + k: v for k, v in scores.items()})
        rows.append(row)
    for task, seed in itertools.product(protocol['tasks'], protocol['seeds']):
        hashes = {records[task, seed, a]['initial_base_sha256'] for a in protocol['arms']}
        if len(hashes) != 1:
            raise ValueError('Arms did not start from the same base weights')
    aggregates, comparisons = [], []
    for task, arm in itertools.product(protocol['tasks'], protocol['arms']):
        group = [r for r in rows if r['task'] == task and r['arm'] == arm]
        agg = dict(task=task, arm=arm, n=len(group))
        for fold, metric in itertools.product(('val', 'test'), METRICS):
            field = fold + '_' + metric
            values = [r[field] for r in group if r[field] is not None]
            agg[field + '_mean'] = float(np.mean(values))
            agg[field + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        aggregates.append(agg)
    for task, (arm, reference) in itertools.product(protocol['tasks'], CONTRASTS):
        comparison = dict(task=task, arm=arm, reference=reference, n=len(protocol['seeds']))
        for fold, metric in itertools.product(('val', 'test'), METRICS):
            deltas = [records[task, s, arm]['metrics'][fold][metric] -
                      records[task, s, reference]['metrics'][fold][metric] for s in protocol['seeds']]
            field = fold + '_' + metric
            comparison[field + '_delta'] = float(np.mean(deltas))
            comparison[field + '_wins'] = sum(d > 0 for d in deltas)
        comparisons.append(comparison)
    save_csv(root / 'per_seed.csv', rows)
    save_csv(root / 'summary.csv', aggregates)
    save_csv(root / 'comparisons.csv', comparisons)
    write_json(root / 'comparisons.json', comparisons)
    report = ['# 注意力机制对照实验', '',
              f"完成 {len(rows)}/{protocol['expected_runs']} 次训练；每组 {len(protocol['seeds'])} 个种子。", '',
              '使用原始节点特征；所有评估均使用训练图。仅以验证集 AUC/AUPR 的均值选择单个 checkpoint。',
              'LncDrug 的训练/验证集重复正关联 (449, 10) 已从本轮训练标签及正反向图边移除；验证集和测试集保留。',
              'MiDrug 的两个训练集内部重复正例按原重数保留，各实验组一致。原始数据文件未改动。', '',
              '| Task | Arm | Val AUC | Val AUPR | Test AUC | Test AUPR | Per-drug test AUC |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        report.append('| %s | %s | %.5f | %.5f | %.5f | %.5f | %.5f |' %
                      (r['task'], r['arm'], r['val_auc_mean'], r['val_aupr_mean'],
                       r['test_auc_mean'], r['test_aupr_mean'], r['test_per_drug_auc_mean']))
    report.extend(['', '## 预设配对对照', '',
                   '| Task | Comparison | Δ test AUC | AUC wins | Δ test AUPR | AUPR wins |',
                   '|---|---|---:|---:|---:|---:|'])
    for c in comparisons:
        report.append('| %s | %s − %s | %+.5f | %d/%d | %+.5f | %d/%d |' %
                      (c['task'], c['arm'], c['reference'], c['test_auc_delta'],
                       c['test_auc_wins'], c['n'], c['test_aupr_delta'], c['test_aupr_wins'], c['n']))
    report.extend(['', '## 验证集筛选', ''])
    for task in protocol['tasks']:
        candidates = [r for r in aggregates if r['task'] == task and r['arm'] in ('base',) + PRIMARY]
        chosen = max(candidates, key=lambda r: (r['val_auc_mean'] + r['val_aupr_mean']) / 2)
        report.append('- %s：在原模型与三个预设注意力方案中，验证集均值选择 `%s`。' % (task, chosen['arm']))
    report.extend(['', '## 解释范围', '',
                   '- 这是固定划分上的探索实验；种子反映优化波动，不能替代独立数据划分。',
                   '- 通道注意力必须对照等参数残差 MLP；稀疏注意力必须对照完整 softmax 和相同 k 的随机筛选；层间注意力必须对照均匀融合。',
                   '- 均匀组冻结未使用的注意力打分参数，实际可训练参数数目记录于 per_seed.csv。',
                   '- 所有残差输出初始化为零；每个种子下所有组的初始 base 参数哈希一致。对同一次编码，加入模块前后的初始预测逐元素一致。',
                   '- GPU 图聚合的浮点归约顺序可有微小差异；checkpoint 验证分数重放差异记录在每次运行的 result.json 中。',
                   '- 注意力权重与嵌入变化幅度已保存，但权重本身不构成生物学机制证据。隐藏通道在不同种子之间不具备天然对齐。',
                   '- 原训练协议允许训练正例同时作为图边；本轮排除了留出目标边。历史不同图或不同选模规则的数值不作为配对基线。',
                   '- 完整标准差与逐种子结果见 summary.csv 和 per_seed.csv。', ''])
    (root / 'REPORT.md').write_text('\n'.join(report))
    print('\n'.join(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--out-dir', required=True)
    init.add_argument('--run-dir', default='/data1/licheng/incrd-hg/incrd-run')
    init.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    init.add_argument('--smoke', action='store_true')
    train = sub.add_parser('run')
    train.add_argument('--out-dir', required=True)
    train.add_argument('--device', default='cuda:0')
    train.add_argument('--threads', type=int, default=4)
    train.add_argument('--shard-index', type=int, default=0)
    train.add_argument('--shard-count', type=int, default=1)
    summary = sub.add_parser('summarize')
    summary.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if args.command == 'init':
        initialise(args)
    elif args.command == 'run':
        run(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
