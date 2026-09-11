"""Controlled drug-conditioned RNA-neighbour attention on frozen link splits.

The released two-layer GeneralConv encoder and decoder are retained. Three
additive controls use the SAME RNA neighbours, projections, degree features and
MLP: uniform pooling, RNA-only queries, and RNA+candidate-drug queries. A scalar
gate starts at zero, so every variant initially reproduces the base model.

Only RNA->RNA relations enter the added branch. Target RNA-drug edges are never
attention-mask inputs. All evaluation uses the TRAINING graph, including test
evaluation; validation/test positive edges are checked for overlap with it.
The released training labels also occur as message-passing edges, as in the
author's transductive training protocol. This experiment retains that property.

Example (run from the repository root):
  python -m improve.drug_condition_attention run --task MiDrug --device cuda:0 \
    --seeds 666 0 1 2 3 --out-dir improve/results/drug_condition_attention_20260907
  python -m improve.drug_condition_attention summarize --out-dir ...
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

from improve.model import Model


ARMS = ('base', 'uniform', 'node_attention', 'drug_attention')
CONFIG = {
    'MiDrug': dict(rna='miRNA', hidden=256, aggr='mean', lr=1e-4, epochs=300),
    'LncDrug': dict(rna='lncRNA', hidden=64, aggr='min', lr=5e-4, epochs=100),
}
METRICS = ('auc', 'aupr', 'per_drug_auc', 'per_drug_aupr')


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def pair_set(indices):
    return set(map(tuple, indices.detach().cpu().t().tolist()))


def load_data(run_dir, task):
    key = (CONFIG[task]['rna'], task, 'drug')
    root = Path(run_dir) / '3.heter_data'
    paths = {fold: root / (task + '_' + fold + '_data.pth')
             for fold in ('train', 'val', 'test')}
    splits = {fold: torch.load(path, map_location='cpu')
              for fold, path in paths.items()}
    train = splits['train']
    positive_sets = {}
    for fold, data in splits.items():
        for nt, x in train.x_dict.items():
            assert torch.equal(x, data[nt].x), 'Node features changed between splits'
            assert torch.isfinite(x).all(), 'Nonfinite node features'
        idx, y = data[key].edge_label_index, data[key].edge_label
        assert set(y.tolist()) == {0.0, 1.0}
        # The current release repeats two training positives. Preserve their
        # original multiplicity identically in every arm; held-out rows must be
        # unique, and cross-fold overlap is rejected below.
        if fold != 'train':
            assert len(pair_set(idx)) == idx.shape[1], 'Duplicate held-out pairs'
        positive_sets[fold] = pair_set(idx[:, y == 1])
    all_positives = set().union(*positive_sets.values())
    graph_pairs = pair_set(train[key].edge_index)
    assert graph_pairs == positive_sets['train'], 'Unexpected training graph labels'
    reverse = ('drug', 'rev_' + task, key[0])
    assert pair_set(train[reverse].edge_index[[1, 0]]) == graph_pairs
    for fold, data in splits.items():
        idx, y = data[key].edge_label_index, data[key].edge_label
        assert not (pair_set(idx[:, y == 0]) & all_positives), 'Negative is a known positive'
        if fold != 'train':
            assert not (positive_sets[fold] & graph_pairs), 'Held-out edge in training graph'
    assert not (positive_sets['val'] & positive_sets['test'])
    # Check labels do not occur in more than one fold, including negatives.
    labelled = {fold: pair_set(data[key].edge_label_index)
                for fold, data in splits.items()}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not (labelled[a] & labelled[b]), 'Labelled pairs overlap between folds'
    audit = {
        'task': task,
        'node_shapes': {nt: list(x.shape) for nt, x in train.x_dict.items()},
        'training_edges': {'|'.join(k): v.size(1) for k, v in train.edge_index_dict.items()},
        'folds': {fold: {'rows': int(data[key].edge_label.numel()),
                         'positives': int(data[key].edge_label.sum()),
                         'duplicate_rows_preserved': int(data[key].edge_label.numel()) -
                         len(pair_set(data[key].edge_label_index))}
                  for fold, data in splits.items()},
        'split_sha256': {fold: file_hash(path) for fold, path in paths.items()},
        'evaluation_graph': 'training graph for both validation and test',
        'training_protocol': 'released training positives also serve as message-passing edges',
        'checks': 'identical finite features; disjoint labels; no held-out positive in training graph',
    }
    return splits, key, audit


def rna_neighbour_masks(data, rna_type):
    """Return destination-by-source masks from existing RNA->RNA graph edges."""
    relations, masks = [], []
    for relation, edges in data.edge_index_dict.items():
        source, _, destination = relation
        if destination != rna_type or source == 'drug':
            continue
        mask = torch.zeros(data[rna_type].num_nodes, data[source].num_nodes,
                           dtype=torch.bool)
        edge = edges.cpu()
        mask[edge[1], edge[0]] = True
        if source == destination:
            mask.fill_diagonal_(False)
        if mask.any():
            relations.append(relation)
            masks.append(mask)
    if not masks:
        raise ValueError('No RNA neighbours in this graph')
    return relations, masks


def masked_weights(logits, mask):
    """A distribution over allowed neighbours; empty rows produce exact zeros."""
    masked = logits.masked_fill(~mask, -1e4)
    weights = masked.softmax(dim=-1) * mask.to(logits.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class NeighbourBranch(nn.Module):
    def __init__(self, hidden, rna_type, relations, masks, mode, width=64):
        super().__init__()
        if mode not in ARMS[1:]:
            raise ValueError(mode)
        self.mode, self.rna_type = mode, rna_type
        self.relations, self.width = list(relations), width
        self.query_r = nn.Linear(hidden, width, bias=False)
        self.query_d = nn.Linear(hidden, width, bias=False)
        self.keys = nn.ModuleList([nn.Linear(hidden, width, bias=False)
                                   for _ in relations])
        self.values = nn.ModuleList([nn.Linear(hidden, width, bias=False)
                                     for _ in relations])
        for i, mask in enumerate(masks):
            self.register_buffer('mask_' + str(i), mask.clone())
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + len(relations) * (width + 1), 128),
            nn.LeakyReLU(0.1), nn.Linear(128, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, 1))
        self.gate = nn.Parameter(torch.zeros(()))

    def prepare(self, z):
        # Normalisation affects only the new branch, leaving the base unchanged.
        norm = {nt: F.layer_norm(x, (x.shape[1],)) for nt, x in z.items()}
        keys = [layer(norm[rel[0]]) for rel, layer in zip(self.relations, self.keys)]
        values = [layer(norm[rel[0]]) for rel, layer in zip(self.relations, self.values)]
        return norm, keys, values

    def pool(self, prepared, pairs, return_weights=False):
        norm, keys, values = prepared
        r, d = pairs
        zr, zd = norm[self.rna_type][r], norm['drug'][d]
        query = self.query_r(zr)
        if self.mode == 'drug_attention':
            query = query + self.query_d(zd)
        pooled, counts, distributions = [], [], []
        for i, (key, value) in enumerate(zip(keys, values)):
            mask = getattr(self, 'mask_' + str(i))[r]
            if self.mode == 'uniform':
                weights = mask.to(zr.dtype)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
            else:
                logits = query @ key.t() / math.sqrt(self.width)
                weights = masked_weights(logits, mask)
            pooled.append(weights @ value)
            counts.append(mask.sum(-1, keepdim=True).to(zr.dtype).log1p())
            if return_weights:
                distributions.append(weights)
        parts = torch.cat([zr, zd] + pooled + counts, dim=-1)
        return (parts, distributions) if return_weights else parts

    def forward(self, prepared, pairs):
        return self.gate.tanh() * self.head(self.pool(prepared, pairs)).view(-1)


class AttentionModel(nn.Module):
    def __init__(self, base, relations, masks, mode, hidden, rna_type, width=64):
        super().__init__()
        self.base = base
        self.branch = None if mode == 'base' else NeighbourBranch(
            hidden, rna_type, relations, masks, mode, width)

    def prepare(self, x, edges):
        z = self.base.encoder(x, edges)
        return z, None if self.branch is None else self.branch.prepare(z)

    def decode(self, prepared, pairs):
        z, extra = prepared
        score = self.base.decoder(z, pairs)
        return score if self.branch is None else score + self.branch(extra, pairs)


def metrics(y, scores, pairs, by_drug=True):
    result = {'auc': float(roc_auc_score(y, scores)),
              'aupr': float(average_precision_score(y, scores))}
    if not by_drug:
        return result
    per_auc, per_ap = [], []
    for drug in np.unique(pairs[1]):
        take = pairs[1] == drug
        if np.unique(y[take]).size != 2:
            continue
        per_auc.append(roc_auc_score(y[take], scores[take]))
        per_ap.append(average_precision_score(y[take], scores[take]))
    result.update(per_drug_auc=float(np.mean(per_auc)) if per_auc else None,
                  per_drug_aupr=float(np.mean(per_ap)) if per_ap else None,
                  evaluable_drugs=len(per_auc))
    return result


@torch.no_grad()
def predict(model, x, edges, pairs, batch_size):
    model.eval()
    prepared = model.prepare(x, edges)
    return torch.cat([model.decode(prepared, pairs[:, start:start + batch_size]).cpu()
                      for start in range(0, pairs.shape[1], batch_size)]).numpy()


def make_base(data, config, seed, device):
    set_seed(seed)
    model = Model(data, config['hidden'], config['aggr'], config['rna']).to(device)
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict,
              data[config['rna'], 'MiDrug' if config['rna'] == 'miRNA' else 'LncDrug',
                   'drug'].edge_label_index[:, :2])
    return model


def run_one(args, splits, key, audit, arm, seed):
    config = dict(CONFIG[args.task])
    config['epochs'] = args.epochs or config['epochs']
    if args.lr is not None:
        config['lr'] = args.lr
    out = Path(args.out_dir) / args.task / ('seed_' + str(seed)) / arm
    signature = {'arm': arm, 'seed': seed, 'task': args.task,
                 'config': config, 'patience': args.patience, 'width': args.width,
                 'batch_size': args.batch_size, 'grad_clip': args.grad_clip,
                 'split_sha256': audit['split_sha256'],
                 'code_sha256': file_hash(__file__),
                 'base_code_sha256': file_hash(Path(__file__).with_name('model.py')),
                 'selection': 'mean validation AUC/AUPR; earliest exact tie'}
    if (out / 'result.json').exists():
        existing = json.loads((out / 'result.json').read_text())
        if existing['signature'] != signature:
            raise ValueError('Existing output has a different experiment signature: ' + str(out))
        print('SKIP', args.task, arm, seed, flush=True)
        return existing
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', signature)
    train = copy.deepcopy(splits['train']).to(args.device)
    x, edges = train.x_dict, train.edge_index_dict
    relations, masks = rna_neighbour_masks(splits['train'], key[0])
    base = make_base(train, config, seed, args.device)
    # All arms start with exactly the same materialised base parameters.
    set_seed(seed + 1000003)
    model = AttentionModel(base, relations, masks, arm, config['hidden'],
                           key[0], args.width).to(args.device)
    pairs = {fold: data[key].edge_label_index.to(args.device)
             for fold, data in splits.items()}
    labels = {fold: data[key].edge_label.cpu().numpy()
              for fold, data in splits.items()}
    target = train[key].edge_label
    with torch.no_grad():
        prepared = model.prepare(x, edges)
        expected = base.decoder(prepared[0], pairs['train'][:, :32])
        actual = model.decode(prepared, pairs['train'][:, :32])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'], alpha=0.99)
    best, best_epoch, trace = -float('inf'), 0, []
    started = time.monotonic()
    best_state = None
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prepared = model.prepare(x, edges)
        loss_total = 0.0
        n = pairs['train'].shape[1]
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            pred = model.decode(prepared, pairs['train'][:, start:end])
            loss = F.binary_cross_entropy_with_logits(pred, target[start:end], reduction='sum') / n
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward(retain_graph=end < n)
            loss_total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip else None
        optimizer.step()
        validation = predict(model, x, edges, pairs['val'], args.batch_size)
        va = metrics(labels['val'], validation, pairs['val'].cpu().numpy(), by_drug=False)
        criterion = (va['auc'] + va['aupr']) / 2
        row = dict(epoch=epoch, loss=loss_total, criterion=criterion,
                   **{'val_' + k: v for k, v in va.items()})
        row['gate'] = 0.0 if model.branch is None else float(model.branch.gate.tanh().detach())
        trace.append(row)
        if criterion > best:
            best, best_epoch = criterion, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % args.log_every == 0 or epoch == config['epochs']:
            print(args.task, arm, 'seed', seed, 'epoch', epoch,
                  'loss %.5f val %.5f/%.5f gate %.4f elapsed %.1fs' %
                  (loss_total, va['auc'], va['aupr'], row['gate'], time.monotonic() - started),
                  flush=True)
            with (out / 'trace.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
                writer.writeheader()
                writer.writerows(trace)
        if args.patience and epoch - best_epoch >= args.patience:
            print('EARLY_STOP', arm, seed, epoch, 'selected', best_epoch, flush=True)
            break
    model.load_state_dict(best_state)
    results, predictions = {}, {}
    for fold in ('val', 'test'):
        predictions[fold] = predict(model, x, edges, pairs[fold], args.batch_size)
        results[fold] = metrics(labels[fold], predictions[fold], pairs[fold].cpu().numpy())
    assert abs((results['val']['auc'] + results['val']['aupr']) / 2 - best) < 5e-5
    query_check = {}
    if model.branch is not None:
        with torch.no_grad():
            _, extra = model.prepare(x, edges)
            check_pairs = pairs['val'][:, :128].clone()
            changed = check_pairs.clone()
            changed[1] = (changed[1] + 1) % train['drug'].num_nodes
            _, first = model.branch.pool(extra, check_pairs, return_weights=True)
            _, second = model.branch.pool(extra, changed, return_weights=True)
            query_check = {'|'.join(rel): float((a - b).abs().sum(-1).mean())
                           for rel, a, b in zip(relations, first, second)}
        if arm != 'drug_attention':
            assert max(query_check.values()) == 0.0
    record = dict(signature=signature, metrics=results, selected_epoch=best_epoch,
                  trained_epochs=epoch, elapsed_seconds=time.monotonic() - started,
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  branch_gate=0.0 if model.branch is None else float(model.branch.gate.tanh().detach()),
                  attention_query_change_l1=query_check,
                  attention_relations=[list(rel) for rel in relations])
    with (out / 'trace.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    torch.save(best_state, out / 'checkpoint.pt')
    np.savez_compressed(out / 'predictions.npz',
                        val=predictions['val'], test=predictions['test'],
                        y_val=labels['val'], y_test=labels['test'],
                        pairs_val=pairs['val'].cpu().numpy(), pairs_test=pairs['test'].cpu().numpy())
    write_json(out / 'result.json', record)
    print('RESULT', json.dumps({'arm': arm, 'seed': seed, 'selected_epoch': best_epoch,
                               'metrics': results}), flush=True)
    del model, optimizer, base, train
    if str(args.device).startswith('cuda'):
        torch.cuda.empty_cache()
    return record


def bootstrap_deltas(paths, reference_paths, fold, replicates=1000):
    """Paired edge bootstrap of fixed, equally weighted seed-mean predictions."""
    predictions, references = [], []
    y, pairs = None, None
    for path, ref in zip(paths, reference_paths):
        with np.load(path) as a, np.load(ref) as b:
            np.testing.assert_array_equal(a['y_' + fold], b['y_' + fold])
            np.testing.assert_array_equal(a['pairs_' + fold], b['pairs_' + fold])
            if y is not None:
                np.testing.assert_array_equal(y, a['y_' + fold])
                np.testing.assert_array_equal(pairs, a['pairs_' + fold])
            y, pairs = a['y_' + fold], a['pairs_' + fold]
            predictions.append(a[fold])
            references.append(b[fold])
    prediction = np.mean(predictions, axis=0)
    reference = np.mean(references, axis=0)
    rng = np.random.RandomState(1701)
    delta = []
    for _ in range(replicates):
        take = rng.randint(0, len(y), len(y))
        if np.unique(y[take]).size < 2:
            continue
        delta.append([roc_auc_score(y[take], prediction[take]) - roc_auc_score(y[take], reference[take]),
                      average_precision_score(y[take], prediction[take]) -
                      average_precision_score(y[take], reference[take])])
    out = {}
    for j, (name, scorer) in enumerate((('auc', roc_auc_score), ('aupr', average_precision_score))):
        lo, hi = np.quantile(np.asarray(delta)[:, j], [0.025, 0.975])
        out[name] = {'delta': float(scorer(y, prediction) - scorer(y, reference)),
                     'ci95': [float(lo), float(hi)]}
    out['limitation'] = 'Fixed-split edge bootstrap; shared-node dependence and split variation are not modelled.'
    return out


def summarize(args):
    root = Path(args.out_dir)
    entries = []
    for path in sorted(root.glob('*/seed_*/*/result.json')):
        record = json.loads(path.read_text())
        signature = record['signature']
        entries.append((path, record, signature))
    if not entries:
        raise ValueError('No completed runs in ' + str(root))
    rows, aggregates, comparisons = [], [], []
    for path, record, signature in entries:
        row = dict(task=signature['task'], arm=signature['arm'], seed=signature['seed'],
                   selected_epoch=record['selected_epoch'], trained_epochs=record['trained_epochs'],
                   branch_gate=record['branch_gate'])
        for fold, values in record['metrics'].items():
            row.update({fold + '_' + k: v for k, v in values.items()})
        rows.append(row)
    for task in sorted(set(r['task'] for r in rows)):
        for arm in ARMS:
            group = [r for r in rows if r['task'] == task and r['arm'] == arm]
            if not group:
                continue
            agg = dict(task=task, arm=arm, n=len(group), seeds=[r['seed'] for r in group])
            for fold in ('val', 'test'):
                for metric in METRICS:
                    name = fold + '_' + metric
                    numbers = [r[name] for r in group if r[name] is not None]
                    agg[name + '_mean'] = float(np.mean(numbers))
                    agg[name + '_std'] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else 0.0
            aggregates.append(agg)
        for arm, reference in (('uniform', 'base'), ('node_attention', 'uniform'),
                               ('drug_attention', 'node_attention'), ('drug_attention', 'base')):
            candidates = {s['seed']: p for p, _, s in entries if s['task'] == task and s['arm'] == arm}
            controls = {s['seed']: p for p, _, s in entries if s['task'] == task and s['arm'] == reference}
            common = sorted(set(candidates) & set(controls))
            if not common:
                continue
            compare = dict(task=task, arm=arm, reference=reference, paired_seeds=common)
            for fold in ('val', 'test'):
                differences = {}
                for metric in METRICS:
                    name = fold + '_' + metric
                    a = {r['seed']: r[name] for r in rows if r['task'] == task and r['arm'] == arm}
                    b = {r['seed']: r[name] for r in rows if r['task'] == task and r['arm'] == reference}
                    ds = [a[s] - b[s] for s in common if a[s] is not None and b[s] is not None]
                    differences[metric] = {'mean': float(np.mean(ds)),
                                           'wins': int(sum(d > 0 for d in ds)), 'deltas': ds}
                compare[fold + '_paired_deltas'] = differences
            if args.bootstrap:
                compare['test_seed_mean_bootstrap'] = bootstrap_deltas(
                    [candidates[s].with_name('predictions.npz') for s in common],
                    [controls[s].with_name('predictions.npz') for s in common],
                    'test', args.bootstrap)
            comparisons.append(compare)
    for filename, data in [('per_seed.csv', rows), ('summary.csv', aggregates)]:
        with (root / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    write_json(root / 'comparisons.json', comparisons)
    report = [
        '# Drug-conditioned RNA-neighbour attention', '',
        'Original node features and labelled splits; all arms evaluate on the training graph.',
        'One checkpoint per run selected by mean validation AUC/AUPR. Test scores do not select a checkpoint.',
        'The additive branch uses only RNA-to-RNA edges; the released encoder and decoder remain trainable.', '',
        '| Task | Arm | Seeds | Val AUC | Val AUPR | Test AUC | Test AUPR | Per-drug test AUC |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in aggregates:
        report.append('| %s | %s | %d | %.5f | %.5f | %.5f | %.5f | %.5f |' %
                      (row['task'], row['arm'], row['n'], row['val_auc_mean'], row['val_aupr_mean'],
                       row['test_auc_mean'], row['test_aupr_mean'], row['test_per_drug_auc_mean']))
    report.extend(['', '## Paired controls', ''])
    for cmp in comparisons:
        ds = cmp['test_paired_deltas']
        report.append('- %s, %s minus %s: test AUC %+.5f (%d/%d seeds), AUPR %+.5f (%d/%d seeds).' %
                      (cmp['task'], cmp['arm'], cmp['reference'], ds['auc']['mean'], ds['auc']['wins'],
                       len(cmp['paired_seeds']), ds['aupr']['mean'], ds['aupr']['wins'], len(cmp['paired_seeds'])))
        if 'test_seed_mean_bootstrap' in cmp:
            bs = cmp['test_seed_mean_bootstrap']
            report.append('  Fixed seed-mean prediction edge-bootstrap 95%% CI: AUC [%+.5f, %+.5f]; AUPR [%+.5f, %+.5f].' %
                          tuple(bs['auc']['ci95'] + bs['aupr']['ci95']))
    report.extend(['', '## Interpretation limits', '',
                   '- This is a fixed-split experiment. Random seeds quantify optimisation variability, not new datasets.',
                   '- Uniform pooling controls for the extra branch. Node-only attention controls for candidate-drug conditioning.',
                   '- Seed-mean bootstrap intervals describe a fixed seed ensemble and assume independent edges, despite shared nodes.',
                   '- Per-drug metrics macro-average only drugs with both positive and negative examples.',
                   '- The original training protocol includes training positives in its encoder graph; no held-out positives enter that graph.',
                   '- The current MiDrug input repeats two training positives; their multiplicity is preserved in all arms and recorded in the audit.',
                   '- Historical results on earlier graphs or test graphs containing validation edges are not matched baselines here.', ''])
    (root / 'REPORT.md').write_text('\n'.join(report))
    print('\n'.join(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--run-dir', default='/data1/licheng/incrd-hg/incrd-run')
    run.add_argument('--task', choices=list(CONFIG), default='MiDrug')
    run.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    run.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    run.add_argument('--epochs', type=int, default=None)
    run.add_argument('--patience', type=int, default=60)
    run.add_argument('--lr', type=float, default=None)
    run.add_argument('--width', type=int, default=64)
    run.add_argument('--batch-size', type=int, default=2048)
    run.add_argument('--grad-clip', type=float, default=0.0)
    run.add_argument('--threads', type=int, default=4)
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--log-every', type=int, default=20)
    run.add_argument('--out-dir', required=True)
    summary = commands.add_parser('summarize')
    summary.add_argument('--out-dir', required=True)
    summary.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.command == 'summarize':
        summarize(args)
        return
    if args.batch_size <= 0 or args.width <= 0 or args.patience < 0 or (args.epochs is not None and args.epochs < 1):
        raise ValueError('Invalid training argument')
    torch.set_num_threads(args.threads)
    splits, key, audit = load_data(args.run_dir, args.task)
    audit['versions'] = {'torch': torch.__version__}
    audit_path = Path(args.out_dir) / (args.task + '_audit.json')
    if audit_path.exists():
        if json.loads(audit_path.read_text()) != audit:
            raise ValueError('Input audit changed for this output directory')
    else:
        write_json(audit_path, audit)
    print('AUDIT', json.dumps(audit), flush=True)
    for seed in args.seeds:
        for arm in args.arms:
            run_one(args, splits, key, audit, arm, seed)


if __name__ == '__main__':
    main()
