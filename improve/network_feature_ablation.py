"""Run and report controlled network/feature experiments on frozen splits.

Example (one task per GPU; use a separate invocation for CPU classifiers):
  python -m improve.network_feature_ablation run --task LncDrug --device cuda:0 \
      --models gnn pair_gnn --out-dir improve/results/network_feature_ablation_20260906
  python -m improve.network_feature_ablation summarize --out-dir ...

GNN: the author's unmodified settings. Pair-GNN: the final additive pair head,
positive weight 2, training DropEdge .2, and 10 fixed evaluation masks. Both
metrics are always measured on ONE checkpoint, selected by mean validation
AUC/AP. Test labels never select a checkpoint, variant, or hyperparameter.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from torch_geometric.nn import GeneralConv, Linear, to_hetero

from improve.hybrid import PairDecoder
from improve.model import EdgeDecoder
from improve.network_feature_inputs import (FEATURE_VARIANTS, digest_array,
    extract_drugs, graph_weights, prepare_task, similarity_matrices,
    split_fingerprints, write_json)
from improve.run import TASKS, set_seed


VARIANTS = ('base', 'drug_unmasked', 'drug_masked', 'expr_pca64_stats',
            'expr_ae64', 'expr_stats3', 'edge_weight')
METRICS = ('val_auc', 'val_aupr', 'test_auc', 'test_aupr')


class WeightedGeneralConv(GeneralConv):
    """GeneralConv with scalar multiplicative weights; no new parameters.

Use the existing edge_attr propagation channel for a scalar message weight.
The parent message receives no edge features, so its learned transformation
is exactly the released GeneralConv transformation. Unit weights recover it.
"""
    def message(self, x_i, x_j, edge_index_i, size_i, edge_attr):
        message = super().message(x_i, x_j, edge_index_i, size_i, None)
        if edge_attr is None:
            return message
        return message * edge_attr.reshape(-1, 1, 1)


class WeightedEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            WeightedGeneralConv((-1, -1), hidden),
            WeightedGeneralConv((-1, -1), hidden)])

    def forward(self, x, edge_index, edge_weight):
        h = self.convs[0](x, edge_index, edge_attr=edge_weight)
        h = F.leaky_relu(h, negative_slope=0.1)
        return self.convs[1](h, edge_index, edge_attr=edge_weight)


class ExperimentModel(torch.nn.Module):
    def __init__(self, data, task, pair_dim=None, mlp=False):
        super().__init__()
        config = TASKS[task]
        hidden = config['channels']
        self.encoder = to_hetero(WeightedEncoder(hidden), data.metadata(),
                                 aggr=config['aggr'], input_map={
                                     'x': 'node', 'edge_index': 'edge', 'edge_weight': 'edge'})
        self.pair = pair_dim is not None
        self.mlp = mlp
        if mlp:
            # A trainable per-node-type expression encoder: the 938-dim CCLE
            # profile (or whichever ncRNA variant) is projected to `hidden`
            # with a learned weight matrix before any message passing, so the
            # model learns which expression/cell-line dimensions matter instead
            # of reading the raw vector.  Lazy Linear resolves on the first
            # forward, so it adapts to the feature variant's input dim.
            self.proj = torch.nn.ModuleDict({
                nt: Linear(-1, hidden)
                for nt in ('lncRNA', 'miRNA') if nt in data.node_types})
        if self.pair:
            self.decoder = PairDecoder(hidden, config['rna'], pair_dim, mode='additive')
        else:
            self.decoder = EdgeDecoder(hidden, config['rna'])

    def forward(self, x, edges, idx, weights, phi=None):
        if self.mlp:
            x = {nt: (F.leaky_relu(self.proj[nt](x[nt]), negative_slope=0.1)
                      if nt in self.proj else x[nt]) for nt in x}
        z = self.encoder(x, edges, weights)
        return self.decoder(z, idx, phi) if self.pair else self.decoder(z, idx)


def masked_graph(edges, weights, rate, seed):
    """A dedicated RNG gives every arm identical masks and cannot alter training RNG."""
    if rate <= 0:
        return edges, weights
    device = next(iter(edges.values())).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    out_edges, out_weights = {}, {}
    for key, edge in edges.items():
        keep = torch.rand(edge.size(1), device=device, generator=generator) >= rate
        out_edges[key] = edge[:, keep]
        out_weights[key] = weights[key][keep]
    return out_edges, out_weights


@torch.no_grad()
def predict(model, data, key, weights, phi, seed, mc=1, rate=0):
    model.eval()
    predictions = []
    for i in range(mc):
        edges, w = masked_graph(data.edge_index_dict, weights, rate, seed + i * 997)
        predictions.append(model(data.x_dict, edges, data[key].edge_label_index,
                                 w, phi).cpu().numpy())
    return np.mean(predictions, axis=0)


def score(y, prediction):
    assert np.isfinite(prediction).all()
    return float(roc_auc_score(y, prediction)), float(average_precision_score(y, prediction))


def load_inputs(args, task, variant):
    feature_variant = 'base' if variant == 'edge_weight' else variant
    root = Path(args.inputs_dir) if args.inputs_dir else Path(args.out_dir) / 'inputs'
    path = root / (task + '_' + feature_variant + '.npz')
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def run_gnn(args, task, model_name, variant, seed, arrays):
    config = TASKS[task]
    key = (config['rna'], task, 'drug')
    pair = model_name == 'pair_gnn'
    set_seed(seed)
    splits = [torch.load(Path(args.run_dir) / '3.heter_data' / (task + '_' + f + '_data.pth'),
                         map_location='cpu') for f in ('train', 'val', 'test')]
    original_fingerprints = split_fingerprints(splits, key)
    inputs_root = Path(args.inputs_dir) if args.inputs_dir else Path(args.out_dir) / 'inputs'
    audit = json.loads((inputs_root / (task + '_audit.json')).read_text())
    assert original_fingerprints == audit['split_fingerprints'], 'Input graph changed since preparation'
    for fold, data in zip(('train', 'val', 'test'), splits):
        np.testing.assert_array_equal(data[key].edge_label.numpy(), arrays['y_' + fold])
        np.testing.assert_array_equal(data[key].edge_label_index.numpy(), arrays['pairs_' + fold])
        for nt in data.node_types:
            data[nt].x = torch.from_numpy(arrays['x_' + nt])
    assert split_fingerprints(splits, key) == original_fingerprints
    splits = [d.to(args.device) for d in splits]
    train, val, test = splits
    matrices = similarity_matrices(args.net_dir)
    weights = [graph_weights(d, matrices, variant == 'edge_weight') for d in splits]
    phi = {f: torch.from_numpy(arrays['phi_' + f]).to(args.device)
           for f in ('train', 'val', 'test')}
    model = ExperimentModel(train, task, phi['train'].shape[1] if pair else None,
                            mlp=(model_name == 'expr_mlp')).to(args.device)
    with torch.no_grad():
        model(train.x_dict, train.edge_index_dict, train[key].edge_label_index,
              weights[0], phi['train'])  # materialize lazy parameters before creating optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'], alpha=0.99)
    epochs = args.epochs or config['epochs'] - 1
    rate, mc = (0.2, 10) if pair else (0.0, 1)
    target = train[key].edge_label
    loss_weights = torch.where(target > 0, 2.0, 1.0) if pair else None
    best_criterion, best_state, best_epoch = -np.inf, None, None
    trace = []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        edges, w = masked_graph(train.edge_index_dict, weights[0], rate,
                                seed * 100003 + epoch)
        pred = model(train.x_dict, edges, train[key].edge_label_index, w, phi['train'])
        loss = F.binary_cross_entropy_with_logits(pred, target, weight=loss_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError('Training loss is not finite')
        loss.backward()
        optimizer.step()
        # The same validation masks are reused across epochs and feature arms.
        pv = predict(model, val, key, weights[1], phi['val'],
                     seed=seed * 100003 + 9000001, mc=mc, rate=rate)
        auc, ap = score(arrays['y_val'], pv)
        criterion = (auc + ap) / 2
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), val_auc=auc,
                          val_aupr=ap, criterion=criterion))
        if criterion > best_criterion:
            best_criterion, best_epoch = criterion, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(task, model_name, variant, seed, 'epoch', epoch,
                  'val %.4f/%.4f' % (auc, ap),
                  'elapsed %.1fs' % (time.monotonic() - start), flush=True)
    model.load_state_dict(best_state)
    pv = predict(model, val, key, weights[1], phi['val'],
                 seed=seed * 100003 + 9000001, mc=mc, rate=rate)
    pt = predict(model, test, key, weights[2], phi['test'],
                 seed=seed * 100003 + 19000001, mc=mc, rate=rate)
    va, vp = score(arrays['y_val'], pv)
    ta, tp = score(arrays['y_test'], pt)
    # The restored checkpoint reproduces the best validation criterion.  CUDA
    # reduction nondeterminism makes two forward passes with identical weights
    # differ by ~1e-4 in score, so the check is loose; a genuine restore bug
    # (wrong state dict) would move scores by orders more.
    assert abs((va + vp) / 2 - best_criterion) < 2e-3, 'Checkpoint restore mismatch'
    result = dict(task=task, model=model_name, variant=variant, seed=seed,
                  val_auc=va, val_aupr=vp, test_auc=ta, test_aupr=tp,
                  selected_epoch=best_epoch, trained_epochs=epochs,
                  selection='mean_validation_auc_aupr', mc=mc,
                  params=sum(p.numel() for p in model.parameters()),
                  elapsed_seconds=time.monotonic() - start,
                  node_dims={nt: int(x.shape[1]) for nt, x in train.x_dict.items()},
                  pair_dim=int(phi['train'].shape[1]))
    return result, pv, pt, trace, best_state if seed == 666 else None


def run_tabular(args, task, model_name, variant, seed, arrays):
    X = {f: arrays['phi_' + f] for f in ('train', 'val', 'test')}
    y = {f: arrays['y_' + f] for f in ('train', 'val', 'test')}
    start = time.monotonic()
    trace = []
    if model_name == 'extratrees':
        model = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                      n_jobs=args.cpu_threads, random_state=seed)
        model.fit(X['train'], y['train'])
        best_epoch, epochs = 0, 0
        selection = 'fixed_hyperparameters_no_validation_selection'
    else:
        model = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2,
                              random_state=seed, early_stopping=False,
                              max_iter=1, learning_rate_init=0.001)
        best_criterion, best_model, best_epoch = -np.inf, None, 0
        limit = args.epochs or 300
        for epoch in range(1, limit + 1):
            model.partial_fit(X['train'], y['train'], classes=np.asarray([0, 1]))
            pv = model.predict_proba(X['val'])[:, 1]
            auc, ap = score(y['val'], pv)
            criterion = (auc + ap) / 2
            trace.append(dict(epoch=epoch, loss=float(model.loss_), val_auc=auc,
                              val_aupr=ap, criterion=criterion))
            if criterion > best_criterion:
                best_criterion, best_epoch = criterion, epoch
                best_model = copy.deepcopy(model)
            if epoch - best_epoch >= 40:
                break
        epochs, model = epoch, best_model
        selection = 'mean_validation_auc_aupr_patience40'
    pv, pt = [model.predict_proba(X[f])[:, 1] for f in ('val', 'test')]
    va, vp = score(y['val'], pv)
    ta, tp = score(y['test'], pt)
    result = dict(task=task, model=model_name, variant=variant, seed=seed,
                  val_auc=va, val_aupr=vp, test_auc=ta, test_aupr=tp,
                  selected_epoch=best_epoch, trained_epochs=epochs,
                  selection=selection, elapsed_seconds=time.monotonic() - start,
                  pair_dim=X['train'].shape[1])
    return result, pv, pt, trace, None


def run(args):
    torch.set_num_threads(args.cpu_threads)
    out = Path(args.out_dir)
    for sub in ('runs', 'scores', 'traces', 'checkpoints'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        for variant in args.variants:
            if variant == 'edge_weight' and model_name not in ('gnn', 'pair_gnn', 'expr_mlp'):
                continue  # the table feature matrix already uses continuous similarities
            arrays = load_inputs(args, args.task, variant)
            for seed in args.seeds:
                tag = '%s_%s_%s_s%d' % (args.task, model_name, variant, seed)
                result_path = out / 'runs' / (tag + '.json')
                if result_path.exists():
                    print('already complete', tag, flush=True)
                    continue
                print('START', tag, flush=True)
                runner = run_gnn if model_name in ('gnn', 'pair_gnn', 'expr_mlp') else run_tabular
                result, pv, pt, trace, checkpoint = runner(
                    args, args.task, model_name, variant, seed, arrays)
                result['source_sha256'] = {
                    p.name: __import__('hashlib').sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), Path(__file__).with_name('network_feature_inputs.py')]}
                np.savez_compressed(out / 'scores' / (tag + '.npz'), val=pv, test=pt,
                                    y_val=arrays['y_val'], y_test=arrays['y_test'],
                                    pairs_val=arrays['pairs_val'], pairs_test=arrays['pairs_test'])
                if trace:
                    pd.DataFrame(trace).to_csv(out / 'traces' / (tag + '.csv'), index=False)
                if checkpoint is not None:
                    torch.save(checkpoint, out / 'checkpoints' / (tag + '.pt'))
                write_json(result_path, result)  # completion marker is written last
                print('DONE', tag, 'val %.4f/%.4f test %.4f/%.4f' % (
                    result['val_auc'], result['val_aupr'],
                    result['test_auc'], result['test_aupr']), flush=True)


def summarize(args):
    out = Path(args.out_dir)
    rows = [json.loads(p.read_text()) for p in sorted((out / 'runs').glob('*.json'))]
    if not rows:
        raise ValueError('No completed runs')
    frame = pd.DataFrame(rows)
    keys = ['task', 'model', 'variant']
    summary, deltas = [], []
    for group, data in frame.groupby(keys, sort=False):
        row = dict(zip(keys, group))
        row['n_seeds'] = len(data)
        row['seeds'] = ','.join(map(str, sorted(data.seed.tolist())))
        for metric in METRICS:
            row[metric + '_mean'] = data[metric].mean()
            row[metric + '_std'] = data[metric].std(ddof=1) if len(data) > 1 else 0
        summary.append(row)
        reference = 'drug_unmasked' if group[2] == 'drug_masked' else 'base'
        if group[2] == reference:
            continue
        base = frame[(frame.task == group[0]) & (frame.model == group[1]) &
                     (frame.variant == reference)]
        paired = data.merge(base, on='seed', suffixes=('', '_reference'))
        for metric in METRICS:
            values = paired[metric] - paired[metric + '_reference']
            if not len(values):
                continue
            deltas.append(dict(task=group[0], model=group[1], variant=group[2],
                               reference=reference, metric=metric, n_seeds=len(values),
                               delta_mean=values.mean(),
                               delta_std=values.std(ddof=1) if len(values) > 1 else 0,
                               seeds_improved=int((values > 0).sum())))
    summary, deltas = pd.DataFrame(summary), pd.DataFrame(deltas)
    frame.drop(columns=['source_sha256', 'node_dims'], errors='ignore').to_csv(out / 'per_seed.csv', index=False)
    summary.to_csv(out / 'summary.csv', index=False)
    deltas.to_csv(out / 'paired_deltas.csv', index=False)
    notes = [
        ('protocol', 'Existing frozen train/val/test pairs, negatives and graph edges retained, including circRNA.'),
        ('selection', 'One GNN/MLP checkpoint maximizes mean validation AUC/AP; both test metrics come from it.'),
        ('test_graph', 'Released test message graph contains validation positives; training graph is used for all pair features.'),
        ('pooling', 'drug_masked must be compared against drug_unmasked; both use the same cached ChemBERTa-77M model.'),
        ('checkpoint', 'Original 768-dimensional ChemBERTa checkpoint was unavailable; cached checkpoint has 384 dimensions.'),
        ('expression', 'Train-node StandardScaler + PCA64 + standardized RNA mean/std/zero fraction; 67 node dimensions.'),
        ('edge_weight', 'Only existing LncLnc/MiMi/DrugDrug messages are weighted; total incoming weight equals original edge count.'),
        ('pair_features', '5-fold construction removes each training row fold positives before all graph-derived features.'),
        ('uncertainty', 'Seed SD describes initialization variability on one fixed split; it is not uncertainty across datasets or splits.'),
        ('models', 'gnn=author settings; pair_gnn=pair branch+positive weight 2+DropEdge .2+MC10; table models use pair features alone.'),
    ]
    workbook = out / '网络与特征消融实验.xlsx'
    with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='均值与标准差', index=False)
        deltas.to_excel(writer, sheet_name='配对增益', index=False)
        frame.drop(columns=['source_sha256', 'node_dims'], errors='ignore').to_excel(
            writer, sheet_name='逐seed结果', index=False)
        pd.DataFrame(notes, columns=['项目', '说明']).to_excel(writer, sheet_name='实验口径', index=False)
        for sheet in writer.book:
            sheet.freeze_panes = 'D2'
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                width = min(75, max(13, max(len(str(cell.value or '')) for cell in column) + 2))
                sheet.column_dimensions[column[0].column_letter].width = width
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = '0.0000'
    print(summary.to_string(index=False))
    print('wrote', workbook)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--run-dir', default='/data1/licheng/incrd-hg/incrd-run')
    parser.add_argument('--net-dir', default='/data1/licheng/incrd-hg/incrd-run/2.construct_network')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--inputs-dir')
    parser.add_argument('--model-dir')
    parser.add_argument('--task', choices=list(TASKS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--models', nargs='+', choices=['gnn', 'pair_gnn', 'expr_mlp',
                                                         'extratrees', 'mlp'],
                        default=['gnn', 'pair_gnn'])
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[666, 0, 1, 2, 3])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.model_dir:
            extract_drugs(args.net_dir, args.model_dir, args.out_dir, args.device)
        for task in [args.task] if args.task else TASKS:
            prepare_task(args.run_dir, args.net_dir, args.out_dir, task)
    elif args.command == 'run':
        if not args.task:
            parser.error('--task is required for run')
        run(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
