"""Observation-likelihood revisions of the cavity and masked-graph families.

The unchanged graph cores learn interactions around the same support-only
Rasch offsets. These are revisions of two existing mechanism families, not
additional innovations attributed to a classical bias model.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import cavity_positive_experiment as cavity
from improve import masked_teacher_experiment as teacher
from improve import residual_graph_experiment as residual

common, precision = residual.common, residual.precision
ROOT = residual.ROOT
LIBRARIES = dict(cavity=cavity, masked_teacher=teacher)
OWN = ('offset_graph_candidates.py', 'test_offset_graph_candidates.py',
       'offset_candidate_validate.py', 'graph_observation_offset.py',
       'residual_graph_experiment.py', 'graph_prediction_precision.py')


class OffsetCandidate(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        for head in (core.rna_degree, core.drug_degree):
            for parameter in head.parameters():
                nn.init.zeros_(parameter)
                parameter.requires_grad_(False)

    def forward(self, xs, packed, pairs):
        graph, offsets, intercept = packed
        value = self.core(xs, graph, pairs)
        r, d = pairs
        return value.double() + offsets[r] + offsets[d] + intercept

    def auxiliary(self, xs, packed, generator):
        return self.core.auxiliary(xs, packed[0], generator)

    def update_teacher(self):
        if hasattr(self.core, 'update_teacher'):
            self.core.update_teacher()


def setup(a, p, kind, device):
    core, xs, graphs, folds = LIBRARIES[p['family']].setup(a, p['core_parameters'], kind, device)
    packed = [(g, torch.as_tensor(a['offset_nodes'][i], device=device),
               torch.as_tensor(a['offset_intercepts'][i], device=device)) for i, g in enumerate(graphs)]
    return OffsetCandidate(core), xs, packed, folds


def prepare(out, family):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new offset candidate directory')
    parent_dir = LIBRARIES[family].OUT
    parent = common.protocol(parent_dir)
    residual.prepare_inputs()
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), method='residual_' + family,
        family=family, smoke=False, tasks=parent['tasks'], kinds=parent['kinds'],
        screen_seed=666, confirm_seeds=[0, 1], epochs=400, check_interval=20,
        lrs=[.001, .003], weight_decay=.001, strong_weight=.1, gnn_weight=.5,
        auxiliary_weight=parent.get('auxiliary_weight', 0.),
        private_rng_offset=17057 if family == 'masked_teacher' else 29015,
        lnc_absolute_tolerance=.001, primary_kind=parent['primary_kind'],
        primary_controls=parent['primary_controls'], core_parameters=parent,
        parent=str(parent_dir), selection=parent['selection'],
        gate='MiDrug mean AUC/AP increase, each in >=2/3 paired confirmations; every LncDrug metric loss <=0.001. The declared primary must beat both its primary mechanism controls and the offset-only null on mean MiDrug AUC/AP. Other arms are component diagnostics.',
        success_scope='Revised mechanism after predecessor failures. Fixed-partition and reserved evaluation remain required; no claim that a predecessor passed.',
        likelihood='Replace trainable marginal degree heads with the same support-fold-only row/column logistic MAP score. The offset is a classical nuisance baseline, not an innovation.',
        boundary='All labels of each fitting query fold are excluded before offset fitting; queried positive facts are excluded before graph construction. Zero observations are not biological negatives.',
        matched_controls=parent['controls'] if 'controls' in parent else parent.get('primary_comparison', ''),
        previous_candidate_results_seen=True, original_test_access=False,
        original_outer_query_label_access=False, existing_default_replaced=False,
        confirmed_innovations=0,
        source_sha256={**parent['source_sha256'], **{n: common.base.digest(ROOT / n) for n in OWN}},
        inputs={str(parent_dir / 'protocol.json'): common.base.digest(parent_dir / 'protocol.json'),
                str(residual.INPUTS / 'manifest.json'): common.base.digest(residual.INPUTS / 'manifest.json')})
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, out / 'source_snapshot' / name)
    for task in p['tasks']:
        source = residual.INPUTS / task / 'arrays.npz'
        dest = out / 'inputs' / task / 'arrays.npz'
        dest.parent.mkdir(parents=True)
        shutil.copy2(source, dest)
        for path in (source, dest):
            p['inputs'][str(path)] = common.base.digest(path)
    common.base.write_json(out / 'protocol.json', p)
    print('FROZEN_OFFSET_CANDIDATE', family, out, flush=True)


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if fixed_epochs is not None and set(a) - residual.TRAIN_KEYS:
        raise ValueError('Fixed offset candidate fitting accepts training inputs only')
    if kind not in p['kinds']:
        raise ValueError('Unknown registered mechanism')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, xs, graphs, folds = setup(a, p, kind, device)
    pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(a['y_train'], device=device)
    query = None if fixed_epochs is not None else torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one fitting step is required')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
    generator = torch.Generator(device=device).manual_seed(seed + p['private_rng_offset'])
    best, selected, state, trace = -np.inf, epochs, None, []
    start = time.monotonic()
    for step in range(1, epochs + 1):
        fold = (step - 1) % len(graphs)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = model(xs, graphs[fold], pairs[:, folds[fold]])
        loss = F.binary_cross_entropy_with_logits(values, labels[folds[fold]])
        if p['auxiliary_weight']:
            auxiliary, _ = model.auxiliary(xs, graphs[fold], generator)
            loss = loss + p['auxiliary_weight'] * auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite offset candidate loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        model.update_teacher()
        if step % p['check_interval'] == 0 or step == epochs:
            row = dict(epoch=step, loss=float(loss.detach()))
            if query is not None:
                prediction = precision.geometry_prediction(model, xs, graphs, query).cpu().numpy()
                composed = .9 * a['anchor_' + str(seed)] + .1 * prediction
                metrics = common.base.metrics(a['y_val'], composed, a['pairs_val'])
                row['anchor_composed'] = metrics
                criterion = (metrics['auc'] + metrics['ap']) / 2
                if criterion > best:
                    best, selected = criterion, step
                    state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
                if step % 100 == 0:
                    print('TRAIN', kind, lr, seed, step, round(metrics['auc'], 9), round(metrics['ap'], 9),
                          round(time.monotonic() - start, 1), flush=True)
            trace.append(row)
    if state is not None:
        model.load_state_dict(state)
    prediction = None if query is None else precision.geometry_prediction(model, xs, graphs, query).cpu().numpy()
    return model, prediction, trace, dict(epoch=selected, seconds=time.monotonic() - start,
        parameters=sum(v.numel() for v in model.parameters()),
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None, query_outcomes_used_for_offset_fitting=False)


def run(out, task, device, stage):
    out = Path(out)
    p = common.protocol(out)
    a = common.load(out / 'inputs' / task / 'arrays.npz')
    ph = common.base.digest(out / 'protocol.json')
    selection = None if stage == 'screen' else json.loads((out / 'SELECTION.json').read_text())
    if selection is not None and selection['protocol_sha256'] != ph:
        raise ValueError('Selection protocol mismatch')
    seeds = [p['screen_seed']] if stage == 'screen' else p['confirm_seeds']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for kind in p['kinds']:
        for lr in p['lrs'] if stage == 'screen' else [selection['chosen'][kind]['lr']]:
            for seed in seeds:
                dest = out / 'runs' / task / kind / ('lr' + str(lr)) / ('s' + str(seed))
                if (dest / 'result.json').exists():
                    common.verify(dest, ph)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                model, prediction, trace, detail = fit(a, p, kind, lr, seed, device)
                torch.save(model.state_dict(), dest / 'model.pt')
                common.base.write_json(dest / 'trace.json', trace)
                loaded, xs, graphs, _ = setup(a, p, kind, device)
                loaded.load_state_dict(torch.load(dest / 'model.pt', weights_only=True, map_location=device))
                query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
                replay = precision.geometry_prediction(loaded, xs, graphs, query.flip(1)).cpu().numpy()[::-1]
                chunks = np.concatenate([precision.geometry_prediction(loaded, xs, graphs, query[:, i:i + 800]).cpu().numpy()
                                         for i in range(0, query.shape[1], 800)])
                for values in (replay, chunks):
                    np.testing.assert_allclose(prediction, values, atol=3e-5, rtol=3e-5)
                scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=prediction, offset=a['offset_val'])
                for name in ('gnn', 'tabicl', 'anchor'):
                    weight = p['gnn_weight'] if name == 'gnn' else p['strong_weight']
                    scores[name] = a[name + '_' + str(seed)]
                    scores[name + '_composed'] = (1 - weight) * scores[name] + weight * prediction
                scores['offset_composed'] = .9 * scores['anchor'] + .1 * scores['offset']
                np.savez_compressed(dest / 'scores.npz', **scores)
                r = dict(task=task, kind=kind, lr=lr, seed=seed, **detail, protocol_sha256=ph,
                    replay_max_abs=float(max(abs(prediction - replay).max(), abs(prediction - chunks).max())),
                    metrics={n: common.base.metrics(scores['y'], v, scores['pairs']) for n, v in scores.items()
                             if n not in ('y', 'pairs', 'offset', 'offset_composed')},
                    offset_metrics={n: common.base.metrics(scores['y'], scores[n], scores['pairs'])
                                    for n in ('offset', 'offset_composed')},
                    files={n: common.base.digest(dest / n) for n in ('model.pt', 'trace.json', 'scores.npz')})
                common.base.write_json(dest / 'result.json', r)
                print('VERIFIED', p['method'], task, kind, lr, seed, detail['epoch'], r['metrics']['anchor_composed'], flush=True)
                del model, loaded, xs, graphs
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'screen', 'select', 'confirm', 'report'))
    parser.add_argument('--family', choices=tuple(LIBRARIES), required=True)
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--out')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    out = args.out or str(ROOT / 'results' / ('residual_' + args.family + '_20260915'))
    if args.command == 'prepare':
        prepare(out, args.family)
    elif args.command in ('screen', 'confirm'):
        run(out, args.task, args.device, args.command)
    elif args.command == 'select':
        common.select(out)
    else:
        residual.report(out)
