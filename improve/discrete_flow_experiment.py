"""Frozen comparison of binary/continuous association flow completion.

The primary comparator is the accepted evidence-tiered TabICL, not a weaker GNN.
OOF calibration and fixed compositions are evaluation plumbing, not additional
innovations. Only development labels are loaded by this module.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from improve import discrete_flow_models as method
from improve import graph_signal_calibrated as calibration
from improve import non_tabular_experiment as base


OUT = base.ROOT / 'results' / 'discrete_flow_20260915'
KINDS = ('discrete', 'continuous', 'homogeneous')
RESEARCH = base.ROOT / 'research' / 'structured_frontier_20260915'


def logit(probability):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def prepare(out, cases):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory')
    names = ('discrete_flow_experiment.py', 'discrete_flow_models.py',
             'test_discrete_flow.py', 'graph_signal_calibrated.py',
             'non_tabular_experiment.py')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
             tasks=list(base.TASKS), cases=list(cases), kinds=list(KINDS),
             seed=20260915, folds=5, epochs=400, width=128, time_dim=32,
             n_steps=10, sampling_steps=2, lr=.001, weight_decay=0.,
             primary='0.9 accepted TabICL logit + 0.1 training-calibrated '
                     'discrete flow logit, with the same rule for both tasks.',
             secondary='Flow alone, one-step readout, and fixed 0.5 GNN / '
                       '0.5 flow logit composition; no coefficient search.',
             architecture='RNA rows, drug columns; two width-128 tanh hidden '
                          'layers, 0.1 dropout, sinusoidal time embedding. '
                          'Condition on the existing 64 attribute/gene-kernel '
                          'RNA coordinates without pair-feature tables.',
             training='400 final-epoch full-batch Adam steps per OOF model. '
                      'MSE on implicit-feedback reconstruction entries. '
                      'OOF query pairs and validation pairs incur no loss. '
                      'The same query outcomes are absent before prior '
                      'estimation and interpolation. No early stopping.',
             unknown_semantics='Unlisted entries are implicit reconstruction '
                               'zeros, not experimental biological negatives.',
             controls='Continuous interpolation/Euler and homogeneous Bernoulli '
                      'prior matched to global density; identical attributes, '
                      'network, RNG seed, losses and budget. The one-step '
                      'readout reuses each model and is not another fit.',
             calibration='Scalar regularized Platt fit on five OOF training '
                         'predictions only; average calibrated validation '
                         'logits from those same five models.',
             gate='MiDrug mean global AUC and AP must exceed accepted TabICL '
                  'and both matched control compositions; each metric must '
                  'improve in at least two of the three historical reference '
                  'seeds (or all cases in a resplit confirmation). LncDrug '
                  'mean and each comparison may lose at most 0.001 absolute '
                  'AUC and AP. GNN-only wins cannot establish this goal.',
             literature='Liu et al. Flow Matching for Collaborative Filtering. '
                        'SIGIR 2025. DOI 10.1145/3711896.3736967; '
                        'arXiv:2502.07303v2. Independent adaptation, not an '
                        'official dataset reproduction.',
             novelty_count='One hypothesis; interpolation controls, calibration, '
                           'mixture and inference step variants never count '
                           'as separate innovations.',
             validation_boundary='Reused/overlapping development data; no '
                                 'independent biological validation claim.',
             original_test_access=False, historical_outer_access=False,
             source_sha256={n: base.digest(base.ROOT / n) for n in names},
             input_sha256={})
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(base.ROOT / name, out / 'source_snapshot' / name)
    for name in ('flowcf.html', 'flowcf_crossref.json',
                 'author_flowcf_model_flowcf.py', 'author_source_manifest.json'):
        source = RESEARCH / name
        p['input_sha256'][str(source)] = base.digest(source)
    for task in p['tasks']:
        source = base.OUT / 'inputs' / task / 'arrays.npz'
        node = load(source)
        p['input_sha256'][str(source)] = base.digest(source)
        for case in cases:
            paths = []
            if case == 'inner':
                a = {k: node[k] for k in ('pairs_train', 'y_train', 'pairs_val', 'y_val')}
                refs = {}
                for seed in (666, 0, 1):
                    path = base.REF / f'predictions/inner/{task}/s{seed}/scores.npz'
                    z = load(path)
                    refs[f'tabicl_{seed}'] = (z['tiered_0p5000'], z['y_val'], z['pairs_val'])
                    paths.append(path)
                    path = base.GRAPH / ('screen' if seed == 666 else 'confirm') / task / f'base/c0/s{seed}/scores.npz'
                    z = load(path)
                    refs[f'gnn_{seed}'] = (z['val'], z['y_val'], z['pairs_val'])
                    paths.append(path)
            else:
                path = calibration.RESPLITS / 'inputs' / task / case / 'arrays.npz'
                z = load(path)
                a = {k: z[k] for k in ('pairs_train', 'y_train', 'pairs_val', 'y_val')}
                paths.append(path)
                refs = {}
                for key, arm in (('tabicl_20260914', 'tiered'), ('gnn_20260914', 'gnn')):
                    path = calibration.RESPLITS / 'runs' / task / case / arm / 'scores.npz'
                    z = load(path)
                    refs[key] = (z['pred'], z['y'], z['pairs'])
                    paths.append(path)
            if set(map(tuple, a['pairs_train'].T)) & set(map(tuple, a['pairs_val'].T)):
                raise ValueError('Training and evaluation pairs overlap')
            for key, (value, y, pairs) in refs.items():
                np.testing.assert_array_equal(y, a['y_val'])
                np.testing.assert_array_equal(pairs, a['pairs_val'])
                if key.startswith('tabicl') and not np.all((value >= 0) & (value <= 1)):
                    raise ValueError('Expected probabilities from the composed TabICL')
                a[key] = value
            a['attributes'] = node['ur']
            a['shape'] = np.array([len(node['ur']), len(node['ud'])])
            a['folds'] = calibration.make_folds(a['pairs_train'], a['y_train'], p['seed'], p['folds'])
            dest = out / 'inputs' / task / case / 'arrays.npz'
            dest.parent.mkdir(parents=True)
            np.savez_compressed(dest, **a)
            paths.append(dest)
            p['input_sha256'].update({str(path): base.digest(path) for path in paths})
    base.write_json(out / 'protocol.json', p)
    print('FROZEN', out, flush=True)


def protocol(out):
    p = json.loads((Path(out) / 'protocol.json').read_text())
    for name, expected in p['source_sha256'].items():
        if base.digest(base.ROOT / name) != expected:
            raise ValueError('Source changed: ' + name)
    for name, expected in p['input_sha256'].items():
        if base.digest(name) != expected:
            raise ValueError('Input changed: ' + name)
    return p


def fit(a, p, kind, fold, device):
    seed = p['seed'] + fold * 101
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    graph, mask = method.support_and_mask(a['pairs_train'], a['y_train'], a['folds'],
                                        fold, tuple(a['shape']), a['pairs_val'])
    x = torch.as_tensor(graph, device=device)
    eligible = torch.as_tensor(mask, device=device)
    attrs = torch.as_tensor(a['attributes'], device=device)
    model = method.BinaryFlowModel(x.shape[1], attrs.shape[1], p['width'], p['time_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    trace = []
    start = time.perf_counter()
    for epoch in range(1, p['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        times = torch.randint(p['n_steps'], (len(x), 1), device=device).float() / p['n_steps']
        prior_random, mask_random = torch.rand_like(x), torch.rand_like(x)
        xt = method.interpolated_state(x, prior_random, mask_random, times, kind)
        prediction = model(xt, attrs, times[:, 0])
        loss = (prediction - x).square()[eligible].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite flow objective')
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            trace.append(dict(epoch=epoch, mse=float(loss), seconds=time.perf_counter() - start))
    model.eval()
    return model, x, attrs, trace, time.perf_counter() - start


def verify(dest, protocol_hash):
    row = json.loads((dest / 'result.json').read_text())
    if row['protocol_sha256'] != protocol_hash:
        raise ValueError('Mismatched run protocol')
    for name, expected in row['files'].items():
        if base.digest(dest / name) != expected:
            raise ValueError('Artifact changed: ' + str(dest / name))
    return row


def run(out, task, device, kinds=None):
    out = Path(out)
    p = protocol(out)
    protocol_hash = base.digest(out / 'protocol.json')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for case in p['cases']:
        a = load(out / 'inputs' / task / case / 'arrays.npz')
        for kind in (kinds or p['kinds']):
            if kind not in p['kinds']:
                raise ValueError('Unknown flow kind')
            root = out / 'runs' / task / case / kind
            oof = {key: np.full(len(a['y_train']), np.nan) for key in ('flow', 'one_step')}
            val = {key: [] for key in oof}
            for fold in range(p['folds']):
                dest = root / str(fold)
                ix = np.flatnonzero(a['folds'] == fold)
                if not (dest / 'result.json').exists():
                    model, x, attrs, trace, seconds = fit(a, p, kind, fold, device)
                    values = {}
                    for key, steps in (('flow', p['sampling_steps']), ('one_step', 1)):
                        score = method.predict(model, x, attrs, kind, p['n_steps'], steps)
                        tq, vq = a['pairs_train'][:, ix], a['pairs_val']
                        values[key + '_oof'] = score[tq[0], tq[1]]
                        values[key + '_val'] = score[vq[0], vq[1]]
                    dest.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), dest / 'model.pt')
                    replay_model = method.BinaryFlowModel(x.shape[1], attrs.shape[1], p['width'], p['time_dim']).to(device)
                    replay_model.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True), strict=True)
                    order = np.arange(len(x) - 1, -1, -1)
                    replay = method.predict(replay_model, x, attrs, kind, p['n_steps'],
                                            p['sampling_steps'], rows=order, batch=37)[::-1]
                    vq = a['pairs_val']
                    replay_values = replay[vq[0], vq[1]]
                    np.testing.assert_allclose(replay_values, values['flow_val'], atol=2e-5, rtol=2e-5)
                    np.savez_compressed(dest / 'scores.npz', oof_indices=ix, **values)
                    base.write_json(dest / 'trace.json', trace)
                    row = dict(task=task, case=case, kind=kind, fold=fold, seconds=seconds,
                               protocol_sha256=protocol_hash,
                               replay_max_abs=float(np.max(np.abs(replay_values - values['flow_val']))),
                               files={n: base.digest(dest / n) for n in ('model.pt', 'scores.npz', 'trace.json')})
                    base.write_json(dest / 'result.json', row)
                    print('FIT', task, case, kind, fold, round(seconds, 2), flush=True)
                    del model, replay_model, x, attrs
                verify(dest, protocol_hash)
                z = load(dest / 'scores.npz')
                np.testing.assert_array_equal(ix, z['oof_indices'])
                for key in oof:
                    oof[key][ix] = z[key + '_oof']
                    val[key].append(z[key + '_val'])
            scores = dict(y=a['y_val'], pairs=a['pairs_val'])
            fitted = {}
            for key in oof:
                fitted[key] = calibration.fit_calibration(oof[key], a['y_train'])
                scores[key] = calibration.apply_calibration(np.stack(val[key]), fitted[key]).mean(0)
                scores[key + '_oof'] = oof[key]
                for name in a:
                    if name.startswith('tabicl_'):
                        scores[name] = logit(a[name])
                        scores[key + '_' + name] = .9 * scores[name] + .1 * scores[key]
                    elif name.startswith('gnn_'):
                        scores[name] = a[name]
                        scores[key + '_' + name] = .5 * a[name] + .5 * scores[key]
            np.savez_compressed(root / 'scores.npz', **scores)
            base.write_json(root / 'calibration.json', fitted)
            base.write_json(root / 'result.json', dict(protocol_sha256=protocol_hash,
                files={n: base.digest(root / n) for n in ('scores.npz', 'calibration.json')}))
            print('DONE', task, case, kind, base.metrics(scores['y'], scores['flow'], scores['pairs']), flush=True)


def report(out):
    out = Path(out)
    p = protocol(out)
    ph = base.digest(out / 'protocol.json')
    rows, contrasts, replay = [], [], []
    for task in p['tasks']:
        for case in p['cases']:
            for kind in p['kinds']:
                root = out / 'runs' / task / case / kind
                verify(root, ph)
                for fold in range(p['folds']):
                    replay.append(verify(root / str(fold), ph)['replay_max_abs'])
                z = load(root / 'scores.npz')
                for key, prediction in z.items():
                    if key in ('y', 'pairs') or key.endswith('_oof'):
                        continue
                    rows.append(dict(task=task, case=case, kind=kind, score=key,
                                     **base.metrics(z['y'], prediction, z['pairs'])))
    table = pd.DataFrame(rows)
    for task in p['tasks']:
        for case in p['cases']:
            part = table[(table.task == task) & (table.case == case)]
            for kind in p['kinds']:
                for reference in ('tabicl', 'gnn'):
                    original = part[(part.kind == kind) & part.score.str.startswith(reference + '_')]
                    candidate = part[(part.kind == kind) & part.score.str.startswith('flow_' + reference + '_')]
                    delta = {metric: candidate.sort_values('score')[metric].to_numpy() -
                             original.sort_values('score')[metric].to_numpy() for metric in ('auc', 'ap')}
                    row = dict(task=task, case=case, kind=kind, reference=reference)
                    for metric in delta:
                        row[metric] = float(candidate[metric].mean())
                        row['delta_' + metric] = float(delta[metric].mean())
                        row['minimum_' + metric] = float(delta[metric].min())
                        row['positive_' + metric] = int((delta[metric] > 0).sum())
                        row['comparisons'] = len(delta[metric])
                    contrasts.append(row)
    contrast = pd.DataFrame(contrasts)
    chosen = contrast[(contrast.kind == 'discrete') & (contrast.reference == 'tabicl')]
    performance = True
    for _, row in chosen.iterrows():
        for metric in ('auc', 'ap'):
            if row.task == 'MiDrug':
                performance &= row['delta_' + metric] > 0 and row['positive_' + metric] >= min(2, row.comparisons)
            else:
                performance &= row['minimum_' + metric] >= -.001 and row['delta_' + metric] >= -.001
    mechanism = {}
    for control in ('continuous', 'homogeneous'):
        mechanism[control] = {}
        control_rows = contrast[(contrast.task == 'MiDrug') & (contrast.kind == control) & (contrast.reference == 'tabicl')]
        for metric in ('auc', 'ap'):
            mechanism[control][metric] = float(chosen[chosen.task == 'MiDrug'][metric].mean() - control_rows[metric].mean())
    mechanism_pass = all(v > 0 for row in mechanism.values() for v in row.values())
    table.to_csv(out / 'metrics.csv', index=False)
    contrast.to_csv(out / 'contrasts.csv', index=False)
    assessment = dict(strong_reference_performance=bool(performance),
                      mechanism_supported=bool(mechanism_pass),
                      shared_gate=bool(performance and mechanism_pass),
                      mechanism_deltas=mechanism, fits=len(replay),
                      replay_max_abs=max(replay), independent_biological_validation=False,
                      newly_confirmed_innovations=0,
                      limitation='A passing inner screen still requires frozen resplit '
                                 'confirmation. This is one flow hypothesis, not '
                                 'three independent techniques.')
    base.write_json(out / 'ASSESSMENT.json', assessment)
    base.write_json(out / 'COMPLETION.json', dict(protocol_sha256=ph,
        source_input_and_output_hashes_verified=True, metrics_recomputed=True,
        neural_fits=len(replay), original_test_access=False,
        accepted_default_replaced=False))
    print(contrast.to_string(index=False))
    print(json.dumps(assessment, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('--out-dir', default=str(OUT))
    parser.add_argument('--cases', nargs='+', default=['inner'], choices=['inner', '42', '2026', '31415'])
    parser.add_argument('--task', choices=base.TASKS)
    parser.add_argument('--kind', nargs='+', choices=KINDS)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out_dir, args.cases)
    elif args.command == 'run':
        run(args.out_dir, args.task, args.device, args.kind)
    else:
        report(args.out_dir)


if __name__ == '__main__':
    main()
