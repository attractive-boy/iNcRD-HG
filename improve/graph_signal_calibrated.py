"""Training-only cross-fitted calibration for a graph-signal diffusion expert.

This follow-up tests one diffusion hypothesis, not a new technique for every
calibration, orientation or mixture. Five support-masked models produce OOF
training predictions. A scalar Platt map sees those training labels only.
The same five models predict evaluation pairs, with a fixed GNN composition.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch

from improve import non_tabular_experiment as base
from improve import graph_signal_diffusion as methods
from improve import graph_signal_experiment as original

OUT = base.ROOT / 'results' / 'graph_signal_calibrated_20260914'
RESPLITS = base.ROOT / 'results' / 'transfer_splits_20260914'
KINDS = ('reverse', 'one_step')


def make_folds(pairs, y, seed, count=5):
    pairs = np.asarray(pairs)
    y = np.asarray(y)
    if pairs.shape != (2, len(y)):
        raise ValueError('Pair/label shape mismatch')
    if len(set(map(tuple, pairs.T))) != len(y):
        raise ValueError('Duplicate pairs would invalidate edge-level cross-fitting')
    folds = np.empty(len(y), dtype=np.int64)
    for f, (_, q) in enumerate(StratifiedKFold(count, shuffle=True, random_state=seed).split(pairs.T, y)):
        folds[q] = f
    return folds


def support_matrix(pairs, y, folds, fold, shape, evaluation_pairs=None):
    """Remove the entire OOF query fold before all graph/spectrum operations."""
    graph = np.zeros(shape, dtype=np.float32)
    q = pairs[:, (folds != fold) & (y == 1)]
    graph[q[0], q[1]] = 1.
    excluded = pairs[:, folds == fold]
    if graph[excluded[0], excluded[1]].any():
        raise ValueError('Cross-fitting query was included in support')
    if evaluation_pairs is not None and graph[evaluation_pairs[0], evaluation_pairs[1]].any():
        raise ValueError('Evaluation pair was included in support')
    return graph


def fit_calibration(prediction, training_labels):
    """Only OOF training scores and training outcomes enter this function."""
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(training_labels)
    if x.shape != y.shape or not np.isfinite(x).all():
        raise ValueError('Invalid OOF calibration arrays')
    center = float(x.mean())
    scale = float(max(x.std(), 1e-12))
    model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, tol=1e-10)
    model.fit(((x-center)/scale)[:, None], y)
    return dict(center=center, scale=scale, coefficient=float(model.coef_[0, 0]),
                intercept=float(model.intercept_[0]), rows=len(y),
                positives=int(y.sum()), fitted_on='OOF training outcomes only')


def apply_calibration(prediction, calibration):
    return ((np.asarray(prediction, dtype=np.float64)-calibration['center']) /
            calibration['scale'] * calibration['coefficient'] + calibration['intercept'])


def prepare(out):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new experiment directory')
    base.load_protocol(base.OUT)
    original.protocol(original.OUT)
    names = ('graph_signal_calibrated.py', 'graph_signal_diffusion.py',
             'graph_signal_experiment.py', 'non_tabular_experiment.py', 'non_tabular_models.py')
    configs = methods.configurations()
    for cfg in configs:
        cfg['epochs'] = 200
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), tasks=list(base.TASKS),
             cases=['inner', '42', '2026', '31415'], seed=20260914, folds=5,
             configs=configs, primary='annotation / reverse / fixed half-logit GNN composition',
             epoch_policy='200 fixed epochs for every task and control, chosen before this follow-up using the old screen; no checkpoint selection on any OOF/evaluation labels.',
             calibration='One regularized scalar Platt map per variant/readout, fitted on five-fold OOF training scores. Mean calibrated logits from the five matching fold models at inference.',
             support='Every OOF positive removed before spectrum and reconstruction. Evaluation pairs absent from all five support graphs. Missing matrix entries remain implicit-feedback zeros, not biological negatives.',
             seed_policy='Five seeds identify cross-fitting models, not five independent biological repeats.',
             composition='0.5 GNN logit + 0.5 calibrated diffusion logit; no coefficient search or task-specific bypass.',
             controls='Support-only and shuffled annotation, each with identical calibration/bagging/epochs. One-step and three-step reverse at the same model weights, each separately calibrated using training outcomes.',
             resplit_policy='All settings frozen before refits. Three cases are overlapping development partitions already used for other methods, not independent external tests.',
             gate='MiDrug both mean global metrics improve with >=2/3 positive signs; LncDrug each metric loses <=0.001 in mean and on every split. Annotation/reverse must outperform support, shuffled and annotation/one-step in both MiDrug mean metrics to support this mechanism.',
             innovation='One graph-signal diffusion adaptation under test. Calibration and mixtures never count as separate innovations.',
             original_test_access=False, historical_outer_query_labels_access=False,
             source_sha256={n:base.digest(base.ROOT/n) for n in names}, input_sha256={})
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for n in names:
        shutil.copy2(base.ROOT/n, out/'source_snapshot'/n)
    for task in p['tasks']:
        node_path = base.OUT/'inputs'/task/'arrays.npz'
        with np.load(node_path, allow_pickle=False) as z:
            nodes = {k:z[k].copy() for k in ('kr', 'kd')}
        p['input_sha256'][str(node_path)] = base.digest(node_path)
        for case in p['cases']:
            source = node_path if case == 'inner' else RESPLITS/'inputs'/task/case/'arrays.npz'
            with np.load(source, allow_pickle=False) as z:
                a = {k:z[k].copy() for k in ('pairs_train', 'y_train', 'pairs_val', 'y_val')}
            p['input_sha256'][str(source)] = base.digest(source)
            if set(map(tuple, a['pairs_train'].T)) & set(map(tuple, a['pairs_val'].T)):
                raise ValueError('Fitting/evaluation overlap')
            a.update(nodes)
            a['folds'] = make_folds(a['pairs_train'], a['y_train'], p['seed'], p['folds'])
            refs = {}
            if case == 'inner':
                for s in (666, 0, 1):
                    refs[str(s)] = base.GRAPH/('screen' if s == 666 else 'confirm')/task/'base/c0'/('s'+str(s))/'scores.npz'
            else:
                refs['20260914'] = RESPLITS/'runs'/task/case/'gnn'/'scores.npz'
            for s, path in refs.items():
                with np.load(path, allow_pickle=False) as z:
                    k = ('pairs_val', 'y_val', 'val') if case == 'inner' else ('pairs', 'y', 'pred')
                    np.testing.assert_array_equal(a['pairs_val'], z[k[0]])
                    np.testing.assert_array_equal(a['y_val'], z[k[1]])
                    a['gnn_'+s] = z[k[2]].copy()
                p['input_sha256'][str(path)] = base.digest(path)
            dest = out/'inputs'/task/case/'arrays.npz'
            dest.parent.mkdir(parents=True)
            np.savez_compressed(dest, **a)
            p['input_sha256'][str(dest)] = base.digest(dest)
    base.write_json(out/'protocol.json', p)
    print('FROZEN', out, flush=True)


def protocol(out):
    p = json.loads((Path(out)/'protocol.json').read_text())
    for n, h in p['source_sha256'].items():
        if base.digest(base.ROOT/n) != h:
            raise ValueError('Source changed: '+n)
    for n, h in p['input_sha256'].items():
        if base.digest(n) != h:
            raise ValueError('Input changed: '+n)
    return p


def construct(graph, kr, kd, cfg, device):
    models = []
    for x, prior in ((graph, kd), (graph.T, kr)):
        vectors, values = methods.spectrum(x, prior, cfg['variant'], cfg['rank'])
        models.append(methods.SignalDiffusion(x.shape[1], vectors, values, cfg['width'],
                                             cfg['steps'], cfg['alpha']).to(device))
    return torch.nn.ModuleList(models)


def fixed_fit(graph, kr, kd, cfg, seed, device):
    start = time.perf_counter()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    models = construct(graph, kr, kd, cfg, device)
    x = torch.as_tensor(graph, device=device)
    opt = torch.optim.AdamW(models.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    trace = []
    for epoch in range(1, cfg['epochs']+1):
        models.train()
        opt.zero_grad(set_to_none=True)
        loss = .5*((models[0].training_prediction(x)-x).square().mean() +
                   (models[1].training_prediction(x.T)-x.T).square().mean())
        if not torch.isfinite(loss):
            raise FloatingPointError('Invalid diffusion loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), 5.)
        opt.step()
        if epoch % 20 == 0 or epoch == cfg['epochs']:
            trace.append(dict(epoch=epoch, training_reconstruction_loss=float(loss)))
    models.eval()
    scores = original.predict(models, x)
    state = {k:v.detach().cpu().clone() for k,v in models.state_dict().items()}
    return scores, state, trace, time.perf_counter()-start


def verify_files(dest, row):
    for n, h in row['files'].items():
        if base.digest(dest/n) != h:
            raise ValueError('Artifact changed: '+str(dest/n))


def run(out, task, cases, device):
    out = Path(out)
    p = protocol(out)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for case in cases:
        if case not in p['cases']:
            raise ValueError('Unknown case')
        with np.load(out/'inputs'/task/case/'arrays.npz', allow_pickle=False) as z:
            a = {n:z[n].copy() for n in z.files}
        for cfg in p['configs']:
            root = out/'runs'/task/case/cfg['id']
            if (root/'result.json').exists():
                verify_files(root, json.loads((root/'result.json').read_text()))
                continue
            oof = {k:np.full(len(a['y_train']), np.nan) for k in KINDS}
            val = {k:[] for k in KINDS}
            total_seconds = 0.
            fold_manifest = {}
            for fold in range(p['folds']):
                dest = root/('fold_'+str(fold))
                dest.mkdir(parents=True, exist_ok=True)
                graph = support_matrix(a['pairs_train'], a['y_train'], a['folds'], fold,
                                       (len(a['kr']), len(a['kd'])), a['pairs_val'])
                ix = np.flatnonzero(a['folds'] == fold)
                tq, vq = a['pairs_train'][:, ix], a['pairs_val']
                if (dest/'result.json').exists():
                    row = json.loads((dest/'result.json').read_text())
                    verify_files(dest, row)
                else:
                    seed = p['seed']+1009*fold
                    scores, state, trace, seconds = fixed_fit(graph, a['kr'], a['kd'], cfg, seed, device)
                    torch.save(state, dest/'model.pt')
                    # Spectrum buffers are part of the checkpoint; recompute to also check inputs.
                    loaded = construct(graph, a['kr'], a['kd'], cfg, device)
                    loaded.load_state_dict(torch.load(dest/'model.pt', map_location=device, weights_only=True))
                    loaded.eval()
                    replay = original.predict(loaded, torch.as_tensor(graph, device=device))
                    err = 0.
                    selected = dict(oof_indices=ix, pairs_oof=tq, pairs_val=vq)
                    for k in KINDS:
                        np.testing.assert_allclose(replay[k], scores[k], atol=2e-6, rtol=2e-6)
                        err = max(err, float(np.abs(replay[k]-scores[k]).max()))
                        selected[k+'_oof'] = scores[k][tq[0], tq[1]]
                        selected[k+'_val'] = scores[k][vq[0], vq[1]]
                    np.savez_compressed(dest/'scores.npz', **selected)
                    base.write_json(dest/'trace.json', trace)
                    row = dict(task=task, case=case, variant=cfg['id'], fold=fold, seed=seed,
                               seconds=seconds, epoch=cfg['epochs'], replay_max_abs=err,
                               fitting_positive_edges=int(graph.sum()), oof_rows=len(ix),
                               oof_query_edges_absent=True, evaluation_query_edges_absent=True,
                               files={n:base.digest(dest/n) for n in ('model.pt', 'scores.npz', 'trace.json')})
                    base.write_json(dest/'result.json', row)
                    print('FOLD', task, case, cfg['id'], fold, round(seconds, 2), flush=True)
                with np.load(dest/'scores.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(ix, z['oof_indices'])
                    np.testing.assert_array_equal(tq, z['pairs_oof'])
                    np.testing.assert_array_equal(vq, z['pairs_val'])
                    for k in KINDS:
                        oof[k][ix] = z[k+'_oof']
                        val[k].append(z[k+'_val'].copy())
                total_seconds += row['seconds']
                for n in (*row['files'], 'result.json'):
                    fold_manifest[str(Path('fold_'+str(fold))/n)] = base.digest(dest/n)
            score_dict = dict(y=a['y_val'], pairs=a['pairs_val'], y_train=a['y_train'],
                              pairs_train=a['pairs_train'], folds=a['folds'])
            calibrations = {}
            for k in KINDS:
                if not np.isfinite(oof[k]).all():
                    raise ValueError('OOF coverage incomplete')
                c = fit_calibration(oof[k], a['y_train'])
                calibrated = apply_calibration(np.stack(val[k]), c).mean(0)
                score_dict[k+'_calibrated'] = calibrated
                score_dict[k+'_oof'] = oof[k]
                score_dict[k+'_fold_val'] = np.stack(val[k])
                for n, g in a.items():
                    if n.startswith('gnn_'):
                        score_dict[n] = g
                        score_dict[k+'_'+n] = .5*g+.5*calibrated
                calibrations[k] = c
            measured = {n:base.metrics(a['y_val'], v, a['pairs_val']) for n,v in score_dict.items()
                        if n.endswith('_calibrated') or 'gnn_' in n}
            np.savez_compressed(root/'scores.npz', **score_dict)
            base.write_json(root/'calibration.json', calibrations)
            files = {n:base.digest(root/n) for n in ('scores.npz', 'calibration.json')}
            files.update(fold_manifest)
            base.write_json(root/'result.json', dict(task=task, case=case, variant=cfg['id'],
                config=cfg, seconds=total_seconds, fold_models=p['folds'], metrics=measured,
                calibration_input='OOF training labels only; no validation label in model/calibrator fitting',
                files=files))
            print('COMPLETE', task, case, cfg['id'], measured, flush=True)


def report(out, cases):
    out = Path(out)
    p = protocol(out)
    rows = []
    for task in p['tasks']:
        for case in cases:
            for cfg in p['configs']:
                root = out/'runs'/task/case/cfg['id']
                r = json.loads((root/'result.json').read_text())
                verify_files(root, r)
                with np.load(root/'scores.npz', allow_pickle=False) as z:
                    for n, old in r['metrics'].items():
                        m = base.metrics(z['y'], z[n], z['pairs'])
                        for k,v in m.items():
                            if abs(v-old[k]) > 1e-12:
                                raise ValueError('Metric mismatch')
                        rows.append(dict(task=task, case=case, variant=cfg['id'], score=n, **m))
    df = pd.DataFrame(rows)
    tag = 'inner' if cases == ['inner'] else 'resplits'
    df.to_csv(out/(tag+'_metrics.csv'), index=False)
    # Average initialization-matched GNN references inside each case first.
    contrasts = []
    for task in p['tasks']:
        for case in cases:
            for cfg in p['configs']:
                part = df[(df.task == task) & (df.case == case) & (df.variant == cfg['id'])]
                ref = part[part.score.str.startswith('gnn_')]
                for k in KINDS:
                    new = part[part.score.str.startswith(k+'_gnn_')]
                    row = dict(task=task, case=case, variant=cfg['id'], kind=k)
                    for m in ('auc', 'ap', 'macro_drug_ap'):
                        row[m] = float(new[m].mean())
                        row['gnn_'+m] = float(ref[m].mean())
                        row['delta_'+m] = row[m]-row['gnn_'+m]
                    contrasts.append(row)
    c = pd.DataFrame(contrasts)
    c.to_csv(out/(tag+'_contrasts.csv'), index=False)
    summary = c.groupby(['task','variant','kind'])[['auc','ap','macro_drug_ap','delta_auc','delta_ap','delta_macro_drug_ap']].mean()
    summary.to_csv(out/(tag+'_summary.csv'))
    print(summary.to_string())
    if tag == 'resplits':
        primary = c[(c.variant == 'annotation') & (c.kind == 'reverse')]
        mi = primary[primary.task == 'MiDrug']
        ln = primary[primary.task == 'LncDrug']
        performance = all(mi['delta_'+m].mean() > 0 and (mi['delta_'+m] > 0).sum() >= 2 for m in ('auc','ap'))
        lnc = all(ln['delta_'+m].mean() >= -.001 and ln['delta_'+m].min() >= -.001 for m in ('auc','ap'))
        controls = {}
        for variant, kind in [('support','reverse'),('shuffled','reverse'),('annotation','one_step')]:
            control = c[(c.task == 'MiDrug') & (c.variant == variant) & (c.kind == kind)]
            controls[variant+'_'+kind] = {m:float(mi[m].mean()-control[m].mean()) for m in ('auc','ap')}
        mechanism = all(v > 0 for pair in controls.values() for v in pair.values())
        assessment = dict(mi_performance=bool(performance), lnc_tolerance=bool(lnc),
                          mechanism_controls=controls, mechanism_supported=mechanism,
                          shared_two_task_gate=bool(performance and lnc and mechanism),
                          independent_biological_validation=False, original_test_access=False,
                          techniques_tested=1, variants_are_not_independent_techniques=True)
        base.write_json(out/'ASSESSMENT.json', assessment)
        print(json.dumps(assessment, ensure_ascii=False, indent=2))
    base.write_json(out/(tag+'_COMPLETION.json'), dict(cases=cases,
        fold_models=len(cases)*len(p['tasks'])*len(p['configs'])*p['folds'],
        all_metrics_recomputed=True, all_artifact_hashes_verified=True,
        artifacts={n:base.digest(out/n) for n in (tag+'_metrics.csv',tag+'_contrasts.csv',tag+'_summary.csv')}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('prepare','run','report'))
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--task', choices=base.TASKS)
    ap.add_argument('--cases', nargs='+', default=['inner'])
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if args.command == 'prepare':
        prepare(args.out_dir)
    elif args.command == 'run':
        run(args.out_dir, args.task, args.cases, args.device)
    else:
        report(args.out_dir, args.cases)


if __name__ == '__main__':
    main()
