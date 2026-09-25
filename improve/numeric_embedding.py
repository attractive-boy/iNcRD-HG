"""Matched numerical-embedding trial on the frozen graph-feature TabM inputs.

Quantile and supervised-tree PLE are compared with scalar TabM and a
feature-wise affine/ReLU embedding control. Selection uses inner data only.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from improve import backbone_replacement as parent
from improve.graph_tabm_modules import GraphTabM, crossfit_features, fit_transform
from improve.numeric_embedding_modules import EmbeddedTabM, fit_bin_edges


ARMS = ('tabm', 'linear', 'ple_quantile', 'ple_tree')
OWN = ('numeric_embedding.py', 'numeric_embedding_modules.py', 'test_numeric_embedding.py',
       'graph_tabm.py', 'graph_tabm_modules.py', 'test_graph_tabm.py')


def sources():
    return dict(parent.sources(), **{name: parent.digest(Path(__file__).with_name(name)) for name in OWN})


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    ref = parent.protocol(reference)
    if ref['smoke']:
        raise ValueError('Use the full parent inputs')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        reference_dir=str(reference), reference_protocol_sha256=ref['sha256'],
        tasks=list(parent.TASKS), arms=list(ARMS), screening_seed=666,
        confirmation_seeds=[0] if args.smoke else [0, 1], refit_seeds=[666] if args.smoke else [666, 0, 1],
        configurations=[dict(id='c'+str(i), lr=lr, weight_decay=1e-4, width=128, depth=3, dropout=.1, members=16, embedding_dim=16, n_bins=16, min_leaf=16)
                        for i, lr in enumerate(([1e-3] if args.smoke else [1e-3, 3e-3]))],
        epochs={t: 3 if args.smoke else c['epochs'] for t, c in parent.TASKS.items()},
        batch_size=1024, folds=5, fold_seed=20260910, smoothing_window=2 if args.smoke else 5,
        source_sha256=sources(),
        selection='Two learning rates per architecture using seed 666 and earliest best trailing-five inner mean(AUC,AP); freeze recipe; confirm seeds 0/1; refit for median selected inner epoch. Never select on outer results.',
        graph_policy='Exactly the parent PCA features and graph. Each training row uses features from a five-fold support graph excluding its entire fold of target positives; all target-dependent counts, paths, SVD and attribute profiles are recomputed. Side features exclude both target directions. Validation uses fitting positives only.',
        network='Scalar TabM, feature-wise affine/ReLU embeddings, quantile PLE, and supervised-tree PLE. Embedded arms use 16 coordinates per feature, 16-bin maximum, then the identical 128-wide three-block 16-member TabM. Independent member BCE, mean probability inference.',
        scope='Only the numerical input representation changes. Identical cross-fitted 36-dimensional graph features, quantile normalization, BCE objective, optimizer and two-learning-rate budget for all arms.',
        binning='Knots are fitted on transformed fitting rows only. Quantile knots are label-free; tree knots use fitting labels with max_leaf_nodes=16 and min_samples_leaf=16. Constant features produce zero ramps; tails clamp at the fitting extrema.',
        primary='Compare ple_quantile and ple_tree separately against scalar TabM. Report all arms on both tasks; no post-hoc outer-based selection of a recipe. A positive two-task result requires mean AUC and AP gains on both tasks, with at least 2/3 seeds positive in every cell.',
        no_test_evaluation=True,
        limitations=['Historical inner and outer validation partitions have been reused.',
            'Three initialization seeds are not independent data splits.',
            'TabM is an internal ensemble with shared weights, not a single ordinary MLP.',
            'Numerical embedding adds parameters; the affine/ReLU control matches the downstream embedding width, but PLE has extra basis-projection weights.',
            'This is an existing numerical-embedding method applied here, not a claim of a new general algorithm.',
            'The graph contains other-task and circRNA information already present in the parent inputs.'])
    out.mkdir(parents=True)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot/name)
    p['sha256'] = parent.json_digest(p)
    parent.write_json(out/'protocol.json', p)
    print('FROZEN', out, flush=True)


def protocol(out):
    p = json.loads((Path(out)/'protocol.json').read_text())
    if p['sha256'] != parent.json_digest({k:v for k,v in p.items() if k!='sha256'}) or p['source_sha256'] != sources():
        raise ValueError('Frozen protocol/source changed')
    ref = parent.protocol(p['reference_dir'])
    if ref['sha256'] != p['reference_protocol_sha256']:
        raise ValueError('Parent input protocol changed')
    return p


def features(out, p, task, stage):
    phase = 'outer' if stage == 'outer' else 'inner'
    path = Path(out) / 'features' / phase / task
    if not (path / 'manifest.json').exists():
        arrays, graph, _ = parent.stage_inputs(p['reference_dir'], task, phase)
        raw, val, names, folds = crossfit_features(graph, task, arrays['pairs_train'], arrays['y_train'],
                                                   arrays['pairs_val'], p['folds'], p['fold_seed'])
        x, xv, transform = fit_transform(raw, val)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path/'arrays.npz', x_train=x, x_val=xv, raw_train=raw, raw_val=val,
            y_train=arrays['y_train'], y_val=arrays['y_val'], pairs_train=arrays['pairs_train'],
            pairs_val=arrays['pairs_val'], fold_ids=folds)
        with (path/'transform.pkl').open('wb') as stream:
            pickle.dump(transform, stream, protocol=4)
        parent.write_json(path/'manifest.json', dict(protocol_sha256=p['sha256'], names=names,
            files={name:parent.digest(path/name) for name in ('arrays.npz', 'transform.pkl')}))
        print('FEATURES', phase, task, x.shape, xv.shape, flush=True)
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['protocol_sha256'] != p['sha256']:
        raise ValueError('Feature cache protocol mismatch')
    for name, sha in manifest['files'].items():
        if parent.digest(path/name) != sha:
            raise ValueError('Feature cache changed')
    with np.load(path/'arrays.npz') as archive:
        result = {key:archive[key].copy() for key in archive.files}
    result['feature_manifest_sha256'] = parent.digest(path/'manifest.json')
    return result


def model_for(arm, dim, cfg, seed, device, x_train, y_train):
    parent.set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kwargs = dict(width=cfg['width'], depth=cfg['depth'], dropout=cfg['dropout'], members=cfg['members'])
    if arm == 'tabm':
        model = GraphTabM(dim, **kwargs)
    elif arm in ('linear', 'ple_quantile', 'ple_tree'):
        edges = None if arm == 'linear' else fit_bin_edges(x_train, y_train,
            n_bins=cfg['n_bins'], method=arm[4:],
            min_leaf=cfg['min_leaf'])
        model = EmbeddedTabM(dim, embedding_dim=cfg['embedding_dim'], edges=edges,
                             n_bins=cfg['n_bins'], **kwargs)
    else:
        raise ValueError(arm)
    return model.to(device)


@torch.no_grad()
def predict(model, x, batch_size):
    model.eval()
    return np.concatenate([torch.sigmoid(model(x[i:i+batch_size])).mean(1).cpu().numpy()
                           for i in range(0, len(x), batch_size)])


def train(a, task, arm, cfg, seed, stage, epochs, p, device):
    started = time.monotonic()
    model = model_for(arm, a['x_train'].shape[1], cfg, seed, device, a['x_train'], a['y_train'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    x, xv = torch.from_numpy(a['x_train']).to(device), torch.from_numpy(a['x_val']).to(device)
    y = torch.from_numpy(a['y_train']).to(device)
    trace, best, selected, state, saved = [], -np.inf, None, None, None
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        for start in range(0, len(x), p['batch_size']):
            output = model(x[start:start+p['batch_size']])
            target = y[start:start+p['batch_size'], None].expand_as(output)
            loss = F.binary_cross_entropy_with_logits(output, target, reduction='sum') / (len(x)*output.shape[1])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite tabular loss')
            loss.backward()
            loss_value += float(loss.detach())
        if not all(bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.grad is not None):
            raise FloatingPointError('Non-finite gradients')
        optimizer.step()
        row = dict(epoch=epoch, loss=loss_value)
        if stage != 'outer':
            prediction = predict(model, xv, p['batch_size'])
            row.update(parent.metrics(a['y_val'], prediction, a['pairs_val']))
            trace.append(row)
            if epoch >= p['smoothing_window']:
                value = float(np.mean([r['criterion'] for r in trace[-p['smoothing_window']:]]))
                if value > best:
                    best, selected, saved = value, epoch, prediction.copy()
                    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            trace.append(row)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print('EPOCH', stage, task, arm, cfg['id'], seed, epoch, round(row.get('criterion', float('nan')), 6),
                   'seconds', round(time.monotonic()-started, 1), flush=True)
    if stage == 'outer':
        selected = epochs
        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        chosen, _ = parent.choose_checkpoint(trace, p['smoothing_window'])
        if chosen != selected:
            raise AssertionError('Checkpoint selection drift')
    model.load_state_dict(state)
    prediction = predict(model, xv, p['batch_size'])
    error = 0.
    if saved is not None:
        np.testing.assert_allclose(prediction, saved, atol=2e-6, rtol=2e-6)
        error = float(np.max(np.abs(prediction-saved)))
    r = dict(task=task, arm=arm, config=cfg, config_id=cfg['id'], seed=seed, stage=stage,
        **parent.metrics(a['y_val'], prediction, a['pairs_val']), selected_epoch=selected,
        selection_score=best if stage != 'outer' else None, trained_epochs=epochs,
        params=sum(v.numel() for v in model.parameters()), elapsed_seconds=time.monotonic()-started,
        restore_max_abs=error, protocol_sha256=p['sha256'], feature_manifest_sha256=a['feature_manifest_sha256'])
    return r, prediction, trace, state


def jobs(out, p, stage):
    selected = parent.decision(out, 'PARAMETERS.json', p)['chosen'] if stage != 'screen' else None
    refits = parent.decision(out, 'REFIT.json', p)['chosen'] if stage == 'outer' else None
    for task in p['tasks']:
        for arm in p['arms']:
            configs = p['configurations'] if stage == 'screen' else [selected[task][arm]]
            seeds = [p['screening_seed']] if stage == 'screen' else p['confirmation_seeds'] if stage == 'confirm' else p['refit_seeds']
            for cfg in configs:
                for seed in seeds:
                    yield task, arm, cfg, seed, refits[task][arm]['epochs'] if stage == 'outer' else p['epochs'][task]


def run(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    planned = list(jobs(out, p, args.stage))
    for task in ([args.task] if args.task else p['tasks']):
        with (out/(task+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            a = features(out, p, task, args.stage)
            for t, arm, cfg, seed, epochs in planned:
                if t != task:
                    continue
                dest = parent.directory(out, args.stage, task, arm, cfg['id'], seed)
                if (dest/'result.json').exists():
                    parent.checked_result(dest, p)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                r, val, trace, state = train(a, task, arm, cfg, seed, args.stage, epochs, p, args.device)
                np.savez_compressed(dest/'scores.npz', val=val, y_val=a['y_val'], pairs_val=a['pairs_val'])
                pd.DataFrame(trace).to_csv(dest/'trace.csv', index=False)
                torch.save(state, dest/'checkpoint.pt')
                r['artifacts_sha256'] = {name:parent.digest(dest/name) for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                parent.write_json(dest/'result.json', r)
                print('DONE', task, arm, cfg['id'], seed, r['auc'], r['ap'], r['selected_epoch'], flush=True)


def select(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if args.stage not in ('screen', 'confirm'):
        raise ValueError('Outer results cannot select a model')
    filename = 'PARAMETERS.json' if args.stage == 'screen' else 'REFIT.json'
    if (out/filename).exists():
        print(json.dumps(parent.decision(out, filename, p)['chosen'], indent=2), flush=True)
        return
    selected = parent.decision(out, 'PARAMETERS.json', p)['chosen'] if args.stage == 'confirm' else None
    obj = dict(protocol_sha256=p['sha256'], chosen={}, results_sha256={})
    for task in p['tasks']:
        obj['chosen'][task] = {}
        for arm in p['arms']:
            if args.stage == 'screen':
                paths = [parent.directory(out, 'screen', task, arm, cfg['id'], p['screening_seed']) for cfg in p['configurations']]
            else:
                cfg = selected[task][arm]
                paths = [parent.directory(out, 'screen', task, arm, cfg['id'], p['screening_seed'])]
                paths += [parent.directory(out, 'confirm', task, arm, cfg['id'], s) for s in p['confirmation_seeds']]
            rows = [parent.checked_result(path, p) for path in paths]
            for path in paths:
                obj['results_sha256'][str((path/'result.json').relative_to(out))] = parent.digest(path/'result.json')
            if args.stage == 'screen':
                obj['chosen'][task][arm] = sorted(rows, key=lambda r:(-r['selection_score'], r['config_id']))[0]['config']
            else:
                obj['chosen'][task][arm] = dict(config=selected[task][arm],
                    epochs=int(np.rint(np.median([r['selected_epoch'] for r in rows]))),
                    inner_auc=float(np.mean([r['auc'] for r in rows])), inner_ap=float(np.mean([r['ap'] for r in rows])))
    obj['sha256'] = parent.json_digest(obj)
    parent.write_json(out/filename, obj)
    print(json.dumps(obj['chosen'], indent=2), flush=True)


def report(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    rows, checks = [], []
    for stage in ('screen', 'confirm', 'outer'):
        for task, arm, cfg, seed, epochs in jobs(out, p, stage):
            dest = parent.directory(out, stage, task, arm, cfg['id'], seed)
            r = parent.checked_result(dest, p)
            a = features(out, p, task, stage)
            if r['feature_manifest_sha256'] != a['feature_manifest_sha256']:
                raise ValueError('Training features changed')
            with np.load(dest/'scores.npz') as archive:
                saved = archive['val'].copy()
                np.testing.assert_array_equal(archive['y_val'], a['y_val'])
                np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
            for key,value in parent.metrics(a['y_val'], saved, a['pairs_val']).items():
                if value is not None:
                    np.testing.assert_allclose(r[key], value, atol=1e-12, rtol=0)
            trace = pd.read_csv(dest/'trace.csv')
            np.testing.assert_array_equal(trace.epoch, np.arange(1, epochs+1))
            if stage != 'outer':
                epoch, smooth = parent.choose_checkpoint(trace.to_dict('records'), p['smoothing_window'])
                if epoch != r['selected_epoch']:
                    raise ValueError('Wrong checkpoint epoch')
                np.testing.assert_allclose(smooth, r['selection_score'], atol=1e-12, rtol=0)
            elif any(key in trace for key in ('auc', 'ap', 'criterion')):
                raise ValueError('Outer evaluation used during training')
            model = model_for(arm, a['x_train'].shape[1], cfg, seed, args.device, a['x_train'], a['y_train'])
            model.load_state_dict(torch.load(dest/'checkpoint.pt', map_location='cpu', weights_only=False))
            replay = predict(model, torch.from_numpy(a['x_val']).to(args.device), p['batch_size'])
            np.testing.assert_allclose(replay, saved, atol=2e-6, rtol=2e-6)
            checks.append(dict(path=str(dest.relative_to(out)), max_abs=float(np.max(np.abs(replay-saved)))))
            rows.append(r)
    pd.DataFrame(rows).to_csv(out/'results.csv', index=False)
    outer = pd.DataFrame([r for r in rows if r['stage']=='outer'])
    outer.groupby(['task','arm'])[['auc','ap','macro_drug_ap']].agg(['mean','std']).to_csv(out/'outer_summary.csv')
    parent.write_json(out/'VALIDATION.json', dict(passed=True, smoke=p['smoke'], completed_runs=len(rows),
        protocol_sha256=p['sha256'], all_checkpoints_replayed=True, no_test_evaluation=True, replays=checks))
    print(outer.groupby(['task','arm'])[['auc','ap','macro_drug_ap']].mean().to_string(), flush=True)
    print('VERIFIED', len(rows), 'checkpoints', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run','select','report'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--reference-dir', default='improve/results/backbone_replacement_20260910')
    parser.add_argument('--stage', choices=('screen','confirm','outer'), default='screen')
    parser.add_argument('--task', choices=parent.TASKS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    {'prepare':prepare,'run':run,'select':select,'report':report}[args.command](args)


if __name__ == '__main__':
    main()
