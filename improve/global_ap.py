"""Global soft-rank AP regularization after the bounded PLE tuning trial.

Run screen -> top-two confirmation -> fixed-epoch refits -> verified report.
Each worker owns disjoint jobs; independent configurations share no RNG state.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F

from improve import backbone_replacement as parent
from improve import numeric_embedding as original
from improve.global_ap_modules import positive_anchors, smooth_ap_loss

ARMS = ('tabm', 'smooth_ap')
METRICS = ('auc', 'ap', 'macro_drug_ap')
OWN = ('global_ap.py', 'global_ap_modules.py', 'test_global_ap.py', 'test_global_ap_training.py')


def sources():
    return dict(original.sources(), **{name: parent.digest(Path(__file__).with_name(name)) for name in OWN})


def configurations(arm):
    if arm == 'tabm':
        recipes = [(64, 2, .1, 1e-4), (128, 3, .1, 1e-4),
                   (128, 3, .3, .01), (256, 3, .1, 1e-4)]
        result = []
        for width, depth, dropout, decay in recipes:
            for lr in (.0005, .001, .003):
                result.append(dict(id='t%02d' % len(result), lr=lr, width=width, depth=depth,
                    dropout=dropout, weight_decay=decay, members=16,
                    n_bins=16, embedding_dim=16, min_leaf=16))
        return result
    if arm != 'smooth_ap':
        raise ValueError(arm)
    result = []
    for temperature in (.01, .05):
        for strength in (.1, .5, 1.):
            for lr in (.001, .003):
                result.append(dict(id='t%02d' % len(result), lr=lr, width=128, depth=3,
                    dropout=.1, weight_decay=1e-4, members=16, ap_lambda=strength,
                    temperature=temperature, n_bins=16, embedding_dim=16, min_leaf=16))
    return result


def horizon(p, task, cfg):
    return 3 if p['smoke'] else p['epoch_limits'][task][str(cfg['lr'])]


def prepare(args):
    out, ref = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    before = original.protocol(ref)
    checked = json.loads((ref/'VALIDATION.json').read_text())
    anchor = json.loads((ref/'ANCHOR_VALIDATION.json').read_text())
    if before['smoke'] or not checked['passed'] or not anchor['passed']:
        raise ValueError('A verified full numerical-embedding reference is required')
    fallback_evidence = {}
    if not args.smoke:
        previous = Path(args.tuning_dir).resolve()
        decision = json.loads((previous/'OUTCOME.json').read_text())
        validation = json.loads((previous/'VALIDATION.json').read_text())
        if decision['smoke'] or not decision['continue_to_new_mechanism'] or not validation['passed']:
            raise ValueError('The preceding PLE trial has not authorized the fallback condition')
        if validation['protocol_sha256'] != decision['protocol_sha256']:
            raise ValueError('Preceding decision does not match its validation')
        fallback_evidence = {str(previous/name): parent.digest(previous/name)
                             for name in ('protocol.json', 'OUTCOME.json', 'VALIDATION.json')}
    files = ['protocol.json', 'VALIDATION.json', 'ANCHOR_VALIDATION.json', 'PARAMETERS.json', 'REFIT.json']
    files += ['features/%s/%s/manifest.json' % (phase, task)
              for phase in ('inner', 'outer') for task in before['tasks']]
    grid = {arm: configurations(arm) for arm in ARMS}
    if args.smoke:
        grid = {arm: [v[0], v[-1]] for arm, v in grid.items()}
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        reference_dir=str(ref), reference_protocol_sha256=before['sha256'],
        reference_artifacts_sha256={name: parent.digest(ref/name) for name in files},
        tasks=before['tasks'], arms=list(ARMS), configurations=grid,
        screening_seed=666, confirmation_seeds=[0] if args.smoke else [0, 1],
        refit_seeds=[666] if args.smoke else [666, 0, 1], shortlist_size=2,
        epoch_limits={'LncDrug': {'0.0005': 300, '0.001': 200, '0.003': 200},
                      'MiDrug': {'0.0005': 600, '0.001': 450, '0.003': 450}},
        batch_size=1024, smoothing_window=2 if args.smoke else 5,
        source_sha256=sources(), no_test_evaluation=True, fallback_evidence=fallback_evidence,
        ap_anchor_budget=256, ap_warmup_epochs=10,
        selection='12 configurations per arm/task on seed 666; top two by earliest best trailing-five mean(AUC,AP); confirm both with seeds 0/1; choose the largest three-seed mean selection score; freeze the median selected epoch. Stable ties use config ID. Outer scores cannot select.',
        search_space='Smooth AP: lambda .1/.5/1.0, temperature .01/.05, learning rate .001/.003, fixed scalar TabM width/depth 128/3 and dropout/decay .1/1e-4. Scalar TabM gets the same 12-configuration search used in the preceding PLE tuning trial. AP candidates have no numerical embedding.',
        feature_policy='Read-only reuse of all four verified 36-dimensional graph-feature caches. A new REFIT decision is required before opening outer arrays. Neither evaluation labels nor query batches define the ranking context; it contains only fitting rows.',
        optimization='The original member BCE with dropout plus lambda times a global soft-rank AP loss on the deterministic mean member probability. AP compares up to 256 uniformly sampled fitting positives against ALL fitting rows; private seeded anchor sampling does not consume dropout RNG. Lambda ramps over the first ten epochs. One AdamW update per epoch; no per-drug ranking or label-dependent negative sampling.',
        comparisons='Soft-rank AP versus previous scalar TabM and versus equally tuned scalar TabM. The repeated scalar baseline has the same search space, training objective and horizons as the preceding tuning round. Comparisons pair identical validation rows and seeds.',
        gate='Mean AUC and AP must both improve, each in at least 2/3 seeds, relative to both previous and equally tuned scalar TabM, on both tasks. Macro AP is reported separately. This is a development screen, not statistical significance.',
        fallback='This independently frozen mechanism follows the preceding PLE tuning trial under the user-requested failure condition.',
        anchors={} if args.smoke else {'tabm': {'t04': 'c0', 't05': 'c1'}},
        limitations=[v for v in before['limitations'] if not v.startswith((
            'Numerical embedding', 'This is an existing numerical-embedding'))] + [
            'This follow-up was motivated by previously seen outer results; outer is reused development evidence.',
            'Three initialization seeds are not new data partitions.',
            'Joint capacity/regularization recipes do not identify the effect of an individual hyperparameter.',
            'A selected epoch near the bounded horizon does not establish convergence.'])
    out.mkdir(parents=True)
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot/name)
    p['sha256'] = parent.json_digest(p)
    parent.write_json(out/'protocol.json', p)
    parent.write_json(out/'ENVIRONMENT.json', dict(executable=sys.executable, python=sys.version,
        torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(),
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    print('FROZEN', out, flush=True)


def protocol(out):
    p = json.loads((Path(out)/'protocol.json').read_text())
    if p['sha256'] != parent.json_digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('Frozen protocol changed')
    if sources() != p['source_sha256']:
        raise ValueError('Frozen source changed')
    ref = Path(p['reference_dir'])
    if original.protocol(ref)['sha256'] != p['reference_protocol_sha256']:
        raise ValueError('Reference protocol changed')
    for name, sha in p['reference_artifacts_sha256'].items():
        if parent.digest(ref/name) != sha:
            raise ValueError('Reference artifact changed: ' + name)
    for name, sha in p['fallback_evidence'].items():
        if parent.digest(name) != sha:
            raise ValueError('Preceding fallback decision changed')
    return p


def features(out, p, task, stage):
    if stage not in ('screen', 'confirm', 'outer'):
        raise ValueError(stage)
    if stage == 'outer':
        parent.decision(out, 'REFIT.json', p)
    ref = Path(p['reference_dir'])
    phase = 'outer' if stage == 'outer' else 'inner'
    path = ref/'features'/phase/task
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['protocol_sha256'] != p['reference_protocol_sha256']:
        raise ValueError('Feature reference belongs to a different protocol')
    for name, sha in manifest['files'].items():
        if parent.digest(path/name) != sha:
            raise ValueError('Feature artifact changed')
    with np.load(path/'arrays.npz') as archive:
        keys = ('x_train', 'x_val', 'y_train', 'y_val', 'pairs_train', 'pairs_val')
        a = {k: archive[k].copy() for k in keys}
    a['feature_manifest_sha256'] = parent.digest(path/'manifest.json')
    return a


def quick_metrics(y, scores):
    auc, ap = roc_auc_score(y, scores), average_precision_score(y, scores)
    return dict(auc=float(auc), ap=float(ap), criterion=float((auc+ap)/2))


predict = original.predict
def model_for(arm, dim, cfg, seed, device, x_train, y_train):
    if arm not in ARMS:
        raise ValueError(arm)
    return original.model_for('tabm', dim, cfg, seed, device, x_train, y_train)

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
        surrogate, ap_weight = 0., 0.
        if arm == 'smooth_ap' and cfg['ap_lambda'] > 0:
            model.eval()
            members = torch.cat([model(x[start:start+p['batch_size']])
                                 for start in range(0, len(x), p['batch_size'])])
            probability = torch.sigmoid(members).mean(1)
            anchors = positive_anchors(a['y_train'], p['ap_anchor_budget'], (seed+1)*1000003+epoch)
            rank_loss = smooth_ap_loss(probability, y, anchors, cfg['temperature'])
            ap_weight = cfg['ap_lambda']*min(1., epoch/p['ap_warmup_epochs'])
            if not bool(torch.isfinite(rank_loss)):
                raise FloatingPointError('Non-finite AP surrogate')
            (ap_weight*rank_loss).backward()
            surrogate = float(rank_loss.detach())
            loss_value += ap_weight*surrogate
            model.train()
        if not all(bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.grad is not None):
            raise FloatingPointError('Non-finite gradients')
        optimizer.step()
        row = dict(epoch=epoch, loss=loss_value)
        if arm == 'smooth_ap':
            row.update(ap_surrogate=surrogate, ap_weight=ap_weight)
        if stage != 'outer':
            prediction = predict(model, xv, p['batch_size'])
            row.update(quick_metrics(a['y_val'], prediction))
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


def rank_screen(rows, k):
    if len({r['config_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate screen configuration')
    if len(rows) < k or any(not np.isfinite(r['selection_score']) for r in rows):
        raise ValueError('Invalid screening scores')
    return [r['config'] for r in sorted(rows,
        key=lambda r: (-r['selection_score'], r['config_id']))[:k]]


def rank_confirm(configs, rows, seeds):
    if set(r['config_id'] for r in rows) != set(c['id'] for c in configs):
        raise ValueError('Unexpected confirmation configuration')
    summaries = []
    for cfg in configs:
        items = [r for r in rows if r['config_id'] == cfg['id']]
        if len(items) != len(seeds) or sorted(r['seed'] for r in items) != sorted(seeds):
            raise ValueError('Missing or duplicated confirmation seed')
        if any(r['config'] != cfg or not np.isfinite(r['selection_score']) for r in items):
            raise ValueError('Invalid confirmation result')
        summaries.append(dict(config=cfg,
            epochs=int(np.rint(np.median([r['selected_epoch'] for r in items]))),
            mean_selection_score=float(np.mean([r['selection_score'] for r in items])),
            inner_auc=float(np.mean([r['auc'] for r in items])),
            inner_ap=float(np.mean([r['ap'] for r in items])),
            inner_macro_drug_ap=float(np.mean([r['macro_drug_ap'] for r in items])),
            selected_epochs={str(r['seed']): r['selected_epoch'] for r in items}))
    summaries.sort(key=lambda r: (-r['mean_selection_score'], r['config']['id']))
    return summaries[0], summaries


def selection(out, p, stage):
    if stage not in ('screen', 'confirm'):
        raise ValueError('Outer scores cannot select a model')
    shortlist = parent.decision(out, 'PARAMETERS.json', p)['chosen'] if stage == 'confirm' else None
    obj = dict(protocol_sha256=p['sha256'], chosen={}, candidates={}, results_sha256={})
    for task in p['tasks']:
        obj['chosen'][task], obj['candidates'][task] = {}, {}
        for arm in p['arms']:
            cfgs = p['configurations'][arm] if stage == 'screen' else shortlist[task][arm]
            paths = [parent.directory(out, 'screen', task, arm, c['id'], p['screening_seed']) for c in cfgs]
            if stage == 'confirm':
                paths += [parent.directory(out, 'confirm', task, arm, c['id'], seed)
                          for c in cfgs for seed in p['confirmation_seeds']]
            rows = [parent.checked_result(path, p) for path in paths]
            for path in paths:
                obj['results_sha256'][str((path/'result.json').relative_to(out))] = parent.digest(path/'result.json')
            if stage == 'screen':
                obj['chosen'][task][arm] = rank_screen(rows, p['shortlist_size'])
            else:
                chosen, candidates = rank_confirm(cfgs, rows,
                    [p['screening_seed']] + p['confirmation_seeds'])
                obj['chosen'][task][arm] = chosen
                obj['candidates'][task][arm] = candidates
    obj['sha256'] = parent.json_digest(obj)
    return obj


def select(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if args.stage not in ('screen', 'confirm'):
        raise ValueError('Outer scores cannot select a model')
    filename = 'PARAMETERS.json' if args.stage == 'screen' else 'REFIT.json'
    with (out/'selection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        obj = selection(out, p, args.stage)
        if (out/filename).exists():
            if obj != parent.decision(out, filename, p):
                raise ValueError('Selection replay disagrees with frozen decision')
        else:
            parent.write_json(out/filename, obj)
    print(json.dumps(obj['chosen'], indent=2), flush=True)


def jobs(out, p, stage):
    if stage == 'screen':
        short = chosen = None
    elif stage == 'confirm':
        short = parent.decision(out, 'PARAMETERS.json', p)['chosen']
        chosen = None
    elif stage == 'outer':
        parent.decision(out, 'PARAMETERS.json', p)
        chosen = parent.decision(out, 'REFIT.json', p)['chosen']
        short = None
    else:
        raise ValueError(stage)
    for task in p['tasks']:
        for arm in p['arms']:
            configs = p['configurations'][arm] if stage == 'screen' else (
                short[task][arm] if stage == 'confirm' else [chosen[task][arm]['config']])
            seeds = [p['screening_seed']] if stage == 'screen' else (
                p['confirmation_seeds'] if stage == 'confirm' else p['refit_seeds'])
            for cfg in configs:
                for seed in seeds:
                    epochs = chosen[task][arm]['epochs'] if stage == 'outer' else horizon(p, task, cfg)
                    yield task, arm, cfg, seed, epochs


def assign_jobs(planned, workers):
    """Deterministic greedy assignment; each job has exactly one worker."""
    if workers < 1:
        raise ValueError('At least one worker is required')
    def cost(job):
        task, arm, cfg, _, epochs = job
        dim = 36
        work = cfg['width']*dim + (cfg['depth']-1)*cfg['width']**2
        return epochs*(1. if task == 'LncDrug' else 2.6)*work*(2.0 if arm == 'smooth_ap' else 1.0)
    order = sorted(range(len(planned)), key=lambda i: (-cost(planned[i]), i))
    queues, loads = [[] for _ in range(workers)], [0.]*workers
    for index in order:
        worker = min(range(workers), key=lambda i: (loads[i], i))
        queues[worker].append(planned[index])
        loads[worker] += cost(planned[index])
    return queues


def run(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    planned = [j for j in jobs(out, p, args.stage)
               if (args.task is None or j[0] == args.task) and (args.arm is None or j[1] == args.arm)]
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError('Invalid worker index')
    cache = {}
    for task, arm, cfg, seed, epochs in assign_jobs(planned, args.worker_count)[args.worker_index]:
        dest = parent.directory(out, args.stage, task, arm, cfg['id'], seed)
        dest.mkdir(parents=True, exist_ok=True)
        with (dest/'job.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (dest/'result.json').exists():
                parent.checked_result(dest, p)
                continue
            if task not in cache:
                cache[task] = features(out, p, task, args.stage)
            a = cache[task]
            r, scores, trace, state = train(a, task, arm, cfg, seed, args.stage, epochs, p, args.device)
            r['device'] = args.device
            np.savez_compressed(dest/'scores.npz', val=scores, y_val=a['y_val'], pairs_val=a['pairs_val'])
            pd.DataFrame(trace).to_csv(dest/'trace.csv', index=False)
            torch.save(state, dest/'checkpoint.pt')
            r['artifacts_sha256'] = {name: parent.digest(dest/name)
                for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
            parent.write_json(dest/'result.json', r)
            print('DONE', task, arm, cfg['id'], seed, r['auc'], r['ap'], r['selected_epoch'], flush=True)


def paired_deltas(left, right, task, arm, control, comparison):
    a = left[(left.task == task) & (left.arm == arm)].set_index('seed').sort_index()
    b = right[(right.task == task) & (right.arm == control)].set_index('seed').sort_index()
    if not a.index.equals(b.index) or not a.index.is_unique or not b.index.is_unique:
        raise ValueError('Comparison seeds do not align')
    return [dict(task=task, arm=arm, control=control, comparison=comparison, seed=int(seed),
                 **{k: float(a.loc[seed, k]-b.loc[seed, k]) for k in METRICS}) for seed in a.index]


def gate(deltas):
    d = pd.DataFrame(deltas)
    return bool(len(d) == 3 and (d[['auc', 'ap']].mean() > 0).all()
                and ((d[['auc', 'ap']] > 0).sum() >= 2).all())


def report(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    for stage, name in (('screen', 'PARAMETERS.json'), ('confirm', 'REFIT.json')):
        if selection(out, p, stage) != parent.decision(out, name, p):
            raise ValueError('Selection does not replay')
    rows, replays, boundary, cache = [], [], [], {}
    for stage in ('screen', 'confirm', 'outer'):
        for task, arm, cfg, seed, epochs in jobs(out, p, stage):
            dest = parent.directory(out, stage, task, arm, cfg['id'], seed)
            r = parent.checked_result(dest, p)
            if (r['task'], r['arm'], r['config'], r['seed'], r['stage'], r['trained_epochs']) != (
                    task, arm, cfg, seed, stage, epochs):
                raise ValueError('Result does not match the planned job')
            key = (task, 'outer' if stage == 'outer' else 'inner')
            if key not in cache:
                cache[key] = features(out, p, task, stage)
            a = cache[key]
            if r['feature_manifest_sha256'] != a['feature_manifest_sha256']:
                raise ValueError('Training feature hash changed')
            with np.load(dest/'scores.npz') as archive:
                saved = archive['val'].copy()
                np.testing.assert_array_equal(archive['y_val'], a['y_val'])
                np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
            for metric, value in parent.metrics(a['y_val'], saved, a['pairs_val']).items():
                if value is not None:
                    np.testing.assert_allclose(r[metric], value, atol=1e-12, rtol=0)
            trace = pd.read_csv(dest/'trace.csv')
            np.testing.assert_array_equal(trace.epoch, np.arange(1, epochs+1))
            if not np.isfinite(trace.loss).all():
                raise ValueError('Non-finite loss trace')
            if stage != 'outer':
                epoch, score = parent.choose_checkpoint(trace.to_dict('records'), p['smoothing_window'])
                if epoch != r['selected_epoch']:
                    raise ValueError('Selected epoch changed')
                np.testing.assert_allclose(score, r['selection_score'], atol=1e-12, rtol=0)
                if epoch >= .95*epochs:
                    boundary.append(dict(task=task, arm=arm, config=cfg['id'], seed=seed,
                        stage=stage, selected_epoch=epoch, limit=epochs))
            elif r['selected_epoch'] != epochs or r['selection_score'] is not None or any(
                    metric in trace for metric in ('auc', 'ap', 'criterion', 'macro_drug_ap')):
                raise ValueError('Outer metrics influenced training')
            model = model_for(arm, a['x_train'].shape[1], cfg, seed, args.device, a['x_train'], a['y_train'])
            model.load_state_dict(torch.load(dest/'checkpoint.pt', map_location='cpu', weights_only=False))
            replay = predict(model, torch.from_numpy(a['x_val']).to(args.device), p['batch_size'])
            np.testing.assert_allclose(replay, saved, atol=2e-6, rtol=2e-6)
            replays.append(dict(path=str(dest.relative_to(out)), max_abs=float(np.max(np.abs(replay-saved)))))
            rows.append(r)
    data = pd.DataFrame(rows)
    outer = data[data.stage == 'outer']
    data.to_csv(out/'results.csv', index=False)
    outer.groupby(['task', 'arm'])[list(METRICS)].agg(['mean', 'std']).to_csv(out/'outer_summary.csv')
    ref, prior = Path(p['reference_dir']), original.protocol(p['reference_dir'])
    old_rows, anchors = [], []
    for task, arm, cfg, seed, _ in original.jobs(ref, prior, 'outer'):
        if arm not in p['arms'] or seed not in p['refit_seeds']:
            continue
        dest = parent.directory(ref, 'outer', task, arm, cfg['id'], seed)
        r = parent.checked_result(dest, prior)
        a = cache[(task, 'outer')]
        with np.load(dest/'scores.npz') as archive:
            np.testing.assert_array_equal(archive['y_val'], a['y_val'])
            np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
            for metric, value in parent.metrics(a['y_val'], archive['val'], a['pairs_val']).items():
                if value is not None:
                    np.testing.assert_allclose(r[metric], value, atol=1e-12, rtol=0)
        old_rows.append(r)
    old = pd.DataFrame(old_rows)
    old.to_csv(out/'reference_outer_results.csv', index=False)
    for arm, recipes in p['anchors'].items():
        for new_id, old_id in recipes.items():
            for task in p['tasks']:
                before = parent.directory(ref, 'screen', task, arm, old_id, p['screening_seed'])
                after = parent.directory(out, 'screen', task, arm, new_id, p['screening_seed'])
                parent.checked_result(before, prior)
                previous = pd.read_csv(before/'trace.csv')
                current = pd.read_csv(after/'trace.csv').iloc[:len(previous)]
                cols = ['epoch', 'loss', 'auc', 'ap', 'criterion']
                np.testing.assert_allclose(current[cols], previous[cols], atol=2e-6, rtol=2e-6)
                anchors.append(dict(task=task, arm=arm, new_id=new_id, old_id=old_id,
                    max_abs=float(np.max(np.abs(current[cols].values-previous[cols].values)))))
    deltas, gates = [], {}
    for task in p['tasks']:
        gates[task] = {}
        for arm in p['arms']:
            controls = [('tabm', 'previous_tabm')]
            if arm in set(old.arm):
                controls.append((arm, 'previous_same_arm'))
            for control, comparison in controls:
                d = paired_deltas(outer, old, task, arm, control, comparison)
                deltas += d
                gates[task][arm+'_vs_'+comparison] = gate(d)
            if arm != 'tabm':
                d = paired_deltas(outer, outer, task, arm, 'tabm', 'tuned_tabm')
                deltas += d
                gates[task][arm+'_vs_tuned_tabm'] = gate(d)
    joint = {arm: all(gates[t][arm+'_vs_'+control] for t in p['tasks']
                     for control in ('previous_tabm', 'tuned_tabm')) for arm in p['arms'] if arm != 'tabm'}
    delta = pd.DataFrame(deltas)
    delta.to_csv(out/'paired_deltas.csv', index=False)
    delta.groupby(['task', 'arm', 'comparison'])[list(METRICS)].agg(['mean', 'std']).to_csv(out/'delta_summary.csv')
    parent.write_json(out/'OUTCOME.json', dict(protocol_sha256=p['sha256'], gates=gates,
        joint=joint, scope_complete=True, smoke=p['smoke']))
    parent.write_json(out/'VALIDATION.json', dict(passed=True, smoke=p['smoke'], protocol_sha256=p['sha256'],
        completed_runs=len(rows), all_checkpoints_replayed=True, selection_replayed=True,
        no_test_evaluation=True, anchors=anchors, near_horizon=boundary, replays=replays))
    means = outer.groupby(['task', 'arm'])[list(METRICS)].mean()
    before = old.groupby(['task', 'arm'])[list(METRICS)].mean()
    lines = ['# 全局 AP 软排序训练结果', '',
        '历史验证划分，seed 666/0/1 均值。所有配置和重训轮数只由内部验证选择；新 AP 组的原指标指原始标量 TabM。', '',
        '| 任务 | 组 | 原 AUC | 调参 AUC | 原 AP | 调参 AP | 原宏 AP | 调参宏 AP |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for task in p['tasks']:
        for arm in p['arms']:
            a, b = means.loc[(task, arm)], before.loc[(task, arm if arm in set(old.arm) else 'tabm')]
            lines.append('| %s | %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |' %
                         (task, arm, b.auc, a.auc, b.ap, a.ap, b.macro_drug_ap, a.macro_drug_ap))
    lines += ['', '## 相对原 TabM 与同预算 TabM 的变化', '',
        '| 任务 | 组 | 对照 | ΔAUC | 正向种子 | ΔAP | 正向种子 | Δ宏 AP |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for (task, arm, comparison), frame in delta.groupby(['task', 'arm', 'comparison']):
        if comparison == 'previous_same_arm':
            continue
        lines.append('| %s | %s | %s | %+.6f | %d/%d | %+.6f | %d/%d | %+.6f |' %
            (task, arm, comparison, frame.auc.mean(), int((frame.auc > 0).sum()), len(frame),
             frame.ap.mean(), int((frame.ap > 0).sum()), len(frame), frame.macro_drug_ap.mean()))
    lines += ['', '## 内部选定配置', '',
        '| 任务 | 组 | ID | AP 权重/温度 | 网络宽/深 | Dropout | 衰减 | 学习率 | 重训轮数 |',
        '|---|---|---|---|---|---:|---:|---:|---:|']
    chosen = parent.decision(out, 'REFIT.json', p)['chosen']
    for task in p['tasks']:
        for arm in p['arms']:
            choice = chosen[task][arm]
            c = choice['config']
            lines.append('| %s | %s | %s | %s | %d/%d | %g | %g | %g | %d |' %
                (task, arm, c['id'], '-' if arm == 'tabm' else '%g/%g' % (c['ap_lambda'], c['temperature']),
                 c['width'], c['depth'], c['dropout'], c['weight_decay'], c['lr'], choice['epochs']))
    lines += ['', '## 判定与核验', '', '预设双任务判定：`'+json.dumps(joint)+'`。', '',
        '- 每组 12 个配置；内部前两名都补 seed 0/1，再按三种子平均选择。',
        '- %d 个检查点全部重放，配置选择、训练轮数和指标全部重算。' % len(rows),
        '- %d 个原配置的训练轨迹前缀复现；四份已验证特征缓存仅只读复用。' % len(anchors),
        '- %d 次选中轮数落在训练上限最后 5%%；不宣称全部配置已收敛。' % len(boundary),
        '- 双任务门槛要求相对旧 TabM 与同预算 TabM 的 AUC/AP 都提高、每项至少 2/3 种子正向。宏 AP 单独报告。',
        '- 同一历史验证划分被重复用于开发；三个初始化不是独立划分。本轮不读取测试集。',
        '- AP 代理优化确定性集成分数；BCE 使用原来的 dropout。固定的 256 个正锚点预算与十轮权重预热没有额外搜索。',
        '- 软排名不等于精确 AP，尤其是在并列分数附近；这里不能据此宣称生物学概率校准。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(means.to_string(), flush=True)
    print('VERIFIED', len(rows), 'checkpoints; joint outcome:', joint, flush=True)


def pipeline(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    devices = args.devices.split(',')
    if not devices or any(not d for d in devices):
        raise ValueError('Specify at least one device')
    log_dir = out/'logs'
    log_dir.mkdir(exist_ok=True)
    execution = []
    begin = time.monotonic()
    for stage in ('screen', 'confirm', 'outer'):
        workers = []
        for index, device in enumerate(devices):
            command = [sys.executable, '-m', 'improve.global_ap', 'run',
                '--out-dir', str(out), '--stage', stage, '--worker-index', str(index),
                '--worker-count', str(len(devices)), '--device', device]
            log = (log_dir/(stage+'_worker'+str(index)+'.log')).open('a')
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            workers.append((command, process, log))
        failed = []
        for command, process, log in workers:
            code = process.wait()
            log.close()
            execution.append(dict(stage=stage, command=command, exit_code=code))
            if code:
                failed.append(command)
        parent.write_json(out/'EXECUTION.json', dict(protocol_sha256=p['sha256'], workers=execution,
            elapsed_seconds=time.monotonic()-begin))
        if failed:
            raise RuntimeError('Stage failed; see worker logs: '+stage)
        print('STAGE_DONE', stage, 'seconds', round(time.monotonic()-begin, 1), flush=True)
        if stage != 'outer':
            args.stage = stage
            select(args)
    args.device = devices[0]
    report(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'select', 'report', 'pipeline'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--reference-dir', default='improve/results/numeric_embedding_20260912')
    parser.add_argument('--tuning-dir', default='improve/results/numeric_embedding_tuning_20260912')
    parser.add_argument('--stage', choices=('screen', 'confirm', 'outer'), default='screen')
    parser.add_argument('--task', choices=parent.TASKS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    dict(prepare=prepare, run=run, select=select, report=report, pipeline=pipeline)[args.command](args)


if __name__ == '__main__':
    main()
