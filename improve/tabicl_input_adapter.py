"""Frozen-protocol experiment for a shared TabICLv2 input adapter.

prepare -> bank -> train (plain/projected) -> run/verify inner -> select.
Only an eligible selection permits run/verify outer. Original test arrays are
never loaded. Existing models, protocols, and installed packages are unchanged.
"""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from improve import tabular_foundation as base
from improve import tabicl_graph_context as graph
from improve import tabicl_graph_context_modules as graph_math
from improve import tabicl_reli as reference
from improve import tabicl_input_adapter_modules as kernels


DEFAULT_OUT = 'improve/results/tabicl_input_adapter_v2_20260913'
TASKS = ('LncDrug', 'MiDrug')
METRICS = ('auc', 'ap', 'macro_drug_ap')
MODES = ('plain', 'projected')
OWN = ('tabicl_input_adapter.py', 'tabicl_input_adapter_modules.py', 'test_tabicl_input_adapter.py')
DEPENDENCIES = ('tabular_foundation.py', 'tabicl_graph_context.py',
    'tabicl_graph_context_modules.py', 'tabicl_graph_context_features.py',
    'tabicl_reli.py', 'tabicl_reli_modules.py')


def sources():
    return {name: base.digest(Path(__file__).with_name(name)) for name in (*OWN, *DEPENDENCIES)}


def prepare(args):
    import tabicl
    out, parent = Path(args.out_dir).resolve(), Path(args.parent_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Preserve earlier trials; use a fresh experiment directory')
    gp = graph.protocol(parent)
    chosen = graph.selection(parent, gp)
    expected = {'LncDrug': 'query_bag_050', 'MiDrug': 'fold_match_050'}
    if any(chosen['chosen'][task]['recipe']['id'] != expected[task] for task in TASKS):
        raise ValueError('Unexpected graph-context reference')
    manifests = {task: Path(gp['feature_reference_dir']) / 'features' / 'inner' / task / 'manifest.json' for task in TASKS}
    names = {task: base.read_json(path)['names'] for task, path in manifests.items()}
    if names['LncDrug'][:31] != names['MiDrug'][:31] or any(len(n) != 36 for n in names.values()):
        raise ValueError('Shared semantic columns changed')
    installed = Path(tabicl.__file__).parent
    implementation = {str(path): base.digest(path) for path in
        [*sorted((installed / '_model').glob('*.py')),
         installed / '_sklearn' / 'classifier.py', installed / '_sklearn' / 'preprocessing.py']}
    steps = [1, 2] if args.smoke else [16, 64]
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        parent_dir=str(parent), parent_protocol_sha256=gp['sha256'],
        parent_selection_sha256=base.digest(parent / 'SELECTION.json'),
        source_sha256=sources(), installed_source_sha256=implementation,
        parameter_sha256=gp['parameter_sha256'], input_configs=gp['input_configs'],
        graph_recipes={task: chosen['chosen'][task]['recipe'] for task in TASKS},
        fitting_manifests={task: dict(path=str(path), sha256=base.digest(path)) for task, path in manifests.items()},
        feature_names=names, tasks=list(TASKS), seeds=[666] if args.smoke else [666, 0, 1],
        training_seed=20260913, modes=list(MODES), steps=steps,
        candidates=[dict(id='%s_e%d' % (mode, step), mode=mode, step=step) for mode in MODES for step in steps],
        rank=8, parameters=496, shared_features=31, inference_members=2 if args.smoke else 32,
        episode_members=2, folds=list(range(5)), mi_episodes_per_fold=1 if args.smoke else 4,
        mi_support=48 if args.smoke else 192, mi_query=16 if args.smoke else 64,
        lnc_support=24 if args.smoke else 96, lnc_query=8 if args.smoke else 32,
        learning_rate=.0003, adam_betas=[.9, .999], weight_decay=0., max_backtracks=8,
        proxy_roundoff_atol=1e-7, replay_atol=1e-6, query_atol=5e-5, query_effect_atol=1e-5,
        baseline_atol=1e-5,
        architecture='One rank-8 residual input adapter on the 31 semantically shared columns. Same weights for both tasks, support/query, graph branches, and ensemble members. No task ID or LncDrug identity bypass.',
        preprocessing='Official support-fitted none/power preprocessing; restore filtered columns with zeros, undo feature shuffle, adapt canonical input, reapply filtering and shuffle. Recompute all backbone representations.',
        training='Same fixed initialization and MiDrug fitting-only episode draws for both trajectories. Objective: binary cross entropy plus smooth global AUC/AP losses. Projected trajectory projects the actual Adam update onto two LncDrug memory-gradient half-spaces; check the two finite-step losses and backtrack or skip. Adam moments retain the MiDrug proposal history even on skipped updates.',
        memory='Five fixed fitting-only LncDrug episodes, one per masked graph fold. Global AUC/AP surrogate over their concatenated pseudo-query margins; no validation labels enter gradients or backtracking.',
        outer_policy='Keep selected shared adapter weights fixed. Reuse the frozen parent outer support/preprocessing states only after own eligible selection; no outer reselection or refitting of adapter.',
        selection='LncDrug mean AUC and AP must be >= both historical graph-context and original TabICLv2 references. MiDrug mean AUC and AP must exceed graph-context, each with >=2/3 positive seed differences. Macro drug AP is reported separately. Prefer projected updates, then fewer steps, then larger MiDrug mean AUC/AP. A smoke run cannot qualify.',
        no_test_evaluation=True,
        limitations=['Historical development splits have repeatedly been inspected.',
            'Inference seeds are not independent data or pretrained-model replicates.',
            'Training-memory surrogate constraints do not guarantee validation or future AUC/AP.',
            'Same single-task graph-side-information protocol as the existing reference; no cross-task transfer claim.']))
    (out / 'source_snapshot').mkdir(parents=True)
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), out / 'source_snapshot' / name)
    base.write_json(out / 'protocol.json', p)
    print('FROZEN', out, p['sha256'], flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out) / 'protocol.json'))
    if p['source_sha256'] != sources():
        raise ValueError('Implementation changed after freeze; start a new trial')
    gp = base.check_seal(base.read_json(Path(p['parent_dir']) / 'protocol.json'))
    if gp['sha256'] != p['parent_protocol_sha256'] or base.digest(Path(p['parent_dir']) / 'SELECTION.json') != p['parent_selection_sha256']:
        raise ValueError('Frozen graph-context reference changed')
    for path, sha in p['installed_source_sha256'].items():
        if base.digest(path) != sha:
            raise ValueError('Installed TabICL changed')
    return p


def base_protocol(p):
    gp = base.read_json(Path(p['parent_dir']) / 'protocol.json')
    return base.protocol(gp['parent_dir'])


def training_arrays(p, task):
    info = p['fitting_manifests'][task]
    path = Path(info['path'])
    if base.digest(path) != info['sha256']:
        raise ValueError('Training manifest changed')
    manifest = base.read_json(path)
    for name, sha in manifest['files'].items():
        if base.digest(path.parent / name) != sha:
            raise ValueError('Fitting data changed')
    with np.load(path.parent / 'arrays.npz', allow_pickle=False) as archive:
        keys = ('raw_train', 'x_train', 'y_train', 'pairs_train', 'fold_ids')
        return {key: archive[key].copy() for key in keys}


def bank(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if (out / 'BANK.json').exists():
        checked_bank(out, p)
        print('EXISTING_BANK_VERIFIED', flush=True)
        return
    episodes, records = {}, []
    for task in TASKS:
        a = training_arrays(p, task)
        key = 'raw_train' if p['input_configs'][task]['input_view'] == 'raw' else 'x_train'
        count = p['mi_episodes_per_fold'] if task == 'MiDrug' else 1
        support_size = p['mi_support'] if task == 'MiDrug' else p['lnc_support']
        query_size = p['mi_query'] if task == 'MiDrug' else p['lnc_query']
        episodes[task] = []
        for fold in p['folds']:
            rows = np.flatnonzero(a['fold_ids'] == fold)
            for index in range(count):
                seed = p['training_seed'] + (2000 if task == 'MiDrug' else 5000) + 100 * fold + index
                rng = np.random.default_rng(seed)
                support, query = kernels.sample_episode(a['y_train'], rows, support_size, query_size, rng)
                sampled = np.r_[support, query]
                if len(query) != query_size or any(len(np.unique(a['y_train'][ix])) != 2 for ix in (support, query)):
                    raise ValueError('Episode lacks rows or both classes')
                if len({tuple(v) for v in a['pairs_train'][:, sampled].T}) != len(sampled):
                    raise ValueError('Duplicate association pair in a training episode')
                episode = kernels.prepare_episode(a[key], a['y_train'], support, query,
                    seed=seed, members=p['episode_members'])
                episode['fold'] = fold
                episodes[task].append(episode)
                records.append(dict(task=task, fold=fold, index=index, seed=seed,
                    support=support.tolist(), query=query.tolist(), columns=episode['retained'].tolist(),
                    support_positive=int(a['y_train'][support].sum()), query_positive=int(a['y_train'][query].sum())))
    torch.save(episodes, out / 'episodes.pt')
    base.write_json(out / 'BANK.json', base.seal(dict(protocol_sha256=p['sha256'],
        files={'episodes.pt': base.digest(out / 'episodes.pt')}, records=records,
        loaded_arrays=['raw_train', 'x_train', 'y_train', 'pairs_train', 'fold_ids'],
        same_fold_support_and_queries=True, validation_labels_used=False)))
    print('BANK_SAVED', {task: len(value) for task, value in episodes.items()}, flush=True)


def checked_bank(out, p):
    record = base.check_seal(base.read_json(Path(out) / 'BANK.json'))
    if record['protocol_sha256'] != p['sha256'] or record['validation_labels_used']:
        raise ValueError('Invalid fitting-only bank')
    for name, sha in record['files'].items():
        if base.digest(Path(out) / name) != sha:
            raise ValueError('Episode bank changed')
    return record


def memory_losses(model, adapter, episodes, device):
    values = [kernels.episode_margin(model, adapter, episode, device) for episode in episodes]
    return kernels.ranking_losses(torch.cat([v[0] for v in values]), torch.cat([v[1] for v in values]))


def state_path(out, mode, step):
    return Path(out) / 'adapters' / mode / ('step_%04d.pt' % step)


def checked_training(out, p, mode=None):
    files = {}
    for current in ([mode] if mode else MODES):
        path = Path(out) / ('TRAINING_' + current + '.json')
        record = base.check_seal(base.read_json(path))
        if record['protocol_sha256'] != p['sha256'] or not record['backbone_unchanged']:
            raise ValueError('Invalid training provenance')
        if record['bank_sha256'] != base.digest(Path(out) / 'BANK.json'):
            raise ValueError('Training bank changed')
        for name, sha in record['files'].items():
            if base.digest(Path(out) / name) != sha:
                raise ValueError('Training state changed')
        files[path.name] = base.digest(path)
    return files


def train(args):
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir), protocol(args.out_dir)
    checked_bank(out, p)
    if (out / ('TRAINING_' + args.mode + '.json')).exists():
        checked_training(out, p, args.mode)
        print('EXISTING_TRAINING_VERIFIED', args.mode, flush=True)
        return
    ep = base_protocol(p)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(p['training_seed'])
    device = torch.device(args.device)
    loader = TabICLClassifier(model_path=ep['checkpoint'], allow_auto_download=False,
        n_estimators=1, device=args.device, use_amp=False, use_fa3=False)
    loader._load_model()
    model = loader.model_.to(device).requires_grad_(False)
    before = base.parameter_digest(loader)
    if before != p['parameter_sha256'] or model.dropout != 0:
        raise ValueError('Expected the unchanged deterministic backbone')
    episodes = torch.load(out / 'episodes.pt', map_location='cpu', weights_only=True)
    # Loading the backbone must not change the adapter's shared initialization.
    torch.manual_seed(p['training_seed'])
    adapter = kernels.SharedInputAdapter(rank=p['rank']).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=p['learning_rate'], betas=tuple(p['adam_betas']))
    if sum(parameter.numel() for parameter in adapter.parameters()) != p['parameters']:
        raise ValueError('Unexpected shared parameter count')
    dest = state_path(out, args.mode, 0).parent
    dest.mkdir(parents=True, exist_ok=True)
    kernels.save_adapter(state_path(out, args.mode, 0), adapter, dict(mode=args.mode, step=0, protocol_sha256=p['sha256']))
    files = {str(state_path(out, args.mode, 0).relative_to(out)): base.digest(state_path(out, args.mode, 0))}
    rng = np.random.default_rng(p['training_seed'])
    draws = rng.integers(len(episodes['MiDrug']), size=max(p['steps']))
    logs, started = [], time.monotonic()
    with torch.no_grad():
        initial_memory = memory_losses(model, adapter, episodes['LncDrug'], device).cpu().numpy()
    for step, draw in enumerate(draws, 1):
        old = kernels.parameter_vector(adapter).clone()
        protected_before, gradients = None, None
        if args.mode == 'projected':
            protection = memory_losses(model, adapter, episodes['LncDrug'], device)
            protected_before = protection.detach()
            gradients = torch.stack([kernels.flatten_gradient(protection[0], adapter, retain_graph=True),
                                     kernels.flatten_gradient(protection[1], adapter)])
            del protection
        optimizer.zero_grad(set_to_none=True)
        margin, labels = kernels.episode_margin(model, adapter, episodes['MiDrug'][int(draw)], device)
        loss = kernels.objective(margin, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite MiDrug objective')
        loss.backward()
        grad_norm = float(torch.linalg.vector_norm(torch.cat([v.grad.reshape(-1) for v in adapter.parameters()])))
        if not np.isfinite(grad_norm):
            raise FloatingPointError('Nonfinite MiDrug gradient')
        optimizer.step()
        proposed = kernels.parameter_vector(adapter) - old
        accepted, reductions, after_memory = True, 0, None
        if args.mode == 'projected':
            kernels.set_parameter_vector(adapter, old)
            projected = kernels.project_update(proposed, gradients)
            accepted = False
            for reductions in range(p['max_backtracks'] + 1):
                kernels.set_parameter_vector(adapter, old + projected * (0.5 ** reductions))
                with torch.no_grad():
                    after_memory = memory_losses(model, adapter, episodes['LncDrug'], device)
                if bool(torch.all(after_memory <= protected_before + p['proxy_roundoff_atol'])):
                    accepted = True
                    break
            if not accepted:
                kernels.set_parameter_vector(adapter, old)
                after_memory = protected_before
        if step in p['steps'] and after_memory is None:
            with torch.no_grad():
                after_memory = memory_losses(model, adapter, episodes['LncDrug'], device)
        actual = kernels.parameter_vector(adapter) - old
        record = dict(step=step, episode=int(draw), mi_objective=float(loss.detach()), gradient_norm=grad_norm,
            proposed_norm=float(proposed.norm()), actual_norm=float(actual.norm()), accepted=accepted,
            backtracks=reductions, memory_before=None if protected_before is None else protected_before.cpu().tolist(),
            memory_after=None if after_memory is None else after_memory.cpu().tolist(),
            gradient_dot_actual=None if gradients is None else (gradients @ actual).detach().cpu().tolist())
        logs.append(record)
        del margin, labels, loss, gradients
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError('Frozen backbone accumulated parameter gradients')
        if step in p['steps']:
            path = state_path(out, args.mode, step)
            kernels.save_adapter(path, adapter, dict(mode=args.mode, step=step,
                protocol_sha256=p['sha256'], bank_sha256=base.digest(out / 'BANK.json')))
            files[str(path.relative_to(out))] = base.digest(path)
        if step == 1 or step % 4 == 0 or step in p['steps']:
            print('TRAIN', args.mode, step, 'mi_loss', round(record['mi_objective'], 6),
                'update', round(record['actual_norm'], 7), 'backtracks', reductions,
                'seconds', round(time.monotonic() - started, 1), flush=True)
    if base.parameter_digest(loader) != before:
        raise ValueError('Backbone changed during adapter training')
    log_path = out / ('training_' + args.mode + '.json')
    base.write_json(log_path, dict(initial_memory=initial_memory.tolist(), steps=logs,
        optimizer='Adam proposals; only the actual parameter deltas are projected/backtracked'))
    files[log_path.name] = base.digest(log_path)
    optimizer_path = dest / 'optimizer_last.pt'
    torch.save(optimizer.state_dict(), optimizer_path)
    files[str(optimizer_path.relative_to(out))] = base.digest(optimizer_path)
    base.write_json(out / ('TRAINING_' + args.mode + '.json'), base.seal(dict(
        protocol_sha256=p['sha256'], mode=args.mode, bank_sha256=base.digest(out / 'BANK.json'),
        files=files, backbone_unchanged=True, parameter_sha256=before,
        shared_task_parameters=True, validation_labels_used=False, optimizer_steps=len(logs),
        skipped_steps=sum(not row['accepted'] for row in logs),
        elapsed_seconds=time.monotonic() - started)))
    print('TRAINING_SAVED', args.mode, 'seconds', round(time.monotonic() - started, 1), flush=True)


def selection(out, p):
    choice = base.check_seal(base.read_json(Path(out) / 'SELECTION.json'))
    if choice['protocol_sha256'] != p['sha256'] or not choice['eligible'] or p['smoke']:
        raise ValueError('No eligible formal inner selection; outer loading is prohibited')
    for name, sha in choice['files'].items():
        if base.digest(Path(out) / name) != sha:
            raise ValueError('Selected inner evidence changed')
    return choice


def stage_arrays(out, p, task, phase):
    if task not in TASKS or phase not in ('inner', 'outer'):
        raise ValueError('Unknown task or phase')
    if phase == 'outer':
        selection(out, p)
    gp = graph.protocol(p['parent_dir'])
    a = graph.features(p['parent_dir'], gp, task, phase)
    if p['smoke']:
        keep = np.sort(np.concatenate([np.flatnonzero(a['fold_ids'] == fold)[base.balanced_subset(
            a['y_train'][a['fold_ids'] == fold], 32, p['training_seed'] + fold)] for fold in p['folds']]))
        for prefix in ('raw', 'x', 'y'):
            a[prefix + '_train'] = a[prefix + '_train'][keep]
        a['pairs_train'], a['fold_ids'] = a['pairs_train'][:, keep], a['fold_ids'][keep]
        keep = base.balanced_subset(a['y_val'], 40, p['training_seed'])
        for prefix in ('raw', 'x', 'y'):
            a[prefix + '_val'] = a[prefix + '_val'][keep]
        a['pairs_val'] = a['pairs_val'][:, keep]
        for prefix in ('raw', 'x'):
            a[prefix + '_query_views'] = a[prefix + '_query_views'][:, keep]
    base.validate_arrays(a)
    return a


def candidates(out, p, phase):
    return p['candidates'] if phase == 'inner' else [selection(out, p)['recipe']]


def requested_seeds(args, p):
    if args.seed is None:
        return p['seeds']
    if args.seed not in p['seeds']:
        raise ValueError('Unregistered seed')
    return [args.seed]


def directory(out, phase, task, seed, kind):
    return Path(out) / phase / task / ('s%d' % seed) / kind


def checked_result(dest, p):
    result = base.read_json(Path(dest) / 'result.json')
    if result['protocol_sha256'] != p['sha256']:
        raise ValueError('Result from another trial')
    for name, sha in result['files'].items():
        if base.digest(Path(dest) / name) != sha:
            raise ValueError('Saved result changed')
    return result


def parent_predictions(source, task, kind, labels, pairs):
    """Read the historical graph-context format before expensive inference."""
    with np.load(Path(source) / 'scores.npz', allow_pickle=False) as archive:
        if set(archive.files) != {'views', 'y_val', 'pairs_val'}:
            raise ValueError('Unexpected historical graph-context prediction schema')
        np.testing.assert_array_equal(archive['y_val'], labels)
        np.testing.assert_array_equal(archive['pairs_val'], pairs)
        values = archive['views'].copy()
    if values.shape != (6 if kind == 'full' else 2, len(labels)) or not np.isfinite(values).all():
        raise ValueError('Wrong historical query views')
    if kind == 'full':
        return values if task == 'LncDrug' else values[:1]
    return values[1:2]


def loaded_adapters(out, p, configs, device):
    checked_training(out, p)
    result = {'zero': kernels.SharedInputAdapter(rank=p['rank']).to(device).eval()}
    for cfg in configs:
        adapter, meta = kernels.load_adapter(state_path(out, cfg['mode'], cfg['step']), device)
        if meta['protocol_sha256'] != p['sha256'] or meta['mode'] != cfg['mode'] or meta['step'] != cfg['step']:
            raise ValueError('Adapter checkpoint identity differs')
        result[cfg['id']] = adapter
    return result


def predict(clf, adapter, queries):
    with kernels.install_adapter(clf, adapter):
        return base.query_scores(clf, queries)


def run(args):
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir), protocol(args.out_dir)
    configs = candidates(out, p, args.phase)
    adapters = loaded_adapters(out, p, configs, args.device)
    a = stage_arrays(out, p, args.task, args.phase)
    gp = graph.protocol(p['parent_dir'])
    for seed in requested_seeds(args, p):
        for kind in reference.kinds(args.task):
            dest = directory(out, args.phase, args.task, seed, kind)
            if (dest / 'result.json').exists():
                checked_result(dest, p)
                continue
            started = time.monotonic()
            x, labels, _, views = reference.job_inputs(p, args.task, a, kind)
            source = graph.directory(p['parent_dir'], args.phase, args.task, seed, kind)
            if p['smoke']:
                clf = base.fit_context(base.estimator(dict(n_estimators=p['inference_members']), seed,
                    base_protocol(p), args.device), x, labels)
                source_hash = None
                expected = None
            else:
                graph.checked_result(source, gp)
                expected = parent_predictions(source, args.task, kind, a['y_val'], a['pairs_val'])
                clf = TabICLClassifier.load(source / 'fitted.pkl', device=args.device)
                source_hash = base.digest(source / 'fitted.pkl')
            if clf.n_estimators != p['inference_members'] or clf.random_state != seed:
                raise ValueError('Wrong frozen ensemble identity')
            if len(clf.ensemble_generator_.y_) != len(labels):
                raise ValueError('Wrong support context size')
            if base.parameter_digest(clf) != p['parameter_sha256']:
                raise ValueError('Wrong backbone checkpoint')
            values = {}
            flat = views.reshape(-1, 36)
            print('RUN', args.phase, args.task, seed, kind, len(x), len(flat), flush=True)
            native = base.query_scores(clf, flat) if p['smoke'] else None
            for name, adapter in adapters.items():
                values[name] = predict(clf, adapter, flat).reshape(views.shape[:2])
                print('PREDICTED', args.task, seed, kind, name, round(time.monotonic() - started, 1), flush=True)
            zero_error = None
            if native is not None:
                zero_error = float(np.max(np.abs(values['zero'].reshape(-1) - native)))
                np.testing.assert_array_equal(values['zero'].reshape(-1), native)
            else:
                np.testing.assert_allclose(values['zero'], expected, atol=p['baseline_atol'], rtol=1e-5)
                zero_error = float(np.max(np.abs(values['zero'] - expected)))
            if base.parameter_digest(clf) != p['parameter_sha256']:
                raise ValueError('Backbone changed during input adaptation')
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / 'scores.npz', scores=np.stack(list(values.values())),
                recipe_ids=np.array(list(values)), y_val=a['y_val'], pairs_val=a['pairs_val'])
            clf.save(dest / 'fitted.pkl', save_model_weights=False, save_training_data=True, save_kv_cache=False)
            base.write_json(dest / 'result.json', dict(protocol_sha256=p['sha256'], phase=args.phase,
                task=args.task, seed=seed, kind=kind, configurations=configs,
                source_fitted_sha256=source_hash, baseline_max_abs=zero_error,
                context_rows=len(x), query_views=len(views), elapsed_seconds=time.monotonic() - started,
                files={name: base.digest(dest / name) for name in ('scores.npz', 'fitted.pkl')}))
            print('SAVED', args.phase, args.task, seed, kind, round(time.monotonic() - started, 1), flush=True)
            del clf
            gc.collect()
            torch.cuda.empty_cache()


def verify(args):
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir), protocol(args.out_dir)
    configs = candidates(out, p, args.phase)
    adapters = loaded_adapters(out, p, configs, args.device)
    a = stage_arrays(out, p, args.task, args.phase)
    for seed in requested_seeds(args, p):
        records = []
        for kind in reference.kinds(args.task):
            dest = directory(out, args.phase, args.task, seed, kind)
            checked_result(dest, p)
            _, _, _, views = reference.job_inputs(p, args.task, a, kind)
            flat = views.reshape(-1, 36)
            clf = TabICLClassifier.load(dest / 'fitted.pkl', device=args.device)
            with np.load(dest / 'scores.npz', allow_pickle=False) as archive:
                saved = dict(zip(archive['recipe_ids'].tolist(), archive['scores']))
            error = 0.
            for name, adapter in adapters.items():
                replay = predict(clf, adapter, flat).reshape(views.shape[:2])
                np.testing.assert_allclose(replay, saved[name], atol=p['replay_atol'], rtol=0)
                error = max(error, float(np.max(np.abs(replay - saved[name]))))
            if base.parameter_digest(clf) != p['parameter_sha256']:
                raise ValueError('Replayed backbone changed')
            records.append(dict(task=args.task, seed=seed, kind=kind, max_abs=error,
                result_sha256=base.digest(dest / 'result.json')))
            print('REPLAYED', args.phase, args.task, seed, kind, error, flush=True)
            del clf
            gc.collect()
            torch.cuda.empty_cache()
        base.write_json(out / ('REPLAY_%s_%s_s%d.json' % (args.phase, args.task, seed)),
            base.seal(dict(protocol_sha256=p['sha256'], records=records, passed=True)))


def audit(args):
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir), protocol(args.out_dir)
    configs = candidates(out, p, args.phase)
    adapters = loaded_adapters(out, p, configs, args.device)
    a = stage_arrays(out, p, args.task, args.phase)
    seed = p['seeds'][0]
    dest = directory(out, args.phase, args.task, seed, 'full')
    checked_result(dest, p)
    _, _, _, views = reference.job_inputs(p, args.task, a, 'full')
    flat = views.reshape(-1, 36)
    rng = np.random.default_rng(p['training_seed'])
    ix = rng.permutation(len(flat))[:min(65, len(flat))]
    ix = np.r_[ix, ix[:3]]
    clf = TabICLClassifier.load(dest / 'fitted.pkl', device=args.device)
    with np.load(dest / 'scores.npz', allow_pickle=False) as archive:
        full = {name: value.reshape(-1) for name, value in zip(archive['recipe_ids'].tolist(), archive['scores'])}
    scores, records = {}, []
    for name, adapter in adapters.items():
        scores[name] = predict(clf, adapter, flat[ix])
        probability_error = float(np.max(np.abs(scores[name] - full[name][ix])))
        effect_error = float(np.max(np.abs((scores[name] - scores['zero']) - (full[name][ix] - full['zero'][ix]))))
        np.testing.assert_allclose(scores[name], full[name][ix], atol=p['query_atol'], rtol=1e-5)
        np.testing.assert_allclose(scores[name] - scores['zero'], full[name][ix] - full['zero'][ix],
            atol=p['query_effect_atol'], rtol=1e-5)
        records.append(dict(recipe=name, probability_max_abs=probability_error, effect_max_abs=effect_error))
    base.write_json(out / ('QUERY_AUDIT_%s_%s.json' % (args.phase, args.task)), base.seal(dict(
        protocol_sha256=p['sha256'], passed=True, task=args.task, records=records,
        includes_reordering_subsetting_and_duplicates=True, result_sha256=base.digest(dest / 'result.json'))))
    print('QUERY_AUDIT_PASSED', args.phase, args.task, max(row['effect_max_abs'] for row in records), flush=True)


def assembled_scores(out, p, phase, task, seed, a):
    configs = candidates(out, p, phase)
    names = ['zero'] + [cfg['id'] for cfg in configs]
    states, hashes = {}, {}
    for kind in reference.kinds(task):
        dest = directory(out, phase, task, seed, kind)
        checked_result(dest, p)
        with np.load(dest / 'scores.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['recipe_ids'], names)
            np.testing.assert_array_equal(archive['y_val'], a['y_val'])
            np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
            states[kind] = archive['scores'].copy()
        hashes[str((dest / 'result.json').relative_to(out))] = base.digest(dest / 'result.json')
    original = states['full'][:, 0]
    component = states['full'][:, 1:].mean(1) if task == 'LncDrug' else np.stack(
        [states['fold%d' % fold][:, 0] for fold in p['folds']]).mean(0)
    return dict(zip(names, .5 * original + .5 * component)), original[0], hashes


def metric_table(out, p, phase):
    rows, files = [], {}
    gp = graph.protocol(p['parent_dir'])
    for task in TASKS:
        a = stage_arrays(out, p, task, phase)
        for seed in p['seeds']:
            scores, own_original, hashes = assembled_scores(out, p, phase, task, seed, a)
            files.update(hashes)
            if p['smoke']:
                original, anchor = own_original, scores['zero']
            else:
                original, components, _ = graph.components(p['parent_dir'], gp, phase, task, seed, a)
                anchor = graph_math.combine(original, components, p['graph_recipes'][task])
                np.testing.assert_allclose(scores['zero'], anchor, atol=p['baseline_atol'], rtol=1e-5)
            for name, value in [('original_tabicl', original), ('graph_reference', anchor), *scores.items()]:
                rows.append(dict(task=task, seed=seed, recipe=name, **base.metrics(a['y_val'], value, a['pairs_val'])))
    table = pd.DataFrame(rows)
    table.to_csv(Path(out) / (phase + '_metrics.csv'), index=False)
    table.groupby(['task', 'recipe'])[list(METRICS)].agg(['mean', 'std']).to_csv(Path(out) / (phase + '_summary.csv'))
    return table, files


def collect_verification(out, p, phase):
    files, records = checked_training(out, p), []
    for task in TASKS:
        for seed in p['seeds']:
            path = Path(out) / ('REPLAY_%s_%s_s%d.json' % (phase, task, seed))
            record = base.check_seal(base.read_json(path))
            if record['protocol_sha256'] != p['sha256'] or not record['passed']:
                raise ValueError('Missing verified inference replay')
            expected = set(reference.kinds(task))
            if {entry['kind'] for entry in record['records']} != expected or len(record['records']) != len(expected):
                raise ValueError('Incomplete replay')
            for entry in record['records']:
                result = directory(out, phase, task, seed, entry['kind']) / 'result.json'
                if (entry['task'], entry['seed']) != (task, seed) or entry['result_sha256'] != base.digest(result):
                    raise ValueError('Replayed result changed')
            files[path.name] = base.digest(path)
            records.extend(record['records'])
        path = Path(out) / ('QUERY_AUDIT_%s_%s.json' % (phase, task))
        audit_record = base.check_seal(base.read_json(path))
        result = directory(out, phase, task, p['seeds'][0], 'full') / 'result.json'
        if audit_record['protocol_sha256'] != p['sha256'] or not audit_record['passed'] or audit_record['result_sha256'] != base.digest(result):
            raise ValueError('Invalid query independence audit')
        files[path.name] = base.digest(path)
    return files, records


def assess(table, cfg, p):
    means = table.groupby(['task', 'recipe'])[list(METRICS)].mean()
    deltas, failures = {}, []
    for task in TASKS:
        deltas[task] = {}
        for metric in METRICS:
            value = float(means.loc[(task, cfg['id']), metric])
            anchor = float(means.loc[(task, 'graph_reference'), metric])
            original = float(means.loc[(task, 'original_tabicl'), metric])
            values = table[(table.task == task) & (table.recipe == cfg['id'])].sort_values('seed')[metric].to_numpy()
            references = table[(table.task == task) & (table.recipe == 'graph_reference')].sort_values('seed')[metric].to_numpy()
            difference = values - references
            deltas[task][metric] = dict(vs_graph=value - anchor, vs_original=value - original,
                positive_seeds=int(np.sum(difference > 0)), minimum_seed_delta=float(difference.min()))
            if not np.isfinite(value):
                failures.append(task + ':' + metric + ':nonfinite')
            elif metric in METRICS[:2]:
                if task == 'LncDrug' and value < max(anchor, original):
                    failures.append(task + ':' + metric)
                if task == 'MiDrug' and (value <= anchor or np.sum(difference > 0) < (1 if p['smoke'] else 2)):
                    failures.append(task + ':' + metric)
    return dict(recipe=cfg, eligible=not failures, failed_conditions=failures, deltas=deltas,
        criterion=float(means.loc[('MiDrug', cfg['id']), ['auc', 'ap']].mean()))


def select(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if (out / 'SELECTION.json').exists():
        raise ValueError('Selection already frozen')
    files, records = collect_verification(out, p, 'inner')
    table, result_files = metric_table(out, p, 'inner')
    files.update(result_files)
    rows = [assess(table, cfg, p) for cfg in p['candidates']]
    eligible = [row for row in rows if row['eligible'] and not p['smoke']]
    best = min(eligible, key=lambda row: (row['recipe']['mode'] != 'projected', row['recipe']['step'], -row['criterion'])) if eligible else None
    for name in ('inner_metrics.csv', 'inner_summary.csv'):
        files[name] = base.digest(out / name)
    choice = base.seal(dict(protocol_sha256=p['sha256'], eligible=best is not None,
        recipe=best['recipe'] if best else {'id': 'zero'}, candidates=rows, files=files,
        protected_lnc_metrics=['auc', 'ap'], macro_drug_ap='report separately',
        smoke=p['smoke'], no_test_evaluation=True, replayed_context_states=len(records)))
    base.write_json(out / 'SELECTION.json', choice)
    print('SELECTED', choice['recipe'], 'eligible', choice['eligible'], flush=True)
    for row in rows:
        print(row['recipe']['id'], row['failed_conditions'], row['deltas'], flush=True)


def report(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    choice = base.check_seal(base.read_json(out / 'SELECTION.json'))
    table, _ = metric_table(out, p, args.phase)
    _, replay = collect_verification(out, p, args.phase)
    means = table.groupby(['task', 'recipe'])[list(METRICS)].mean()
    outcome = dict(protocol_sha256=p['sha256'], eligible_inner=choice['eligible'], phase=args.phase,
        smoke=p['smoke'], no_test_evaluation=True, replayed_context_states=len(replay),
        max_replay_error=max(row['max_abs'] for row in replay), active_reference='Existing graph-context TabICLv2')
    if args.phase == 'outer':
        result = assess(table, choice['recipe'], p)
        outcome.update(passed_outer=result['eligible'], deltas=result['deltas'], failed_conditions=result['failed_conditions'])
    lines = ['# TabICLv2 共享输入适配实验', '',
        '工程短流程；分数不作为性能证据。' if p['smoke'] else '历史开发划分；推理种子 666/0/1，每种子 32 个成员。', '',
        '内部通过：`%s`；选择：`%s`。' % (choice['eligible'], choice['recipe']['id']),
        'LncDrug AUC、AP 逐项不得低于历史图上下文与原始 TabICLv2 参考；按药物宏 AP 单独报告。', '',
        '| 任务 | 方案 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for (task, recipe), row in means.iterrows():
        lines.append('| %s | %s | %.9f | %.9f | %.9f |' % (task, recipe, row.auc, row.ap, row.macro_drug_ap))
    lines += ['', '第一轮固定 4 个候选：无约束/投影更新，各取第 16/64 步（工程短流程为第 1/2 步）。',
        '适配器共 496 参数，只作用于两任务前 31 个语义对应列；后 5 列沿用原输入。',
        '支持拟合预处理、列过滤和特征置换均显式对齐；每个候选重新计算全部编码，未复用旧表示缓存。',
        '两个任务、全部支持/查询和图分支共用新增参数，没有按任务身份回退。',
        '无约束与投影轨迹使用相同初始权重和 MiDrug 训练 episode 序列。投影约束取自五个固定 LncDrug 拟合 episode，且检查有限步长后的代理损失。',
        '保护代理是训练记忆的平滑 AUC/AP，不是验证 AUC/AP；优化器不读取验证标签。',
        '比较使用历史参考的原始分数；单独列出本轮零适配复算，避免将浮点重算误差算作改进。', '',
        '已完整重放 %d 个上下文状态，每个状态的全部候选均复算，最大误差 %.3g。' % (len(replay), outcome['max_replay_error']),
        '通过查询重排、子集和重复行检查。原始测试集未使用。', '',
        '历史内外层分区已反复用于开发；推理种子不能当作独立数据重复。此轮不提供跨任务迁移或未来分布不下降的保证。']
    if args.phase == 'outer':
        lines += ['', '外层硬约束通过：`%s`。未依据外层结果重新选择参数。' % outcome['passed_outer']]
    elif not choice['eligible']:
        lines += ['', '没有候选通过内部硬约束，本轮不进入外层验证，现有模型保留。']
    (out / ('REPORT_' + args.phase + '.md')).write_text('\n'.join(lines) + '\n')
    base.write_json(out / ('OUTCOME_' + args.phase + '.json'), base.seal(outcome))
    print('REPORT_SAVED', args.phase, outcome, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'bank', 'train', 'run', 'verify', 'audit', 'select', 'report'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--parent-dir', default=reference.DEFAULT_PARENT)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=MODES, default='projected')
    parser.add_argument('--task', choices=TASKS, default='MiDrug')
    parser.add_argument('--phase', choices=('inner', 'outer'), default='inner')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
