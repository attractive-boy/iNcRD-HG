"""Train and evaluate one conditional relation-attention adapter for both RNA tasks.

prepare -> cache (both tasks) -> train -> run inner -> verify -> select
An eligible frozen selection alone permits outer evaluation. No original test
array is loaded, and no inner validation target is used by the optimizer.
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
from sklearn.model_selection import StratifiedKFold

from improve import tabular_foundation as base
from improve import tabicl_graph_context as graph
from improve import tabicl_graph_context_modules as graph_math
from improve import tabicl_reli as reference
from improve import tabicl_care_modules as care


DEFAULT_OUT = 'improve/results/tabicl_care_20260913'
TASKS = ('LncDrug', 'MiDrug')
OWN = ('tabicl_care.py', 'tabicl_care_modules.py', 'test_tabicl_care.py')
DEPENDENCIES = ('tabular_foundation.py', 'tabicl_graph_context.py',
                'tabicl_graph_context_modules.py', 'tabicl_reli.py', 'tabicl_reli_modules.py')
METRICS = ('auc', 'ap', 'macro_drug_ap')


def sources():
    return {name: base.digest(Path(__file__).with_name(name)) for name in (*OWN, *DEPENDENCIES)}


def identifier(trust, step, strength):
    return 'care_t%g_e%d_a%s' % (trust, step, str(strength).replace('.', 'p'))


def prepare(args):
    out, parent, inputs = (Path(v).resolve() for v in (args.out_dir, args.parent_dir, args.inputs_dir))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Preserve existing runs; use a fresh output directory')
    gp = graph.protocol(parent)
    selected = graph.selection(parent, gp)
    expected = dict(LncDrug='query_bag_050', MiDrug='fold_match_050')
    if any(selected['chosen'][t]['recipe']['id'] != expected[t] for t in TASKS):
        raise ValueError('Unexpected frozen graph-context reference')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        parent_dir=str(parent), parent_protocol_sha256=gp['sha256'],
        parent_selection_sha256=base.digest(parent / 'SELECTION.json'),
        inputs_dir=str(inputs), kernels_sha256=base.digest(inputs / 'kernels.npz'),
        source_sha256=sources(), input_configs=gp['input_configs'], parameter_sha256=gp['parameter_sha256'],
        tasks=list(TASKS), seeds=[666] if args.smoke else [666, 0, 1],
        inference_members=2 if args.smoke else 32, episode_members=2 if args.smoke else 4,
        episode_folds=[0] if args.smoke else list(range(5)), episode_query_cap=24 if args.smoke else 256,
        training_seed=20260913, steps=[2, 4] if args.smoke else [50, 200],
        trusts=[1.] if args.smoke else [1., 10.], strengths=[.125, 1.] if args.smoke else [.03125, .125, .5, 1.],
        learning_rate=.01, weight_decay=.001, training_query_batch=128,
        training_attention_backend='PyTorch mathematical SDPA for supported mask gradients; official kernels retained for inference.',
        training_modes=['conditional', 'pooled', 'constant'], protect_macro=args.protect_macro,
        graph_recipes={t: selected['chosen'][t]['recipe'] for t in TASKS},
        mechanism='One learned query-conditioned, bounded relation bias in final ICL attention. Both tasks share every adapter parameter; no task ID, RNA type, length or identity embedding enters the controller.',
        training_policy='All pseudo-query and context rows belong to the same original graph fold, whose entire target-edge set is excluded from their feature graph. Two complementary stratified half-fold episodes. Only fitting labels are loaded. Both tasks alternate equally; LncDrug has fourfold teacher KL weight, with the same supervised loss. Backbone stays frozen.',
        selection_policy='LncDrug mean AUC/AP must be no lower than either original TabICLv2 or graph-context reference; optionally macro drug AP too. MiDrug mean AUC/AP must exceed graph-context, each with at least two positive paired inference seeds. Among eligible candidates prefer lower strength, greater trust, fewer training steps, then larger MiDrug criterion. No outer reselection.',
        limitations=['Historical split has already been used repeatedly for development.',
            'Inference seeds share one checkpoint and one data split; they are not independent generalization replicates.',
            'Shared weights are trained on both tasks, not a demonstrated cross-task transfer benchmark.',
            'Future-distribution non-degradation cannot be guaranteed by this finite evaluation.'],
        no_test_evaluation=True, replay_atol=1e-6, query_atol=5e-5, query_effect_atol=1e-5)
    p['candidates'] = [dict(id=identifier(t, e, a), trust=t, step=e, strength=a, mode='conditional')
                       for t in p['trusts'] for e in p['steps'] for a in p['strengths']]
    p = base.seal(p)
    (out / 'source_snapshot').mkdir(parents=True)
    for name in (*OWN, *DEPENDENCIES):
        shutil.copy2(Path(__file__).with_name(name), out / 'source_snapshot' / name)
    base.write_json(out / 'protocol.json', p)
    print('FROZEN', out, p['sha256'], flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out) / 'protocol.json'))
    if p['source_sha256'] != sources():
        raise ValueError('Source changed after protocol freeze; start a new run')
    if base.digest(Path(p['inputs_dir']) / 'kernels.npz') != p['kernels_sha256']:
        raise ValueError('Frozen entity evidence changed')
    gp = base.check_seal(base.read_json(Path(p['parent_dir']) / 'protocol.json'))
    if gp['sha256'] != p['parent_protocol_sha256'] or base.digest(Path(p['parent_dir']) / 'SELECTION.json') != p['parent_selection_sha256']:
        raise ValueError('Frozen baseline changed')
    return p


def evidence(p, task, mode='conditional'):
    key = 'pooled' if mode == 'pooled' else 'local'
    with np.load(Path(p['inputs_dir']) / 'kernels.npz', allow_pickle=False) as archive:
        return care.RelationEvidence(archive[task + '_' + key], archive['drug_' + key],
                                     archive[task + '_present'], archive['drug_present'])


def training_arrays(p, task):
    """Deliberately load only fitting arrays; do not deserialize validation targets."""
    gp = base.read_json(Path(p['parent_dir']) / 'protocol.json')
    dest = Path(gp['feature_reference_dir']) / 'features' / 'inner' / task
    m = base.read_json(dest / 'manifest.json')
    for name, expected in m['files'].items():
        if base.digest(dest / name) != expected:
            raise ValueError('Training feature cache changed')
    with np.load(dest / 'arrays.npz', allow_pickle=False) as archive:
        keys = ('raw_train', 'x_train', 'y_train', 'pairs_train', 'fold_ids')
        data = {key: archive[key].copy() for key in keys}
    return data


def base_protocol(p):
    gp = base.read_json(Path(p['parent_dir']) / 'protocol.json')
    return base.protocol(gp['parent_dir'])


def cache(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    a, ep = training_arrays(p, args.task), base_protocol(p)
    prefix = 'raw' if p['input_configs'][args.task]['input_view'] == 'raw' else 'x'
    local, pooled = evidence(p, args.task), evidence(p, args.task, 'pooled')
    dest = out / 'training_cache' / args.task
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'manifest.json').exists():
        checked_cache(dest, p)
        print('CACHE_ALREADY_DONE', args.task, flush=True)
        return
    records = []
    for fold in p['episode_folds']:
        rows = np.flatnonzero(a['fold_ids'] == fold)
        if p['smoke']:
            rows = rows[base.balanced_subset(a['y_train'][rows], 80, p['training_seed'] + fold)]
        split = StratifiedKFold(2, shuffle=True, random_state=p['training_seed'] + fold)
        for episode, (context, query) in enumerate(split.split(rows, a['y_train'][rows])):
            support = rows[context]
            query = rows[query]
            query = query[base.balanced_subset(a['y_train'][query], p['episode_query_cap'], p['training_seed'] + episode)]
            if np.intersect1d(support, query).size or not np.all(a['fold_ids'][np.r_[support, query]] == fold):
                raise ValueError('Training episode crossed its masked support graph')
            relation, info = local.features(a['pairs_train'][:, query], a['pairs_train'][:, support])
            relation_pooled, info_pooled = pooled.features(a['pairs_train'][:, query], a['pairs_train'][:, support])
            clf = base.fit_context(base.estimator(dict(n_estimators=p['episode_members']),
                p['training_seed'] + fold * 2 + episode, ep, args.device), a[prefix + '_train'][support], a['y_train'][support])
            if base.parameter_digest(clf) != p['parameter_sha256']:
                raise ValueError('Unexpected pretrained model')
            item = care.capture_training_episode(clf, a[prefix + '_train'][query], relation, info, a['y_train'][query])
            item.update(relation_pooled=torch.from_numpy(relation_pooled), evidence_pooled=torch.from_numpy(info_pooled),
                        support_indices=torch.from_numpy(support), query_indices=torch.from_numpy(query))
            name = 'f%d_e%d.pt' % (fold, episode)
            torch.save(item, dest / name)
            records.append(dict(file=name, sha256=base.digest(dest / name), fold=fold, episode=episode,
                                support_rows=len(support), query_rows=len(query),
                                context_query_overlap=0, all_targets_excluded_from_feature_graph=True))
            print('CACHED', args.task, name, len(support), len(query), flush=True)
            del clf, item
            gc.collect()
            torch.cuda.empty_cache()
    base.write_json(dest / 'manifest.json', base.seal(dict(protocol_sha256=p['sha256'], task=args.task,
        episodes=records, loaded_array_keys=['raw_train', 'x_train', 'y_train', 'pairs_train', 'fold_ids'],
        validation_labels_loaded=False, no_test_evaluation=True)))


def checked_cache(dest, p):
    m = base.check_seal(base.read_json(dest / 'manifest.json'))
    if m['protocol_sha256'] != p['sha256'] or m['validation_labels_loaded']:
        raise ValueError('Invalid training cache')
    for row in m['episodes']:
        if base.digest(dest / row['file']) != row['sha256']:
            raise ValueError('Training episode changed')
    return m


def state_path(out, mode, trust, step):
    return Path(out) / 'adapters' / mode / ('trust%g' % trust) / ('step%04d.pt' % step)


def train(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if (out / 'TRAINING.json').exists():
        checked_training(out, p)
        print('TRAINING_ALREADY_DONE', flush=True)
        return
    episodes, manifests = {}, {}
    for task in TASKS:
        dest = out / 'training_cache' / task
        m = checked_cache(dest, p)
        manifests[task] = base.digest(dest / 'manifest.json')
        episodes[task] = [torch.load(dest / row['file'], map_location='cpu', weights_only=True) for row in m['episodes']]
    a, ep = training_arrays(p, 'MiDrug'), base_protocol(p)
    keep = base.balanced_subset(a['y_train'], 16, p['training_seed'])
    clf = base.fit_context(base.estimator(dict(n_estimators=2), p['training_seed'], ep, args.device),
                           a['raw_train'][keep], a['y_train'][keep])
    clf.model_.to(args.device).eval().requires_grad_(False)
    predictor = clf.model_.icl_predictor
    files, paths = {}, []
    for mode in p['training_modes']:
        for trust in p['trusts']:
            torch.manual_seed(p['training_seed'])
            rng = np.random.default_rng(p['training_seed'])
            adapter = care.ConditionalRelationAdapter(conditional=mode != 'constant').to(args.device)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
            trace, started = [], time.monotonic()
            for step in range(1, max(p['steps']) + 1):
                task = TASKS[(step - 1) % len(TASKS)]
                # Episode sampling is independent of validation metrics.
                item = episodes[task][int(rng.integers(len(episodes[task])))]
                selected = np.sort(rng.choice(len(item['target']), min(p['training_query_batch'], len(item['target'])), replace=False))
                cache_ = {key: item[key][:, selected].to(args.device) if key in ('query', 'native_logits') else item[key].to(args.device)
                          for key in ('query', 'native_logits', 'key', 'value')}
                suffix = '_pooled' if mode == 'pooled' else ''
                relation, info = item['relation' + suffix][selected].to(args.device), item['evidence' + suffix][selected].to(args.device)
                optimizer.zero_grad(set_to_none=True)
                logits = care.adapted_logits(predictor, cache_, relation, info, adapter)
                loss, terms = care.adaptation_loss(logits, cache_['native_logits'], item['target'][selected].to(args.device),
                    item['permutations'].to(args.device), trust=trust * (4. if task == 'LncDrug' else 1.))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite adapter objective')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                trace.append(dict(step=step, task=task, loss=float(loss.detach()), grad_norm=float(norm),
                                  **{k: float(v.detach()) for k, v in terms.items()}))
                if step in p['steps']:
                    path = state_path(out, mode, trust, step)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    state = dict(config=adapter.config(), state_dict={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                                 mode=mode, trust=trust, step=step, protocol_sha256=p['sha256'],
                                 training_seed=p['training_seed'], tasks=list(TASKS))
                    torch.save(state, path)
                    files[str(path.relative_to(out))] = base.digest(path)
                if step % 50 == 0 or step == max(p['steps']):
                    print('TRAIN', mode, trust, step, 'loss', round(float(loss.detach()), 6),
                          'seconds', round(time.monotonic() - started, 1), flush=True)
            trace_path = state_path(out, mode, trust, 0).parent / 'trace.csv'
            pd.DataFrame(trace).to_csv(trace_path, index=False)
            files[str(trace_path.relative_to(out))] = base.digest(trace_path)
            paths.append(dict(mode=mode, trust=trust, seconds=time.monotonic() - started,
                              trainable_parameters=sum(v.numel() for v in adapter.parameters() if v.requires_grad)))
    if base.parameter_digest(clf) != p['parameter_sha256'] or any(v.grad is not None for v in clf.model_.parameters()):
        raise ValueError('The backbone was modified by adaptation')
    base.write_json(out / 'TRAINING.json', base.seal(dict(protocol_sha256=p['sha256'], files=files,
        cache_manifests=manifests, trajectories=paths, backbone_unchanged=True, shared_task_parameters=True,
        validation_labels_used=False, outer_refit_policy='Shared adapter stays frozen; outer TabICL contexts use the full fitting partition.')))


def checked_training(out, p):
    m = base.check_seal(base.read_json(Path(out) / 'TRAINING.json'))
    if m['protocol_sha256'] != p['sha256'] or m['validation_labels_used'] or not m['backbone_unchanged']:
        raise ValueError('Invalid adapter training record')
    for name, expected in m['files'].items():
        if base.digest(Path(out) / name) != expected:
            raise ValueError('Frozen adapter state changed')
    return m


def selection(out, p):
    s = base.check_seal(base.read_json(Path(out) / 'SELECTION.json'))
    if s['protocol_sha256'] != p['sha256'] or not s['eligible'] or p['smoke']:
        raise ValueError('No eligible formal inner selection; outer loading is prohibited')
    for name, expected in s['files'].items():
        if base.digest(Path(out) / name) != expected:
            raise ValueError('Inner evidence changed after selection')
    return s


def stage_arrays(out, p, task, phase):
    if task not in TASKS or phase not in ('inner', 'outer'):
        raise ValueError('Unknown task or evaluation phase')
    if phase == 'outer':
        selection(out, p)  # Must precede all parent feature and outer array IO.
    gp = graph.protocol(p['parent_dir'])
    a = graph.features(p['parent_dir'], gp, task, phase)
    if p['smoke']:
        keep = np.sort(np.concatenate([np.flatnonzero(a['fold_ids'] == fold)[base.balanced_subset(
            a['y_train'][a['fold_ids'] == fold], 32, p['training_seed'] + fold)] for fold in range(5)]))
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


def recipes(out, p, phase, controls=False):
    if phase == 'inner' and not controls:
        return p['candidates']
    chosen = selection(out, p)['recipe']
    return [chosen, *[dict(chosen, id=mode + '_control', mode=mode) for mode in ('pooled', 'constant')]]


def loaded_variants(out, p, configs, device):
    checked_training(out, p)
    return {cfg['id']: (care.load_adapter(state_path(out, cfg['mode'], cfg['trust'], cfg['step']), device)[0], cfg['strength'])
            for cfg in configs}


def directory(out, phase, task, seed, kind, controls=False):
    return Path(out) / (phase + ('_controls' if controls else '')) / task / ('s%d' % seed) / kind


def checked_result(dest, p):
    r = base.read_json(dest / 'result.json')
    if r['protocol_sha256'] != p['sha256']:
        raise ValueError('Result belongs to another protocol')
    for name, expected in r['files'].items():
        if base.digest(dest / name) != expected:
            raise ValueError('Saved inference state changed')
    return r


def infer(clf, p, task, a, kind, configs, variants):
    _, _, support_pairs, queries = reference.job_inputs(p, task, a, kind)
    query_pairs = np.tile(a['pairs_val'], (1, len(queries)))
    values, numerical = {}, {}
    for mode in dict.fromkeys(cfg['mode'] for cfg in configs):
        relation, info = evidence(p, task, mode).features(query_pairs, support_pairs)
        names = [cfg['id'] for cfg in configs if cfg['mode'] == mode]
        predictions, audit = care.predict_variants(clf, queries.reshape(-1, 36), relation, info,
                                                   {name: variants[name] for name in names})
        if 'zero' in values:
            np.testing.assert_array_equal(values['zero'], predictions['zero'])
        values.update(predictions)
        numerical[mode] = audit
    return values, numerical, queries


def run(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    configs = recipes(out, p, args.phase, args.controls)
    variants = loaded_variants(out, p, configs, args.device)
    a, ep = stage_arrays(out, p, args.task, args.phase), base_protocol(p)
    for seed in p['seeds']:
        for kind in reference.kinds(args.task):
            dest = directory(out, args.phase, args.task, seed, kind, args.controls)
            if (dest / 'result.json').exists():
                checked_result(dest, p)
                continue
            started = time.monotonic()
            x, labels, _, _ = reference.job_inputs(p, args.task, a, kind)
            clf = base.fit_context(base.estimator(dict(n_estimators=p['inference_members']), seed, ep, args.device), x, labels)
            print('RUN', args.phase, args.task, seed, kind, 'support', len(x), flush=True)
            scores, numerical, queries = infer(clf, p, args.task, a, kind, configs, variants)
            if base.parameter_digest(clf) != p['parameter_sha256']:
                raise ValueError('Backbone changed during inference')
            names = ['zero'] + [cfg['id'] for cfg in configs]
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / 'scores.npz', scores=np.stack([scores[name].reshape(queries.shape[:2]) for name in names]),
                                recipe_ids=np.array(names), pairs_val=a['pairs_val'], y_val=a['y_val'])
            clf.save(dest / 'fitted.pkl', save_model_weights=False, save_training_data=True, save_kv_cache=False)
            base.write_json(dest / 'result.json', dict(protocol_sha256=p['sha256'], task=args.task, phase=args.phase,
                seed=seed, kind=kind, configurations=configs, numerical=numerical, context_rows=len(x),
                elapsed_seconds=time.monotonic() - started,
                files={name: base.digest(dest / name) for name in ('scores.npz', 'fitted.pkl')}))
            print('SAVED', args.task, seed, kind, round(time.monotonic() - started, 1), flush=True)
            del clf
            gc.collect()
            torch.cuda.empty_cache()


def assembled_scores(out, p, phase, task, seed, a, controls=False):
    configs = recipes(out, p, phase, controls)
    names, states, hashes = ['zero'] + [cfg['id'] for cfg in configs], {}, {}
    for kind in reference.kinds(task):
        dest = directory(out, phase, task, seed, kind, controls)
        checked_result(dest, p)
        with np.load(dest / 'scores.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['recipe_ids'], names)
            np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
            np.testing.assert_array_equal(archive['y_val'], a['y_val'])
            states[kind] = archive['scores'].copy()
        hashes[str((dest / 'result.json').relative_to(out))] = base.digest(dest / 'result.json')
    original = states['full'][:, 0]
    component = states['full'][:, 1:].mean(1) if task == 'LncDrug' else np.stack(
        [states['fold%d' % fold][:, 0] for fold in range(5)]).mean(0)
    combined = .5 * original + .5 * component
    return dict(zip(names, combined)), original[0], hashes


def verify(args):
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir), protocol(args.out_dir)
    configs = recipes(out, p, args.phase, args.controls)
    variants = loaded_variants(out, p, configs, args.device)
    records = []
    for task in ([args.task] if args.task else TASKS):
        a = stage_arrays(out, p, task, args.phase)
        for seed in p['seeds']:
            for kind in reference.kinds(task):
                dest = directory(out, args.phase, task, seed, kind, args.controls)
                checked_result(dest, p)
                clf = TabICLClassifier.load(dest / 'fitted.pkl', device=args.device)
                scores, numerical, queries = infer(clf, p, task, a, kind, configs, variants)
                names = ['zero'] + [cfg['id'] for cfg in configs]
                with np.load(dest / 'scores.npz', allow_pickle=False) as archive:
                    replay = np.stack([scores[name].reshape(queries.shape[:2]) for name in names])
                    np.testing.assert_allclose(replay, archive['scores'], atol=p['replay_atol'], rtol=0)
                    error = float(np.max(np.abs(replay - archive['scores'])))
                records.append(dict(task=task, seed=seed, kind=kind, max_abs=error,
                    result_sha256=base.digest(dest / 'result.json'), numerical=numerical))
                print('REPLAYED', args.phase, task, seed, kind, error, flush=True)
                del clf
                gc.collect()
                torch.cuda.empty_cache()
    name = 'REPLAY_%s%s%s.json' % (args.phase, '_controls' if args.controls else '', '_' + args.task if args.task else '')
    base.write_json(out / name, base.seal(dict(protocol_sha256=p['sha256'], records=records, passed=True)))


def audit(args):
    """Audit query order/subsets against the saved full, label-free predictions."""
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir), protocol(args.out_dir)
    configs = recipes(out, p, args.phase, args.controls)
    variants = loaded_variants(out, p, configs, args.device)
    for task in ([args.task] if args.task else TASKS):
        a = stage_arrays(out, p, task, args.phase)
        dest = directory(out, args.phase, task, p['seeds'][0], 'full', args.controls)
        checked_result(dest, p)
        _, _, support, queries = reference.job_inputs(p, task, a, 'full')
        flat_queries = queries.reshape(-1, 36)
        pairs = np.tile(a['pairs_val'], (1, len(queries)))
        idx = np.random.default_rng(p['training_seed']).permutation(len(flat_queries))[:min(257, len(flat_queries))]
        # Duplicates must receive the same score, without changing other rows.
        idx = np.r_[idx, idx[:3]]
        clf = TabICLClassifier.load(dest / 'fitted.pkl', device=args.device)
        with np.load(dest / 'scores.npz', allow_pickle=False) as archive:
            full = {name: value.reshape(-1) for name, value in zip(archive['recipe_ids'].tolist(), archive['scores'])}
        records = []
        for mode in dict.fromkeys(cfg['mode'] for cfg in configs):
            relation, info = evidence(p, task, mode).features(pairs[:, idx], support)
            names = [cfg['id'] for cfg in configs if cfg['mode'] == mode]
            scores, numerical = care.predict_variants(clf, flat_queries[idx], relation, info,
                                                      {name: variants[name] for name in names})
            for name, score in scores.items():
                np.testing.assert_allclose(score, full[name][idx], atol=p['query_atol'], rtol=1e-5)
                effect = score - scores['zero']
                full_effect = full[name][idx] - full['zero'][idx]
                np.testing.assert_allclose(effect, full_effect, atol=p['query_effect_atol'], rtol=1e-5)
                records.append(dict(recipe=name, query_probability_max_abs=float(np.max(np.abs(score - full[name][idx]))),
                    query_effect_max_abs=float(np.max(np.abs(effect - full_effect))), numerical=numerical))
        suffix = '_controls' if args.controls else ''
        base.write_json(out / ('QUERY_AUDIT_' + args.phase + suffix + '_' + task + '.json'), base.seal(dict(
            protocol_sha256=p['sha256'], task=task, query_rows=len(idx), records=records, passed=True,
            result_sha256=base.digest(dest / 'result.json'))))
        print('QUERY_AUDIT_PASSED', args.phase, task, len(idx), max(r['query_effect_max_abs'] for r in records), flush=True)
        del clf
        gc.collect()
        torch.cuda.empty_cache()


def metric_table(out, p, phase, controls=False):
    rows, hashes = [], {}
    gp = base.read_json(Path(p['parent_dir']) / 'protocol.json')
    for task in TASKS:
        a = stage_arrays(out, p, task, phase)
        for seed in p['seeds']:
            scores, original, result_hashes = assembled_scores(out, p, phase, task, seed, a, controls)
            hashes.update(result_hashes)
            for name, values in [('original_tabicl', original), *scores.items()]:
                rows.append(dict(task=task, seed=seed, recipe=name, **base.metrics(a['y_val'], values, a['pairs_val'])))
            if not p['smoke']:
                parent_original, components, _ = graph.components(p['parent_dir'], gp, phase, task, seed, a)
                parent_graph = graph_math.combine(parent_original, components, p['graph_recipes'][task])
                np.testing.assert_allclose(original, parent_original, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(scores['zero'], parent_graph, atol=1e-5, rtol=1e-5)
    table = pd.DataFrame(rows)
    table.to_csv(Path(out) / (phase + ('_controls' if controls else '') + '_metrics.csv'), index=False)
    return table, hashes


def collect_replay(out, p, phase, controls=False):
    stem = 'REPLAY_' + phase + ('_controls' if controls else '')
    paths = [Path(out) / (stem + '.json')]
    if not paths[0].exists():
        paths = [Path(out) / (stem + '_' + task + '.json') for task in TASKS]
    records, files = [], {}
    for path in paths:
        record = base.check_seal(base.read_json(path))
        if record['protocol_sha256'] != p['sha256'] or not record['passed']:
            raise ValueError('Invalid replay evidence')
        records.extend(record['records'])
        files[str(path.relative_to(out))] = base.digest(path)
    expected = {(task, seed, kind) for task in TASKS for seed in p['seeds'] for kind in reference.kinds(task)}
    actual = [(r['task'], r['seed'], r['kind']) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicated replay records')
    for r in records:
        dest = directory(out, phase, r['task'], r['seed'], r['kind'], controls)
        if base.digest(dest / 'result.json') != r['result_sha256']:
            raise ValueError('Replayed result changed')
    return files, records


def select(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if (out / 'SELECTION.json').exists():
        raise ValueError('Selection is already frozen')
    files, _ = collect_replay(out, p, 'inner')
    for task in TASKS:
        path = out / ('QUERY_AUDIT_inner_' + task + '.json')
        record = base.check_seal(base.read_json(path))
        result = directory(out, 'inner', task, p['seeds'][0], 'full') / 'result.json'
        if record['protocol_sha256'] != p['sha256'] or not record['passed'] or record['result_sha256'] != base.digest(result):
            raise ValueError('Missing or invalid query-independence audit')
        files[str(path.relative_to(out))] = base.digest(path)
    table, result_files = metric_table(out, p, 'inner')
    files.update(result_files)
    files['TRAINING.json'] = base.digest(out / 'TRAINING.json')
    means = table.groupby(['task', 'recipe'])[list(METRICS)].mean()
    rows = []
    protected = METRICS if p['protect_macro'] else METRICS[:2]
    for cfg in p['candidates']:
        failures, deltas = [], {}
        for task in TASKS:
            deltas[task] = {}
            for metric in METRICS:
                value = float(means.loc[(task, cfg['id']), metric])
                graph_value = float(means.loc[(task, 'zero'), metric])
                original = float(means.loc[(task, 'original_tabicl'), metric])
                delta = table[(table.task == task) & (table.recipe == cfg['id'])].sort_values('seed')[metric].to_numpy() - table[
                    (table.task == task) & (table.recipe == 'zero')].sort_values('seed')[metric].to_numpy()
                deltas[task][metric] = dict(vs_graph=value - graph_value, vs_original=value - original,
                                           positive_seeds=int(np.sum(delta > 0)), minimum_seed_delta=float(delta.min()))
                if not np.isfinite(value):
                    failures.append(task + ':' + metric + ':nonfinite')
                elif task == 'LncDrug' and metric in protected and value < max(graph_value, original):
                    failures.append(task + ':' + metric)
                elif task == 'MiDrug' and metric in METRICS[:2] and (value <= graph_value or np.sum(delta > 0) < (1 if p['smoke'] else 2)):
                    failures.append(task + ':' + metric)
        rows.append(dict(recipe=cfg, eligible=not failures, failed_conditions=failures, deltas=deltas,
                         criterion=float(means.loc[('MiDrug', cfg['id']), ['auc', 'ap']].mean())))
    eligible = [r for r in rows if r['eligible'] and not p['smoke']]
    best = min(eligible, key=lambda r: (r['recipe']['strength'], -r['recipe']['trust'], r['recipe']['step'], -r['criterion'])) if eligible else None
    choice = base.seal(dict(protocol_sha256=p['sha256'], eligible=bool(best),
        recipe=best['recipe'] if best else dict(id='zero'), candidates=rows, files=files,
        protected_lnc_metrics=list(protected), smoke=p['smoke'], no_test_evaluation=True))
    base.write_json(out / 'SELECTION.json', choice)
    print('SELECTED', choice['recipe'], 'eligible', choice['eligible'], flush=True)
    for row in rows:
        print(row['recipe']['id'], row['failed_conditions'], row['deltas'], flush=True)


def report(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    choice = base.check_seal(base.read_json(out / 'SELECTION.json'))
    table, _ = metric_table(out, p, args.phase, args.controls)
    replay_files, replay_records = collect_replay(out, p, args.phase, args.controls)
    means = table.groupby(['task', 'recipe'])[list(METRICS)].mean()
    lines = ['# CARE-TabICL 共享条件关系注意力', '',
        '工程短流程，分数不作为性能证据。' if p['smoke'] else '同一历史划分；三个推理种子 666/0/1；每种子 32 个成员。', '',
        '内部冻结选择：`%s`；通过：`%s`。' % (choice['recipe']['id'], choice['eligible']),
        'LncDrug 硬约束：' + '、'.join(choice['protected_lnc_metrics']) + '；逐指标同时比较原始 TabICLv2 和图上下文参考。', '',
        '| 任务 | 方案 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for (task, recipe), row in means.iterrows():
        lines.append('| %s | %s | %.9f | %.9f | %.9f |' % (task, recipe, row.auc, row.ap, row.macro_drug_ap))
    outcome = dict(protocol_sha256=p['sha256'], phase=args.phase, eligible_inner=choice['eligible'],
                   all_states_replayed=True, max_replay_error=max(r['max_abs'] for r in replay_records),
                   no_test_evaluation=True, smoke=p['smoke'])
    if args.phase == 'outer':
        passed, differences = True, {}
        for task in TASKS:
            differences[task] = {}
            for metric in METRICS:
                new, anchor, original = (float(means.loc[(task, name), metric]) for name in (choice['recipe']['id'], 'zero', 'original_tabicl'))
                differences[task][metric] = dict(vs_graph=new - anchor, vs_original=new - original)
                if task == 'LncDrug' and metric in choice['protected_lnc_metrics']:
                    passed &= new >= max(anchor, original)
                if task == 'MiDrug' and metric in METRICS[:2]:
                    delta = table[(table.task == task) & (table.recipe == choice['recipe']['id'])].sort_values('seed')[metric].to_numpy() - table[
                        (table.task == task) & (table.recipe == 'zero')].sort_values('seed')[metric].to_numpy()
                    passed &= new > anchor and np.sum(delta > 0) >= 2
        outcome.update(passed_outer=bool(passed), deltas=differences)
        lines += ['', '外层预设条件通过：`%s`。外层结果未用于更换适配器、强度或训练步数。' % bool(passed)]
    lines += ['', '新增模块在最终 ICL 查询到支持案例的注意力中工作；两任务使用相同参数。',
        '训练 episode 只读取拟合数据，同一 episode 的支持和伪查询都来自同一遮边折，整个折的目标边都被排除。',
        '每条训练轨迹交替使用两任务；LncDrug 的教师 KL 权重为 MiDrug 的四倍。验证标签不进入优化器。',
        '外层只扩充冻结 TabICL 的支持上下文；共享适配器保持内部训练后的同一权重。', '',
        '已重放 %d 个拟合状态；最大预测误差 %.3g。' % (len(replay_records), outcome['max_replay_error']), '',
        '历史验证已多次参与开发；三个推理种子不是独立数据重复。本轮没有原始测试集、家族/骨架留出或跨任务参数迁移结论。']
    name = 'REPORT_' + args.phase + ('_controls' if args.controls else '') + '.md'
    (out / name).write_text('\n'.join(lines) + '\n')
    base.write_json(out / ('OUTCOME_' + args.phase + '.json'), base.seal(outcome))
    print('REPORT', out / name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'cache', 'train', 'run', 'verify', 'audit', 'select', 'report'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--parent-dir', default=reference.DEFAULT_PARENT)
    parser.add_argument('--inputs-dir', default=reference.DEFAULT_INPUTS)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--phase', choices=('inner', 'outer'), default='inner')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--protect-macro', action='store_true')
    args = parser.parse_args()
    if args.command in ('cache', 'run') and args.task is None:
        parser.error('cache and run require --task')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
