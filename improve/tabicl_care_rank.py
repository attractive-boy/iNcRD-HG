"""Shared support-rate control and teacher-order retention for CARE-TabICL.

Earlier negative trials remain unchanged. This developmental follow-up trains
new shared weights from fitting-only cached episodes, selects only on inner
validation, and reports the already reused historical outer validation plainly.
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
from improve import tabicl_care as core
from improve import tabicl_care_modules as care
from improve import tabicl_care_rank_modules as rank
from improve import tabicl_reli as reference


DEFAULT_OUT = 'improve/results/tabicl_care_rank_20260913'
OWN = ('tabicl_care_rank.py', 'tabicl_care_rank_modules.py', 'test_tabicl_care_rank.py')


def sources():
    return dict(core.sources(), **{name: base.digest(Path(__file__).with_name(name)) for name in OWN})


def prepare(args):
    out, parent = Path(args.out_dir).resolve(), Path(args.training_parent).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh follow-up directory')
    previous = core.protocol(parent)
    core.checked_training(parent, previous)
    development = Path(args.development_parent).resolve()
    last = base.check_seal(base.read_json(development / 'OUTCOME_outer.json'))
    if last.get('passed_outer', True) or not last['all_states_replayed']:
        raise ValueError('Preserve and finish the previous negative outer evaluation first')
    p = {k: v for k, v in previous.items() if k != 'sha256'}
    p.update(created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources(), smoke=args.smoke,
        training_parent=str(parent), training_parent_protocol_sha256=previous['sha256'],
        training_parent_manifest_sha256=base.digest(parent / 'TRAINING.json'),
        development_parent=str(development), development_outcome_sha256=base.digest(development / 'OUTCOME_outer.json'),
        seeds=[666] if args.smoke else previous['seeds'], inference_members=2 if args.smoke else 32,
        trusts=[1.] if args.smoke else [1., 10.], steps=[2, 4] if args.smoke else [50, 200],
        strengths=[1., 2.] if args.smoke else [1., 2., 4.], gate_power=.5,
        training_modes=['conditional', 'unscaled', 'no_order'], order_weight=20., support_reference_size=512.,
        architecture='CARE-RO: context-rate-normalized conditional relation attention with teacher-order retention.',
        training_policy='Reuse verified fitting-only same-graph episodes. Both tasks contribute gradients before every optimizer step. LncDrug has fourfold KL and a close-correct-pair teacher-order penalty. Two matched controls remove support-rate normalization or teacher-order retention. No validation label is passed to the optimizer.',
        development_history='The preceding CARE outer result was observed and failed LncDrug AUC. This follow-up targets a context-size mismatch hypothesis, not an independent replication; the historical outer split is reused development data.',
        selection_policy='Unchanged two-reference LncDrug metric envelope and positive MiDrug AUC/AP with at least two positive inference seeds. Eligible candidates prefer lower correction strength, larger KL, then fewer optimizer steps. No outer reselection.',
        no_test_evaluation=True)
    p['candidates'] = [dict(id='ro_t%g_e%d_a%g' % (trust, step, strength), trust=trust, step=step,
                            gate_power=.5, strength=strength, mode='conditional')
                        for trust in p['trusts'] for step in p['steps'] for strength in p['strengths']]
    p = base.seal(p)
    (out / 'source_snapshot').mkdir(parents=True)
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), out / 'source_snapshot' / name)
    base.write_json(out / 'protocol.json', p)
    print('FROZEN_CARE_RO', out, p['sha256'], flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out) / 'protocol.json'))
    if p['source_sha256'] != sources():
        raise ValueError('Frozen follow-up implementation changed')
    parent = Path(p['training_parent'])
    original = core.protocol(parent)
    if original['sha256'] != p['training_parent_protocol_sha256']:
        raise ValueError('Original training episode source changed')
    if base.digest(parent / 'TRAINING.json') != p['training_parent_manifest_sha256']:
        raise ValueError('Source training manifest changed')
    if base.digest(Path(p['development_parent']) / 'OUTCOME_outer.json') != p['development_outcome_sha256']:
        raise ValueError('Previous negative outcome changed')
    return p


def variants(out, p, configs, device):
    core.checked_training(out, p)
    result = {}
    for cfg in configs:
        model, _ = rank.load_adapter(core.state_path(out, cfg['mode'], cfg['trust'], cfg['step']), device)
        result[cfg['id']] = (model, cfg['strength'])
    return result


def train(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if (out / 'TRAINING.json').exists():
        core.checked_training(out, p)
        return
    original = core.protocol(p['training_parent'])
    episodes, hashes = {}, {}
    for task in core.TASKS:
        dest = Path(p['training_parent']) / 'training_cache' / task
        manifest = core.checked_cache(dest, original)
        episodes[task] = [torch.load(dest / row['file'], weights_only=True, map_location='cpu') for row in manifest['episodes']]
        hashes[task] = base.digest(dest / 'manifest.json')
    a, ep = core.training_arrays(p, 'MiDrug'), core.base_protocol(p)
    keep = base.balanced_subset(a['y_train'], 16, p['training_seed'])
    clf = base.fit_context(base.estimator(dict(n_estimators=2), p['training_seed'], ep, args.device),
                           a['raw_train'][keep], a['y_train'][keep])
    clf.model_.to(args.device).eval().requires_grad_(False)
    predictor, files, trajectories = clf.model_.icl_predictor, {}, []
    for mode in p['training_modes']:
        for trust in p['trusts']:
            torch.manual_seed(p['training_seed'])
            rng = np.random.default_rng(p['training_seed'])
            model = rank.ScaleStableAdapter(normalize_support=mode != 'unscaled', gate_power=p['gate_power']).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
            trace, started = [], time.monotonic()
            for step in range(1, max(p['steps']) + 1):
                optimizer.zero_grad(set_to_none=True)
                for task in core.TASKS:
                    item = episodes[task][int(rng.integers(len(episodes[task])))]
                    size = min(24 if p['smoke'] else p['training_query_batch'], len(item['target']))
                    idx = np.sort(rng.choice(len(item['target']), size, replace=False))
                    cache = {key: item[key][:, idx].to(args.device) if key in ('query', 'native_logits') else item[key].to(args.device)
                             for key in ('query', 'native_logits', 'key', 'value')}
                    relation, info = item['relation'][idx].to(args.device), item['evidence'][idx].to(args.device)
                    target, permutations = item['target'][idx].to(args.device), item['permutations'].to(args.device)
                    logits = care.adapted_logits(predictor, cache, relation, info, model)
                    objective, terms = care.adaptation_loss(logits, cache['native_logits'], target, permutations,
                                                          trust=trust * (4. if task == 'LncDrug' else 1.))
                    order, protected = rank.teacher_order_loss(logits, cache['native_logits'], target, permutations)
                    weight = p['order_weight'] if task == 'LncDrug' and mode != 'no_order' else 0.
                    loss = .5 * (objective + weight * order)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite rank-retention objective')
                    loss.backward()
                    trace.append(dict(step=step, task=task, loss=float(loss.detach()), order_loss=float(order.detach()),
                        protected_pairs=protected, order_weight=weight, **{k: float(v.detach()) for k, v in terms.items()}))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if step in p['steps']:
                    path = core.state_path(out, mode, trust, step)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(config=model.config(), state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        mode=mode, trust=trust, step=step, tasks=list(core.TASKS), protocol_sha256=p['sha256']), path)
                    files[str(path.relative_to(out))] = base.digest(path)
                if step % 50 == 0 or step == max(p['steps']):
                    print('TRAIN_RO', mode, trust, step, 'seconds', round(time.monotonic() - started, 1), flush=True)
            path = core.state_path(out, mode, trust, 0).parent / 'trace.csv'
            pd.DataFrame(trace).to_csv(path, index=False)
            files[str(path.relative_to(out))] = base.digest(path)
            trajectories.append(dict(mode=mode, trust=trust, seconds=time.monotonic() - started,
                trainable_parameters=sum(value.numel() for value in model.parameters()),
                max_order_penalty=max(row['order_loss'] for row in trace), protected_pairs=sum(row['protected_pairs'] for row in trace)))
    if base.parameter_digest(clf) != p['parameter_sha256'] or any(value.grad is not None for value in clf.model_.parameters()):
        raise ValueError('Frozen TabICL changed')
    base.write_json(out / 'TRAINING.json', base.seal(dict(protocol_sha256=p['sha256'], files=files,
        source_cache_manifests=hashes, trajectories=trajectories, backbone_unchanged=True,
        validation_labels_used=False, shared_task_parameters=True, optimizer_steps=max(p['steps']))))


def phase_recipes(out, p, phase):
    if phase == 'inner':
        return p['candidates']
    chosen = core.selection(out, p)['recipe']
    return [chosen, *[dict(chosen, id=mode + '_control', mode=mode) for mode in ('unscaled', 'no_order')]]


def inference(clf, p, task, a, kind, configs, models):
    _, _, support, queries = reference.job_inputs(p, task, a, kind)
    relation, info = core.evidence(p, task).features(np.tile(a['pairs_val'], (1, len(queries))), support)
    # All three mechanisms use exactly the same local evidence. Reuse one
    # official prefix across primary and controls; only their adapters differ.
    scores, numerical = care.predict_variants(clf, queries.reshape(-1, 36), relation, info, models)
    return scores, numerical, queries


def requested_seeds(args, p):
    if args.seed is None:
        return p['seeds']
    if args.seed not in p['seeds']:
        raise ValueError('Unregistered inference seed')
    return [args.seed]


def run(args):
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir), protocol(args.out_dir)
    configs = phase_recipes(out, p, args.phase)
    models = variants(out, p, configs, args.device)
    a, ep = core.stage_arrays(out, p, args.task, args.phase), core.base_protocol(p)
    for seed in requested_seeds(args, p):
        for kind in reference.kinds(args.task):
            dest = core.directory(out, args.phase, args.task, seed, kind)
            if (dest / 'result.json').exists():
                core.checked_result(dest, p)
                continue
            started = time.monotonic()
            x, labels, _, _ = reference.job_inputs(p, args.task, a, kind)
            # Formal inner runs reuse the exact previously verified fitted
            # preprocessing/context state. Smoke uses its own small contexts.
            if args.phase == 'inner' and not p['smoke']:
                source = core.directory(p['training_parent'], 'inner', args.task, seed, kind)
                core.checked_result(source, core.protocol(p['training_parent']))
                clf = TabICLClassifier.load(source / 'fitted.pkl', device=args.device)
            else:
                clf = base.fit_context(base.estimator(dict(n_estimators=p['inference_members']), seed, ep, args.device), x, labels)
            print('RUN', args.phase, args.task, seed, kind, len(x), flush=True)
            scores, numerical, queries = inference(clf, p, args.task, a, kind, configs, models)
            if base.parameter_digest(clf) != p['parameter_sha256']:
                raise ValueError('Backbone changed')
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


def verify(args):
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir), protocol(args.out_dir)
    configs = phase_recipes(out, p, args.phase)
    models = variants(out, p, configs, args.device)
    for task in ([args.task] if args.task else core.TASKS):
        a = core.stage_arrays(out, p, task, args.phase)
        for seed in requested_seeds(args, p):
            records = []
            for kind in reference.kinds(task):
                dest = core.directory(out, args.phase, task, seed, kind)
                core.checked_result(dest, p)
                clf = TabICLClassifier.load(dest / 'fitted.pkl', device=args.device)
                scores, numerical, queries = inference(clf, p, task, a, kind, configs, models)
                names = ['zero'] + [cfg['id'] for cfg in configs]
                with np.load(dest / 'scores.npz', allow_pickle=False) as archive:
                    replay = np.stack([scores[name].reshape(queries.shape[:2]) for name in names])
                    np.testing.assert_allclose(replay, archive['scores'], atol=p['replay_atol'], rtol=0)
                    error = float(np.max(np.abs(replay - archive['scores'])))
                if base.parameter_digest(clf) != p['parameter_sha256']:
                    raise ValueError('Replayed backbone changed')
                records.append(dict(task=task, seed=seed, kind=kind, max_abs=error,
                    result_sha256=base.digest(dest / 'result.json'), numerical=numerical))
                print('REPLAYED', args.phase, task, seed, kind, error, flush=True)
                del clf
                gc.collect()
                torch.cuda.empty_cache()
            base.write_json(out / ('REPLAY_%s_%s_s%d.json' % (args.phase, task, seed)), base.seal(dict(
                protocol_sha256=p['sha256'], records=records, passed=True)))


def collect_replay(out, p, phase):
    for task in core.TASKS:
        records = []
        for seed in p['seeds']:
            path = Path(out) / ('REPLAY_%s_%s_s%d.json' % (phase, task, seed))
            state = base.check_seal(base.read_json(path))
            if state['protocol_sha256'] != p['sha256'] or not state['passed']:
                raise ValueError('Invalid per-seed replay record')
            records.extend(state['records'])
        target = Path(out) / ('REPLAY_%s_%s.json' % (phase, task))
        combined = base.seal(dict(protocol_sha256=p['sha256'], records=records, passed=True))
        if target.exists() and base.read_json(target) != combined:
            raise ValueError('Existing consolidated replay differs')
        if not target.exists():
            base.write_json(target, combined)
    return core.collect_replay(out, p, phase)


def audit(args):
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir), protocol(args.out_dir)
    configs = phase_recipes(out, p, args.phase)
    models = variants(out, p, configs, args.device)
    for task in ([args.task] if args.task else core.TASKS):
        a = core.stage_arrays(out, p, task, args.phase)
        dest = core.directory(out, args.phase, task, p['seeds'][0], 'full')
        core.checked_result(dest, p)
        _, _, support, queries = reference.job_inputs(p, task, a, 'full')
        flat = queries.reshape(-1, 36)
        pairs = np.tile(a['pairs_val'], (1, len(queries)))
        idx = np.random.default_rng(p['training_seed']).permutation(len(flat))[:min(257, len(flat))]
        idx = np.r_[idx, idx[:3]]
        clf = TabICLClassifier.load(dest / 'fitted.pkl', device=args.device)
        with np.load(dest / 'scores.npz', allow_pickle=False) as archive:
            full = {name: score.reshape(-1) for name, score in zip(archive['recipe_ids'].tolist(), archive['scores'])}
        records = []
        for mode in ('conditional',):
            relation, info = core.evidence(p, task, mode).features(pairs[:, idx], support)
            names = [cfg['id'] for cfg in configs]
            scores, numerical = care.predict_variants(clf, flat[idx], relation, info, {name: models[name] for name in names})
            for name, score in scores.items():
                effect, full_effect = score - scores['zero'], full[name][idx] - full['zero'][idx]
                np.testing.assert_allclose(score, full[name][idx], atol=p['query_atol'], rtol=1e-5)
                np.testing.assert_allclose(effect, full_effect, atol=p['query_effect_atol'], rtol=1e-5)
                records.append(dict(recipe=name, query_probability_max_abs=float(np.max(np.abs(score - full[name][idx]))),
                                    query_effect_max_abs=float(np.max(np.abs(effect - full_effect)))))
        base.write_json(out / ('QUERY_AUDIT_' + args.phase + '_' + task + '.json'), base.seal(dict(
            protocol_sha256=p['sha256'], task=task, query_rows=len(idx), records=records, passed=True,
            result_sha256=base.digest(dest / 'result.json'))))
        print('QUERY_AUDIT_PASSED', args.phase, task, max(r['query_effect_max_abs'] for r in records), flush=True)
        del clf
        gc.collect()
        torch.cuda.empty_cache()


def assembled_scores(out, p, phase, task, seed, a):
    names = ['zero'] + [cfg['id'] for cfg in phase_recipes(out, p, phase)]
    states, files = {}, {}
    for kind in reference.kinds(task):
        dest = core.directory(out, phase, task, seed, kind)
        core.checked_result(dest, p)
        with np.load(dest / 'scores.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['recipe_ids'], names)
            np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
            np.testing.assert_array_equal(archive['y_val'], a['y_val'])
            states[kind] = archive['scores'].copy()
        files[str((dest / 'result.json').relative_to(out))] = base.digest(dest / 'result.json')
    original = states['full'][:, 0]
    component = states['full'][:, 1:].mean(1) if task == 'LncDrug' else np.stack(
        [states['fold%d' % fold][:, 0] for fold in range(5)]).mean(0)
    return dict(zip(names, .5 * original + .5 * component)), original[0], files


def metric_table(out, p, phase):
    rows, files = [], {}
    gp = base.read_json(Path(p['parent_dir']) / 'protocol.json')
    for task in core.TASKS:
        a = core.stage_arrays(out, p, task, phase)
        for seed in p['seeds']:
            scores, original, hashes = assembled_scores(out, p, phase, task, seed, a)
            files.update(hashes)
            for name, values in [('original_tabicl', original), *scores.items()]:
                rows.append(dict(task=task, seed=seed, recipe=name, **base.metrics(a['y_val'], values, a['pairs_val'])))
            if not p['smoke']:
                old, parts, _ = core.graph.components(p['parent_dir'], gp, phase, task, seed, a)
                old_graph = core.graph_math.combine(old, parts, p['graph_recipes'][task])
                np.testing.assert_allclose(original, old, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(scores['zero'], old_graph, atol=1e-5, rtol=1e-5)
    table = pd.DataFrame(rows)
    table.to_csv(Path(out) / (phase + '_metrics.csv'), index=False)
    return table, files


def select(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if (out / 'SELECTION.json').exists():
        raise ValueError('Selection is already frozen')
    files, _ = collect_replay(out, p, 'inner')
    for task in core.TASKS:
        path = out / ('QUERY_AUDIT_inner_' + task + '.json')
        record = base.check_seal(base.read_json(path))
        source = core.directory(out, 'inner', task, p['seeds'][0], 'full') / 'result.json'
        if record['protocol_sha256'] != p['sha256'] or not record['passed'] or record['result_sha256'] != base.digest(source):
            raise ValueError('Invalid query-independence audit')
        files[str(path.relative_to(out))] = base.digest(path)
    table, result_files = metric_table(out, p, 'inner')
    files.update(result_files)
    files['TRAINING.json'] = base.digest(out / 'TRAINING.json')
    means = table.groupby(['task', 'recipe'])[list(core.METRICS)].mean()
    rows, protected = [], core.METRICS if p['protect_macro'] else core.METRICS[:2]
    for cfg in p['candidates']:
        failures, deltas = [], {}
        for task in core.TASKS:
            deltas[task] = {}
            for metric in core.METRICS:
                value, anchor, original = (float(means.loc[(task, name), metric]) for name in (cfg['id'], 'zero', 'original_tabicl'))
                delta = table[(table.task == task) & (table.recipe == cfg['id'])].sort_values('seed')[metric].to_numpy() - table[
                    (table.task == task) & (table.recipe == 'zero')].sort_values('seed')[metric].to_numpy()
                deltas[task][metric] = dict(vs_graph=value - anchor, vs_original=value - original,
                                           positive_seeds=int(np.sum(delta > 0)), minimum_seed_delta=float(delta.min()))
                if not np.isfinite(value) or (task == 'LncDrug' and metric in protected and value < max(anchor, original)):
                    failures.append(task + ':' + metric)
                elif task == 'MiDrug' and metric in core.METRICS[:2] and (value <= anchor or np.sum(delta > 0) < (1 if p['smoke'] else 2)):
                    failures.append(task + ':' + metric)
        rows.append(dict(recipe=cfg, eligible=not failures, failed_conditions=failures, deltas=deltas,
                         criterion=float(means.loc[('MiDrug', cfg['id']), ['auc', 'ap']].mean())))
    eligible = [r for r in rows if r['eligible'] and not p['smoke']]
    best = min(eligible, key=lambda r: (r['recipe']['strength'], -r['recipe']['gate_power'], -r['recipe']['trust'],
                                       r['recipe']['step'], -r['criterion'])) if eligible else None
    choice = base.seal(dict(protocol_sha256=p['sha256'], eligible=bool(best), recipe=best['recipe'] if best else dict(id='zero'),
        candidates=rows, files=files, protected_lnc_metrics=list(protected), smoke=p['smoke'], no_test_evaluation=True))
    base.write_json(out / 'SELECTION.json', choice)
    print('SELECTED', choice['recipe'], 'eligible', choice['eligible'], flush=True)
    for row in rows:
        print(row['recipe']['id'], row['failed_conditions'], row['deltas'], flush=True)


def report(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    choice = base.check_seal(base.read_json(out / 'SELECTION.json'))
    table, _ = metric_table(out, p, args.phase)
    _, replay = collect_replay(out, p, args.phase)
    means = table.groupby(['task', 'recipe'])[list(core.METRICS)].mean()
    lines = ['# CARE-RO：支持比例控制与教师排序保持', '',
        '工程短流程，分数不构成绩效证据。' if p['smoke'] else '历史划分，seed 666/0/1，各 32 个推理成员；两任务共用同一适配器。', '',
        '内部冻结选择：`%s`；通过：`%s`。' % (choice['recipe']['id'], choice['eligible']), '',
        '| 任务 | 方案 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for (task, name), row in means.iterrows():
        lines.append('| %s | %s | %.9f | %.9f | %.9f |' % (task, name, row.auc, row.ap, row.macro_drug_ap))
    outcome = dict(protocol_sha256=p['sha256'], phase=args.phase, eligible_inner=choice['eligible'],
                   all_states_replayed=True, max_replay_error=max(r['max_abs'] for r in replay), no_test_evaluation=True, smoke=p['smoke'])
    if args.phase == 'outer':
        passed, deltas = True, {}
        for task in core.TASKS:
            deltas[task] = {}
            for metric in core.METRICS:
                new, anchor, original = (float(means.loc[(task, name), metric]) for name in (choice['recipe']['id'], 'zero', 'original_tabicl'))
                pairs = table[(table.task == task) & (table.recipe == choice['recipe']['id'])].sort_values('seed')[metric].to_numpy() - table[
                    (table.task == task) & (table.recipe == 'zero')].sort_values('seed')[metric].to_numpy()
                deltas[task][metric] = dict(vs_graph=new - anchor, vs_original=new - original,
                                           positive_seeds=int(np.sum(pairs > 0)), minimum_seed_delta=float(pairs.min()))
                if task == 'LncDrug' and metric in choice['protected_lnc_metrics']:
                    passed &= new >= max(anchor, original)
                if task == 'MiDrug' and metric in core.METRICS[:2]:
                    passed &= new > anchor and np.sum(pairs > 0) >= 2
        outcome.update(passed_outer=bool(passed), deltas=deltas)
        lines += ['', '外层预设条件通过：`%s`。' % bool(passed)]
    lines += ['', '既有两个负结果保留。本轮重新训练共享参数；支持数量先按固定 512 案例的参考规模归一化。',
        '每次优化同时累积两个任务的梯度；LncDrug 对原模型已正确排列的近邻正负案例增加排序保持损失。它是训练约束，不保证验证或未来样本不下降。',
        '外层使用同一共享参数，并报告不做比例归一化、去掉排序保持的同输入同参数量对照；不根据外层重选模型。',
        '已重放 %d 个拟合状态，最大预测差 %.3g。' % (len(replay), outcome['max_replay_error']), '',
        '这一轮是在看到前轮外层 LncDrug 退化后提出，历史外层属于复用的开发验证，不能称为独立确认。三个推理种子不是三份独立划分。未打开原始测试集，未评价家族/骨架留出或跨任务权重迁移。']
    (out / ('REPORT_' + args.phase + '.md')).write_text('\n'.join(lines) + '\n')
    base.write_json(out / ('OUTCOME_' + args.phase + '.json'), base.seal(outcome))
    print('OUTCOME', outcome, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'run', 'verify', 'audit', 'select', 'report'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--training-parent', default=core.DEFAULT_OUT)
    parser.add_argument('--development-parent', default='improve/results/tabicl_care_refine_20260913')
    parser.add_argument('--task', choices=core.TASKS)
    parser.add_argument('--phase', choices=('inner', 'outer'), default='inner')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'run' and args.task is None:
        parser.error('run requires --task')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
