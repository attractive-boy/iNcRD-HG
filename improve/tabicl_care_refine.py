"""A separately frozen strength/uncertainty study of the trained CARE architecture.

The first CARE trial is retained unchanged. This follow-up reuses its shared
weights and expands only the bounded inference correction. No outer labels have
been inspected when this grid is registered; outer data remain gated on this
trial's own selection. Model states and data are otherwise identical.
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
from torch import nn

from improve import tabular_foundation as base
from improve import tabicl_care as core
from improve import tabicl_care_modules as care
from improve import tabicl_reli as reference


DEFAULT_OUT = 'improve/results/tabicl_care_refine_20260913'
OWN = ('tabicl_care_refine.py', 'test_tabicl_care_refine.py')


class RefinedAdapter(nn.Module):
    """The same learned coefficients, with an explicit bounded confidence exponent."""
    def __init__(self, original, gate_power=1.):
        super().__init__()
        if gate_power not in (.5, 1.):
            raise ValueError('Unregistered confidence exponent')
        self.original, self.gate_power = original, float(gate_power)

    def forward(self, relation, evidence, native_logits, strength=1.):
        if not np.isfinite(strength) or not 0. <= strength <= 4.:
            raise ValueError('Correction strength must lie in [0, 4]')
        if self.gate_power == 1.:
            # Exactly reproduce the initial implementation at strength=1.
            return strength * self.original(relation, evidence, native_logits, 1.)
        margin = (native_logits[..., 1] - native_logits[..., 0]).abs().detach()
        uncertainty = 4. * torch.sigmoid(margin) * torch.sigmoid(-margin)
        inputs = torch.cat((evidence[None].expand(len(margin), -1, -1), uncertainty[..., None],
                            (margin / (1. + margin))[..., None]), -1)
        coefficients = self.original.global_coefficients[None, None]
        if self.original.conditional:
            coefficients = coefficients + self.original.controller(inputs)
        coefficients = torch.tanh(coefficients).expand(len(margin), len(evidence), -1)
        raw = torch.einsum('bqf,qsf->bqs', coefficients, relation)
        bias = 2. * torch.tanh(raw / (2. * care.RELATION_DIM ** .5))
        return strength * uncertainty.pow(self.gate_power)[..., None] * bias


def sources():
    return dict(core.sources(), **{name: base.digest(Path(__file__).with_name(name)) for name in OWN})


def prepare(args):
    out, parent = Path(args.out_dir).resolve(), Path(args.training_parent).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh follow-up directory')
    previous = core.protocol(parent)
    previous_choice = base.check_seal(base.read_json(parent / 'SELECTION.json'))
    if previous_choice['eligible']:
        raise ValueError('This follow-up assumes the recorded first-trial failure')
    trained = core.checked_training(parent, previous)
    p = {k: v for k, v in previous.items() if k != 'sha256'}
    p.update(created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources(), smoke=args.smoke,
        training_parent=str(parent), training_parent_protocol_sha256=previous['sha256'],
        training_parent_selection_sha256=base.digest(parent / 'SELECTION.json'),
        training_parent_manifest_sha256=base.digest(parent / 'TRAINING.json'),
        seeds=[666] if args.smoke else previous['seeds'], inference_members=2 if args.smoke else 32,
        gate_powers=[1., .5], refinement_strengths=[1., 2., 4.],
        refinement_checkpoints=[dict(trust=1., step=50), dict(trust=10., step=200)],
        refinement_reason='The first trial improved both tasks over graph-context in one candidate, but missed the stronger original LncDrug AP reference. Retain that failure; test a predeclared bounded correction range and a softer uncertainty exponent.',
        training_updates_this_round=0,
        selection_policy='Same reference envelopes and no-degradation rule as the first trial. Among eligible candidates prefer smaller strength, larger uncertainty exponent, greater trust, then fewer training steps; never choose by outer scores.',
        no_test_evaluation=True)
    p['candidates'] = [dict(id='ref_t%g_e%d_g%g_a%g' % (checkpoint['trust'], checkpoint['step'], power, strength),
                            **checkpoint, gate_power=power, strength=strength, mode='conditional')
                        for checkpoint in p['refinement_checkpoints'] for power in p['gate_powers']
                        for strength in p['refinement_strengths']]
    if args.smoke:
        p['candidates'] = [p['candidates'][0], p['candidates'][4]]
    p = base.seal(p)
    (out / 'source_snapshot').mkdir(parents=True)
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), out / 'source_snapshot' / name)
    base.write_json(out / 'protocol.json', p)
    files = {}
    for name, digest in trained['files'].items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
        if base.digest(target) != digest:
            raise ValueError('Adapter copy differs from original training state')
        files[name] = digest
    base.write_json(out / 'TRAINING.json', base.seal(dict(protocol_sha256=p['sha256'], files=files,
        backbone_unchanged=True, validation_labels_used=False, shared_task_parameters=True,
        training_updates_this_round=0, source_training_manifest_sha256=p['training_parent_manifest_sha256'])))
    print('FROZEN_FOLLOWUP', out, p['sha256'], flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out) / 'protocol.json'))
    if p['source_sha256'] != sources():
        raise ValueError('Frozen follow-up implementation changed')
    parent = Path(p['training_parent'])
    original = core.protocol(parent)
    if original['sha256'] != p['training_parent_protocol_sha256'] or base.digest(parent / 'SELECTION.json') != p['training_parent_selection_sha256']:
        raise ValueError('Original trial or its negative result changed')
    if base.digest(parent / 'TRAINING.json') != p['training_parent_manifest_sha256']:
        raise ValueError('Source training manifest changed')
    core.checked_training(out, p)
    return p


def variants(out, p, configs, device):
    result = {}
    for cfg in configs:
        original, _ = care.load_adapter(core.state_path(out, cfg['mode'], cfg['trust'], cfg['step']), device)
        result[cfg['id']] = (RefinedAdapter(original, cfg['gate_power']).to(device).eval(), cfg['strength'])
    return result


def requested_seeds(args, p):
    if args.seed is None:
        return p['seeds']
    if args.seed not in p['seeds']:
        raise ValueError('Unregistered inference seed')
    return [args.seed]


def run(args):
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir), protocol(args.out_dir)
    configs = core.recipes(out, p, args.phase)
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
            scores, numerical, queries = core.infer(clf, p, args.task, a, kind, configs, models)
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
    configs = core.recipes(out, p, args.phase)
    models = variants(out, p, configs, args.device)
    for task in ([args.task] if args.task else core.TASKS):
        a = core.stage_arrays(out, p, task, args.phase)
        for seed in requested_seeds(args, p):
            records = []
            for kind in reference.kinds(task):
                dest = core.directory(out, args.phase, task, seed, kind)
                core.checked_result(dest, p)
                clf = TabICLClassifier.load(dest / 'fitted.pkl', device=args.device)
                scores, numerical, queries = core.infer(clf, p, task, a, kind, configs, models)
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
    configs = core.recipes(out, p, args.phase)
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
        for mode in dict.fromkeys(cfg['mode'] for cfg in configs):
            relation, info = core.evidence(p, task, mode).features(pairs[:, idx], support)
            names = [cfg['id'] for cfg in configs if cfg['mode'] == mode]
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
    table, result_files = core.metric_table(out, p, 'inner')
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
    table, _ = core.metric_table(out, p, args.phase)
    _, replay = collect_replay(out, p, args.phase)
    means = table.groupby(['task', 'recipe'])[list(core.METRICS)].mean()
    lines = ['# CARE-TabICL 有界强度与置信度调整', '',
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
    lines += ['', '首轮结果完整保留。本轮在读取外层标签前冻结 12 个修正强度/置信度候选；没有重新训练参数或降低性能门槛。',
        '控制器仍使用原不确定度输入；最终注意力偏置乘以不确定度的 1 或 0.5 次方。每个 logit 的修正绝对值不超过 2α，α 最大为 4。',
        '外层仍冻结同一共享参数，仅扩充原 TabICL 支持案例；同时报告池化、常系数对照，不根据外层更换模型。',
        '已重放 %d 个拟合状态，最大预测差 %.3g。' % (len(replay), outcome['max_replay_error']), '',
        '历史验证已多次参与开发；三个推理种子不是三份独立划分。未打开原始测试集，未评价家族/骨架留出或跨任务权重迁移。']
    (out / ('REPORT_' + args.phase + '.md')).write_text('\n'.join(lines) + '\n')
    base.write_json(out / ('OUTCOME_' + args.phase + '.json'), base.seal(outcome))
    print('OUTCOME', outcome, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'verify', 'audit', 'select', 'report'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--training-parent', default=core.DEFAULT_OUT)
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
