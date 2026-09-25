"""Equal-budget TabM tuning on the audited, fixed structural-program features.

prepare -> screen -> shortlist -> confirm -> freeze -> outer -> report.
The old experiment and its source files remain immutable. This runner reuses
its feature artifacts, and never selects configurations or epochs on outer data.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from improve import backbone_replacement as parent
from improve import graph_tabm as table
from improve import structure_program_search as original


ARMS = original.ARMS
OWN = ('structure_program_tuning.py', 'test_structure_program_tuning.py')
METRICS = ('auc', 'ap', 'macro_drug_ap', 'macro_recall10_sampled')


def sources():
    return dict(original.sources(), **{
        name: parent.digest(Path(__file__).with_name(name)) for name in OWN})


def configurations():
    result = []
    for width, depth in ((128, 3), (64, 2)):
        for dropout, decay in ((.1, 1e-4), (.3, .01)):
            for lr in (.001, .003, .0005):
                result.append(dict(id='t%02d' % len(result), width=width,
                    depth=depth, dropout=dropout, weight_decay=decay,
                    lr=lr, members=16))
    return result


def horizon(p, task, cfg):
    if p['smoke']:
        return 3
    return p['epoch_limits'][task][str(cfg['lr'])]


def prepare(args):
    out = Path(args.out_dir).resolve()
    reference = Path(args.reference_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    prior = original.protocol(reference)
    for name in ('VALIDATION.json', 'FEATURE_RECHECK.json'):
        check = json.loads((reference/name).read_text())
        if prior['smoke'] or not check['passed'] or check['protocol_sha256'] != prior['sha256']:
            raise ValueError('A verified full structural-program reference is required')
    files = ['protocol.json', 'VALIDATION.json', 'FEATURE_RECHECK.json',
             'PARAMETERS.json', 'REFIT.json']
    # Pin manifests now; evaluation arrays are opened only after our REFIT exists.
    files += ['features/%s/%s/manifest.json' % (phase, task)
              for phase in ('inner', 'outer') for task in prior['tasks']]
    grid = configurations()
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        reference_dir=str(reference), reference_protocol_sha256=prior['sha256'],
        reference_artifacts_sha256={name: parent.digest(reference/name) for name in files},
        tasks=prior['tasks'], arms=list(ARMS), source_sha256=sources(),
        configurations=[grid[0], grid[-1]] if args.smoke else grid,
        screening_seed=666, confirmation_seeds=[0] if args.smoke else [0, 1],
        refit_seeds=[666] if args.smoke else [666, 0, 1], shortlist_size=2,
        epoch_limits={'LncDrug': {'0.0005': 400, '0.001': 250, '0.003': 200},
                      'MiDrug': {'0.0005': 900, '0.001': 600, '0.003': 450}},
        batch_size=prior['batch_size'], smoothing_window=2 if args.smoke else 5,
        anchors={} if args.smoke else {'t00': 'c0', 't01': 'c1'},
        feature_policy='Read-only reuse of the audited 36/48-dimensional features. Program IDs, feature count, fold assignments, quantile transforms, null samples and null definitions are unchanged. Validate feature and dependency hashes on loading.',
        selection='Same 12 configurations per task and arm. Screen seed 666 by earliest best trailing-five mean(AUC,AP); shortlist the top two by score, then config ID. Confirm BOTH with seeds 0/1. Select the largest three-seed mean selection_score (config ID breaks ties); freeze median selected epoch of that config. The selected checkpoint supplies both metrics; no metric-wise checkpoint choice.',
        search_space='Two joint capacity settings (width/depth 128/3, 64/2), two joint regularization settings (dropout/decay .1/1e-4, .3/.01), three learning rates (.0005,.001,.003). This is a bounded 2x2x3 factorial search, not a separate causal estimate for depth, dropout or decay.',
        optimizer='Unchanged AdamW and constant learning rate; one accumulated full-data mean member-BCE update per epoch. Same per-learning-rate horizon across arms; 16 members and batch size 1024 remain fixed.',
        outer_policy='This runner rejects opening outer feature arrays before its own frozen REFIT. Outer epochs never compute validation metrics. Compare only the inner-selected recipe for each arm; do not select arms/configs/seeds using outer scores.',
        comparisons='Predeclared: tuned arm versus its previous recipe; tuned arm versus previous base; random/search/excess versus equally tuned base; search versus random; excess versus search. Paired by the identical seed and validation pairs.',
        performance_gate='Mean AUC and AP both increase, each in at least 2/3 seeds, and mean macro drug AP does not decrease. Mechanism comparisons omit the macro condition. Cross-task support requires both tasks; seeds are initialization variation only.',
        no_test_evaluation=True,
        limitations=prior['limitations'] + [
            'This follow-up was motivated by the previous outer results, so outer validation is reused development evidence.',
            'More inner searches can overfit the same inner partition; two-candidate confirmation does not create independent data.',
            'Capacity and regularization components are varied jointly, so one hyperparameter cannot be credited in isolation.',
            'Low-learning-rate horizons are bounded; a checkpoint at the limit is not evidence of convergence.'])
    out.mkdir(parents=True)
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot/name)
    p['sha256'] = parent.json_digest(p)
    parent.write_json(out/'protocol.json', p)
    import scipy
    import sklearn
    import torch_geometric
    parent.write_json(out/'ENVIRONMENT.json', dict(python=sys.version,
        executable=sys.executable, platform=platform.platform(), numpy=np.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__, torch=torch.__version__,
        pyg=torch_geometric.__version__,
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        threads=torch.get_num_threads(), gpu_assignment={'base': 0, 'random': 1, 'search': 0, 'excess': 1}))
    print('FROZEN', out, 'configurations', len(p['configurations']), flush=True)


def protocol(out):
    p = json.loads((Path(out)/'protocol.json').read_text())
    if p['sha256'] != parent.json_digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('Frozen protocol changed')
    if p['source_sha256'] != sources():
        raise ValueError('Frozen source changed')
    reference = Path(p['reference_dir'])
    if original.protocol(reference)['sha256'] != p['reference_protocol_sha256']:
        raise ValueError('Reference protocol changed')
    for name, sha in p['reference_artifacts_sha256'].items():
        if parent.digest(reference/name) != sha:
            raise ValueError('Reference artifact changed: '+name)
    return p


def features(out, p, task, stage, arm):
    if stage not in ('screen', 'confirm', 'outer') or arm not in p['arms']:
        raise ValueError('Unknown stage or arm')
    if stage == 'outer':
        parent.decision(out, 'REFIT.json', p)
    phase = 'outer' if stage == 'outer' else 'inner'
    reference = Path(p['reference_dir'])
    path = reference/'features'/phase/task
    original.checked_manifest(path, original.protocol(reference))
    with np.load(path/'arrays.npz') as archive:
        a = {key: archive[key].copy() for key in ('y_train', 'y_val', 'pairs_train', 'pairs_val')}
        a['x_train'] = archive[arm+'_x_train'].copy()
        a['x_val'] = archive[arm+'_x_val'].copy()
    dim = 36 if arm == 'base' else 48
    if a['x_train'].shape != (len(a['y_train']), dim) or a['x_val'].shape != (len(a['y_val']), dim):
        raise ValueError('Unexpected feature shape')
    if not np.isfinite(a['x_train']).all() or not np.isfinite(a['x_val']).all():
        raise ValueError('Non-finite features')
    a['feature_manifest_sha256'] = parent.digest(path/'manifest.json')
    return a


def jobs(out, p, stage):
    if stage == 'screen':
        selected = refits = None
    elif stage == 'confirm':
        selected = parent.decision(out, 'PARAMETERS.json', p)['chosen']
        refits = None
    elif stage == 'outer':
        parent.decision(out, 'PARAMETERS.json', p)
        refits = parent.decision(out, 'REFIT.json', p)['chosen']
        selected = None
    else:
        raise ValueError('Unknown stage')
    for task in p['tasks']:
        for arm in p['arms']:
            cfgs = p['configurations'] if stage == 'screen' else (
                selected[task][arm] if stage == 'confirm' else [refits[task][arm]['config']])
            seeds = [p['screening_seed']] if stage == 'screen' else (
                p['confirmation_seeds'] if stage == 'confirm' else p['refit_seeds'])
            for cfg in cfgs:
                for seed in seeds:
                    epochs = refits[task][arm]['epochs'] if stage == 'outer' else horizon(p, task, cfg)
                    yield task, arm, cfg, seed, epochs


def run(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    planned = list(jobs(out, p, args.stage))
    for task in ([args.task] if args.task else p['tasks']):
        for arm in ([args.arm] if args.arm else p['arms']):
            a = features(out, p, task, args.stage, arm)
            for t, ar, cfg, seed, epochs in planned:
                if (t, ar) != (task, arm):
                    continue
                dest = parent.directory(out, args.stage, task, arm, cfg['id'], seed)
                dest.mkdir(parents=True, exist_ok=True)
                with (dest/'job.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if (dest/'result.json').exists():
                        parent.checked_result(dest, p)
                        continue
                    r, scores, trace, state = table.train(a, task, 'tabm', cfg, seed,
                        args.stage, epochs, p, args.device)
                    r.update(arm=arm, input_dim=a['x_train'].shape[1], device=args.device)
                    np.savez_compressed(dest/'scores.npz', val=scores,
                        y_val=a['y_val'], pairs_val=a['pairs_val'])
                    pd.DataFrame(trace).to_csv(dest/'trace.csv', index=False)
                    torch.save(state, dest/'checkpoint.pt')
                    r['artifacts_sha256'] = {name: parent.digest(dest/name)
                        for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                    parent.write_json(dest/'result.json', r)
                    print('DONE', task, arm, cfg['id'], seed, r['auc'], r['ap'],
                          r['selected_epoch'], flush=True)


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
            cfgs = p['configurations'] if stage == 'screen' else shortlist[task][arm]
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


def compare(left, right, task, arm, control, kind):
    x = left[(left.task == task) & (left.arm == arm)].set_index('seed').sort_index()
    y = right[(right.task == task) & (right.arm == control)].set_index('seed').sort_index()
    if not x.index.equals(y.index) or not x.index.is_unique or not y.index.is_unique:
        raise ValueError('Mismatched comparison seeds')
    return [dict(task=task, arm=arm, control=control, comparison=kind, seed=int(seed),
        **{m: float(x.loc[seed, m]-y.loc[seed, m]) for m in METRICS}) for seed in x.index]


def performance_gate(deltas, macro=True):
    d = pd.DataFrame(deltas)
    return bool(len(d) == 3 and (d[['auc', 'ap']].mean() > 0).all()
        and ((d[['auc', 'ap']] > 0).sum() >= 2).all()
        and (not macro or d.macro_drug_ap.mean() >= 0))


def report(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    for stage, filename in (('screen', 'PARAMETERS.json'), ('confirm', 'REFIT.json')):
        if selection(out, p, stage) != parent.decision(out, filename, p):
            raise ValueError('Configuration/epoch selection replay failed')
    rows, replays, drug_rows, cache, boundary = [], [], [], {}, []
    for stage in ('screen', 'confirm', 'outer'):
        for task, arm, cfg, seed, epochs in jobs(out, p, stage):
            dest = parent.directory(out, stage, task, arm, cfg['id'], seed)
            r = parent.checked_result(dest, p)
            identity = (r['task'], r['arm'], r['config'], r['seed'], r['stage'], r['trained_epochs'])
            if identity != (task, arm, cfg, seed, stage, epochs):
                raise ValueError('Result does not match planned job')
            key = (task, 'outer' if stage == 'outer' else 'inner', arm)
            if key not in cache:
                cache[key] = features(out, p, task, stage, arm)
            a = cache[key]
            if r['feature_manifest_sha256'] != a['feature_manifest_sha256']:
                raise ValueError('Training feature hash changed')
            with np.load(dest/'scores.npz') as archive:
                saved = archive['val'].copy()
                np.testing.assert_array_equal(archive['y_val'], a['y_val'])
                np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
            for metric, value in parent.metrics(a['y_val'], saved, a['pairs_val']).items():
                if value is not None:
                    np.testing.assert_allclose(r[metric], value, rtol=0, atol=1e-12)
            trace = pd.read_csv(dest/'trace.csv')
            np.testing.assert_array_equal(trace.epoch, np.arange(1, epochs+1))
            if not np.isfinite(trace.loss).all():
                raise ValueError('Non-finite loss trace')
            if stage != 'outer':
                epoch, score = parent.choose_checkpoint(trace.to_dict('records'), p['smoothing_window'])
                if epoch != r['selected_epoch']:
                    raise ValueError('Checkpoint epoch mismatch')
                np.testing.assert_allclose(score, r['selection_score'], rtol=0, atol=1e-12)
                if epoch >= .95*epochs:
                    boundary.append(dict(task=task, arm=arm, config_id=cfg['id'],
                        seed=seed, stage=stage, selected_epoch=epoch, limit=epochs))
            elif r['selected_epoch'] != epochs or r['selection_score'] is not None or any(
                    metric in trace for metric in ('auc', 'ap', 'criterion', 'macro_drug_ap')):
                raise ValueError('Outer refit did not follow fixed-epoch policy')
            model = table.model_for('tabm', a['x_train'].shape[1], cfg, seed, args.device)
            model.load_state_dict(torch.load(dest/'checkpoint.pt', map_location='cpu', weights_only=False))
            replay = table.predict(model, torch.from_numpy(a['x_val']).to(args.device), p['batch_size'])
            np.testing.assert_allclose(replay, saved, rtol=2e-6, atol=2e-6)
            replays.append(dict(path=str(dest.relative_to(out)),
                max_abs=float(np.max(np.abs(replay-saved)))))
            if stage == 'outer':
                extras, drugs = original.extra_metrics(a['y_val'], saved, a['pairs_val'])
                r.update(extras)
                drug_rows += [dict(task=task, arm=arm, seed=seed, **v) for v in drugs]
            rows.append(r)
    data = pd.DataFrame(rows)
    outer = data[data.stage == 'outer']
    data.to_csv(out/'results.csv', index=False)
    outer.groupby(['task', 'arm'])[list(METRICS)].agg(['mean', 'std']).to_csv(out/'outer_summary.csv')
    pd.DataFrame(drug_rows).to_csv(out/'per_drug.csv', index=False)

    # Recompute prior metrics from validated, pair-aligned predictions, not a CSV.
    reference = Path(p['reference_dir'])
    prior = original.protocol(reference)
    old_rows, anchors = [], []
    for task, arm, cfg, seed, _ in table.jobs(reference, prior, 'outer'):
        if seed not in p['refit_seeds']:
            continue
        dest = parent.directory(reference, 'outer', task, arm, cfg['id'], seed)
        r = parent.checked_result(dest, prior)
        a = cache[(task, 'outer', arm)]
        with np.load(dest/'scores.npz') as archive:
            np.testing.assert_array_equal(a['y_val'], archive['y_val'])
            np.testing.assert_array_equal(a['pairs_val'], archive['pairs_val'])
            for metric, value in parent.metrics(a['y_val'], archive['val'], a['pairs_val']).items():
                if value is not None:
                    np.testing.assert_allclose(r[metric], value, atol=1e-12, rtol=0)
            extras, _ = original.extra_metrics(a['y_val'], archive['val'], a['pairs_val'])
        r.update(extras)
        old_rows.append(r)
    old = pd.DataFrame(old_rows)
    old.to_csv(out/'reference_outer_results.csv', index=False)
    for new_id, old_id in p['anchors'].items():
        for task in p['tasks']:
            for arm in p['arms']:
                before = parent.directory(reference, 'screen', task, arm, old_id, p['screening_seed'])
                after = parent.directory(out, 'screen', task, arm, new_id, p['screening_seed'])
                parent.checked_result(before, prior)
                tr = pd.read_csv(before/'trace.csv')
                tr2 = pd.read_csv(after/'trace.csv').iloc[:len(tr)]
                np.testing.assert_allclose(tr.to_numpy(), tr2.to_numpy(), atol=2e-6, rtol=2e-6)
                anchors.append(dict(task=task, arm=arm, new_config=new_id,
                    previous_config=old_id, epochs=len(tr),
                    max_abs=float(np.max(np.abs(tr.to_numpy()-tr2.to_numpy())))))

    deltas, gates = [], {}
    for task in p['tasks']:
        gates[task] = {}
        for arm in p['arms']:
            for control, kind in ((arm, 'vs_previous_same_arm'), ('base', 'vs_previous_base')):
                d = compare(outer, old, task, arm, control, kind)
                deltas += d
                gates[task][arm+'_'+kind] = performance_gate(d)
        for arm, control in (('random', 'base'), ('search', 'base'), ('excess', 'base'),
                             ('search', 'random'), ('excess', 'search')):
            d = compare(outer, outer, task, arm, control, 'vs_tuned_'+control)
            deltas += d
            gates[task][arm+'_vs_tuned_'+control] = performance_gate(d, macro=control == 'base')
    delta = pd.DataFrame(deltas)
    delta.to_csv(out/'paired_deltas.csv', index=False)
    delta.groupby(['task', 'arm', 'comparison', 'control'])[list(METRICS)].agg(
        ['mean', 'std', lambda v: int((v > 0).sum())]).to_csv(out/'delta_summary.csv')
    parent.write_json(out/'VALIDATION.json', dict(passed=True, smoke=p['smoke'],
        protocol_sha256=p['sha256'], completed_runs=len(rows), no_test_evaluation=True,
        all_checkpoints_replayed=True, selection_replayed=True,
        reference_feature_audit_sha256=p['reference_artifacts_sha256']['FEATURE_RECHECK.json'],
        feature_policy='Unchanged audited artifacts; all feature/dependency hashes verified.',
        old_trace_prefix_checks=anchors, near_horizon=boundary, gates=gates, replays=replays))
    means = outer.groupby(['task', 'arm'])[list(METRICS)].mean()
    old_means = old.groupby(['task', 'arm'])[list(METRICS)].mean()
    lines = ['# 固定结构程序的同预算调参实验', '',
        '历史外层验证，seed 666/0/1 的均值；AP 是 average precision。此表不用于选配置。', '',
        '| 任务 | 方案 | 原 AUC | 调参 AUC | 原 AP | 调参 AP | 原宏 AP | 调参宏 AP |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for task in p['tasks']:
        for arm in p['arms']:
            v, before = means.loc[(task, arm)], old_means.loc[(task, arm)]
            lines.append('| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
                task, arm, before.auc, v.auc, before.ap, v.ap, before.macro_drug_ap, v.macro_drug_ap))
    lines += ['', '## 内层选定的配置', '',
        '| 任务 | 方案 | 配置 | 宽/深 | Dropout | 衰减 | 学习率 | 外层轮数 |',
        '|---|---|---|---|---:|---:|---:|---:|']
    refit = parent.decision(out, 'REFIT.json', p)['chosen']
    for task in p['tasks']:
        for arm in p['arms']:
            v = refit[task][arm]
            c = v['config']
            lines.append('| {} | {} | {} | {}/{} | {} | {} | {} | {} |'.format(
                task, arm, c['id'], c['width'], c['depth'], c['dropout'], c['weight_decay'], c['lr'], v['epochs']))
    lines += ['', '## 预设门槛', '',
        '性能门槛：AUC/AP 均值同时提高、各至少 2/3 种子提高、宏 AP 均值不下降。机制比较不要求宏 AP。', '']
    for task, values in gates.items():
        lines.append('- '+task+'：'+json.dumps(values, ensure_ascii=False))
    lines += ['', '## 核验与边界', '',
        '- {} 次训练的检查点全部重放；指标、轮数、两阶段配置选择重算通过。'.format(len(rows)),
        '- 旧特征缓存及依赖哈希一致；复用此前 24 个支持图的精确重建核验，没有修改规则、空模型或数据划分。',
        '- {} 个旧配置的训练轨迹前缀通过复现检查。'.format(len(anchors)),
        '- {} 次内层训练的选中轮数位于上限最后 5%；详见 VALIDATION.json 的 near_horizon。'.format(len(boundary)),
        '- 外层只按冻结轮数训练，循环内没有验证指标；没有测试集评估。',
        '- 历史内外层都已多次用于开发；三个种子只反映初始化变动，不是独立划分或泛化证明。',
        '- 容量与正则化参数成组改变，不能将收益单独归因于某一个超参数。',
        '- 空模型退化与跨任务侧信息等限制沿用原实验，调网络参数不会解决这些机制限制。', '',
        '完整配置、选择依据、标准差和逐种子配对差见 protocol.json、PARAMETERS.json、REFIT.json、outer_summary.csv、paired_deltas.csv。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(means.to_string(), flush=True)
    print('VERIFIED', len(rows), 'checkpoints', json.dumps(gates), flush=True)


def pipeline(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    execution = []
    started = time.monotonic()
    for stage in ('screen', 'confirm', 'outer'):
        workers = []
        for arm, gpu in (('base', 0), ('random', 1), ('search', 0), ('excess', 1)):
            command = [sys.executable, '-m', 'improve.structure_program_tuning', 'run',
                '--out-dir', str(out), '--stage', stage, '--arm', arm, '--device', 'cuda:'+str(gpu)]
            log = (out/(stage+'_'+arm+'.log')).open('a')
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            workers.append((arm, command, proc, log, time.monotonic()))
        failed = []
        for arm, command, proc, log, begin in workers:
            code = proc.wait()
            log.close()
            execution.append(dict(stage=stage, arm=arm, command=command,
                exit_code=code, elapsed_seconds=time.monotonic()-begin))
            if code:
                failed.append(arm)
        parent.write_json(out/'EXECUTION.json', dict(protocol_sha256=p['sha256'],
            elapsed_seconds=time.monotonic()-started, workers=execution))
        if failed:
            raise RuntimeError('Failed '+stage+' arms: '+', '.join(failed))
        print('STAGE_DONE', stage, 'elapsed', round(time.monotonic()-started, 1), flush=True)
        if stage != 'outer':
            args.stage = stage
            select(args)
    report(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'select', 'report', 'pipeline'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--reference-dir', default='improve/results/structure_program_20260911')
    parser.add_argument('--stage', choices=('screen', 'confirm', 'outer'), default='screen')
    parser.add_argument('--task', choices=parent.TASKS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    dict(prepare=prepare, run=run, select=select, report=report, pipeline=pipeline)[args.command](args)


if __name__ == '__main__':
    main()
