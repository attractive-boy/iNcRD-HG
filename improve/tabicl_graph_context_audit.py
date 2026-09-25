"""Explicit numerical-audit amendment for the frozen graph-context experiment.

The original runner mistakenly applies its 1e-6 SAME-SHAPE replay tolerance to
the baseline column predicted alongside five extra query views. Its protocol
already fixes 1e-5 for CROSS-SHAPE query comparisons. This adapter applies that
existing bound only to the latter check. Model fitting, queries, all candidate
scores, saved-state replay and the already-frozen inner selection are unchanged.
The original source, protocol, artifacts and failed run are retained.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from improve import tabular_foundation as base
from improve import tabicl_graph_context as experiment


DOCUMENT = 'NUMERICAL_AUDIT_PROTOCOL.json'


def freeze(out):
    p = experiment.protocol(out)
    experiment.selection(out, p)
    if (out/DOCUMENT).exists():
        audit_protocol(out, p)
        return
    audit = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=p['sha256'], selection_sha256=base.digest(out/'SELECTION.json'),
        source_sha256={Path(__file__).name:base.digest(__file__)},
        change='Use the already-frozen query_invariance_atol/rtol for cross-query-shape comparisons to original baseline scores. Keep 1e-6 saved-state replay, every prediction computation and the frozen inner selection unchanged.',
        cause='Full-context inference concatenates six query views. A baseline-only original call has a different GPU batch shape; FP32 rounding reached 1.6689300537109375e-6 for outer LncDrug seed 0 and failed the original same-shape assertion.',
        baseline_cross_shape_atol=p['query_invariance_atol'],
        baseline_cross_shape_rtol=p['query_invariance_rtol'],
        saved_state_atol=p['replay_atol'], saved_state_rtol=p['replay_rtol'],
        no_model_or_recipe_change=True, selection_was_already_frozen=True,
        some_outer_outputs_already_observed=True))
    shutil.copy2(__file__, out/'source_snapshot'/Path(__file__).name)
    base.write_json(out/DOCUMENT, audit)
    print('FROZEN_NUMERICAL_AUDIT', audit['sha256'], flush=True)


def audit_protocol(out, p):
    a = base.check_seal(base.read_json(out/DOCUMENT))
    if a['protocol_sha256'] != p['sha256'] or a['selection_sha256'] != base.digest(out/'SELECTION.json'):
        raise ValueError('The frozen experimental selection changed')
    if a['source_sha256'] != {Path(__file__).name:base.digest(__file__)}:
        raise ValueError('Numerical audit source changed')
    if (a['baseline_cross_shape_atol'], a['baseline_cross_shape_rtol']) != (p['query_invariance_atol'], p['query_invariance_rtol']):
        raise ValueError('Cross-shape bound must match the original predeclared query bound')
    if (a['saved_state_atol'], a['saved_state_rtol']) != (p['replay_atol'], p['replay_rtol']):
        raise ValueError('Saved-state replay tolerance cannot change')
    experiment.selection(out, p)
    return a


def full_context_job(out, p, audit, phase, task, seed, a, device):
    """Same full-context computation as the frozen runner; only its audit differs."""
    import torch
    dest = experiment.directory(out, phase, task, seed, 'full')
    if (dest/'result.json').exists():
        return experiment.checked_result(dest, p)
    x, y, queries = experiment.context_and_queries(p, task, a, 'full')
    tp = base.read_json(Path(p['parent_dir'])/'protocol.json')
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    clf = base.fit_context(base.estimator(dict(n_estimators=p['inference_members']), seed, tp, device), x, y)
    if base.parameter_digest(clf) != p['parameter_sha256']:
        raise ValueError('Official pretrained parameters changed')
    score = base.query_scores(clf, queries.reshape(-1, x.shape[1])).reshape(queries.shape[:2])
    if base.parameter_digest(clf) != p['parameter_sha256']:
        raise ValueError('Inference updated model parameters')
    baseline_error = None
    if not p['smoke']:
        original = experiment.baseline_score(p, phase, task, seed, a)
        np.testing.assert_allclose(score[0], original, atol=audit['baseline_cross_shape_atol'],
                                   rtol=audit['baseline_cross_shape_rtol'])
        baseline_error = float(np.max(np.abs(score[0]-original)))
    dest.mkdir(parents=True, exist_ok=True)
    clf.save(dest/'fitted.pkl', save_model_weights=False, save_training_data=True, save_kv_cache=False)
    np.savez_compressed(dest/'scores.npz', views=score, y_val=a['y_val'], pairs_val=a['pairs_val'])
    result = dict(protocol_sha256=p['sha256'], phase=phase, task=task, seed=seed, kind='full',
        context_rows=len(y), n_query_views=len(queries), inference_members=p['inference_members'],
        parameter_sha256=p['parameter_sha256'], graph_manifest_sha256=a['graph_manifest_sha256'],
        baseline_replay_max_abs=baseline_error, elapsed_seconds=time.monotonic()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device), pretrained_parameter_updates=0,
        numerical_audit_protocol_sha256=audit['sha256'],
        artifacts_sha256={name:base.digest(dest/name) for name in ('scores.npz', 'fitted.pkl')})
    base.write_json(dest/'result.json', result)
    print('SAVED_AUDITED', phase, task, seed, 'full', 'baseline_cross_shape_error', baseline_error, flush=True)
    del clf
    gc.collect()
    torch.cuda.empty_cache()
    return result


def resume(args):
    import torch
    out = Path(args.out_dir).resolve()
    p = experiment.protocol(out)
    audit = audit_protocol(out, p)
    if args.task not in p['tasks']:
        raise ValueError('Unknown task')
    torch.cuda.set_device(args.device)
    with (out/('outer_%s.lock' % args.task)).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a = experiment.features(out, p, args.task, 'outer')
        for seed in p['seeds']:
            for kind in experiment.KINDS:
                if kind == 'full':
                    full_context_job(out, p, audit, 'outer', args.task, seed, a, args.device)
                else:
                    experiment.run_job(out, p, 'outer', args.task, seed, kind, a, args.device)


def verify(args):
    import torch
    from tabicl import TabICLClassifier
    out = Path(args.out_dir).resolve()
    p = experiment.protocol(out)
    audit = audit_protocol(out, p)
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    replays = []
    for phase in ('inner', 'outer'):
        a = experiment.features(out, p, args.task, phase)
        for seed in p['seeds']:
            for kind in experiment.KINDS:
                dest = experiment.directory(out, phase, args.task, seed, kind)
                r = experiment.checked_result(dest, p)
                if (r['phase'], r['task'], r['seed'], r['kind']) != (phase, args.task, seed, kind):
                    raise ValueError('Wrong estimator identity')
                x, y, query = experiment.context_and_queries(p, args.task, a, kind)
                clf = TabICLClassifier.load(dest/'fitted.pkl', device=args.device)
                if clf.n_estimators != p['inference_members'] or clf.random_state != seed or clf.allow_auto_download:
                    raise ValueError('Saved estimator settings changed')
                if base.parameter_digest(clf) != p['parameter_sha256'] or clf.n_samples_in_ != len(y):
                    raise ValueError('Saved parameters or context row count changed')
                np.testing.assert_array_equal(clf.ensemble_generator_.y_, y)
                expected = clf.ensemble_generator_.unique_filter_.transform(clf.X_encoder_.transform(x))
                np.testing.assert_allclose(clf.ensemble_generator_.X_, expected, atol=1e-7, rtol=1e-7)
                flat = query.reshape(-1, x.shape[1])
                replay = base.query_scores(clf, flat).reshape(query.shape[:2])
                with np.load(dest/'scores.npz', allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive['y_val'], a['y_val'])
                    np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
                    saved = archive['views'].copy()
                np.testing.assert_allclose(replay, saved, atol=audit['saved_state_atol'], rtol=audit['saved_state_rtol'])
                entry = dict(path=str(dest.relative_to(out)), result_sha256=base.digest(dest/'result.json'),
                             max_abs=float(np.max(np.abs(replay-saved))))
                if kind == 'full' and not p['smoke']:
                    original = experiment.baseline_score(p, phase, args.task, seed, a)
                    np.testing.assert_allclose(replay[0], original, atol=audit['baseline_cross_shape_atol'],
                                               rtol=audit['baseline_cross_shape_rtol'])
                    entry['baseline_cross_shape_max_abs'] = float(np.max(np.abs(replay[0]-original)))
                if p['smoke']:
                    reverse = base.query_scores(clf, flat[::-1])[::-1]
                    split = np.concatenate([base.query_scores(clf, part) for part in np.array_split(flat, 3)])
                    np.testing.assert_allclose(reverse, replay.ravel(), atol=p['query_invariance_atol'], rtol=p['query_invariance_rtol'])
                    np.testing.assert_allclose(split, replay.ravel(), atol=p['query_invariance_atol'], rtol=p['query_invariance_rtol'])
                    entry.update(query_permutation_max_abs=float(np.max(np.abs(reverse-replay.ravel()))),
                                 query_splitting_max_abs=float(np.max(np.abs(split-replay.ravel()))))
                replays.append(entry)
                print('REPLAYED_AUDITED', phase, args.task, seed, kind, entry['max_abs'], flush=True)
                del clf
                gc.collect()
                torch.cuda.empty_cache()
    base.write_json(out/('VERIFY_%s.json' % args.task), dict(protocol_sha256=p['sha256'],
        numerical_audit_protocol_sha256=audit['sha256'], task=args.task, passed=True, replays=replays))


def report(out):
    p = experiment.protocol(out)
    audit = audit_protocol(out, p)
    for task in p['tasks']:
        v = base.read_json(out/('VERIFY_%s.json' % task))
        if v.get('numerical_audit_protocol_sha256') != audit['sha256']:
            raise ValueError('Missing amended numerical verification')
    experiment.report(out)
    validation = base.read_json(out/'VALIDATION.json')
    validation.update(numerical_audit_protocol_sha256=audit['sha256'],
        baseline_cross_shape_max_abs=max((row.get('baseline_cross_shape_max_abs', 0.) for row in validation['replays']), default=0.),
        original_inner_selection_unchanged=True, prediction_computation_unchanged=True)
    base.write_json(out/'VALIDATION.json', validation)
    with (out/'REPORT.md').open('a') as stream:
        stream.write('\n## 数值核验修订\n\n'
            '原运行器把等形状重放的 1e-6 容差也用于“原基线单批查询”与“拼接六种查询视图”的比较，'
            '完整规模的 LncDrug seed 0 因约 1.7e-6 的 FP32 差异触发断言。'
            '该比较现使用原协议已预设的分批查询容差（atol=rtol=1e-5），'
            '等形状保存状态重放仍为 1e-6。模型、输入、所有预测计算及已冻结的内部选择均未更改。'
            '修订前已经观察到部分外层输出，修订仅涉及数值审核。'
            '原源码与协议保留，修订单独冻结为 NUMERICAL_AUDIT_PROTOCOL.json。\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'resume', 'verify', 'report', 'pipeline'))
    parser.add_argument('--out-dir', default=experiment.DEFAULT_OUT)
    parser.add_argument('--task', choices=('LncDrug', 'MiDrug'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    if args.command == 'freeze':
        freeze(out)
    elif args.command == 'resume':
        resume(args)
    elif args.command == 'verify':
        verify(args)
    elif args.command == 'report':
        report(out)
    else:
        p = experiment.protocol(out)
        audit_protocol(out, p)
        devices = args.devices.split(',')
        for stage in ('resume', 'verify'):
            experiment.workers([[sys.executable, '-u', '-m', 'improve.tabicl_graph_context_audit', stage,
                '--out-dir', str(out), '--task', task, '--device', devices[i % len(devices)]]
                for i, task in enumerate(p['tasks'])], 'audited_'+stage)
        report(out)


if __name__ == '__main__':
    main()
