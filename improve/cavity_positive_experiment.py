"""Positive-support NBA confirmation, chosen before additional initializations.

The predecessor found sampled-zero propagation harmful. This version reuses
its unchanged positive-only screening fits and supplies input-matched nulls.
Observation counts remain available to every arm; no zero-labeled edge carries
a message. Source and result records of the predecessor are retained.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from improve import cavity_experiment as predecessor

common, runner = predecessor.common, predecessor.runner
OUT = common.base.ROOT / 'results/cavity_positive_v2_20260915'
KINDS = ('full', 'backtracking', 'edge_mlp')


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unknown positive-support cavity control')
    model, xs, graphs, folds = predecessor.setup(a, p, 'positive_only', device)
    model.kind = kind
    return model, xs, graphs, folds


def prepare(out):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new output directory')
    parent = common.protocol(predecessor.OUT)
    p = {k: v for k, v in parent.items() if k not in ('source_sha256', 'inputs')}
    own = ('cavity_positive_experiment.py', 'test_cavity_positive_experiment.py')
    p.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
        method='cavity_positive', executor_module='improve.cavity_positive_experiment', kinds=list(KINDS),
        primary_kind='full', primary_controls=['backtracking'],
        predecessor=str(predecessor.OUT), predecessor_protocol_sha256=common.base.digest(predecessor.OUT / 'protocol.json'),
        selection_before_confirmation='Only predecessor seed 666 screen outcomes were used to select positive-only message propagation. No seeds 0/1 for this version and no partition predictions for this version exist at registration.',
        boundary='Positive fitting-query facts are deleted before inverse arcs, grouping and all messages. Sampled-zero fitting-query facts are deleted before counts. Sampled zeros supply marginal observation counts only, identically in all arms. All message graphs contain existing physical relations and positive support only.',
        adaptation=parent['adaptation'].replace('Adds sampled zero-labeled fitting pairs as an observation relation; this is not biological negative evidence.',
            'Sampled zero-labeled fitting pairs enter marginal observation counts only, never propagation; this is not biological negative evidence.'),
        controls='Backtracking uses identical graphs, node statistics, initialized parameters and training budgets, but permits return messages. Edge-MLP uses the same graphs and edge initial states but disables cross-edge recursion. No claimed benefit from zero-labeled propagation.',
        gate='MiDrug mean AUC/AP improve, each in >=2/3 paired confirmations; every LncDrug metric loss <=0.001. Full must beat the identical-input backtracking control on mean MiDrug AUC/AP. Edge-MLP is separately reported as a propagation diagnostic.',
        reuse_policy='Four existing positive_only seed-666 fits map to full with unchanged parameters, traces, scores and selected steps. Saved-model inference is checked through this executor before any new fitting. Reused fits are not counted as new training.',
        source_sha256={**parent['source_sha256'], **{n: common.base.digest(common.base.ROOT / n) for n in own}},
        inputs={str(predecessor.OUT / 'protocol.json'): common.base.digest(predecessor.OUT / 'protocol.json')})
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(common.base.ROOT / name, out / 'source_snapshot' / name)
    for task in p['tasks']:
        source = predecessor.OUT / 'inputs' / task / 'arrays.npz'
        dest = out / 'inputs' / task / 'arrays.npz'
        dest.parent.mkdir(parents=True)
        shutil.copy2(source, dest)
        for path in (source, dest):
            p['inputs'][str(path)] = common.base.digest(path)
        for lr in p['lrs']:
            source = predecessor.OUT / 'runs' / task / 'positive_only' / ('lr' + str(lr)) / 's666'
            common.verify(source, p['predecessor_protocol_sha256'])
            for path in source.iterdir():
                if path.is_file():
                    p['inputs'][str(path)] = common.base.digest(path)
    common.base.write_json(out / 'protocol.json', p)
    ph = common.base.digest(out / 'protocol.json')
    for task in p['tasks']:
        for lr in p['lrs']:
            source = predecessor.OUT / 'runs' / task / 'positive_only' / ('lr' + str(lr)) / 's666'
            dest = out / 'runs' / task / 'full' / ('lr' + str(lr)) / 's666'
            dest.mkdir(parents=True)
            r = json.loads((source / 'result.json').read_text())
            for name in r['files']:
                shutil.copy2(source / name, dest / name)
            r.update(kind='full', protocol_sha256=ph, reused_from=str(source),
                     reused_result_sha256=common.base.digest(source / 'result.json'))
            common.base.write_json(dest / 'result.json', r)
    print('FROZEN_POSITIVE_CAVITY', out, 'reused fits=4', flush=True)


def audit_reuse(out, device):
    out = Path(out)
    p = common.protocol(out)
    torch.set_num_threads(4)
    checked = []
    for task in p['tasks']:
        a = common.load(out / 'inputs' / task / 'arrays.npz')
        for lr in p['lrs']:
            dest = out / 'runs' / task / 'full' / ('lr' + str(lr)) / 's666'
            model, xs, graphs, _ = setup(a, p, 'full', device)
            model.load_state_dict(torch.load(dest / 'model.pt', weights_only=True, map_location=device))
            q = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
            replay = runner.precision.geometry_prediction(model, xs, graphs, q).cpu().numpy()
            stored = common.load(dest / 'scores.npz')['standalone']
            np.testing.assert_allclose(replay, stored, atol=1e-10, rtol=1e-10)
            checked.append(dict(task=task, lr=lr, max_abs=float(abs(replay - stored).max()),
                                result_sha256=common.base.digest(dest / 'result.json')))
            del model, xs, graphs
            torch.cuda.empty_cache()
    common.base.write_json(out / 'REUSE_AUDIT.json', dict(protocol_sha256=common.base.digest(out / 'protocol.json'),
                                                      replayed=checked, new_training_runs=0))
    print('REUSED_FITS_REPLAYED', checked, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'audit-reuse', 'screen', 'select', 'confirm', 'report'))
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--task', choices=common.base.TASKS)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.command == 'prepare':
        prepare(args.out)
    elif args.command == 'audit-reuse':
        audit_reuse(args.out, args.device)
    elif args.command in ('screen', 'confirm'):
        if not (Path(args.out) / 'REUSE_AUDIT.json').exists():
            raise ValueError('Existing fits must be replayed through this executor first')
        runner.run(args.out, args.task, args.device, args.command)
    elif args.command == 'select':
        common.select(args.out)
    else:
        common.report(args.out)
