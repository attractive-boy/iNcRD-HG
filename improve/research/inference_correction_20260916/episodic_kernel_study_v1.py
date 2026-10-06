"""Prospective checks and gated execution for episodic conditional graph learning."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_priority_graph_v1 as experiment
from improve import episodic_kernel_graph_v1 as core
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

ROOT = experiment.ROOT
HERE = Path(__file__).resolve().parent
FAMILY = 'episodic_kernel'
STUDY = ROOT / 'results/anchor_priority_episodic_kernel_v1_20260916'
CORE = ROOT / 'results/episodic_kernel_core_v1_20260916'


def now():
    return datetime.now(timezone.utc).isoformat()


def smoke():
    target = HERE / 'EPISODIC_KERNEL_SMOKE_V1.json'
    if target.exists():
        raise ValueError('Preserve smoke evidence')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family=FAMILY, implementation_module='episodic_kernel_graph_v1',
        source_sha256={'episodic_kernel_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5,
        check_interval=20, weight_decay=.001, strong_weight=.1)
    source = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs/MiDrug/arrays.npz'
    with np.load(source, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in experiment.TRAIN_KEYS}
    runs = []
    for kind in core.KINDS:
        torch.cuda.reset_peak_memory_stats('cuda:1')
        model, _, trace, detail = experiment.fit(a, p, kind, .001, 31677, 'cuda:1', fixed_epochs=5)
        loaded, xs, episodes, _ = experiment.setup(a, p, kind, 'cuda:1')
        loaded.load_state_dict(model.state_dict())
        query = torch.as_tensor(a['pairs_train'][:, :31] + a['pair_offset'][:, None], device='cuda:1')
        first = experiment.predict(model, xs, episodes, query, p).cpu().numpy()
        replay = experiment.predict(loaded, xs, episodes, query.flip(1), p).cpu().numpy()[::-1]
        np.testing.assert_allclose(first, replay, atol=3e-5, rtol=3e-5)
        if not np.isfinite(first).all() or abs(first).max() > 2.:
            raise ValueError('Invalid bounded prediction')
        runs.append(dict(kind=kind, detail=detail, replay_max_abs=float(abs(first-replay).max()),
                         peak_memory_bytes=torch.cuda.max_memory_allocated('cuda:1')))
        print('EPISODIC_SMOKE', runs[-1], flush=True)
        del model, loaded, xs, episodes
        torch.cuda.empty_cache()
    audits, files = [], {str(source): base.digest(source)}
    for split in ('historical', '61453', '7919', '104729'):
        path = source if split == 'historical' else ROOT / ('results/mechanism_validation_20260915/inputs/MiDrug/' + split + '/arrays.npz')
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k].copy() for k in experiment.TRAIN_KEYS if k in z.files}
        _, _, episodes, folds = core.setup(a, core.parameters(), 'full', 'cpu')
        for episode, ids in zip(episodes, folds):
            forbidden = a['pairs_train'][:, np.r_[episode['support_ids'], ids.numpy()]] + a['pair_offset'][:, None]
            arcs = episode['graph'][0].numpy()
            codes = arcs[0] * int(a['node_counts'].sum()) + arcs[1]
            for q in (forbidden, forbidden[::-1]):
                if np.isin(q[0] * int(a['node_counts'].sum()) + q[1], codes).any():
                    raise ValueError('Held-out fact visible in an episode')
            audits.append(dict(split=split, query_fold=episode['query_fold'], support_fold=episode['support_fold'],
                               query_rows=len(ids), support_rows=len(episode['support_ids']), forbidden_overlap=0))
        files[str(path)] = base.digest(path)
    base.write_json(target, base.seal(dict(created_at_utc=now(), runs=runs, episode_audits=audits,
        performance_metrics_computed=False, validation_outcomes_loaded=False, formal_fits=0,
        source_sha256={str(Path(core.__file__)): base.digest(core.__file__), str(Path(__file__)): base.digest(__file__)},
        input_sha256=files)))


def freeze():
    if STUDY.exists() or CORE.exists():
        raise ValueError('Preserve existing protocol')
    unit = base.check_seal(base.read_json(HERE / 'EPISODIC_KERNEL_UNIT_V1.json'))
    short = base.check_seal(base.read_json(HERE / 'EPISODIC_KERNEL_SMOKE_V1.json'))
    if not unit['passed'] or unit['tests_run'] != 4 or len(short['episode_audits']) != 80:
        raise ValueError('Implementation checks incomplete')
    for record in (unit, short):
        for path, digest in record['source_sha256'].items():
            if base.digest(path) != digest:
                raise ValueError('Checked implementation changed')
    names = tuple(dict.fromkeys(experiment.OWN + ('episodic_kernel_graph_v1.py',
        'test_episodic_kernel_graph_v1.py', 'sharpness_graph_v1.py', 'overlap_block_models.py')))
    evidence = [Path(__file__), HERE / 'EPISODIC_KERNEL_ADAPTATION_V1.md',
                HERE / 'EPISODIC_KERNEL_UNIT_V1.json', HERE / 'EPISODIC_KERNEL_SMOKE_V1.json',
                HERE / '2604_22031.pdf', HERE / 'mochi_commits_v1.json', HERE / 'MOCHI_AUTHOR_FETCH_V1.json']
    evidence.extend(HERE.glob('author_*'))
    p = dict(created_at_utc=now(), family=FAMILY, implementation_module='episodic_kernel_graph_v1',
        tasks=['MiDrug', 'LncDrug'], primary_kind='full', primary_controls=['stop_support_gradient'],
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        literature=dict(arxiv='2604.22031v2', status='preprint', author_commit='420966d106c28fde6583bc128b872b26263fb146'),
        local_adaptation='Typed graph and learned pair embeddings; support/query/graph triple separation; primal FP64 ridge; bounded residual around unchanged reference; 20 support/graph configurations; no author checkpoint.',
        selection='Same per-arm two-learning-rate and every-20-step global AUC/AP selection as the existing reference-residual protocol.',
        source_sha256={n: base.digest(ROOT/n) for n in names},
        inputs={str(path): base.digest(path) for path in evidence},
        no_performance_observed=True, original_test_source_access=False, confirmed_innovations=0)
    CORE.mkdir(parents=True)
    (CORE/'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(ROOT/name, CORE/'source_snapshot'/name)
    base.write_json(CORE/'protocol.json', p)
    experiment.prepare(STUDY, FAMILY, experiment.nested.DEFAULT, CORE)


def run():
    target = HERE/'EPISODIC_KERNEL_QUEUE_RESULT_V1.json'
    if target.exists():
        raise ValueError('Preserve completed queue')
    p = experiment.common.protocol(STUDY)
    for stage in ('screen', 'confirm'):
        for task in ('LncDrug', 'MiDrug'):
            with (STUDY/(stage+'_'+task+'.log')).open('a', buffering=1) as stream:
                subprocess.run([sys.executable, '-u', '-m', 'improve.anchor_priority_graph_v1', stage,
                    '--out', str(STUDY), '--family', FAMILY, '--task', task, '--device', 'cuda:1'],
                    stdout=stream, stderr=subprocess.STDOUT, check=True)
        with (STUDY/(stage+'_assessment.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
            if stage == 'screen':
                experiment.select(STUDY)
            assessment = audit.assess(STUDY, stage)
        print('EPISODIC_KERNEL_GATES', stage, assessment['performance_gate'], assessment['primary_mechanism_gate'], flush=True)
        if not assessment['declared_gate']:
            break
    records = [base.read_json(x) for x in (STUDY/'runs').glob('*/*/*/*/result.json')]
    base.write_json(target, base.seal(dict(created_at_utc=now(), stage=stage,
        protocol_sha256=base.digest(STUDY/'protocol.json'), declared_gate=assessment['declared_gate'],
        new_neural_fits=sum(r['trained_neural_model'] for r in records),
        identity_scoring_records=sum(not r['trained_neural_model'] for r in records),
        partition_verification_pending=assessment['declared_gate'], confirmed_final_innovations=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke','freeze','run'))
    args = parser.parse_args()
    globals()[args.command]()
