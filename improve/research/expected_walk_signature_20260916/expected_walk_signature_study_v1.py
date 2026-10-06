"""Freeze, execute and audit the prospective expected walk signature study."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import torch

from improve import anchor_expected_walk_signature_experiment_v1 as experiment
from improve import expected_walk_signature_graph_v1 as core
from improve import assess_declared_graph as assessor
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT
CORE = ROOT/'results/expected_walk_signature_frozen_core_v1_20260916'
OUT = ROOT/'results/anchor_priority_expected_walk_signature_v1_20260916'


def prepare():
    smoke_path, unit_path = HERE/'EXPECTED_WALK_SIGNATURE_SMOKE_V1.json', HERE/'EXPECTED_WALK_SIGNATURE_UNIT_V1.json'
    smoke, unit = [base.check_seal(base.read_json(p)) for p in (smoke_path, unit_path)]
    assert unit['success'] and unit['tests'] == 6
    assert smoke['formal_fits'] == 0 and smoke['short_neural_probes'] == 3
    assert not smoke['evaluation_labels_read'] and not smoke['performance_metrics_computed']
    assert smoke['core_parameters'] == core.parameters()
    assert len(smoke['records']) == 6 and len(smoke['fold_audits']) == 40
    for records in (smoke['source_sha256'], smoke['input_sha256'], unit['source_sha256']):
        for path, digest in records.items(): assert base.digest(path) == digest, path
    if CORE.exists() or OUT.exists(): raise ValueError('Use a new version; do not replace studies')
    sources = ['expected_walk_signature_graph_v1.py', 'anchor_expected_walk_signature_experiment_v1.py',
        'test_expected_walk_signature_graph_v1.py', 'test_nested_anchor_v2.py', 'sharpness_graph_v1.py',
        'overlap_block_models.py', 'overlap_block_experiment.py', 'non_tabular_experiment.py',
        'tabular_foundation.py', 'graph_prediction_precision.py', 'nested_anchor_inputs_v2.py',
        'residual_graph_experiment.py', 'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    inputs = [smoke_path, unit_path, Path(__file__), HERE/'expected_walk_signature_smoke_v1.py',
        HERE/'EXPECTED_WALK_SIGNATURE_ADAPTATION_V1.md', HERE/'DISCOVERY_V1.json',
        HERE/'SOURCE_RETRIEVAL_V1.json', HERE/'PDF_RETRIEVAL_V1.json', HERE/'FOLLOWUP_RETRIEVAL_V1.json',
        HERE/'hypoelliptic_exact_crossref_v1.json', HERE/'hypoelliptic_v1.pdf', HERE/'hypoelliptic_v1.txt',
        HERE/'signature_thesis_v2.pdf', HERE/'signature_thesis_v2.txt',
        HERE/'pretraining_row_sum_roundoff_v1/FAILURE.json']
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='expected_walk_signature',
        implementation_module='expected_walk_signature_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Ordered walk-bridge second-level signature primary; symmetrized endpoint moment and first-level mandatory controls. Learning rates 0.001/0.003, 400 steps, 20-step checkpoints; no horizon, degree, rank, mass-floor or fusion sweep.',
        required_executor='improve.anchor_expected_walk_signature_experiment_v1',
        source_sha256={name:base.digest(ROOT/name) for name in sources},
        inputs={str(path.resolve()):base.digest(path) for path in inputs},
        literature=dict(foundational_doi='10.52202/068431-2812', foundational_year=2022,
            later_exposition='2506.17634v2', primary_fulltext='2205.14092v1',
            adaptation_is_independent=True, author_algorithms_or_benchmarks_reproduced=False),
        mechanism='Exact conditional second-level signatures of all four-step support-graph walks between RNA and drug endpoints, retaining ordered Levy area',
        distinctness='Must beat symmetrized and first-level controls on both global MiDrug metrics; only one expected-signature family.',
        numerical_gate='Float64 Markov probabilities and signature recurrence; explicit all-path enumeration and derivatives verified; finite conditional moments, loss, gradients, weights and optimizer; exact-zero missing bridge mask; replay 3e-5.',
        training_budget='400 AdamW updates per formal fit; all arms share parameter counts, initialization and dropout streams. Sparse operations and backward FLOPs differ and are not claimed equal; first-level control has inactive degree-two decoder columns.',
        evaluation='Historical screen, eligible three-initialization confirmation, then fixed development partitions and frozen old-outer recheck; stop at any failed gate.',
        previous_failures_preserved=True, previous_outer_exposure_known=True,
        no_new_performance_observed=True, original_test_source_access=False,
        confirmed_innovations=0, existing_default_replaced=False, smoke=False)
    CORE.mkdir(parents=True)
    (CORE/'source_snapshot').mkdir()
    for name in sources: shutil.copy2(ROOT/name, CORE/'source_snapshot'/name)
    base.write_json(CORE/'protocol.json', p)
    experiment.prepare(OUT, 'expected_walk_signature', ROOT/'results/nested_anchor_v2_20260915/historical', CORE)
    q = base.read_json(OUT/'protocol.json')
    q['inputs'].update(p['inputs'])
    for key in ('required_executor', 'training_budget', 'numerical_gate', 'distinctness'):
        q[key] = p[key]
    base.write_json(OUT/'protocol.json', q)
    base.write_json(HERE/'EXPECTED_WALK_SIGNATURE_QUEUE_PROTOCOL_V1.json', base.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), family='expected_walk_signature',
        protocol_sha256=base.digest(OUT/'protocol.json'), study_source_sha256=base.digest(__file__),
        device='cuda:1', stages=['screen', 'confirm_if_screen_passes'],
        expected_new_neural_fits_per_stage=6, expected_identity_records_per_stage=6,
        stop_on_failed_performance_mechanism_or_numerical_gate=True,
        final_success_requires_partitions_and_outer=True)))
    print('EXPECTED_WALK_SIGNATURE_FROZEN', base.digest(OUT/'protocol.json'), flush=True)


def check_queue():
    queue = base.check_seal(base.read_json(HERE/'EXPECTED_WALK_SIGNATURE_QUEUE_PROTOCOL_V1.json'))
    assert queue['study_source_sha256'] == base.digest(__file__)
    assert queue['protocol_sha256'] == base.digest(OUT/'protocol.json')
    experiment.common.protocol(OUT)
    return queue


def run_queue():
    queue = check_queue()
    destination = HERE/'EXPECTED_WALK_SIGNATURE_QUEUE_RESULT_V1.json'
    if destination.exists(): raise ValueError('Queue already completed; use audit')
    for stage in ('screen', 'confirm'):
        print('EXPECTED_WALK_SIGNATURE_STAGE', stage, datetime.now(timezone.utc).isoformat(), flush=True)
        for task in ('MiDrug', 'LncDrug'):
            experiment.run(OUT, task, 'cuda:1', stage)
        if not (OUT/'SELECTION.json').exists(): experiment.select(OUT)
        with (OUT/(stage+'_assessment.log')).open('a') as log, redirect_stdout(log):
            result = assessor.assess(OUT, stage)
        print('EXPECTED_WALK_SIGNATURE_GATES', stage, result['performance_gate'],
              result['primary_mechanism_gate'], result['offset_null_gate'], flush=True)
        if not result['declared_gate']:
            base.write_json(OUT/(stage.upper()+'_STOP_DECISION.json'), base.seal(dict(
                created_at_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                decision='Stop this frozen version; do not substitute a better-looking control',
                performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
                offset_gate=result['offset_null_gate'], confirmed_final_innovations=0)))
            break
    records = [base.read_json(p) for p in (OUT/'runs').glob('*/*/*/*/result.json')]
    base.write_json(destination, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=queue['sha256'], last_stage=stage, declared_gate=result['declared_gate'],
        new_neural_fits=sum(r['trained_neural_model'] for r in records),
        identity_scoring_records=sum(not r['trained_neural_model'] for r in records),
        new_partition_fits=0, new_outer_fits=0, partition_verification_pending=result['declared_gate'],
        confirmed_final_innovations=0, existing_default_replaced=False)))
    print('EXPECTED_WALK_SIGNATURE_QUEUE_COMPLETE', stage, result['declared_gate'], flush=True)


def audit():
    check_queue()
    queue = base.check_seal(base.read_json(HERE/'EXPECTED_WALK_SIGNATURE_QUEUE_RESULT_V1.json'))
    p = experiment.common.protocol(OUT)
    ph = base.digest(OUT/'protocol.json')
    table = experiment.common.all_rows(OUT)
    selection = base.read_json(OUT/'SELECTION.json')
    assert selection['protocol_sha256'] == ph
    rows, random_states = [], {}
    for path in sorted((OUT/'runs').glob('*/*/*/*')):
        r = experiment.common.verify(path, ph)
        trace = base.read_json(path/'trace.json')
        scores = experiment.common.load(path/'scores.npz')
        state = torch.load(path/'model.pt', map_location='cpu', weights_only=True)
        if r['trained_neural_model']:
            assert r['task'] == 'MiDrug'
            assert r['training_steps'] == r['gradient_evaluations'] == r['optimizer_steps'] == 400
            assert len(trace) == 20 and [v['epoch'] for v in trace] == list(range(20, 401, 20))
            assert r['signature_diagnostics']['forwards'] == 505
            recurrences = 505*core.parameters()['walk_steps'] if r['kind'] == 'full' else 0
            assert r['signature_diagnostics']['bridge_recurrences'] == recurrences
            assert r['signature_diagnostics']['bridge_sparse_products'] == 3*recurrences
            assert all(np.isfinite(v) for v in r['signature_diagnostics'].values())
            final = torch.load(path/'training_final.pt', map_location='cpu', weights_only=True)
            assert final['steps'] == 400
            for value in list(state.values())+list(final['model'].values()):
                if value.is_floating_point(): assert torch.isfinite(value).all()
            indexes = final['optimizer']['param_groups'][0]['params']
            assert len(indexes) == len(final['optimizer_parameter_names']) == len(final['optimizer']['state'])
            for index in indexes:
                o = final['optimizer']['state'][index]
                assert int(o['step']) == 400
                assert all(torch.isfinite(v).all() for v in o.values() if torch.is_tensor(v))
            best = max(trace, key=lambda v:(v['anchor_composed']['auc']+v['anchor_composed']['ap'])/2)
            assert best['epoch'] == r['epoch']
            for metric in ('auc', 'ap', 'macro_drug_ap'):
                assert best['anchor_composed'][metric] == r['metrics']['anchor_composed'][metric]
            key = (r['lr'], r['seed'])
            random_states.setdefault(key, []).append((final['cpu_rng'], final['cuda_rng']))
        else:
            assert r['task'] == 'LncDrug' and not state and not trace
            assert r['training_steps'] == r['parameters'] == r['epoch'] == 0
            assert not (path/'training_final.pt').exists()
            np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
        rows.append(dict(task=r['task'], kind=r['kind'], lr=r['lr'], seed=r['seed'],
            epoch=r['epoch'], trained_neural_model=r['trained_neural_model'],
            metrics=r['metrics']['anchor_composed'], replay_max_abs=r['replay_max_abs'],
            result=str(path/'result.json'), result_sha256=base.digest(path/'result.json')))
    for states in random_states.values():
        for values in states[1:]:
            for x, y in zip(values, states[0]): torch.testing.assert_close(x, y, atol=0, rtol=0)
    name = 'SCREEN_ASSESSMENT.json' if queue['last_stage'] == 'screen' else 'HISTORICAL_ASSESSMENT.json'
    assessment = base.check_seal(base.read_json(OUT/name))
    selected = {}
    seeds = [666]+([0, 1] if queue['last_stage'] == 'confirm' else [])
    for task in p['tasks']:
        for kind in core.KINDS:
            for seed in seeds:
                path = OUT/'runs'/task/kind/('lr'+str(selection['chosen'][kind]['lr']))/('s'+str(seed))
                selected[task, kind, seed] = experiment.common.verify(path, ph)
    expected = assessor.evaluate(p, selected, seeds)
    for key in expected: assert expected[key] == assessment[key], key
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=ph, assessment_sha256=base.digest(OUT/name),
        queue_result_sha256=base.digest(HERE/'EXPECTED_WALK_SIGNATURE_QUEUE_RESULT_V1.json'),
        new_neural_fits=sum(r['trained_neural_model'] for r in rows),
        identity_scoring_records=sum(not r['trained_neural_model'] for r in rows),
        all_metrics_recomputed=True, all_optimizer_steps_verified=True,
        all_checkpoint_selection_verified=True, paired_random_streams_verified=True,
        input_and_source_hashes_verified=True, all_lnc_scores_exact_identity=True,
        maximum_replay_error=max(r['replay_max_abs'] for r in rows),
        assessment_recomputed=True, declared_gate=assessment['declared_gate'], records=rows,
        confirmed_final_innovations=0, original_test_source_access=False))
    destination = HERE/'EXPECTED_WALK_SIGNATURE_CONSOLIDATION_V1.json'
    if destination.exists(): raise ValueError('Preserve earlier audit evidence')
    base.write_json(destination, result)
    table.to_csv(OUT/'ALL_SCORES_RECOMPUTED_V1.csv', index=False)
    print('EXPECTED_WALK_SIGNATURE_AUDIT', result['new_neural_fits'], result['identity_scoring_records'],
          result['declared_gate'], result['maximum_replay_error'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'queue', 'audit'))
    command = parser.parse_args().command
    {'prepare':prepare, 'queue':run_queue, 'audit':audit}[command]()
