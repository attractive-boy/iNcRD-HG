"""Register the narrower joint-tuple hypothesis before new control performance."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_spectral_persistence_experiment_v1 as experiment
from improve import persistence_pairing_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    unit_path, smoke_path = HERE / 'PERSISTENCE_PAIRING_UNIT_V1.json', HERE / 'PERSISTENCE_PAIRING_SMOKE_V1.json'
    unit, smoke = (base.check_seal(base.read_json(path)) for path in (unit_path, smoke_path))
    assert unit['success'] and unit['tests'] == 4
    assert smoke['new_short_neural_fits'] == 1 and smoke['new_formal_fits'] == 0
    assert not smoke['new_performance_metrics_computed'] and not smoke['new_evaluation_labels_read']
    assert smoke['core_parameters'] == core.parameters() and smoke['inherited_fold_audit_count'] == 40
    assert smoke['primary_prediction_bitwise_identical'] and smoke['final_control_rng_matches_parent_primary']
    for mapping in (unit['source_sha256'], smoke['source_sha256'], smoke['input_sha256']):
        for path, digest in mapping.items():
            assert base.digest(path) == digest, path
    parent_dir = experiment.ROOT / 'results/anchor_priority_spectral_persistence_v1_20260916'
    previous = experiment.common.protocol(parent_dir)
    core_out = experiment.ROOT / 'results/persistence_pairing_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_persistence_pairing_v1_20260916'
    queue_path = HERE / 'PERSISTENCE_PAIRING_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve earlier studies; use a new version')
    sources = sorted(set(previous['source_sha256']) | {
        'persistence_pairing_graph_v1.py', 'test_persistence_pairing_graph_v1.py',
        'adopt_persistence_pairing_v1.py'})
    inputs = [unit_path, smoke_path, HERE / 'PERSISTENCE_PAIRING_ADAPTATION_V1.md',
        HERE / 'persistence_pairing_smoke_v1.py', HERE / 'persistence_pairing_unit_v1.log', Path(__file__),
        HERE / 'run_persistence_pairing_queue_v1.py', HERE / 'PERSISTENCE_CONSOLIDATION_V1.json',
        HERE / 'PERSISTENCE_SMOKE_V3.json', HERE / 'PERSISTENCE_UNIT_V2.json', parent_dir / 'protocol.json']
    budget = ('400 AdamW steps and gradient evaluations and 20 common checkpoints for every newly fitted model; '
              'two unchanged seed-666 primary neural fits and two identity records are adopted by strict verification. '
              'Same initialized modules, native computations and random streams in both arms.')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='persistence_pairing',
        implementation_module='persistence_pairing_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=['marginal_only'],
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Primary exactly adopts the previously observed no_spectrum arm. Before new control performance, compare only a coordinate-marginal coupling with the same modules and budget. No filtration, pooling, neighborhood or native-algorithm changes.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={**previous['inputs'], **{str(path.resolve()): base.digest(path) for path in inputs}},
        required_executor='improve.anchor_spectral_persistence_experiment_v1', training_budget=budget,
        mechanism='Joint zero-dimensional persistence event attributes beyond their separate empirical marginals',
        local_adaptation='Exact no-spectrum parent primary; control sorts each tuple coordinate separately within the query, preserving all marginals and cycle events. The synthetic coupling need not describe a graph.',
        numerical_gate='Inherit frozen FP64 native checks; primary CPU weights/optimizer/trace/RNG identical; real parent predictions bitwise equal; query replay absolute tolerance 3e-5.',
        literature=dict(title='Graph Persistence goes Spectral', arxiv='2506.06571v3',
            definition_used='Section 2.2 Definition 2.3 and the already executed pinned author code',
            author_commit='90ad637b00481a33318306d776882dce8e2012e8',
            no_claim_original_2023_experiments_reproduced=True),
        evaluation='Observed primary screen with one new control; eligible seeds 0/1 confirmation; three fixed development partitions; then frozen outer. Stop on any failed joint gate.',
        parent_study=str(parent_dir), parent_kind='no_spectrum', prior_parent_outcomes_seen=True,
        independent_screen=False, no_new_control_performance_observed=True, smoke=False,
        innovation_family='persistent_topology', original_test_source_access=False,
        prior_failures_preserved=True, existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'persistence_pairing', experiment.ROOT / 'results/nested_anchor_v2_20260915/historical', core_out)
    historical = base.read_json(out / 'protocol.json')
    historical.update(required_executor=p['required_executor'], training_budget=budget,
        local_adaptation=p['local_adaptation'], numerical_gate=p['numerical_gate'],
        parent_study=str(parent_dir), parent_kind='no_spectrum', prior_parent_outcomes_seen=True,
        independent_screen=False, no_new_control_performance_observed=True, innovation_family='persistent_topology')
    historical['inputs'].update(p['inputs'])
    base.write_json(out / 'protocol.json', historical)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', unit_path, smoke_path,
                    HERE / 'PERSISTENCE_PAIRING_ADAPTATION_V1.md', parent_dir / 'protocol.json']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='persistence_pairing',
        source_sha256=base.digest(HERE / 'run_persistence_pairing_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs}, device='cuda:1',
        stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 2, 'confirm_if_eligible': 4},
        expected_new_identity_records={'screen': 2, 'confirm_if_eligible': 4},
        expected_adopted_neural_fits=2, expected_adopted_identity_records=2,
        identity_records_count_as_neural_fits=False, independent_screen=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('PERSISTENCE_PAIRING_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
