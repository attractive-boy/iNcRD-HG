"""Freeze the declared implicit-equilibrium mechanism before performance use."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_priority_graph_v1 as experiment
from improve import equilibrium_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    smoke_path = HERE / 'EQUILIBRIUM_SMOKE_V1.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    unit_path = HERE / 'EQUILIBRIUM_UNIT_V1.json'
    unit = base.check_seal(base.read_json(unit_path))
    assert smoke['formal_fits'] == 0 and not smoke['performance_metrics_computed']
    assert not smoke['evaluation_labels_read']
    assert smoke['core_parameters'] == core.parameters() and len(smoke['fold_audits']) == 40
    assert unit['passed'] and unit['tests_run'] == 5
    for source in (smoke['source_sha256'], unit['input_sha256']):
        for path, digest in source.items():
            if base.digest(path) != digest:
                raise ValueError('Pre-performance evidence changed: ' + path)
    core_out = experiment.ROOT / 'results/equilibrium_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_equilibrium_v1_20260916'
    queue_path = HERE / 'EQUILIBRIUM_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve previous studies; use a new version')
    sources = ['equilibrium_graph_v1.py', 'test_equilibrium_graph_v1.py',
        'anchor_equilibrium_experiment_v1.py', 'sharpness_graph_v1.py',
        'test_anchor_priority_graph_v1.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py',
        'tabular_foundation.py', 'graph_prediction_precision.py',
        'nested_anchor_inputs_v2.py', 'residual_graph_experiment.py',
        'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    inputs = [smoke_path, unit_path, HERE / 'EQUILIBRIUM_ADAPTATION_V1.md',
        HERE / 'equilibrium_smoke_v1.py', HERE / 'equilibrium_unit_v1.log', Path(__file__),
        HERE / 'run_equilibrium_queue_v1.py', HERE / 'megnn_tree_v1.json',
        HERE / 'MEGNN_TREE_FETCH_V1.json', HERE / 'MEGNN_SOURCE_FETCH_V3.json',
        HERE / 'MEGNN_SOURCE_FETCH_V4.json', HERE / 'MEGNN_PUBLISHER_ACCESS_V1.json',
        HERE / 'EQUILIBRIUM_AUTHOR_SOURCE_V1.json', HERE / 'megnn_author_models.py',
        HERE / 'megnn_crossref_v1.json', HERE / 'megnn_epmc_v1.json',
        HERE / 'megnn_semanticscholar_v1.json']
    for entry in base.read_json(HERE / 'MEGNN_SOURCE_FETCH_V3.json'):
        path = Path(entry['file']).resolve()
        if not entry['verified'] or base.digest(path) != entry['sha256']:
            raise ValueError('Author source integrity failed')
        inputs.append(path)
    for entry in base.read_json(HERE / 'MEGNN_SOURCE_FETCH_V4.json'):
        path = HERE / ('megnn_author_v4_' + entry['name'])
        if base.digest(path) != entry['sha256']:
            raise ValueError('Author data source integrity failed')
        inputs.append(path)
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='equilibrium',
        implementation_module='equilibrium_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=['two_hops'],
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Primary and two-hop control fixed before new performance. Two learning rates, 400 steps, every 20-step checkpoint, identical common criterion. No architecture or contraction sweep.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='Multi-Channel Equilibrium Graph Neural Network for Multi-View Semi-Supervised Learning',
            doi='10.1109/TPAMI.2025.3587216', pmid='40690351', venue='TPAMI 2025',
            author_commit='eb47cba61fdf0e4f7f8e7ee3fa12153da89008b8', journal_full_text_obtained=False,
            evidence='Verified metadata/abstract and pinned released code; independent adaptation'),
        mechanism='Learned contractive relation-channel fixed points with exact implicit adjoints versus parameter-matched two-hop truncation',
        required_executor='improve.anchor_equilibrium_experiment_v1',
        numerical_gate='FP64 forward and backward relative residual <=1e-8; fail rather than return an unconverged iterate',
        evaluation='Historical seed 666 screen, seeds 0/1 confirmation, three fixed development partitions, then frozen outer recheck; stop this version at any failed gate.',
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        prior_failures_preserved=True, existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'equilibrium', experiment.ROOT / 'results/nested_anchor_v2_20260915/historical', core_out)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path, unit_path,
                     HERE / 'EQUILIBRIUM_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='equilibrium',
        source_sha256=base.digest(HERE / 'run_equilibrium_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 4, 'confirm_if_eligible': 4},
        expected_identity_records={'screen': 4, 'confirm_if_eligible': 4},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False,
        confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('EQUILIBRIUM_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
