"""Freeze the spectral persistence adapter after training-only numerical checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_spectral_persistence_experiment_v1 as experiment
from improve import spectral_persistence_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    smoke_path, unit_path = HERE / 'PERSISTENCE_SMOKE_V3.json', HERE / 'PERSISTENCE_UNIT_V2.json'
    smoke, unit = (base.check_seal(base.read_json(path)) for path in (smoke_path, unit_path))
    assert unit['success'] and unit['tests'] == 8
    assert smoke['formal_fits'] == 0 and not smoke['performance_metrics_computed']
    assert not smoke['evaluation_labels_read'] and smoke['core_parameters'] == core.parameters()
    assert len(smoke['fold_audits']) == 40 and smoke['cuda_random_stream_equal_across_arms']
    assert smoke['uncached_repeatability']['real_training_diagrams_cache_bitwise_equal']
    for mapping in (unit['source_sha256'], smoke['source_sha256']):
        for path, digest in mapping.items():
            assert base.digest(path) == digest, path
    core_out = experiment.ROOT / 'results/spectral_persistence_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_spectral_persistence_v1_20260916'
    queue_path = HERE / 'PERSISTENCE_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve prior studies; use a new version')
    sources = ['spectral_persistence_graph_v1.py', 'test_spectral_persistence_graph_v1.py',
        'spectral_persistence_native_v1.cpp', 'spectral_persistence_build_v1.py',
        'anchor_spectral_persistence_experiment_v1.py', 'sharpness_graph_v1.py',
        'test_anchor_priority_graph_v1.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py',
        'tabular_foundation.py', 'graph_prediction_precision.py',
        'nested_anchor_inputs_v2.py', 'residual_graph_experiment.py',
        'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    inputs = [smoke_path, unit_path, HERE / 'SPECTRE_ADAPTATION_V1.md',
        HERE / 'persistence_smoke_v3.py', HERE / 'persistence_unit_v2.log', Path(__file__),
        HERE / 'run_persistence_queue_v1.py', HERE / 'spectre_paper_v1.pdf',
        HERE / 'spectre_paper_v1.txt', HERE / 'spectre_commit_v1.json',
        HERE / 'spectre_tree_v1.json', HERE / 'SPECTRE_AUTHOR_SOURCE_V1.json',
        HERE / 'SPECTRE_ADDITIONAL_SOURCE_V1.json', HERE / 'PERSISTENCE_LOCAL_SIZE_PROBE_V1.json',
        HERE / 'PERSISTENCE_SMOKE_V1.json', HERE / 'persistence_smoke_v2.log',
        experiment.ROOT / 'results/spectral_persistence_smoke_v2_20260916/FAILURE.json']
    for manifest in ('SPECTRE_AUTHOR_SOURCE_V1.json', 'SPECTRE_ADDITIONAL_SOURCE_V1.json'):
        for record in base.read_json(HERE / manifest)['records']:
            if record['status'] != 'downloaded':
                continue
            path = Path(record['file'])
            assert record['git_blob_verified'] and base.digest(path) == record['sha256']
            inputs.append(path)
    builds = [experiment.ROOT / 'results/spectral_persistence_native_v1_20260916/BUILD.json',
              HERE / 'author_build_v1/BUILD.json']
    for path in builds:
        build = base.read_json(path)
        assert base.digest(build['binary']) == build['binary_sha256']
        for source, digest in build['recipe']['source_sha256'].items():
            assert base.digest(source) == digest
        inputs.extend((path, Path(build['binary'])))
    budget = ('400 AdamW steps and gradient evaluations per model, one omitted training fold per step; '
              '20 common checkpoints. Identical initialized modules and RNG streams in every arm; '
              'all arms compute full persistence events and spectra. Exact cache counts can differ.')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='spectral_persistence',
        implementation_module='spectral_persistence_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Primary and two spectrum controls fixed before new performance; two learning rates, 400 steps and every 20-step checkpoint. No filtration, neighborhood or spectrum-subset sweep.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='Graph Persistence goes Spectral', arxiv='2506.06571v3',
            arxiv_version_date='2026-02-04', author_reported_venue='NeurIPS 2025',
            publisher_venue_independently_verified=False,
            author_commit='90ad637b00481a33318306d776882dce8e2012e8', full_text_obtained=True,
            pinned_author_cpu_code_executed=True, full_original_experiment_reproduced=False),
        mechanism='Learned persistence events carrying their time-specific connected-component spectra',
        required_executor='improve.anchor_spectral_persistence_experiment_v1',
        training_budget=budget,
        local_adaptation='Complete one-hop common-neighbor induced query graph; final typed GNN layer; one filtration; FP64 spectra; component-specific simultaneous events; fixed log1p spectrum transform; mean-zero-dimensional and sum-cycle pooling; zero-initialized residual head.',
        numerical_gate='Finite FP64 unweighted spectra, PSD, rank, trace, second moment and Euler checks; exact matrix memoization; prediction replay absolute tolerance 3e-5.',
        native_build_sha256={str(path): base.digest(path) for path in builds},
        evaluation='Historical seed 666 screen; seeds 0/1 confirmation if eligible; three fixed development partitions; then frozen outer. Stop this version at the first failed joint gate.',
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        prior_failures_preserved=True, existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'spectral_persistence',
        experiment.ROOT / 'results/nested_anchor_v2_20260915/historical', core_out)
    # Complete this new, unused protocol before sealing the queue. Include the
    # native binaries and build recipes in the direct runtime input checks.
    historical = base.read_json(out / 'protocol.json')
    historical.update(required_executor=p['required_executor'], training_budget=budget,
        local_adaptation=p['local_adaptation'], numerical_gate=p['numerical_gate'])
    historical['inputs'].update(p['inputs'])
    base.write_json(out / 'protocol.json', historical)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path, unit_path,
                     HERE / 'SPECTRE_ADAPTATION_V1.md'] + builds
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='spectral_persistence',
        source_sha256=base.digest(HERE / 'run_persistence_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('PERSISTENCE_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
