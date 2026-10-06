"""Freeze attribute-initialized phase dynamics after training-only checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

from improve import anchor_kuramoto_phase_experiment_v1 as experiment
from improve import kuramoto_phase_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    smoke_path, unit_path = HERE / 'KURAMOTO_PHASE_SMOKE_V1.json', HERE / 'KURAMOTO_PHASE_UNIT_V1.json'
    smoke, unit = (base.check_seal(base.read_json(path)) for path in (smoke_path, unit_path))
    assert unit['success'] and unit['tests'] == 7
    assert smoke['formal_fits'] == 0 and not smoke['performance_metrics_computed']
    assert not smoke['evaluation_labels_read'] and smoke['core_parameters'] == core.parameters()
    assert len(smoke['fold_audits']) == 40 and len(smoke['records']) == 6
    for key in ('common_initial_parameters_equal', 'initial_random_streams_equal',
        'final_random_streams_equal_across_arms', 'initial_residual_exact_zero',
        'every_active_parameter_gradient_finite', 'fixed_parameters_unchanged',
        'phase_and_message_counts_equal_across_arms', 'fixed_fit_rejects_evaluation_keys',
        'replay_uses_only_omitted_fold'):
        assert smoke[key], key
    for mapping in (unit['source_sha256'], smoke['source_sha256'], smoke['input_sha256']):
        for path, digest in mapping.items():
            assert base.digest(path) == digest, path
    core_out = experiment.ROOT / 'results/kuramoto_phase_frozen_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_kuramoto_phase_v1_20260916'
    queue_path = HERE / 'KURAMOTO_PHASE_QUEUE_PROTOCOL_V1.json'
    selection_path = HERE / 'KURAMOTO_PHASE_SELECTION_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists() or selection_path.exists():
        raise ValueError('Preserve earlier studies; use a new version')
    sources = ['kuramoto_phase_graph_v1.py', 'test_kuramoto_phase_graph_v1.py',
        'anchor_kuramoto_phase_experiment_v1.py', 'sharpness_graph_v1.py',
        'test_nested_anchor_v2.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py', 'tabular_foundation.py',
        'graph_prediction_precision.py', 'nested_anchor_inputs_v2.py',
        'residual_graph_experiment.py', 'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    names = ['KURAMOTO_PHASE_ADAPTATION_V1.md', 'kuramoto_phase_smoke_v1.py',
        'kuramoto_phase_smoke_v1.log', 'kuramoto_phase_unit_v1.log',
        'run_kuramoto_phase_queue_v1.py', 'consolidate_kuramoto_phase_v1.py',
        'kope_v2.pdf', 'kope_v2.txt', 'kope_repo_v1.json',
        'kope_author_commit_v1.json', 'kope_author_tree_v1.json',
        'OSCILLATORY_SECONDARY_SOURCES_V1.json', 'OSCILLATORY_AUTHOR_DISCOVERY_V1.json',
        'KOPE_AUTHOR_SOURCE_V1.json']
    inputs = [smoke_path, unit_path, Path(__file__)] + [HERE / name for name in names]
    for record in base.read_json(HERE / 'KOPE_AUTHOR_SOURCE_V1.json'):
        path = Path(record['file']).resolve()
        assert record['git_blob_verified'] and base.digest(path) == record['sha256']
        assert record['commit'] == '0eade3cebc76978d2944f3ebb9f50f57f1b7ee53'
        inputs.append(path)
    selection = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family='kuramoto_phase', primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lrs=[.001, .003], epochs=400, check_interval=20,
        screen_seed=666, confirm_seeds=[0, 1], strong_weight=.1, weight_decay=.001,
        lnc_policy='identity_strong_reference', new_performance_observed=False,
        one_broader_oscillatory_family_only=True,
        adaptation_sha256=base.digest(HERE / 'KURAMOTO_PHASE_ADAPTATION_V1.md'),
        unit_sha256=base.digest(unit_path), smoke_sha256=base.digest(smoke_path)))
    base.write_json(selection_path, selection)
    inputs.append(selection_path)
    budget = ('400 AdamW steps and gradient evaluations; one omitted query fold per step; '
        '20 checkpoints and 505 model forwards, each with four attention layers and four phase updates. '
        'All arms compute phase initialization, mixing, feature rotations and shared coupling on the same graph. '
        'Initial stored tensors and dropout streams match; active parameter dimensions differ and are audited. '
        'Mixed-normalization clamp counts are data-dependent and are not equal-budget quantities.')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='kuramoto_phase',
        implementation_module='kuramoto_phase_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Full, static-phase and no-phase-rotation fixed before performance; two learning rates and 400 steps with 20-step checkpoints. No gamma, width, depth, initialization, normalization or fusion sweep.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(arxiv='2604.07904v2', updated='2026-06-24',
            publication_status='ICML2026 reported by authors; publisher record not independently verified',
            author_repository='microsoft/Neuro-inspired_Phase_Encoding',
            author_commit='0eade3cebc76978d2944f3ebb9f50f57f1b7ee53', full_text_obtained=True,
            unchanged_selected_author_attention_coupling_and_rotation_bodies_executed=True,
            full_ViT_and_benchmarks_reproduced=False, larger_simdinov2_source_downloaded=False),
        mechanism='Attribute-initialized unit-circle phase synchronization that modifies typed sparse Q/K/V attention by pair rotations and inverse output rotation',
        related_prior_family='Broader oscillatory family; Stuart-Landau failure known and preserved',
        distinctness_gate='Must beat static attribute phases and ordinary typed attention on both MiDrug mean metrics; oscillatory revisions count at most once.',
        required_executor='improve.anchor_kuramoto_phase_experiment_v1', training_budget=budget,
        local_adaptation='Width64 four-layer typed attention; learned attribute phases replace 2D RoPE, source-phase rotation of edge keys, gamma0.05, shared coupling, ordinary zero-initialized pair decoder and local common linear/LayerNorm/FFN conventions.',
        numerical_gate='Finite scores, phases, states, loss, gradients and optimizer; phase unit-norm error <=1e-5; pinned-author and independent sparse NumPy forward/gradient checks; frozen-parameter conservation and absolute replay tolerance 3e-5. Save exact partial state on failure.',
        inference_claim='Sparse graph adaptation with all phase branches computed, including unused final phase update. No full-ViT reproduction, equal active capacity, backward FLOP equality, speedup or first graph-application claim.',
        evaluation='Historical seed666 screen; seeds0/1 only if eligible; then three fixed development partitions and separately frozen outer. Stop at first failed joint or numerical gate.',
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        prior_failures_preserved=True, existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'kuramoto_phase',
        experiment.ROOT / 'results/nested_anchor_v2_20260915/historical', core_out)
    for task in p['tasks']:
        training = experiment.nested.load_selected(out / 'inputs' / task / 'arrays.npz', experiment.TRAIN_KEYS)
        checked = experiment.nested.load_selected(experiment.ROOT /
            'results/anchor_priority_morphological_alignment_v1_20260916/inputs' / task / 'arrays.npz', experiment.TRAIN_KEYS)
        assert set(training) == set(checked)
        for key in training:
            np.testing.assert_array_equal(training[key], checked[key])
    historical = base.read_json(out / 'protocol.json')
    for key in ('required_executor', 'training_budget', 'local_adaptation', 'numerical_gate',
                'inference_claim', 'related_prior_family', 'distinctness_gate'):
        historical[key] = p[key]
    historical['inputs'].update(p['inputs'])
    base.write_json(out / 'protocol.json', historical)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path, unit_path,
                    HERE / 'KURAMOTO_PHASE_ADAPTATION_V1.md', selection_path]
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='kuramoto_phase',
        source_sha256=base.digest(HERE / 'run_kuramoto_phase_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', workers_run_serially=True, stages=['screen', 'confirm'],
        stop_on_any_failed_gate=True, expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6}, adopted_neural_fits=0,
        identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('KURAMOTO_PHASE_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
