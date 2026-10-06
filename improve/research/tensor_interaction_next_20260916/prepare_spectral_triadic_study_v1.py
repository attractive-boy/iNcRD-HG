"""Freeze spectral quadratic sharing after formula and training-only checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

from improve import anchor_spectral_triadic_experiment_v1 as experiment
from improve import spectral_triadic_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    smoke_path, unit_path = HERE / 'SPECTRAL_TRIADIC_SMOKE_V1.json', HERE / 'SPECTRAL_TRIADIC_UNIT_V1.json'
    smoke, unit = (base.check_seal(base.read_json(path)) for path in (smoke_path, unit_path))
    assert unit['success'] and unit['tests'] == 6
    assert smoke['formal_fits'] == 0 and not smoke['performance_metrics_computed']
    assert not smoke['evaluation_labels_read'] and smoke['core_parameters'] == core.parameters()
    assert len(smoke['fold_audits']) == 40 and len(smoke['records']) == 6
    for key in ('common_initial_parameters_and_effective_weights_equal', 'initial_random_streams_equal',
        'final_random_streams_equal_across_arms', 'initial_residual_exact_zero',
        'every_active_parameter_gradient_finite', 'fixed_parameters_unchanged',
        'query_and_quadratic_counts_equal_across_arms', 'fixed_fit_rejects_evaluation_keys',
        'replay_uses_only_omitted_fold'):
        assert smoke[key], key
    for mapping in (unit['source_sha256'], smoke['source_sha256'], smoke['input_sha256']):
        for path, digest in mapping.items():
            assert base.digest(path) == digest, path
    core_out = experiment.ROOT / 'results/spectral_triadic_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_spectral_triadic_v1_20260916'
    queue_path = HERE / 'SPECTRAL_TRIADIC_QUEUE_PROTOCOL_V1.json'
    selection_path = HERE / 'SPECTRAL_TRIADIC_SELECTION_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists() or selection_path.exists():
        raise ValueError('Preserve earlier studies; use a new version')
    sources = ['spectral_triadic_graph_v1.py', 'test_spectral_triadic_graph_v1.py',
        'anchor_spectral_triadic_experiment_v1.py', 'sharpness_graph_v1.py',
        'test_nested_anchor_v2.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py', 'tabular_foundation.py',
        'graph_prediction_precision.py', 'nested_anchor_inputs_v2.py',
        'residual_graph_experiment.py', 'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    names = ['SPECTRAL_TRIADIC_ADAPTATION_V1.md', 'spectral_triadic_smoke_v1.py',
        'spectral_triadic_smoke_v1.log', 'spectral_triadic_unit_v1.log',
        'run_spectral_triadic_queue_v1.py', 'consolidate_spectral_triadic_v1.py',
        'shonn_v1.pdf', 'shonn_v1.txt', 'shonn_crossref_v1.json', 'shonn_author_commit_v1.json',
        'shonn_author_tree_v1.json', 'SHONN_DISCOVERY_V1.json', 'SHONN_AUTHOR_DISCOVERY_V1.json',
        'SHONN_AUTHOR_SOURCE_V1.json']
    inputs = [smoke_path, unit_path, Path(__file__)] + [HERE / name for name in names]
    for record in base.read_json(HERE / 'SHONN_AUTHOR_SOURCE_V1.json'):
        path = (experiment.ROOT.parent / record['file']).resolve()
        assert record['git_blob_verified'] and base.digest(path) == record['sha256']
        assert record['commit'] == 'bf38adfcfc441d4bf71e9eff43d87e41a0f1e7a7'
        inputs.append(path)
    selection = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family='spectral_triadic', primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lrs=[.001, .003], epochs=400, check_interval=20,
        screen_seed=666, confirm_seeds=[0, 1], strong_weight=.1, weight_decay=.001,
        lnc_policy='identity_strong_reference', new_performance_observed=False,
        prior_failed_polynormer_seen=True, one_broader_feature_interaction_family_only=True,
        adaptation_sha256=base.digest(HERE / 'SPECTRAL_TRIADIC_ADAPTATION_V1.md'),
        unit_sha256=base.digest(unit_path), smoke_sha256=base.digest(smoke_path)))
    base.write_json(selection_path, selection)
    inputs.append(selection_path)
    budget = ('400 AdamW steps and gradient evaluations per model; one omitted query fold per step; '
        '20 common checkpoints and 505 model forwards. Same graphs, query rows, quadratic monomials, '
        'initial RNG and dropout streams. Full and direct-quadratic have the same initial effective '
        'weights but intentionally different active parameter dimensions. Fixed tensors remain stored.')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='spectral_triadic',
        implementation_module='spectral_triadic_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Full, linear-only and direct-quadratic fixed before performance; two learning rates and 400 steps with each 20-step checkpoint. No width, depth, initialization, normalization or fusion sweep.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(arxiv='2603.28420v1', date='2026-03-30', publication_status='preprint',
            exact_crossref_doi_not_established=True, author_repository='gianluca-peri/hyperspectral',
            author_commit='bf38adfcfc441d4bf71e9eff43d87e41a0f1e7a7',
            full_text_obtained=True, unchanged_author_layer_module_executed=True,
            original_benchmark_experiment_reproduced=False),
        mechanism='Shared input-pair and output spectral factors of a fixed random quadratic coefficient tensor',
        related_prior_family='Broader feature-interaction family; previous failed Polynormer used featurewise multiplication in local/global graph propagation',
        distinctness_gate='Must beat both a nonquadratic decoder and directly optimized quadratic coefficients; polynomial history, residual infrastructure, controls and revisions do not count separately.',
        required_executor='improve.anchor_spectral_triadic_experiment_v1', training_budget=budget,
        local_adaptation='Common typed attention backbone; concatenate endpoint states; 128-to-64 and 64-to-64 spectral-triadic layers, LayerNorm/GELU/dropout after each, and zero-initialized output. Complete quadratic pairs including squares; bias only linear.',
        numerical_gate='Finite forward, active gradients and optimizer state; pinned-author layer/gradient, independent polynomial Hessian and gradcheck; fixed parameter conservation, replay tolerance 3e-5. Save exact partial state on training failure.',
        inference_claim='Finite graph-conditioned quadratic decoder. No source universal-approximation guarantee, graph-spectral interpretation, reduced quadratic multiplication/storage complexity or biological interaction-discovery claim.',
        evaluation='Historical seed666 screen; seeds0/1 only if eligible; then three fixed development partitions and separately frozen outer. Stop at first failed joint or numerical gate.',
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        prior_failures_preserved=True, existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'spectral_triadic',
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
                    HERE / 'SPECTRAL_TRIADIC_ADAPTATION_V1.md', selection_path]
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='spectral_triadic',
        source_sha256=base.digest(HERE / 'run_spectral_triadic_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', workers_run_serially=True, stages=['screen', 'confirm'],
        stop_on_any_failed_gate=True, expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6}, adopted_neural_fits=0,
        identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('SPECTRAL_TRIADIC_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
