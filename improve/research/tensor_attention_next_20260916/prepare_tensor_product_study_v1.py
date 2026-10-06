"""Freeze contextual tensor products after formula and training-only checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

from improve import anchor_tensor_product_experiment_v1 as experiment
from improve import tensor_product_attention_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    smoke_path, unit_path = HERE / 'TENSOR_PRODUCT_SMOKE_V1.json', HERE / 'TENSOR_PRODUCT_UNIT_V1.json'
    smoke, unit = (base.check_seal(base.read_json(path)) for path in (smoke_path, unit_path))
    assert unit['success'] and unit['tests'] == 7
    assert smoke['formal_fits'] == 0 and not smoke['performance_metrics_computed']
    assert not smoke['evaluation_labels_read'] and smoke['core_parameters'] == core.parameters()
    assert len(smoke['fold_audits']) == 40 and len(smoke['records']) == 6
    for key in ('common_initial_parameters_equal', 'initial_random_streams_equal',
        'final_random_streams_equal_across_arms', 'initial_residual_exact_zero',
        'every_active_parameter_gradient_finite', 'fixed_parameters_unchanged',
        'tensor_and_message_counts_equal_across_arms', 'fixed_fit_rejects_evaluation_keys',
        'replay_uses_only_omitted_fold'):
        assert smoke[key], key
    for mapping in (unit['source_sha256'], smoke['source_sha256'], smoke['input_sha256']):
        for path, digest in mapping.items():
            assert base.digest(path) == digest, path
    core_out = experiment.ROOT / 'results/tensor_product_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_tensor_product_v1_20260916'
    queue_path = HERE / 'TENSOR_PRODUCT_QUEUE_PROTOCOL_V1.json'
    selection_path = HERE / 'TENSOR_PRODUCT_SELECTION_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists() or selection_path.exists():
        raise ValueError('Preserve earlier studies; use a new version')
    sources = ['tensor_product_attention_graph_v1.py', 'test_tensor_product_attention_graph_v1.py',
        'anchor_tensor_product_experiment_v1.py', 'sharpness_graph_v1.py',
        'test_nested_anchor_v2.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py', 'tabular_foundation.py',
        'graph_prediction_precision.py', 'nested_anchor_inputs_v2.py',
        'residual_graph_experiment.py', 'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    names = ['TENSOR_PRODUCT_ADAPTATION_V1.md', 'tensor_product_smoke_v1.py',
        'tensor_product_smoke_v1.log', 'tensor_product_unit_v1.log',
        'run_tensor_product_queue_v1.py', 'consolidate_tensor_product_v1.py',
        'tpa_v7.pdf', 'tpa_v7.txt', 'tpa_metadata_v1.xml', 'tpa_crossref_v1.json',
        'tpa_crossref_exact_v1.json', 'tpa_github_v1.json',
        'tpa_author_commit_v1.json', 'tpa_author_tree_v1.json',
        'TPA_SOURCE_RETRIEVAL_V1.json', 'TPA_AUTHOR_DISCOVERY_V1.json', 'TPA_AUTHOR_SOURCE_V1.json']
    inputs = [smoke_path, unit_path, Path(__file__)] + [HERE / name for name in names]
    for record in base.read_json(HERE / 'TPA_AUTHOR_SOURCE_V1.json'):
        path = (experiment.ROOT.parent / record['file']).resolve()
        assert record['git_blob_verified'] and base.digest(path) == record['sha256']
        assert record['commit'] == 'c276c80d5ad807881dedb4707d8d3c20b4e97ec6'
        inputs.append(path)
    selection = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family='tensor_product_attention', primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lrs=[.001, .003], epochs=400, check_interval=20,
        screen_seed=666, confirm_seeds=[0, 1], strong_weight=.1, weight_decay=.001,
        lnc_policy='identity_strong_reference', new_performance_observed=False,
        one_broader_feature_interaction_family_only=True,
        adaptation_sha256=base.digest(HERE / 'TENSOR_PRODUCT_ADAPTATION_V1.md'),
        unit_sha256=base.digest(unit_path), smoke_sha256=base.digest(smoke_path)))
    base.write_json(selection_path, selection)
    inputs.append(selection_path)
    budget = ('400 AdamW steps and gradient evaluations; one omitted query fold per step; '
        '20 checkpoints and 505 model forwards, each with two attention layers. All arms compute '
        'contextual and static tensor products and dense projections on the same nodes and graph edges. '
        'Initial stored tensors and dropout streams match; active parameter dimensions differ and are audited.')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='tensor_product_attention',
        implementation_module='tensor_product_attention_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Full, noncontextual-A and dense-QKV fixed before performance; two learning rates and 400 steps with 20-step checkpoints. No rank, width, depth, initialization, normalization or fusion sweep.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(arxiv='2501.06425v7', updated='2026-01-11', first_posted='2025-01-11',
            publication_status='NeurIPS2025 proceedings; exact Crossref verified',
            doi='10.52202/085713-3748', author_repository='tensorgi/TPA',
            author_commit='c276c80d5ad807881dedb4707d8d3c20b4e97ec6', full_text_obtained=True,
            unchanged_selected_author_projection_bodies_executed=True,
            rotary_provider='identity, all-zero positions', full_T6_and_benchmarks_reproduced=False,
            preexisting_GraphTPA_repository_seen=True),
        mechanism='Node-conditioned low-rank tensor-product query, key and value activations in typed sparse graph attention',
        related_prior_family='Broader feature-interaction family; previous Polynormer and SHONN failures are known and preserved',
        distinctness_gate='Must beat noncontextual head factors and free dense linear QKV; projections and revisions do not count separately.',
        required_executor='improve.anchor_tensor_product_experiment_v1', training_budget=budget,
        local_adaptation='Common width64 two-layer typed attention, ranks6/2/2, source CPLinear initializer, no RoPE, common pair decoder with zero output; static-A and dense controls start at equivalent linear functions.',
        numerical_gate='Finite projections, attention, states, losses, gradients and optimizer; pinned-author and independent tensor/Hessian/attention checks; frozen-parameter conservation and absolute replay tolerance 3e-5. Save exact partial state on training failure.',
        inference_claim='Materialized graph QKV with all control branches computed. No KV-cache, FlashTPA, autoregressive acceleration, equal active capacity or first graph-TPA claim.',
        evaluation='Historical seed666 screen; seeds0/1 only if eligible; then three fixed development partitions and separately frozen outer. Stop at first failed joint or numerical gate.',
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        prior_failures_preserved=True, existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'tensor_product_attention',
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
                    HERE / 'TENSOR_PRODUCT_ADAPTATION_V1.md', selection_path]
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='tensor_product_attention',
        source_sha256=base.digest(HERE / 'run_tensor_product_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', workers_run_serially=True, stages=['screen', 'confirm'],
        stop_on_any_failed_gate=True, expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6}, adopted_neural_fits=0,
        identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('TENSOR_PRODUCT_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
