"""Freeze resistance-curvature flow after independent and training-only checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_resistance_flow_experiment_v1 as experiment
from improve import resistance_flow_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    smoke_path, unit_path = HERE / 'RESISTANCE_FLOW_SMOKE_V1.json', HERE / 'RESISTANCE_FLOW_UNIT_V2.json'
    smoke, unit = (base.check_seal(base.read_json(path)) for path in (smoke_path, unit_path))
    assert unit['success'] and unit['tests'] == 5
    assert smoke['formal_fits'] == 0 and not smoke['performance_metrics_computed']
    assert not smoke['evaluation_labels_read'] and smoke['core_parameters'] == core.parameters()
    assert len(smoke['fold_audits']) == 40 and len(smoke['records']) == 6
    assert all(smoke[key] for key in ('final_random_streams_equal_across_arms',
        'initialized_parameters_and_random_streams_equal', 'replay_uses_only_omitted_fold',
        'fixed_fit_rejects_evaluation_keys', 'initial_residual_exact_zero',
        'every_parameter_gradient_finite', 'inverse_calls_equal_across_arms'))
    for mapping in (unit['source_sha256'], smoke['source_sha256'], smoke['input_sha256']):
        for path, digest in mapping.items():
            assert base.digest(path) == digest, path
    core_out = experiment.ROOT / 'results/resistance_flow_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_resistance_flow_v1_20260916'
    queue_path = HERE / 'RESISTANCE_FLOW_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve earlier studies; use a new version')
    sources = ['resistance_flow_graph_v1.py', 'test_resistance_flow_graph_v1.py',
        'anchor_resistance_flow_experiment_v1.py', 'sharpness_graph_v1.py',
        'nonlinear_pair_field_v1.py', 'test_anchor_priority_graph_v1.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py', 'tabular_foundation.py',
        'graph_prediction_precision.py', 'nested_anchor_inputs_v2.py',
        'residual_graph_experiment.py', 'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    inputs = [smoke_path, unit_path, HERE / 'RESISTANCE_FLOW_ADAPTATION_V1.md',
        HERE / 'CURVATURE_SELECTION_V1.json', HERE / 'resistance_flow_smoke_v1.py',
        HERE / 'resistance_flow_smoke_v1.log', HERE / 'resistance_flow_unit_v2.log', Path(__file__),
        HERE / 'run_resistance_flow_queue_v1.py', HERE / 'consolidate_resistance_flow_v1.py',
        HERE / 'resistance_flow_v1.pdf', HERE / 'resistance_flow_v1.txt',
        HERE / 'resistance_curvature_commit_v1.json', HERE / 'RESISTANCE_CURVATURE_PINNED_TREE_V1.json',
        HERE / 'RESISTANCE_AUTHOR_SOURCE_V1.json', HERE / 'rcf_commit_v1.json',
        HERE / 'BEC_AUTHOR_AUDIT_V2.json', HERE / 'RESISTANCE_FLOW_UNIT_V1.json',
        HERE / 'ARXIV_DISCOVERY_V1.json', HERE / 'ARXIV_DISCOVERY_V2.json',
        HERE / 'CROSSREF_DISCOVERY_V1.json', HERE / 'CURVATURE_FULLTEXT_V1.json']
    for record in base.read_json(HERE / 'RESISTANCE_AUTHOR_SOURCE_V1.json'):
        path = (experiment.ROOT.parent / record['file']).resolve()
        assert record['git_blob_verified'] and base.digest(path) == record['sha256']
        assert record['commit'] == '30aad3daf194c26359146e25a5c193d1066f02c1'
        inputs.append(path)
    budget = ('400 AdamW steps and gradient evaluations per model, one omitted training fold per step; '
        '20 common checkpoints. Equal initialized modules, parameter counts and random streams. '
        'Three FP64 inverse/flow steps on each of two entity axes per forward in every arm, including '
        'zero-weighted unused control paths. 505 training/selection forwards, 1010 entity flows, '
        '3030 inverse/flow steps, before independent saved-model replay.')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='resistance_flow',
        implementation_module='resistance_flow_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Full, local resistance and no flow fixed before performance; two learning rates, 400 steps and each 20-step checkpoint. No neighborhood, bandwidth, flow length, step size, ridge or fusion sweep.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='Dynamic Graph Structure Learning via Resistance Curvature Flow',
            arxiv='2601.08149v1', arxiv_version_date='2026-01-13', publication_status='preprint',
            application_repository='cqfei/RCF', application_repository_empty=True,
            used_repository='cqfei/resistance_curvature',
            author_commit='30aad3daf194c26359146e25a5c193d1066f02c1',
            full_text_obtained=True, pinned_library_forward_executed=True,
            author_backward_failed_inplace_version_check=True,
            full_original_experiment_reproduced=False),
        mechanism='Differentiable all-path resistance-curvature evolution of learned entity affinities',
        related_prior_family='Graph structure learning including the already evaluated cDGM adaptation',
        distinctness_gate='All-path metric evolution must beat both direct-edge-resistance flow and initial affinity. Renaming learned graph structure does not count as another successful family.',
        required_executor='improve.anchor_resistance_flow_experiment_v1', training_budget=budget,
        local_adaptation='Fixed-attribute symmetric 10NN supports with all boundary ties; typed-attention learned unit metric, exp(-squared distance), 3 steps per axis on miRNA and drug only; author-library min-resistance scaling, leaf convention and distance-weighted center; non-inplace active-edge arithmetic; zero-initialized pair residual head.',
        numerical_gate='Every inverse relative residual <=1e-10, positive finite regularized effective edge resistance and flow multipliers; stop if violated; saved-model replay tolerance 3e-5.',
        inference_claim='Dense ridge-regularized Laplacian Cholesky inverses. No unregularized exact-pseudoinverse, original RCF benchmark, near-linear solver or inherited complexity claim.',
        evaluation='Historical seed 666 screen; seeds 0/1 confirmation if eligible; three fixed development partitions; then frozen outer. Stop this version at the first failed joint gate.',
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        prior_failures_preserved=True, existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'resistance_flow',
        experiment.ROOT / 'results/nested_anchor_v2_20260915/historical', core_out)
    historical = base.read_json(out / 'protocol.json')
    for key in ('required_executor', 'training_budget', 'local_adaptation', 'numerical_gate',
                'inference_claim', 'related_prior_family', 'distinctness_gate'):
        historical[key] = p[key]
    historical['inputs'].update(p['inputs'])
    base.write_json(out / 'protocol.json', historical)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path, unit_path,
                    HERE / 'RESISTANCE_FLOW_ADAPTATION_V1.md', HERE / 'CURVATURE_SELECTION_V1.json']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='resistance_flow',
        source_sha256=base.digest(HERE / 'run_resistance_flow_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', workers_run_serially=True, stages=['screen', 'confirm'],
        stop_on_any_failed_gate=True, expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6}, adopted_neural_fits=0,
        identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('RESISTANCE_FLOW_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
