"""Freeze Fourier quantile aggregation after independent and training-only checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

from improve import anchor_fourier_quantile_experiment_v1 as experiment
from improve import fourier_quantile_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    smoke_path, unit_path = HERE / 'FOURIER_QUANTILE_SMOKE_V1.json', HERE / 'FOURIER_QUANTILE_UNIT_V2.json'
    smoke, unit = (base.check_seal(base.read_json(path)) for path in (smoke_path, unit_path))
    assert unit['success'] and unit['tests'] == 5
    assert smoke['formal_fits'] == 0 and not smoke['performance_metrics_computed']
    assert not smoke['evaluation_labels_read'] and smoke['core_parameters'] == core.parameters()
    assert len(smoke['fold_audits']) == 40 and len(smoke['records']) == 6
    assert all(smoke[key] for key in ('final_random_streams_equal_across_arms',
        'initialized_parameters_and_random_streams_equal', 'replay_uses_only_omitted_fold',
        'fixed_fit_rejects_evaluation_keys', 'initial_residual_exact_zero',
        'every_parameter_gradient_finite', 'sort_and_message_counts_equal_across_arms'))
    for mapping in (unit['source_sha256'], smoke['source_sha256'], smoke['input_sha256']):
        for path, digest in mapping.items():
            assert base.digest(path) == digest, path
    core_out = experiment.ROOT / 'results/fourier_quantile_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_fourier_quantile_v1_20260916'
    queue_path = HERE / 'FOURIER_QUANTILE_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve earlier studies; use a new version')
    sources = ['fourier_quantile_graph_v1.py', 'test_fourier_quantile_graph_v1.py',
        'anchor_fourier_quantile_experiment_v1.py', 'sharpness_graph_v1.py',
        'test_anchor_priority_graph_v1.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py', 'tabular_foundation.py',
        'graph_prediction_precision.py', 'nested_anchor_inputs_v2.py',
        'residual_graph_experiment.py', 'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    names=['FOURIER_QUANTILE_ADAPTATION_V1.md','FOURIER_QUANTILE_SELECTION_V1.json',
        'fourier_quantile_smoke_v1.py','fourier_quantile_smoke_v1.log','fourier_quantile_unit_v2.log',
        'run_fourier_quantile_queue_v1.py','consolidate_fourier_quantile_v1.py',
        'fsw_multiset_2024_v3.pdf','fsw_multiset_2024_v3.txt','fsw_gnn_v2_export.pdf','fsw_gnn_v2_export.txt',
        'fsw_gnn_commit_v1.json','fsw_gnn_pinned_tree_v1.json','FSW_GNN_AUTHOR_SOURCE_V1.json',
        'FSW_PUBLISHER_V1.json','fsw_iclr2025_publisher_v1.html','DISCOVERY_V1.json',
        'DISCOVERY_DEDUPLICATED_V1.json','FSW_FULLTEXT_V1.json','FSW_SOURCE_FOLLOWUP_V2.json',
        'FSW_NEAR_ZERO_GRADIENT_AUDIT_V1.json','FOURIER_QUANTILE_UNIT_V1.json']
    inputs=[smoke_path,unit_path,Path(__file__)]+[HERE/name for name in names]
    for record in base.read_json(HERE/'FSW_GNN_AUTHOR_SOURCE_V1.json'):
        path=(experiment.ROOT.parent/record['file']).resolve()
        assert record['git_blob_verified'] and base.digest(path)==record['sha256']
        assert record['commit']=='596ae091528bc10698eed73d60ffc00140d61d4a'
        inputs.append(path)
    budget=('400 AdamW steps and gradient evaluations per model, one omitted fold per step; '
        '20 common checkpoints. Equal modules, parameter counts, initial states and random streams; '
        'every arm computes the same full-neighborhood sorts and moments. 505 training/selection '
        'forwards, 1010 message-passing layers before independent saved-weight replay.')
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family='fourier_quantile',
        implementation_module='fourier_quantile_graph_v1',tasks=['MiDrug','LncDrug'],
        kinds=list(core.KINDS),primary_kind='full',primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(),lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Full, mean-only and projected mean/std fixed before performance; two learning rates, 400 steps and each 20-step checkpoint. No embedding width, frequency, depth, neighborhood or fusion sweep.',
        source_sha256={name:base.digest(experiment.ROOT/name) for name in sources},
        inputs={str(path.resolve()):base.digest(path) for path in inputs},
        literature=dict(embedding_arxiv='2405.16519v3',embedding_venue='ICLR 2025',
            embedding_venue_publisher_verified=True,duplicate_alias='2504.02544v2',
            graph_arxiv='2410.09118v2',graph_version_date='2026-03-02',
            graph_venue_author_reported='LoG 2025',graph_venue_publisher_verified=False,
            author_repository='yonatansverdlov/Over-squashing',
            author_commit='596ae091528bc10698eed73d60ffc00140d61d4a',
            full_text_obtained=True,unchanged_dense_helper_ast_executed=True,
            full_original_GNN_experiment_reproduced=False),
        mechanism='Distribution-sensitive Fourier integration of projected neighborhood quantile functions in typed message passing',
        related_prior_family='Related to multiset pooling and optimal transport; previous cross-RNA alignment and determinant subset selection remain separately recorded',
        distinctness_gate='Must beat both projected means and projected mean/std; literature aliases, numerical correction and common residual infrastructure do not count separately.',
        required_executor='improve.anchor_fourier_quantile_experiment_v1',training_budget=budget,
        local_adaptation='Two layers, width64, edge embedding8, 128 unit directions/positive softplus frequencies; spread initialization; all non-diagonal messages with unit mass; power-of-two bucket storage without truncation or padding mass; exact interval integral in FP64; sinc sixth-order series for |x|<1e-3; log1p degree, residual/LayerNorm and zero pair head.',
        numerical_gate='Finite forward values, every parameter gradient and optimizer state; exact empty/zero-head limits; independent cosine quadrature, near-zero derivative and pinned-author dense-helper checks; saved-model replay tolerance 3e-5.',
        inference_claim='Finite-width learned link-prediction adaptation. No source dimension/depth bi-Lipschitz or full WL-equivalence guarantee, unbiased SW-distance estimate, or reproduced source benchmark claim.',
        evaluation='Historical seed666 screen; seeds0/1 only if eligible; then three fixed development partitions and separately frozen outer. Stop at first failed joint or numerical gate.',
        no_new_performance_observed=True,smoke=False,original_test_source_access=False,
        prior_failures_preserved=True,existing_default_replaced=False,confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'fourier_quantile',
        experiment.ROOT / 'results/nested_anchor_v2_20260915/historical', core_out)
    for task in p['tasks']:
        training=experiment.nested.load_selected(out/'inputs'/task/'arrays.npz',experiment.TRAIN_KEYS)
        checked=experiment.nested.load_selected(experiment.ROOT/'results/anchor_priority_morphological_alignment_v1_20260916/inputs'/task/'arrays.npz',experiment.TRAIN_KEYS)
        assert set(training)==set(checked)
        for key in training:np.testing.assert_array_equal(training[key],checked[key])
    historical = base.read_json(out / 'protocol.json')
    for key in ('required_executor', 'training_budget', 'local_adaptation', 'numerical_gate',
                'inference_claim', 'related_prior_family', 'distinctness_gate'):
        historical[key] = p[key]
    historical['inputs'].update(p['inputs'])
    base.write_json(out / 'protocol.json', historical)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path, unit_path,
                    HERE / 'FOURIER_QUANTILE_ADAPTATION_V1.md', HERE / 'FOURIER_QUANTILE_SELECTION_V1.json']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='fourier_quantile',
        source_sha256=base.digest(HERE / 'run_fourier_quantile_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', workers_run_serially=True, stages=['screen', 'confirm'],
        stop_on_any_failed_gate=True, expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6}, adopted_neural_fits=0,
        identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('FOURIER_QUANTILE_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
