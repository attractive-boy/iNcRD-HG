"""Freeze conditional determinant graph selection after training-only checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_determinantal_selection_experiment_v1 as experiment
from improve import determinantal_selection_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    smoke_path, unit_path = HERE / 'DETERMINANTAL_SMOKE_V1.json', HERE / 'DETERMINANTAL_UNIT_V2.json'
    smoke, unit = (base.check_seal(base.read_json(path)) for path in (smoke_path, unit_path))
    assert unit['success'] and unit['tests'] == 7
    assert smoke['formal_fits'] == 0 and not smoke['performance_metrics_computed']
    assert not smoke['evaluation_labels_read'] and smoke['core_parameters'] == core.parameters()
    assert len(smoke['fold_audits']) == 40 and smoke['final_random_streams_equal_across_arms']
    assert smoke['initialized_parameters_and_random_streams_equal'] and smoke['replay_uses_only_omitted_fold']
    for mapping in (unit['source_sha256'], smoke['source_sha256'], smoke['input_sha256']):
        for path, digest in mapping.items():
            assert base.digest(path) == digest, path
    core_out = experiment.ROOT / 'results/determinantal_selection_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_determinantal_selection_v1_20260916'
    queue_path = HERE / 'DETERMINANTAL_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve earlier studies; use a new version')
    sources = ['determinantal_selection_graph_v1.py', 'test_determinantal_selection_graph_v1.py',
        'anchor_determinantal_selection_experiment_v1.py', 'sharpness_graph_v1.py',
        'test_anchor_priority_graph_v1.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py',
        'tabular_foundation.py', 'graph_prediction_precision.py',
        'nested_anchor_inputs_v2.py', 'residual_graph_experiment.py',
        'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    inputs = [smoke_path, unit_path, HERE / 'CDPRUNER_ADAPTATION_V1.md',
        HERE / 'determinantal_smoke_v1.py', HERE / 'determinantal_unit_v2.log', Path(__file__),
        HERE / 'run_determinantal_queue_v1.py', HERE / 'conditional_diversity_paper_v1.pdf',
        HERE / 'conditional_diversity_paper_v1.txt', HERE / 'cdpruner_commit_v1.json',
        HERE / 'cdpruner_tree_v1.json', HERE / 'CDPRUNER_AUTHOR_SOURCE_V1.json',
        HERE / 'DIVERSITY_RESOURCE_PROBE_V1.json', HERE / 'DETERMINANTAL_UNIT_V1.json']
    for record in base.read_json(HERE / 'CDPRUNER_AUTHOR_SOURCE_V1.json')['records']:
        if record['status'] != 'downloaded':
            continue
        path = Path(record['file'])
        assert record['git_blob_verified'] and base.digest(path) == record['sha256']
        inputs.append(path)
    budget = ('400 AdamW steps and gradient evaluations per model, one omitted training fold per step; '
              '20 common checkpoints. Identical initialized modules, parameter counts, '
              'graph/query inputs and random streams in every arm; no additional fitting in hard selection.')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='determinantal_selection',
        implementation_module='determinantal_selection_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Full, no-diversity and no-condition fixed before new performance; two learning rates, 400 steps and every 20-step checkpoint. No retention, quality, ridge, backbone or selection-rule sweep.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs',
            arxiv='2506.10967v2', arxiv_version_date='2025-07-01', author_reported_venue='NeurIPS 2025',
            publisher_venue_independently_verified=False,
            author_commit='9541616c40fcd5625de1cdb8ea6c33c129eb7864', full_text_obtained=True,
            pinned_author_method_ast_executed=True, full_original_experiment_reproduced=False),
        mechanism='Query-conditioned diversity of a hard-selected subset of graph-neighborhood evidence',
        required_executor='improve.anchor_determinantal_selection_experiment_v1', training_budget=budget,
        local_adaptation='All one-hop union candidates after deleting the entire query fold; positive endpoint-sum cosine with per-query normalization; 1e-4 quality floor and 1e-5 diagonal ridge; FP64 matrix-free greedy Cholesky, 16 retained neighbors; hard detached indices with gathered embedding gradients; zero-initialized residual head.',
        numerical_gate='Positive finite FP64 pivots and PSD residuals; unit/zero vectors; exact dense greedy subsets on synthetic data; prediction replay absolute tolerance 3e-5.',
        inference_claim='Greedy MAP heuristic; no exact MAP, DPP sampling, or general approximation-ratio claim.',
        evaluation='Historical seed 666 screen; seeds 0/1 confirmation if eligible; three fixed development partitions; then frozen outer. Stop this version at the first failed joint gate.',
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        prior_failures_preserved=True, existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'determinantal_selection',
        experiment.ROOT / 'results/nested_anchor_v2_20260915/historical', core_out)
    historical = base.read_json(out / 'protocol.json')
    historical.update(required_executor=p['required_executor'], training_budget=budget,
        local_adaptation=p['local_adaptation'], numerical_gate=p['numerical_gate'],
        inference_claim=p['inference_claim'])
    historical['inputs'].update(p['inputs'])
    base.write_json(out / 'protocol.json', historical)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path, unit_path,
                     HERE / 'CDPRUNER_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='determinantal_selection',
        source_sha256=base.digest(HERE / 'run_determinantal_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('DETERMINANTAL_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
