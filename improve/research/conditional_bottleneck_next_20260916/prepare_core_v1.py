"""Freeze CORE-derived bottleneck screening after training-only implementation checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_bottleneck_experiment_v1 as experiment
from improve import information_bottleneck_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    family = 'information_bottleneck'
    smoke_path = HERE / 'INFORMATION_BOTTLENECK_SMOKE_V1.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    if (smoke['performance_metrics_computed'] or smoke['formal_fits']
            or smoke['new_short_neural_fits'] != 3 or smoke['identity_replays'] != 3
            or len(smoke['fold_audits']) != 40 or smoke['core_parameters'] != core.parameters()):
        raise ValueError('Invalid training-only smoke audit')
    for path, digest in {**smoke['source_sha256'], **smoke['input_sha256']}.items():
        if base.digest(path) != digest:
            raise ValueError('Smoke source or input changed')
    for row in smoke['fold_audits']:
        if not row['identity'] and (not row['query_label_flip_invariant']
                or any(row[k] for k in ('original_forbidden_overlap',
                                        'augmented_forbidden_overlap', 'virtual_forbidden_overlap'))):
            raise ValueError('A query fold failed isolation')
    unit_path = HERE / 'CORE_EXECUTOR_UNIT_V1.json'
    unit = base.check_seal(base.read_json(unit_path))
    if unit['tests_run'] != 3 or unit['failures'] or unit['exit_code']:
        raise ValueError('Executor unit checks did not pass')
    for path, digest in unit['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Unit-checked source changed')
    root = experiment.ROOT
    core_out = root / 'results/information_bottleneck_core_v1_20260916'
    out = root / 'results/anchor_priority_information_bottleneck_v1_20260916'
    queue_path = HERE / 'INFORMATION_BOTTLENECK_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing studies')
    names = ('information_bottleneck_graph_v1.py', 'test_information_bottleneck_graph_v1.py',
        'anchor_bottleneck_experiment_v1.py', 'test_anchor_bottleneck_experiment_v1.py',
        'bicentric_subgraph.py', 'sharpness_graph_v1.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py', 'tabular_foundation.py',
        'graph_prediction_precision.py', 'nested_anchor_inputs_v2.py',
        'residual_graph_experiment.py', 'anchor_priority_graph_v1.py',
        'assess_declared_graph.py', 'consolidate_declared_history_v1.py')
    inputs = [Path(__file__), smoke_path, unit_path, HERE / 'core_smoke_v1.py',
        HERE / 'CORE_ADAPTATION_V1.md', HERE / 'CORE_RESOURCE_V1.json',
        HERE / 'core_resource_v1.py', HERE / 'core_paper_v1.pdf', HERE / 'core_paper_v1.txt',
        HERE / 'core_crossref_v1.json', HERE / 'CORE_AUTHOR_SEARCH_V2.json',
        HERE / 'BOTTLENECK_FULLTEXT_ACQUISITION_V1.json',
        HERE / 'run_core_queue_v1.py', HERE / 'core_unit_checks_v3.log']
    if base.digest(HERE / 'core_paper_v1.pdf') != '27ca9e936d4f39ebce446bf67a5bb44097c83930b4accc2bf37288414651c33d':
        raise ValueError('Paper identity changed')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        implementation_module='information_bottleneck_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=['no_kl', 'no_completion'],
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Within each run choose a checkpoint by the mean of global AUC and sklearn AP for anchor + 0.1 residual, checking every 20 steps. Choose one shared learning rate per arm by the MiDrug-up/LncDrug-tolerance gate, then mean MiDrug gain, then smaller learning rate. Only the two common learning rates are searched.',
        source_sha256={name: base.digest(root / name) for name in names},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='CORE: Data Augmentation for Link Prediction via Information Bottleneck',
            authors=['Kaiwen Dong', 'Zhichun Guo', 'Nitesh V. Chawla'],
            arxiv_fulltext_read='2404.11032v1', arxiv_date='2024-04-17',
            crossref_doi='10.1145/3789200', crossref_published_online='2026-03-07',
            crossref_venue='ACM Transactions on Knowledge Discovery from Data',
            journal_fulltext_verified=False, author_code_verified=False,
            implementation='Independent equation-based adaptation'),
        mechanism_family='query_conditioned_information_bottleneck_graph_completion',
        local_adaptation='Resource-allocation completion restricted to same-node-type virtual edges, all cutoff ties kept, query-deleted one-hop marked subgraphs, shared two-layer GCN, nested-reference BCE residual, fixed inflated-graph degree normalization, relaxed Bernoulli masks in training, sigmoid probability weights in inference, and exact LncDrug identity. Original/rank origin embeddings and per-graph edge-sum KL are retained.',
        completion_is_supervision=False, target_query_virtual_edges_forbidden=True,
        inference_is_exact_marginalization=False, no_causal_or_theorem_guarantee_claim=True,
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        evaluation='Historical screen, three initializations, three fixed partitions, frozen outer recheck; every gate required.',
        confirmed_innovations=0, existing_default_replaced=False)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(root / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, family, root / 'results/nested_anchor_v2_20260915/historical', core_out)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path,
                     HERE / 'CORE_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        source_sha256=base.digest(HERE / 'run_core_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs}, device='cuda:1',
        stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False,
        confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('INFORMATION_BOTTLENECK_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
