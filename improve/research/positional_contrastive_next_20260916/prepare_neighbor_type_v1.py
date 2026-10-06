"""Seal the new contrastive objective before any performance computation."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_contrastive_experiment_v1 as experiment
from improve import neighbor_type_contrastive_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    family = 'neighbor_type_contrastive'
    smoke_path = HERE / 'NEIGHBOR_TYPE_CONTRASTIVE_SMOKE_V2.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    if smoke['performance_metrics_computed'] or smoke['formal_fits'] or len(smoke['fold_audits']) != 40:
        raise ValueError('Invalid training-only smoke audit')
    if smoke['core_parameters'] != core.parameters():
        raise ValueError('Core parameters changed')
    for path, digest in smoke['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Smoke source changed')
    root = experiment.ROOT
    core_out = root / 'results/neighbor_type_contrastive_core_v1_20260916'
    out = root / 'results/anchor_priority_neighbor_type_contrastive_v1_20260916'
    queue_path = HERE / 'NEIGHBOR_TYPE_CONTRASTIVE_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing studies')
    names = ('neighbor_type_contrastive_v1.py', 'test_neighbor_type_contrastive_v1.py',
        'sharpness_graph_v1.py', 'test_anchor_contrastive_experiment_v1.py',
        'overlap_block_models.py', 'overlap_block_experiment.py',
        'non_tabular_experiment.py', 'tabular_foundation.py', 'graph_prediction_precision.py',
        'nested_anchor_inputs_v2.py', 'residual_graph_experiment.py',
        'anchor_contrastive_experiment_v1.py', 'assess_declared_graph.py')
    inputs = [Path(__file__), smoke_path, HERE / 'neighbor_type_smoke_v2.py',
        HERE / 'NEIGHBOR_TYPE_ADAPTATION_V1.md', HERE / 'DISCOVERY_ARXIV_V1.xml',
        HERE / 'ntssm_v1.pdf', HERE / 'ntssm_v1.txt', HERE / 'DOWNLOAD_MANIFEST_V2.json',
        HERE / 'ntssm_author_tree_v1.json', HERE / 'NTSSM_AUTHOR_SOURCE_V1.json',
        HERE / 'NTSSM_PUBLISHER_CHECK_V1.json', HERE / 'NTSSM_VENUE_CROSSREF_V1.json']
    downloads = base.read_json(HERE / 'DOWNLOAD_MANIFEST_V2.json')['records']
    paper = next(r for r in downloads if r['key'] == 'ntssm')
    if paper['status'] != 'open_access_downloaded' or base.digest(paper['path']) != paper['sha256']:
        raise ValueError('Paper verification failed')
    for entry in base.read_json(HERE / 'NTSSM_AUTHOR_SOURCE_V1.json')['records']:
        if entry['status'] != 'downloaded' or base.digest(entry['file']) != entry['sha256']:
            raise ValueError('Author source incomplete or changed')
        inputs.append(Path(entry['file']))
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        implementation_module='neighbor_type_contrastive_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=['type_shared', 'bce'],
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Fixed primary/controls, external Table 5 coefficients and two learning rates before performance; no architecture or coefficient search.',
        auxiliary_objective='0.1 times bidirectional source-type negative-logit contrastive loss; positive-only training query batches; duplicate-endpoint false negatives masked. Type-shared control preserves direction-specific mean alpha; BCE disables auxiliary gradient.',
        source_sha256={name: base.digest(root / name) for name in names},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy',
            arxiv='2605.24015v1', author_claimed_venue='ICML 2026',
            publisher_landing_verified=False, official_source_tree='da8655ef0b331e5fb2b6e5ceb05756b2f8c8aa05'),
        mechanism_family='graph_self_supervised_contrastive',
        mechanism='Exact source-type decomposition of linear bipartite propagation and distinct negative-logit coefficients in a symmetric contrastive objective.',
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
                     HERE / 'NEIGHBOR_TYPE_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        source_sha256=base.digest(HERE / 'run_neighbor_type_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs}, device='cuda:1',
        stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('NEIGHBOR_TYPE_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
