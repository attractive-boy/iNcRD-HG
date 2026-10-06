"""Freeze the independently adapted expected-graph auxiliary experiment."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_expected_graph_experiment_v1 as experiment
from improve import expected_graph_selftraining_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    family = 'expected_graph_selftraining'
    smoke_path = HERE / 'EXPECTED_GRAPH_SELFTRAINING_SMOKE_V1.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    if (smoke['performance_metrics_computed'] or smoke['formal_fits']
            or smoke['new_short_neural_fits'] != 3 or smoke['identity_replays'] != 3
            or smoke['short_training_steps_per_fit'] != 55 or len(smoke['fold_audits']) != 40
            or smoke['core_parameters'] != core.parameters()):
        raise ValueError('Invalid stage-crossing smoke audit')
    for path, digest in {**smoke['source_sha256'], **smoke['input_sha256']}.items():
        if base.digest(path) != digest:
            raise ValueError('Smoke source or input changed')
    for row in smoke['fold_audits']:
        if not row['identity'] and (row['forbidden_overlap'] or not row['label_flip_invariant']
                                    or row['reference_offsets_passed_to_proposals']):
            raise ValueError('Expected-graph isolation failed')
    for filename, count in [('pull_core_unit_v1.log', 4), ('pull_executor_unit_v2.log', 3)]:
        log = (HERE / filename).read_text()
        if f'Ran {count} tests' not in log or not log.rstrip().endswith('OK'):
            raise ValueError('Missing successful targeted checks')
    author = base.read_json(HERE / 'PULL_AUTHOR_SOURCE_V1.json')
    if author['tree_sha'] != '5901af0ffa51ba86e9368b023792ad79558a7b9c' or len(author['records']) != 4:
        raise ValueError('Author implementation revision changed')
    author_paths = []
    for row in author['records']:
        if row['status'] != 200 or base.digest(row['file']) != row['sha256']:
            raise ValueError('Author source verification failed')
        author_paths.append(Path(row['file']).resolve())
    root = experiment.ROOT
    core_out = root / 'results/expected_graph_selftraining_core_v1_20260916'
    out = root / 'results/anchor_priority_expected_graph_selftraining_v1_20260916'
    queue_path = HERE / 'EXPECTED_GRAPH_SELFTRAINING_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing studies')
    names = ('expected_graph_selftraining_v1.py', 'test_expected_graph_selftraining_v1.py',
        'anchor_expected_graph_experiment_v1.py', 'test_anchor_expected_graph_experiment_v1.py',
        'sharpness_graph_v1.py', 'overlap_block_models.py', 'overlap_block_experiment.py',
        'non_tabular_experiment.py', 'tabular_foundation.py', 'graph_prediction_precision.py',
        'nested_anchor_inputs_v2.py', 'residual_graph_experiment.py', 'anchor_priority_graph_v1.py',
        'assess_declared_graph.py', 'consolidate_declared_history_v1.py')
    inputs = [Path(__file__), smoke_path, HERE / 'pull_smoke_v1.py', HERE / 'PULL_ADAPTATION_V1.md',
        HERE / 'pull_core_unit_v1.log', HERE / 'pull_executor_unit_v2.log',
        HERE / 'pull_paper_v1.pdf', HERE / 'pull_paper_v1.txt', HERE / 'crossref_pull_v1.json',
        HERE / 'FOCUSED_FULLTEXT_V1.json', HERE / 'FOCUSED_METADATA_V1.json',
        HERE / 'PULL_AUTHOR_DISCOVERY_V1.json', HERE / 'PULL_AUTHOR_SOURCE_V1.json',
        HERE / 'pull_author_tree_v1.json', HERE / 'run_pull_queue_v1.py', *author_paths]
    if base.digest(HERE / 'pull_paper_v1.pdf') != 'f50fc89c5fd272c14b5572cab40a5ff55cf519b28d0efc1cb67413e3a5763d2a':
        raise ValueError('Paper identity changed')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        implementation_module='expected_graph_selftraining_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=['no_correction', 'observed_only'],
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Choose a checkpoint by mean global ROC AUC and sklearn AP of anchor + 0.1 residual from steps 60,80,...,400 only, after the first real augmentation refresh. Steps 20/40 are recorded but ineligible. Select one common learning rate per arm from 0.001/0.003 by the MiDrug-up/LncDrug-tolerance gate, then mean MiDrug gain, then smaller learning rate.',
        source_sha256={name: base.digest(root / name) for name in names},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='Accurate Link Prediction for Edge-Incomplete Graphs via PU Learning',
            arxiv_fulltext_read='2405.11911v2', arxiv_version_date='2024-12-12',
            crossref_doi='10.1609/aaai.v39i17.33966', crossref_date='2025-04-11', venue='AAAI 2025',
            author_tree=author['tree_sha'], author_files_read=4, author_code_executed=False),
        mechanism_family='auxiliary_expected_graph_selftraining',
        local_adaptation='Typed RMS-normalized attributes; shared weighted GCN with a zero-initialized residual head and a separate absolute-probability head; strict cyclic query deletion; virtual candidates restricted to other fitting-zero rows; all boundary ties retained; confidence threshold applied to current unknown query targets; 50-step teacher stages within 400 steps; 0.1 auxiliary weight; equal-total-weight no-correction control; original-graph residual inference; exact LncDrug identity.',
        residual_is_edge_probability=False, reference_offsets_used_for_proposals=False,
        virtual_edges_are_biological_facts=False, no_unbiased_pu_or_causal_guarantee_claim=True,
        relation_to_other_families='Related to graph augmentation and self-training. Shared completion ideas cannot be counted twice as independent components.',
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
                     HERE / 'PULL_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        source_sha256=base.digest(HERE / 'run_pull_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs}, device='cuda:1',
        stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6}, adopted_neural_fits=0,
        teacher_snapshots_count_as_neural_fits=False, identity_records_count_as_neural_fits=False,
        confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('EXPECTED_GRAPH_SELFTRAINING_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
