"""Seal differential attention after training-only implementation checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_priority_graph_v1 as experiment
from improve import differential_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    family = 'differential_attention'
    smoke_path = HERE / 'DIFFERENTIAL_ATTENTION_SMOKE_V1.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    if smoke['performance_metrics_computed'] or smoke['formal_fits'] or len(smoke['fold_audits']) != 40:
        raise ValueError('Invalid training-only smoke audit')
    if smoke['core_parameters'] != core.parameters():
        raise ValueError('Core parameters changed')
    for path, digest in smoke['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Smoke source changed')
    root = experiment.ROOT
    core_out = root / 'results/differential_attention_core_v1_20260916'
    out = root / 'results/anchor_priority_differential_attention_v1_20260916'
    queue_path = HERE / 'DIFFERENTIAL_ATTENTION_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing studies')
    names = ('differential_graph_v1.py', 'test_differential_graph_v1.py',
        'sharpness_graph_v1.py', 'test_anchor_priority_graph_v1.py',
        'overlap_block_models.py', 'overlap_block_experiment.py',
        'non_tabular_experiment.py', 'tabular_foundation.py', 'graph_prediction_precision.py',
        'nested_anchor_inputs_v2.py', 'residual_graph_experiment.py',
        'anchor_priority_graph_v1.py', 'assess_declared_graph.py')
    inputs = [Path(__file__), smoke_path, HERE / 'differential_smoke_v1.py',
        HERE / 'DIFFERENTIAL_ATTENTION_ADAPTATION_V1.md', HERE / 'arxiv_v1.xml',
        HERE / 'differential_arxiv_v2.pdf', HERE / 'differential_arxiv_v2.txt',
        HERE / 'PDF_VERIFICATION_V2.json', HERE / 'author_tree_v1.json', HERE / 'AUTHOR_SOURCE_V1.json']
    for entry in base.read_json(HERE / 'AUTHOR_SOURCE_V1.json')['records']:
        if entry['status'] != 'downloaded' or base.digest(entry['file']) != entry['sha256']:
            raise ValueError('Author source incomplete or changed')
        inputs.append(Path(entry['file']))
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        implementation_module='differential_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=['additive_attention'],
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Fixed primary/control and two learning rates before performance; no architecture search.',
        source_sha256={name: base.digest(root / name) for name in names},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='Differential Transformer', arxiv='2410.05258v2',
            original_paper_venue='ICLR 2025', actual_attention_ancestor='Author Diff Transformer V2 code release',
            v2_peer_reviewed_status_claimed=False),
        mechanism='Difference of two support attention maps with shared K/V, node/head sigmoid gates and matched coefficient normalization.',
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
                     HERE / 'DIFFERENTIAL_ATTENTION_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        source_sha256=base.digest(HERE / 'run_differential_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs}, device='cuda:1',
        stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 4, 'confirm_if_eligible': 4},
        expected_identity_records={'screen': 4, 'confirm_if_eligible': 4},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('DIFFERENTIAL_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
