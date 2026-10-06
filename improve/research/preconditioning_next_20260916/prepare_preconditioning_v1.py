"""Freeze the optimizer experiment after the independent training-only checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_preconditioning_experiment_v1 as experiment
from improve import preconditioned_graph_v1 as core
from improve import kronecker_preconditioning_v1 as optimizer
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    family = 'kronecker_preconditioning'
    smoke_path, unit_path = HERE / 'PRECONDITIONING_SMOKE_V1.json', HERE / 'PRECONDITIONING_UNIT_V1.json'
    smoke, unit = base.check_seal(base.read_json(smoke_path)), base.check_seal(base.read_json(unit_path))
    if (smoke['performance_metrics_computed'] or smoke['formal_fits']
            or smoke['new_short_neural_fits'] != 4 or smoke['identity_replays'] != 4
            or len(smoke['fold_audits']) != 40 or smoke['core_parameters'] != core.parameters()
            or smoke['optimizer_parameters'] != optimizer.parameters()):
        raise ValueError('Invalid training-only smoke audit')
    if (not unit['passed'] or unit['failures'] or unit['errors']
            or unit['core_tests_run'] != 6 or unit['executor_tests_run'] != 3):
        raise ValueError('Numerical/executor unit checks did not pass')
    for path, digest in {**smoke['source_sha256'], **smoke['input_sha256'], **unit['source_sha256']}.items():
        if base.digest(path) != digest:
            raise ValueError('Checked source or input changed')
    for row in smoke['fold_audits']:
        if not row['identity'] and (row['forbidden_overlap'] or not row['query_labels_invariant']):
            raise ValueError('A support-fold boundary failed')
    root = experiment.ROOT
    core_out = root / 'results/kronecker_preconditioning_core_v1_20260916'
    out = root / 'results/anchor_priority_kronecker_preconditioning_v1_20260916'
    queue_path = HERE / 'PRECONDITIONING_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing studies')
    names = tuple(dict.fromkeys(experiment.OWN + ('overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py', 'tabular_foundation.py',
        'consolidate_declared_history_v1.py')))
    inputs = [Path(__file__), smoke_path, unit_path, HERE / 'preconditioning_smoke_v1.py',
        HERE / 'PRECONDITIONING_ADAPTATION_V1.md', HERE / 'run_preconditioning_queue_v1.py',
        HERE / 'PRECONDITIONING_DISCOVERY_V1.json', HERE / 'PRECONDITIONING_FULLTEXT_FETCH_V1.json',
        HERE / 'PRECONDITIONING_AUTHOR_DISCOVERY_V1.json', HERE / 'PRECONDITIONING_AUTHOR_SOURCE_V1.json',
        HERE / 'PRECONDITIONING_PUBLICATION_CHECK_V1.json', HERE / 'soap_paper_v1.pdf',
        HERE / 'soap_paper_v1.txt', HERE / 'kl_shampoo_paper_v1.pdf', HERE / 'kl_shampoo_paper_v1.txt']
    for record in base.read_json(HERE / 'PRECONDITIONING_AUTHOR_SOURCE_V1.json'):
        path = Path(record['file'])
        if base.digest(path) != record['sha256']:
            raise ValueError('Pinned author source changed')
        inputs.append(path)
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        implementation_module='preconditioned_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=['scalar_coupling', 'soap', 'adamw'],
        core_parameters=core.parameters(), optimizer_parameters=optimizer.parameters(),
        lnc_policy='identity_strong_reference', identity_records_are_not_neural_fits=True,
        selection='Select each run checkpoint by mean global AUC and sklearn AP for anchor + 0.1 residual every 20 updates. Choose one shared learning rate per arm by MiDrug improvement and LncDrug tolerance, then mean MiDrug gain, then smaller learning rate. Only lr=0.001/0.003 are searched.',
        source_sha256={name: base.digest(root / name) for name in names},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(primary_title='Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization',
            arxiv_fulltext_read='2509.03378v10', version_date='2026-06-22',
            author_claimed_venue='ICLR 2026 (earlier version)', publisher_status_verified=False,
            author_code_commit='a02e6221556301cf84f9607c73f327910afb2b95',
            author_code_read=True, author_code_executed=False,
            soap_fulltext_read='2409.11321v2', soap_author_code_commit='a1e553530fde97d0e6b307d7c82ac6d38b072340',
            implementation='Independent matrix-only equation/code-informed adaptation'),
        mechanism_family='two_sided_kronecker_covariance_preconditioning',
        local_adaptation='Unchanged graph residual architecture; only nondegenerate Linear matrices are preconditioned. AdamW handles other parameters. All arms consume one initialization-only gradient then 400 updates. Common betas (0.9,0.98), fp32 states, decay before update, no grafting for KL matrix updates. Scalar-coupling is a purpose-built directional ablation, not official Shampoo. LncDrug is exact identity.',
        no_exact_eigendecomposition_or_kl_optimum_claim=True,
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        evaluation='Historical screen, three initializations, three fixed partitions, frozen outer recheck; every gate is mandatory.',
        confirmed_innovations=0, existing_default_replaced=False)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(root / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, family, root / 'results/nested_anchor_v2_20260915/historical', core_out)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path,
                     HERE / 'PRECONDITIONING_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        source_sha256=base.digest(HERE / 'run_preconditioning_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs}, device='cuda:1',
        stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 8, 'confirm_if_eligible': 8},
        expected_identity_records={'screen': 8, 'confirm_if_eligible': 8},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False,
        confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('PRECONDITIONING_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
