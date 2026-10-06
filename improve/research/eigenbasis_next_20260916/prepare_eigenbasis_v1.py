"""Freeze a narrower SOAP study, preserving the failed KL-Shampoo protocol."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_eigenbasis_experiment_v1 as experiment
from improve import eigenbasis_adam_v1 as optimizer
from improve import eigenbasis_graph_v1 as core
from improve import adopt_eigenbasis_parent_v1 as adoption
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    root = experiment.ROOT
    smoke_path, unit_path = HERE/'EIGENBASIS_SMOKE_V1.json', HERE/'EIGENBASIS_UNIT_V1.json'
    smoke, unit = base.check_seal(base.read_json(smoke_path)), base.check_seal(base.read_json(unit_path))
    if (not unit['passed'] or smoke['new_short_neural_fits'] != 1 or smoke['identity_replays'] != 3
            or smoke['performance_metrics_computed'] or smoke['formal_fits']
            or len(smoke['unchanged_graph_fold_audits_reused']) != 40
            or smoke['core_parameters'] != core.parameters() or smoke['optimizer_parameters'] != optimizer.parameters()):
        raise ValueError('Implementation checks failed')
    for path, digest in {**unit['source_sha256'], **smoke['source_sha256'], **smoke['input_sha256']}.items():
        if base.digest(path) != digest:
            raise ValueError('Checked source or input changed')
    core_out = root/'results/eigenbasis_adam_core_v1_20260916'
    out = root/'results/anchor_priority_eigenbasis_adam_v1_20260916'
    queue_path = HERE/'EIGENBASIS_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing study artifacts')
    parent = root/'results/anchor_priority_kronecker_preconditioning_v1_20260916'
    parent_core = root/'results/kronecker_preconditioning_core_v1_20260916'
    p = dict(experiment.common.protocol(parent_core))
    experiment.common.protocol(parent)
    names = tuple(dict.fromkeys(experiment.OWN+('adopt_eigenbasis_parent_v1.py',)))
    inputs = [Path(__file__), smoke_path, unit_path, HERE/'eigenbasis_smoke_v1.py',
        HERE/'EIGENBASIS_ADAPTATION_V1.md', HERE/'run_eigenbasis_queue_v1.py',
        HERE/'EXECUTOR_DERIVATION_V1.json', parent/'protocol.json', parent/'SCREEN_ASSESSMENT.json',
        parent_core/'protocol.json']
    p.update(created_at_utc=datetime.now(timezone.utc).isoformat(), family='eigenbasis_adam',
        implementation_module='eigenbasis_graph_v1', kinds=list(core.KINDS), primary_kind='full',
        primary_controls=['separable_variance', 'adamw'], core_parameters=core.parameters(),
        optimizer_parameters=optimizer.parameters(), parent_screening_results_seen=True,
        no_new_performance_observed=False, independent_screen=False,
        literature=dict(title='SOAP: Improving and Stabilizing Shampoo using Adam',
            arxiv_fulltext_read='2409.11321v2', author_code_commit='a1e553530fde97d0e6b307d7c82ac6d38b072340',
            author_code_read=True, author_code_executed=False, publisher_status_verified=False),
        mechanism_family='coordinate_second_moments_in_learned_gradient_eigenbasis',
        local_adaptation='Exact prior SOAP optimizer and graph. New separable_variance changes only the second-moment denominator to its row/column rank-one factorization; it keeps full state for matched bookkeeping. All budgets and LncDrug identity match the parent.',
        baseline_adoption=dict(parent=str(parent), mapping={'full':'soap', 'adamw':'adamw'},
            required_equal_inputs=True, required_exact_replay=True, parent_outcomes_seen=True,
            expected_adopted_neural_fits=4, expected_adopted_identity_records=4),
        source_sha256={**p['source_sha256'], **{name:base.digest(root/name) for name in names}},
        inputs={**p['inputs'], **{str(path.resolve()):base.digest(path) for path in inputs}})
    core_out.mkdir(parents=True)
    (core_out/'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(root/name, core_out/'source_snapshot'/name)
    base.write_json(core_out/'protocol.json', p)
    experiment.prepare(out, 'eigenbasis_adam', root/'results/nested_anchor_v2_20260915/historical', core_out)
    adoption.adopt(out)
    frozen = [core_out/'protocol.json', out/'protocol.json', out/'PARENT_ADOPTION_V1.json',
              smoke_path, HERE/'EIGENBASIS_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='eigenbasis_adam',
        source_sha256=base.digest(HERE/'run_eigenbasis_queue_v1.py'),
        input_sha256={str(path):base.digest(path) for path in frozen}, device='cuda:1',
        stages=['screen','confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen':2,'confirm_if_eligible':6}, adopted_neural_fits=4,
        expected_identity_records={'screen':6,'confirm_if_eligible':6}, adopted_identity_records=4,
        identity_records_count_as_neural_fits=False, independent_screen=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('EIGENBASIS_FROZEN', base.digest(out/'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
