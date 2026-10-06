"""Freeze the self-taught meta-loss study after numerical and boundary probes."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_meta_loss_experiment_v1 as experiment
from improve import meta_loss_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    smoke_path, unit_path = HERE / 'META_LOSS_SMOKE_V2.json', HERE / 'META_LOSS_UNIT_V2.json'
    smoke, unit = (base.check_seal(base.read_json(path)) for path in (smoke_path, unit_path))
    assert smoke['formal_fits'] == 0 and not smoke['performance_metrics_computed']
    assert not smoke['evaluation_labels_read'] and smoke['episode_rng_matched'] and smoke['dropout_rng_matched']
    assert smoke['core_parameters'] == core.parameters() and len(smoke['fold_audits']) == 40
    assert unit['passed'] and unit['tests_run'] == 6
    for mapping in (smoke['source_sha256'], unit['input_sha256'], smoke['reference_manifest_sha256']):
        for path, digest in mapping.items():
            if base.digest(path) != digest:
                raise ValueError('Pre-performance evidence changed: ' + path)
    core_out = experiment.ROOT / 'results/meta_loss_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_meta_loss_v1_20260916'
    queue_path = HERE / 'META_LOSS_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve previous studies; use a new version')
    sources = ['meta_loss_graph_v1.py', 'test_meta_loss_graph_v1.py',
        'anchor_meta_loss_experiment_v1.py', 'sharpness_graph_v1.py',
        'test_anchor_priority_graph_v1.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py',
        'tabular_foundation.py', 'graph_prediction_precision.py',
        'nested_anchor_inputs_v2.py', 'residual_graph_experiment.py',
        'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    inputs = [smoke_path, unit_path, HERE / 'STORM_ADAPTATION_V1.md',
        HERE / 'meta_loss_smoke_v1.py', HERE / 'meta_loss_unit_attempt2.log', Path(__file__),
        HERE / 'run_meta_loss_queue_v1.py', HERE / 'storm_author_tree_v1.json',
        HERE / 'storm_author_commit_v1.json', HERE / 'STORM_AUTHOR_SOURCE_V1.json',
        HERE / 'metaloss_paper_v1.pdf', HERE / 'metaloss_paper_v1.txt',
        HERE / 'metaloss_crossref_v1.json', HERE / 'metaloss_publisher_v1.html',
        HERE / 'meta_loss_numeric_diagnosis_v1.log',
        experiment.ROOT / 'results/meta_loss_smoke_v1_20260916/failed_attempt1_source/FAILURE.json']
    for record in base.read_json(HERE / 'STORM_AUTHOR_SOURCE_V1.json')['records']:
        path = Path(record['file'])
        if record['status'] != 'downloaded' or not record['git_blob_verified'] or base.digest(path) != record['sha256']:
            raise ValueError('Pinned author source integrity failed')
        inputs.append(path)
    budget = ('400 task AdamW updates with persistent moments, 400 task-gradient requests, '
        '400 total rescaler-gradient requests and 400 direct-gradient diagnostics; '
        '2400 dropout forwards. Independent inner/meta halves of the same omitted fold. '
        'Same initialization, episode/dropout streams and checkpoint budget in all arms. '
        'Gradient ablations have lower actual backward cost, not equal second-order FLOPs.')
    likelihood = ('Inner mean BCE(anchor_oof + residual) weighted by 2*Omega except uniform; '
        'meta mean BCE after the actual differentiable AdamW step weighted by 2*Omega except '
        'no_outer_rescaling/uniform. no_meta_gradient detaches updated task parameters only '
        'for the meta forward. Detached three-dropout statistics and separate training memories; '
        'no held-out labels or cross-fold reference offsets enter the episode.')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='meta_loss',
        implementation_module='meta_loss_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Primary and three controls fixed before performance; two task learning rates, fixed rescaler lr 0.01, 400 updates and every 20-step checkpoint. No architecture sweep.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='Learning from Noisy Labels via Self-Taught On-the-Fly Meta Loss Rescaling',
            doi='10.1609/aaai.v39i22.34580', venue='AAAI 2025', publisher_date='2025-04-11',
            author_commit='5d579b10c14786d81ab887524b8ae5a97014c628', full_text_obtained=True,
            full_original_reproduction=False),
        mechanism='Self-taught dual-path meta loss rescaling from training-only model statistics',
        required_executor='improve.anchor_meta_loss_experiment_v1',
        training_budget=budget, likelihood=likelihood,
        local_adaptation='Query-masked typed graph residual; same-fold disjoint inner/meta queries; persistent differentiable AdamW; initial 2*Omega=1; mean losses; numerically stable sample variance.',
        evaluation='Historical seed 666 screen, seeds 0/1 confirmation, three fixed development partitions, then frozen outer recheck; stop this version at any failed gate.',
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        pre_performance_numeric_failure_preserved=True, prior_failures_preserved=True,
        existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'meta_loss', experiment.ROOT / 'results/nested_anchor_v2_20260915/historical', core_out)
    # Finish this new, still unused protocol before sealing the execution queue.
    historical = base.read_json(out / 'protocol.json')
    historical.update(required_executor=p['required_executor'], training_budget=budget,
        likelihood=likelihood, local_adaptation=p['local_adaptation'])
    base.write_json(out / 'protocol.json', historical)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path, unit_path,
                     HERE / 'STORM_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='meta_loss',
        source_sha256=base.digest(HERE / 'run_meta_loss_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 8, 'confirm_if_eligible': 8},
        expected_identity_records={'screen': 8, 'confirm_if_eligible': 8},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False,
        confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('META_LOSS_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
