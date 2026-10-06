"""Freeze an explicitly observed ranking revision and audit screen-only reuse."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_ranking_experiment_v2 as experiment
from improve import staco_ranking_v2 as core
from improve import adopt_nonproximal_ranking_v1 as adoption
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    family = 'nonproximal_ranking'
    root = experiment.ROOT
    old_core = root / 'results/staco_ranking_core_v1_20260916'
    old = root / 'results/anchor_priority_staco_ranking_v1_20260916'
    parent = experiment.common.protocol(old_core)
    previous = experiment.common.protocol(old)
    selection = base.read_json(old / 'SELECTION.json')
    smoke_path = HERE / 'NONPROXIMAL_RANKING_SMOKE_V1.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    if (smoke['performance_metrics_computed'] or smoke['formal_fits']
            or smoke['new_short_neural_fits'] != 2 or smoke['identity_replays'] != 2
            or len(smoke['fold_audits']) != 40):
        raise ValueError('Invalid smoke boundary')
    if smoke['core_parameters'] != core.parameters():
        raise ValueError('Core parameters changed')
    for path, digest in smoke['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Smoke source changed')
    core_out = root / 'results/nonproximal_ranking_core_v1_20260916'
    out = root / 'results/anchor_priority_nonproximal_ranking_v1_20260916'
    queue_path = HERE / 'NONPROXIMAL_RANKING_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing studies')
    source_records = {}
    for task in previous['tasks']:
        for lr, seed in adoption.cells(previous, selection, 'screen'):
            path = old / 'runs' / task / 'no_proximal' / ('lr'+str(lr)) / ('s'+str(seed)) / 'result.json'
            experiment.common.verify(path.parent, base.digest(old / 'protocol.json'))
            source_records[str(path)] = base.digest(path)
    additions = ('staco_ranking_v2.py', 'anchor_ranking_experiment_v2.py',
        'test_anchor_ranking_experiment_v2.py', 'adopt_nonproximal_ranking_v1.py',
        'test_adopt_nonproximal_ranking_v1.py', 'adopt_cooperative_local_v1.py',
        'anchor_priority_graph_v1.py')
    sources = {**parent['source_sha256'], **{name: base.digest(root / name) for name in additions}}
    input_files = [old_core / 'protocol.json', old / 'protocol.json', old / 'SELECTION.json',
        old / 'SCREEN_ASSESSMENT.json', smoke_path, Path(__file__),
        HERE / 'NONPROXIMAL_RANKING_ADAPTATION_V1.md', HERE / 'STACO_ADAPTATION_V1.md']
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        mechanism_family='ranking_risk', implementation_module='staco_ranking_v2',
        tasks=['MiDrug', 'LncDrug'], kinds=list(core.KINDS), primary_kind='full',
        primary_controls=['bce'], core_parameters=core.parameters(),
        lnc_policy='identity_strong_reference', identity_records_are_not_neural_fits=True,
        selection='Full is the already observed v1 no_proximal arm. New BCE control has the same zero proximal terms and stage averaging. Screen reuse is not independent discovery; learning-rate and stage-checkpoint rules remain unchanged.',
        source_sha256=sources, inputs={**parent['inputs'], **source_records,
            **{str(path): base.digest(path) for path in input_files}},
        literature=parent['literature'],
        ranking_objective='BCE plus 0.1 times double-block two-way tail-risk saddle gradient. Both parameter and threshold proximal terms are identically zero. Independent negative batches, projected positive-row dual state and 50-step stage averages remain. BCE removes only the ranking gradient.',
        mechanism='Nonproximal per-positive primal/dual block ranking with independent negative estimates and averaged stages; one ranking-risk family.',
        proximal_terms_disabled=True, gamma_retained_only_for_source_recipe_comparison=True,
        is_staco2=False, theorem_claimed_for_nonconvex_adamw=False,
        same_innovation_family_as_prior_version=True, revision_of=str(old),
        full_screen_predictions_previously_observed=True,
        full_confirmation_and_partition_predictions_previously_observed=False,
        new_control_scores_observed=False, no_independent_screening_claim=True,
        baseline_adoption=dict(source=str(old), source_kind='no_proximal', target_kind='full',
            cpu_forward_gradient_rng_equivalence_test_passed=True,
            full_training_trace_and_risk_state_equivalence_test_passed=True,
            result_sha256=source_records, expected_screen_neural_adoptions=2,
            expected_confirm_neural_adoptions=0),
        evaluation='Every historical, fixed-partition and outer gate remains mandatory; the earlier failed version stays failed.',
        smoke=False, original_test_source_access=False, confirmed_innovations=0, existing_default_replaced=False)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(root / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, family, root / 'results/nested_anchor_v2_20260915/historical', core_out)
    adoption.adopt(out, 'screen')
    inputs = [core_out / 'protocol.json', out / 'protocol.json', out / 'RANK_NONPROX_ADOPTION_SCREEN_V1.json',
              smoke_path, HERE / 'NONPROXIMAL_RANKING_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        source_sha256=base.digest(HERE / 'run_nonproximal_ranking_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in inputs}, device='cuda:1',
        stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 2, 'confirm_if_eligible': 4},
        expected_adopted_neural_fits={'screen': 2, 'confirm_if_eligible': 0},
        identity_records={'screen': 4, 'confirm_if_eligible': 4},
        adopted_identity_records={'screen': 2, 'confirm_if_eligible': 0},
        same_mechanism_family_as_previous=True, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('NONPROXIMAL_RANKING_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
