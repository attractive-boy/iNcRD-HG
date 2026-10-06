"""Register the narrower family revision before fitting its new control."""
from datetime import datetime,timezone
from pathlib import Path
import shutil

from improve import anchor_variational_feature_experiment_v1 as experiment
from improve import variational_feature_learning_v1 as core
from improve import adopt_variational_feature_v1 as adoption
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    root = experiment.ROOT
    core_out = root/'results/variational_feature_learning_core_v1_20260916'
    out = root/'results/anchor_priority_variational_feature_learning_v1_20260916'
    queue_path = HERE/'VARIATIONAL_FEATURE_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing study artifacts')
    smoke_path,unit_path = HERE/'VARIATIONAL_FEATURE_SMOKE_V1.json',HERE/'VARIATIONAL_FEATURE_UNIT_V1.json'
    smoke,unit = base.check_seal(base.read_json(smoke_path)),base.check_seal(base.read_json(unit_path))
    if (not unit['passed'] or unit['tests_run'] != 4 or smoke['new_short_neural_fits'] != 1
            or smoke['identity_replays'] != 3 or smoke['performance_metrics_computed'] or smoke['formal_fits']
            or smoke['new_fold_audits'] or len(smoke['reused_fold_audits']) != 40):
        raise ValueError('Implementation checks incomplete')
    for path,digest in {**unit['source_sha256'],**smoke['source_sha256'],**smoke['input_sha256']}.items():
        if base.digest(path) != digest:
            raise ValueError('Checked source or input changed')
    parent_core = root/'results/variational_lastlayer_core_v1_20260916'
    parent = root/'results/anchor_priority_variational_lastlayer_v1_20260916'
    p = dict(experiment.common.protocol(parent_core))
    experiment.common.protocol(parent)
    inputs = [Path(__file__),smoke_path,unit_path,HERE/'variational_feature_smoke_v1.py',
              HERE/'VARIATIONAL_FEATURE_ADAPTATION_V1.md',HERE/'run_variational_feature_queue_v1.py',
              HERE/'FEATURE_EXECUTOR_DERIVATION_V1.json',parent/'protocol.json',parent/'SCREEN_ASSESSMENT.json',
              HERE/'VBLL_SCREEN_CONSOLIDATION_V1.json']
    p.update(created_at_utc=datetime.now(timezone.utc).isoformat(),family='variational_feature_learning',
        implementation_module='variational_feature_learning_v1',kinds=list(core.KINDS),primary_kind='full',
        primary_controls=['no_feature_variance','map'],core_parameters=core.parameters(),
        parent_screening_results_seen=True,no_new_performance_observed=False,independent_screen=False,
        mechanism_family='variational_classifier_weight_learning',
        local_adaptation='Diagonal VBLL and MAP retain the exact previous numerical path. Only the new control detaches features in the marginal-variance term, retaining the mean path and head gradients. It is a partial-gradient ablation, not a proper alternative ELBO. No dense-covariance benefit is claimed.',
        baseline_adoption=dict(parent=str(parent),mapping={'full':'diagonal','map':'map'},
            expected_adopted_neural_fits=4,expected_adopted_identity_records=4,parent_outcomes_seen=True,
            required_equal_inputs=True,required_exact_prediction_replay=True),
        source_sha256={**p['source_sha256'],**{name:base.digest(root/name) for name in experiment.OWN}},
        inputs={**p['inputs'],**{str(path):base.digest(path) for path in inputs}})
    core_out.mkdir(parents=True)
    (core_out/'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(root/name,core_out/'source_snapshot'/name)
    base.write_json(core_out/'protocol.json',p)
    experiment.prepare(out,'variational_feature_learning',root/'results/nested_anchor_v2_20260915/historical',core_out)
    adoption.adopt(out)
    sources = [core_out/'protocol.json',out/'protocol.json',out/'PARENT_ADOPTION_V1.json',
               smoke_path,HERE/'VARIATIONAL_FEATURE_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family='variational_feature_learning',
        source_sha256=base.digest(HERE/'run_variational_feature_queue_v1.py'),
        input_sha256={str(path):base.digest(path) for path in sources},device='cuda:1',
        stages=['screen','confirm'],stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen':2,'confirm_if_eligible':6},adopted_neural_fits=4,
        expected_identity_records={'screen':6,'confirm_if_eligible':6},adopted_identity_records=4,
        identity_records_count_as_neural_fits=False,independent_screen=False,confirmed_final_innovations=0))
    base.write_json(queue_path,queue)
    print('VARIATIONAL_FEATURE_FROZEN',base.digest(out/'protocol.json'),queue['sha256'],flush=True)


if __name__ == '__main__':
    main()
