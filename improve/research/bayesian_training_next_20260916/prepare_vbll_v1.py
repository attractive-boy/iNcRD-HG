"""Seal the Bayesian-weight hypothesis after training-only implementation checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_variational_lastlayer_experiment_v1 as experiment
from improve import variational_lastlayer_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    family = 'variational_lastlayer'
    root = experiment.ROOT
    core_out = root/'results/variational_lastlayer_core_v1_20260916'
    out = root/'results/anchor_priority_variational_lastlayer_v1_20260916'
    queue_path = HERE/'VBLL_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve completed or partial studies')
    smoke_path, unit_path = HERE/'VBLL_SMOKE_V1.json', HERE/'VBLL_EXECUTOR_UNIT_V1.json'
    smoke, unit = base.check_seal(base.read_json(smoke_path)), base.check_seal(base.read_json(unit_path))
    if (not unit['passed'] or unit['tests_run'] != 3 or smoke['formal_fits'] or smoke['performance_metrics_computed']
            or smoke['new_short_neural_fits'] != 3 or smoke['identity_replays'] != 3
            or len(smoke['fold_audits']) != 40 or smoke['core_parameters'] != core.parameters()):
        raise ValueError('Implementation checks incomplete')
    for path,digest in {**unit['source_sha256'],**smoke['source_sha256'],**smoke['input_sha256']}.items():
        if base.digest(path) != digest:
            raise ValueError('Checked source or input changed')
    for row in smoke['fold_audits']:
        if not row['identity'] and (row['forbidden_overlap'] or not row['query_labels_invariant']):
            raise ValueError('Support boundary audit failed')
    names = tuple(dict.fromkeys(experiment.OWN+('overlap_block_experiment.py','non_tabular_experiment.py',
                                               'tabular_foundation.py','consolidate_declared_history_v1.py')))
    inputs = [Path(__file__),smoke_path,unit_path,HERE/'vbll_core_unit_v1.log',HERE/'vbll_smoke_v1.py',
              HERE/'VBLL_ADAPTATION_V1.md',HERE/'run_vbll_queue_v1.py',
              HERE/'BAYESIAN_AND_SAMPLING_DISCOVERY_V1.json',HERE/'BAYESIAN_AND_SAMPLING_FULLTEXT_V1.json',
              HERE/'BAYESIAN_AND_SAMPLING_AUTHOR_DISCOVERY_V1.json',HERE/'BAYESIAN_AND_SAMPLING_AUTHOR_SOURCE_V2.json',
              HERE/'vbll_paper_v1.pdf',HERE/'vbll_paper_v1.txt']
    inputs += sorted(HERE.glob('vbll_author_*.py'))
    inputs += [HERE/'vbll_author_README.md']
    author_manifest = base.read_json(HERE/'BAYESIAN_AND_SAMPLING_AUTHOR_SOURCE_V2.json')
    for row in author_manifest['records']:
        if row['name'] == 'vbll':
            path = root.parent/row['file']
            if (row['commit'] != '142f21edd606c1584a15b003bbfb732b4ba2affc'
                    or not row['git_blob_verified'] or base.digest(path) != row['sha256']):
                raise ValueError('Pinned author evidence changed')
    if base.digest(HERE/'vbll_paper_v1.pdf') != '07e3574d2ce591d2eefd1d38fa45073f66030942743f69638714ed7d841ca5e6':
        raise ValueError('Pinned full text changed')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family=family,
        implementation_module='variational_lastlayer_graph_v1',tasks=['MiDrug','LncDrug'],
        kinds=list(core.KINDS),primary_kind='full',primary_controls=['diagonal','map'],
        core_parameters=core.parameters(),lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Every 20 updates select each run checkpoint by mean global ROC AUC and sklearn AP of reference + 0.1 residual. Select one shared learning rate per arm by MiDrug gain subject to LncDrug tolerance, then mean MiDrug gain, then smaller learning rate. Only 0.001 and 0.003 are searched.',
        literature=dict(title='Variational Bayesian Last Layers',arxiv_fulltext_read='2404.11599v1',
            author_claimed_venue='ICLR 2024',publisher_status_verified=False,
            author_code_commit='142f21edd606c1584a15b003bbfb732b4ba2affc',
            author_code_read=True,author_code_executed=False),
        local_adaptation='Two independent class Gaussian weight posteriors on jointly learned graph features, bias augmentation, zero means and diagonal covariance initialization, zero logit noise, nested reference likelihood offset, analytic Gaussian KL/N, full vs diagonal vs MAP mean-prior control. Fixed predictive support aggregation and Gauss-Hermite quadrature replace author Monte Carlo prediction. LncDrug is exact identity.',
        mechanism_family='correlated_variational_classifier_weights',
        no_new_performance_observed=True,smoke=False,original_test_source_access=False,
        evaluation='Screen, three initializations, three frozen development partitions and frozen outer recheck; all gates mandatory.',
        source_sha256={name:base.digest(root/name) for name in names},
        inputs={str(path.resolve()):base.digest(path) for path in inputs},
        confirmed_innovations=0,existing_default_replaced=False)
    core_out.mkdir(parents=True)
    (core_out/'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(root/name,core_out/'source_snapshot'/name)
    base.write_json(core_out/'protocol.json',p)
    experiment.prepare(out,family,root/'results/nested_anchor_v2_20260915/historical',core_out)
    frozen = [core_out/'protocol.json',out/'protocol.json',smoke_path,HERE/'VBLL_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family=family,
        source_sha256=base.digest(HERE/'run_vbll_queue_v1.py'),
        input_sha256={str(path):base.digest(path) for path in frozen},device='cuda:1',
        stages=['screen','confirm'],stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen':6,'confirm_if_eligible':6},
        expected_identity_records={'screen':6,'confirm_if_eligible':6},adopted_neural_fits=0,
        identity_records_count_as_neural_fits=False,confirmed_final_innovations=0))
    base.write_json(queue_path,queue)
    print('VBLL_FROZEN',base.digest(out/'protocol.json'),queue['sha256'],flush=True)


if __name__ == '__main__':
    main()
