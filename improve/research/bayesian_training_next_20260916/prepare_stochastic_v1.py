"""Seal score-based augmentation only after numerical and training-only audits."""
from datetime import datetime,timezone
from pathlib import Path
import shutil

from improve import anchor_stochastic_augmentation_v1 as experiment
from improve import stochastic_augmentation_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    root = experiment.ROOT
    family = 'stochastic_augmentation'
    core_out = root/'results/stochastic_augmentation_core_v1_20260916'
    out = root/'results/anchor_priority_stochastic_augmentation_v1_20260916'
    queue_path = HERE/'STOCHASTIC_AUGMENTATION_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve study artifacts')
    smoke_path,unit_path = HERE/'STOCHASTIC_AUGMENTATION_SMOKE_V1.json',HERE/'STOCHASTIC_AUGMENTATION_UNIT_V1.json'
    smoke,unit = base.check_seal(base.read_json(smoke_path)),base.check_seal(base.read_json(unit_path))
    if (not unit['passed'] or unit['tests_run'] != 4 or smoke['new_short_neural_fits'] != 3 or smoke['identity_replays'] != 3
            or smoke['performance_metrics_computed'] or smoke['formal_fits'] or len(smoke['fold_audits']) != 40
            or smoke['core_parameters'] != core.parameters()):
        raise ValueError('Implementation checks incomplete')
    for path,digest in {**unit['source_sha256'],**smoke['source_sha256'],**smoke['input_sha256']}.items():
        if base.digest(path) != digest:
            raise ValueError('Checked source or input changed')
    for row in smoke['fold_audits']:
        if not row['identity'] and (row['forbidden_overlap'] or not row['query_labels_invariant']):
            raise ValueError('Support boundary failure')
    inputs = [Path(__file__),smoke_path,unit_path,HERE/'stochastic_smoke_v1.py',HERE/'run_stochastic_queue_v1.py',
              HERE/'SCONE_ADAPTATION_V1.md',HERE/'STOCHASTIC_EXECUTOR_DERIVATION_V1.json',
              HERE/'SCONE_METHOD_SOURCE_V1.json',HERE/'SCONE_METHOD_SOURCE_V2.json',
              HERE/'scone_crossref_verification_v1.json',HERE/'scone_paper_v1.pdf',HERE/'scone_paper_v1.txt',
              HERE/'BAYESIAN_AND_SAMPLING_AUTHOR_SOURCE_V2.json',HERE/'scone_author_model_graph_SCONE.py',HERE/'scone_author_README.md']
    for record in base.read_json(HERE/'SCONE_METHOD_SOURCE_V2.json'):
        path = Path(record['file'])
        if base.digest(path) != record['sha256']:
            raise ValueError('Pinned method source changed')
        if 'source' in record and (not record['git_blob_verified'] or record['commit'] != '37292a56d58d580cbcf73de4aab5b9b5572544fd'):
            raise ValueError('Method source provenance mismatch')
        inputs.append(path)
    for record in base.read_json(HERE/'BAYESIAN_AND_SAMPLING_AUTHOR_SOURCE_V2.json')['records']:
        if record['name'] == 'scone':
            if (not record['git_blob_verified'] or record['commit'] != '37292a56d58d580cbcf73de4aab5b9b5572544fd'
                    or base.digest(root.parent/record['file']) != record['sha256']):
                raise ValueError('Pinned primary author source changed')
    if base.digest(HERE/'scone_paper_v1.pdf') != '27e1a51ba0aef8c2f18d2d6986632b6eda8aedc22c2fbebedab54e4404438d15':
        raise ValueError('Pinned full text changed')
    names = tuple(dict.fromkeys(experiment.OWN+('tabular_foundation.py','consolidate_declared_history_v1.py')))
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family=family,
        implementation_module='stochastic_augmentation_graph_v1',tasks=['MiDrug','LncDrug'],
        kinds=list(core.KINDS),primary_kind='full',primary_controls=['no_injection','no_reverse_drift'],
        core_parameters=core.parameters(),lnc_policy='identity_strong_reference',identity_records_are_not_neural_fits=True,
        selection='Every 20 encoder updates select checkpoint by mean global ROC AUC and sklearn AP of reference + 0.1 residual. Select one shared learning rate per arm among 0.001 and 0.003 subject to MiDrug improvement and LncDrug tolerance, then mean MiDrug gain, then smaller learning rate.',
        literature=dict(title='SCONE: A Novel Stochastic Sampling to Generate Contrastive Views and Hard Negative Samples for Recommendation',
            doi='10.1145/3701551.3703522',venue='WSDM 2025',publisher_status_verified_by='Crossref exact title/DOI/authors/proceedings',
            arxiv_fulltext_read='2405.00287v2',author_code_commit='37292a56d58d580cbcf73de4aab5b9b5572544fd',author_code_read=True,author_code_executed=False),
        mechanism_family='score_based_generative_training_augmentation',
        local_adaptation='Typed graph encoder with nested reference likelihood, actual same-RNA positive/zero-labelled training triplets, auxiliary weights 0.1 ranking/0.05 symmetric InfoNCE, noise-prediction score parameterization, paper variance-increment stochastic sampler with noise and injection at all ten steps; not an exact author-code reproduction. Generator updates use detached features; frozen generator parameters permit input gradients for encoder training.',
        source_sha256={name:base.digest(root/name) for name in names},
        inputs={str(path.resolve()):base.digest(path) for path in inputs},
        no_new_performance_observed=True,smoke=False,original_test_source_access=False,
        evaluation='Screen, three initializations, fixed development partitions, frozen outer recheck; all gates mandatory.',
        confirmed_innovations=0,existing_default_replaced=False)
    core_out.mkdir(parents=True)
    (core_out/'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(root/name,core_out/'source_snapshot'/name)
    base.write_json(core_out/'protocol.json',p)
    experiment.prepare(out,family,root/'results/nested_anchor_v2_20260915/historical',core_out)
    frozen = [core_out/'protocol.json',out/'protocol.json',smoke_path,HERE/'SCONE_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family=family,
        source_sha256=base.digest(HERE/'run_stochastic_queue_v1.py'),input_sha256={str(path):base.digest(path) for path in frozen},
        device='cuda:1',stages=['screen','confirm'],stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen':6,'confirm_if_eligible':6},expected_identity_records={'screen':6,'confirm_if_eligible':6},
        adopted_neural_fits=0,identity_records_count_as_neural_fits=False,confirmed_final_innovations=0))
    base.write_json(queue_path,queue)
    print('STOCHASTIC_AUGMENTATION_FROZEN',base.digest(out/'protocol.json'),queue['sha256'],flush=True)


if __name__ == '__main__':
    main()
