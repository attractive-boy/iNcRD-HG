"""Run the sealed variational-classifier study and stop on any failed gate."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np

from improve import anchor_variational_lastlayer_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
FAMILY = 'variational_lastlayer'


def run_stage(out, stage):
    for task in ('LncDrug', 'MiDrug'):
        with (out/(stage+'_'+task+'.log')).open('a', buffering=1) as stream:
            command = [sys.executable,'-u','-m','improve.anchor_variational_lastlayer_experiment_v1',
                       stage,'--family',FAMILY,'--out',str(out),'--task',task,'--device','cuda:1']
            result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Variational worker failed: '+str((stage,task,result.returncode)))


def classify(out, stage):
    p = experiment.common.protocol(out)
    ph = base.digest(out/'protocol.json')
    expected = {(task,kind,lr,p['screen_seed']) for task in p['tasks']
                for kind in p['kinds'] for lr in p['lrs']}
    if stage == 'confirm':
        selection = base.read_json(out/'SELECTION.json')
        expected |= {(task,kind,selection['chosen'][kind]['lr'],seed) for task in p['tasks']
                     for kind in p['kinds'] for seed in p['confirm_seeds']}
    found, neural, identity = set(), [], []
    for path in sorted((out/'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent,ph)
        found.add((r['task'],r['kind'],r['lr'],r['seed']))
        if r.get('adopted_without_new_training',False):
            raise ValueError('No model adoption is registered')
        item = dict(path=str(path),sha256=base.digest(path),task=r['task'],kind=r['kind'],lr=r['lr'],seed=r['seed'])
        trace = base.read_json(path.parent/'trace.json')
        scores = experiment.common.load(path.parent/'scores.npz')
        posterior = experiment.common.load(path.parent/'posterior_diagnostics.npz')
        if any(not np.isfinite(value).all() for value in posterior.values()) or (posterior['variance'] < 0).any():
            raise ValueError('Invalid saved predictive posterior')
        np.testing.assert_array_equal(scores['standalone'],posterior['residual'])
        np.testing.assert_array_equal(scores['mean_only_composed'],scores['anchor']+.1*posterior['mean_residual'])
        error = float(abs(posterior['reference_residual']-posterior['residual']).max())
        if error != r['quadrature_reference_max_abs'] or error > p['core_parameters']['quadrature_tolerance']:
            raise ValueError('Predictive integration error changed')
        if r['trained_neural_model']:
            if (r['task'] != 'MiDrug' or r['training_steps'] != 400 or r['gradient_evaluations'] != 400
                    or r['head_weight_decay'] != 0. or r['regularizer_is_gaussian_kl'] != (r['kind'] != 'map')):
                raise ValueError('Wrong variational training budget or objective')
            if len(trace) != 400 or [row['epoch'] for row in trace] != list(range(1,401)):
                raise ValueError('Incomplete training trace')
            if [row['epoch'] for row in trace if 'anchor_composed' in row] != list(range(20,401,20)):
                raise ValueError('Checkpoint-selection budget changed')
            for row in trace:
                if (row['omitted_fold'] != (row['epoch']-1)%5 or not np.isfinite(row['loss'])
                        or abs(row['regularizer_unscaled']/row['total_training_rows']-row['regularizer_per_training_row']) > 1e-12):
                    raise ValueError('Invalid likelihood or regularization trace')
            if r['kind'] == 'full' and not any(row['covariance_offdiagonal_norm'] > 0 for row in trace):
                raise ValueError('Full covariance never trained')
            neural.append(item)
        else:
            if r['task'] != 'LncDrug' or trace or any(r[k] for k in
                    ('epoch','parameters','trainable_parameters','training_steps','gradient_evaluations')):
                raise ValueError('Invalid identity record')
            np.testing.assert_array_equal(scores['standalone'],np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor_composed'],scores['anchor'])
            identity.append(item)
    count = len(p['kinds'])*(len(p['lrs'])+(len(p['confirm_seeds']) if stage == 'confirm' else 0))
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or unexpected fit records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family=FAMILY,stage=stage,
        protocol_sha256=ph,cumulative_new_neural_fits=len(neural),cumulative_identity_scoring_records=len(identity),
        neural=neural,identity=identity,adopted_neural_fits=0,identity_records_count_as_training=False,
        mean_only_diagnostic_has_no_separate_training_or_selection=True,confirmed_final_innovations=0))
    path = HERE/('VBLL_RECORD_CLASSIFICATION_'+stage.upper()+'_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        if any(old[k] != result[k] for k in ('neural','identity')):
            raise ValueError('Existing classification disagrees')
    else:
        base.write_json(path,result)
    return path


def main():
    out = experiment.ROOT/'results/anchor_priority_variational_lastlayer_v1_20260916'
    frozen = base.check_seal(base.read_json(HERE/'VBLL_QUEUE_PROTOCOL_V1.json'))
    if frozen['family'] != FAMILY or frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Frozen queue source changed')
    for path,digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen input changed')
    experiment.common.protocol(out)
    target = HERE/'VBLL_QUEUE_RESULT_V1.json'
    if target.exists():
        raise ValueError('Preserve completed queue result')
    for stage in ('screen','confirm'):
        print('START_VBLL_STAGE',stage,datetime.now(timezone.utc).isoformat(),flush=True)
        run_stage(out,stage)
        classification = classify(out,stage)
        with (out/(stage+'_assessment.log')).open('a',buffering=1) as stream, redirect_stdout(stream):
            if stage == 'screen' and not (out/'SELECTION.json').exists():
                experiment.select(out)
            filename = 'SCREEN_ASSESSMENT.json' if stage == 'screen' else 'HISTORICAL_ASSESSMENT.json'
            if (out/filename).exists():
                experiment.common.all_rows(out)
                result = base.check_seal(base.read_json(out/filename))
            else:
                result = audit.assess(out,stage)
        print('VBLL_GATES',stage,result['performance_gate'],result['primary_mechanism_gate'],result['offset_null_gate'],flush=True)
        if not result['declared_gate']:
            stop = out/('SCREEN_STOP_DECISION.json' if stage == 'screen' else 'HISTORICAL_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop,base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    decision='Stop this frozen version before further fitting',stage=stage,
                    assessment_sha256=base.digest(out/filename),confirmed_innovations=0,default_replaced=False)))
            break
    counts = base.check_seal(base.read_json(classification))
    base.write_json(target,base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family=FAMILY,stage=stage,
        queue_protocol_sha256=frozen['sha256'],historical_gate=result['declared_gate'],
        assessment_sha256=base.digest(out/filename),classification_sha256=base.digest(classification),
        new_neural_fits=counts['cumulative_new_neural_fits'],identity_scoring_records=counts['cumulative_identity_scoring_records'],
        adopted_neural_fits=0,partition_verification_pending=bool(result['declared_gate']),
        confirmed_final_innovations=0,existing_default_replaced=False)))
    print('VBLL_QUEUE_COMPLETE',stage,result['declared_gate'],flush=True)


if __name__ == '__main__':
    main()
