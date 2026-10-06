"""Run the sealed variational-classifier study and stop on any failed gate."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_stochastic_augmentation_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
FAMILY = 'stochastic_augmentation'


def run_stage(out, stage):
    for task in ('LncDrug', 'MiDrug'):
        with (out/(stage+'_'+task+'.log')).open('a', buffering=1) as stream:
            command = [sys.executable,'-u','-m','improve.anchor_stochastic_augmentation_v1',
                       stage,'--family',FAMILY,'--out',str(out),'--task',task,'--device','cuda:1']
            result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Variational worker failed: '+str((stage,task,result.returncode)))


def classify(out,stage):
    p = experiment.common.protocol(out)
    ph = base.digest(out/'protocol.json')
    expected = {(task,kind,lr,p['screen_seed']) for task in p['tasks'] for kind in p['kinds'] for lr in p['lrs']}
    if stage == 'confirm':
        selection = base.read_json(out/'SELECTION.json')
        expected |= {(task,kind,selection['chosen'][kind]['lr'],seed) for task in p['tasks']
                     for kind in p['kinds'] for seed in p['confirm_seeds']}
    found,neural,identity = set(),[],[]
    for path in sorted((out/'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent,ph)
        found.add((r['task'],r['kind'],r['lr'],r['seed']))
        if r.get('adopted_without_new_training',False):
            raise ValueError('No model adoption is registered')
        item = dict(path=str(path),sha256=base.digest(path),task=r['task'],kind=r['kind'],lr=r['lr'],seed=r['seed'])
        trace = base.read_json(path.parent/'trace.json')
        state = torch.load(path.parent/'training_final.pt',map_location='cpu',weights_only=True)
        if r['trained_neural_model']:
            if (r['task'] != 'MiDrug' or r['training_steps'] != 400 or r['gradient_evaluations'] != 800
                    or r['encoder_optimizer_steps'] != 400 or r['score_optimizer_steps'] != 400):
                raise ValueError('Wrong two-optimizer training budget')
            if len(trace) != 400 or [row['epoch'] for row in trace] != list(range(1,401)):
                raise ValueError('Incomplete per-step trace')
            if [row['epoch'] for row in trace if 'anchor_composed' in row] != list(range(20,401,20)):
                raise ValueError('Changed checkpoint-selection budget')
            for row in trace:
                step = row['epoch']
                if (row['omitted_fold'] != (step-1)%5 or row['gradient_evaluations'] != 2*step
                        or row['encoder_optimizer_steps'] != step or row['score_optimizer_steps'] != step
                        or not row['same_rna_triplets'] or row['triplet_count'] != p['core_parameters']['triplet_batch']):
                    raise ValueError('Inconsistent stochastic training trace')
                if any(not np.isfinite(row[key]) for key in ('loss','score_loss','ranking','contrastive','bce')):
                    raise ValueError('Nonfinite training objective')
            for group in ('encoder_optimizer','score_optimizer'):
                if not state[group]['state']:
                    raise ValueError('Empty optimizer state')
                for item_state in state[group]['state'].values():
                    if int(item_state['step']) != 400:
                        raise ValueError('Saved optimizer budget changed')
                    if any(not torch.isfinite(value).all() for value in item_state.values() if isinstance(value,torch.Tensor)):
                        raise ValueError('Saved optimizer is nonfinite')
            neural.append(item)
        else:
            if r['task'] != 'LncDrug' or state is not None or trace or any(r[k] for k in
                    ('epoch','parameters','trainable_parameters','training_steps','gradient_evaluations','encoder_optimizer_steps','score_optimizer_steps')):
                raise ValueError('Invalid identity record')
            scores = experiment.common.load(path.parent/'scores.npz')
            np.testing.assert_array_equal(scores['standalone'],np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor_composed'],scores['anchor'])
            identity.append(item)
    count = len(p['kinds'])*(len(p['lrs'])+(len(p['confirm_seeds']) if stage == 'confirm' else 0))
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or unexpected records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family=FAMILY,stage=stage,
        protocol_sha256=ph,cumulative_new_neural_fits=len(neural),cumulative_identity_scoring_records=len(identity),
        neural=neural,identity=identity,adopted_neural_fits=0,identity_records_count_as_training=False,
        encoder_updates=400,score_updates=400,total_backward_evaluations=800,confirmed_final_innovations=0))
    path = HERE/('STOCHASTIC_AUGMENTATION_RECORD_CLASSIFICATION_'+stage.upper()+'_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        if any(old[k] != result[k] for k in ('neural','identity')):
            raise ValueError('Existing classification disagrees')
    else:
        base.write_json(path,result)
    return path


def main():
    out = experiment.ROOT/'results/anchor_priority_stochastic_augmentation_v1_20260916'
    frozen = base.check_seal(base.read_json(HERE/'STOCHASTIC_AUGMENTATION_QUEUE_PROTOCOL_V1.json'))
    if frozen['family'] != FAMILY or frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Frozen queue source changed')
    for path,digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen input changed')
    experiment.common.protocol(out)
    target = HERE/'STOCHASTIC_AUGMENTATION_QUEUE_RESULT_V1.json'
    if target.exists():
        raise ValueError('Preserve completed queue result')
    for stage in ('screen','confirm'):
        print('START_STOCHASTIC_AUGMENTATION_STAGE',stage,datetime.now(timezone.utc).isoformat(),flush=True)
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
        print('STOCHASTIC_AUGMENTATION_GATES',stage,result['performance_gate'],result['primary_mechanism_gate'],result['offset_null_gate'],flush=True)
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
    print('STOCHASTIC_AUGMENTATION_QUEUE_COMPLETE',stage,result['declared_gate'],flush=True)


if __name__ == '__main__':
    main()
