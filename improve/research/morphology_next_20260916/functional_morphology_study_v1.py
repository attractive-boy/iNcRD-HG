"""Complete the pending morphology revision, preserving the failed parent."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_functional_morphology_v1 as experiment
from improve import functional_morphology_v1 as core
from improve import adopt_functional_morphology_v1 as adoption
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base
from improve.research.morphology_next_20260916 import morphology_study_v1 as parent

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT
FAMILY = 'functional_morphology'
STUDY = adoption.CHILD
CORE = ROOT / 'results/functional_morphology_core_v1_20260916'


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze():
    if CORE.exists() or STUDY.exists():
        raise ValueError('Preserve existing study')
    previous = experiment.common.protocol(adoption.PARENT)
    failed = base.check_seal(base.read_json(adoption.PARENT / 'SCREEN_ASSESSMENT.json'))
    if failed['declared_gate'] or failed['primary_mechanism_gate']:
        raise ValueError('Unexpected parent assessment')
    unit = base.check_seal(base.read_json(HERE / 'FUNCTIONAL_MORPHOLOGY_UNIT_V1.json'))
    if not unit['passed'] or unit['tests_run'] != 3:
        raise ValueError('Revision checks incomplete')
    for path, digest in unit['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Checked implementation changed')
    short = parent.verified(HERE / 'MORPHOLOGY_SMOKE_V1.json')
    if len(short['fold_audits']) != 40 or short['performance_metrics_computed']:
        raise ValueError('Parent input-isolation audit missing')
    c = core.parameters()
    core.load_inputs(c)
    source = experiment.common.protocol(parent.CORE)
    names = tuple(dict.fromkeys(experiment.OWN + ('tabular_foundation.py',)))
    evidence = [Path(__file__), HERE / 'FUNCTIONAL_MORPHOLOGY_ADAPTATION_V1.md',
                HERE / 'FUNCTIONAL_MORPHOLOGY_UNIT_V1.json', HERE / 'MORPHOLOGY_SMOKE_V1.json',
                parent.CORE / 'protocol.json', adoption.PARENT / 'protocol.json',
                adoption.PARENT / 'SCREEN_ASSESSMENT.json']
    p = dict(source, created_at_utc=now(), family=FAMILY,
        implementation_module='functional_morphology_v1', kinds=list(core.KINDS),
        primary_kind='full', primary_controls=['shuffled_rna_targets'], core_parameters=c,
        no_new_performance_observed=False, parent_no_alignment_results_observed=True,
        mechanism_family='functional_target_mediated_morphological_alignment',
        local_adaptation='Adaptive narrowing to functional target identity without DTI alignment; real versus globally permuted target identities with matched target counts and phenotype coverage. Full exactly adopts parent no_alignment screens. Missing coverage and LncDrug give exact reference identity.',
        source_sha256={name: base.digest(ROOT / name) for name in names},
        inputs={**source['inputs'], **{str(path): base.digest(path) for path in evidence}},
        confirmed_innovations=0, existing_default_replaced=False)
    CORE.mkdir(parents=True)
    (CORE / 'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(ROOT / name, CORE / 'source_snapshot' / name)
    base.write_json(CORE / 'protocol.json', p)
    experiment.prepare(STUDY, FAMILY, previous['nested_reference'], CORE)
    adoption.compatible(previous, experiment.common.protocol(STUDY))
    base.write_json(HERE / 'FUNCTIONAL_MORPHOLOGY_QUEUE_PROTOCOL_V1.json', base.seal(dict(
        created_at_utc=now(), source_sha256=base.digest(__file__),
        protocol_sha256=base.digest(STUDY / 'protocol.json'),
        expected_new_neural_fits={'screen': 2, 'confirm_if_eligible': 4},
        adopted_neural_fits=2, adopted_identity_records=2,
        stop_on_any_failed_gate=True, confirmed_final_innovations=0)))
    print('FUNCTIONAL_MORPHOLOGY_FROZEN', flush=True)


def classify(stage):
    p = experiment.common.protocol(STUDY)
    ph = base.digest(STUDY / 'protocol.json')
    expected = {(task, kind, lr, p['screen_seed']) for task in p['tasks']
                for kind in p['kinds'] for lr in p['lrs']}
    if stage == 'confirm':
        selection = base.read_json(STUDY / 'SELECTION.json')
        expected |= {(task, kind, selection['chosen'][kind]['lr'], seed)
            for task in p['tasks'] for kind in p['kinds'] for seed in p['confirm_seeds']}
    counts = dict(new_neural_fits=0, adopted_neural_fits=0,
                  new_identity_records=0, adopted_identity_records=0)
    found, files = set(), {}
    for path in sorted((STUDY / 'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent, ph)
        found.add((r['task'], r['kind'], r['lr'], r['seed']))
        state = torch.load(path.parent / 'training_final.pt', map_location='cpu', weights_only=True)
        trace = base.read_json(path.parent / 'trace.json')
        adopted = r.get('adopted_without_new_training', False)
        if adopted:
            if r['kind'] != 'full' or r['seed'] != p['screen_seed'] or r['new_training_steps'] != 0:
                raise ValueError('Invalid adopted run')
            if base.digest(r['adoption_source_result']) != r['adoption_source_result_sha256']:
                raise ValueError('Adoption source changed')
        if r['trained_neural_model']:
            if r['task'] != 'MiDrug':
                raise ValueError('Unregistered neural task')
            parent.check_record(state, trace, r, p['epochs'], p['core_parameters'], r['kind'])
            counts['adopted_neural_fits' if adopted else 'new_neural_fits'] += 1
        else:
            if r['task'] != 'LncDrug' or state is not None or trace or r['epoch'] or r['training_steps']:
                raise ValueError('Invalid identity record')
            z = experiment.common.load(path.parent / 'scores.npz')
            np.testing.assert_array_equal(z['standalone'], np.zeros_like(z['standalone']))
            np.testing.assert_array_equal(z['anchor_composed'], z['anchor'])
            counts['adopted_identity_records' if adopted else 'new_identity_records'] += 1
        files[str(path)] = base.digest(path)
    if found != expected:
        raise ValueError('Incomplete or extra fitting records')
    if counts['adopted_neural_fits'] != 2 or counts['adopted_identity_records'] != 2:
        raise ValueError('Unexpected reuse count')
    if counts['new_neural_fits'] != (2 if stage == 'screen' else 6):
        raise ValueError('Incorrect fitting count')
    result = base.seal(dict(created_at_utc=now(), stage=stage, **counts,
        result_sha256=files, confirmed_final_innovations=0))
    dest = HERE / ('FUNCTIONAL_MORPHOLOGY_CLASSIFICATION_' + stage.upper() + '_V1.json')
    if dest.exists():
        old = base.check_seal(base.read_json(dest))
        if old['result_sha256'] != files:
            raise ValueError('Classification evidence changed')
        return old
    base.write_json(dest, result)
    return result


def run():
    frozen = base.check_seal(base.read_json(HERE / 'FUNCTIONAL_MORPHOLOGY_QUEUE_PROTOCOL_V1.json'))
    target = HERE / 'FUNCTIONAL_MORPHOLOGY_QUEUE_RESULT_V1.json'
    if target.exists() or frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Completed queue or changed source')
    if base.digest(STUDY / 'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('Study changed')
    if not (STUDY / 'ADOPTION_AUDIT.json').exists():
        adoption.adopt()
    for stage in ('screen', 'confirm'):
        print('START_FUNCTIONAL_MORPHOLOGY', stage, now(), flush=True)
        for task in ('LncDrug', 'MiDrug'):
            with (STUDY / (stage + '_' + task + '.log')).open('a', buffering=1) as stream:
                subprocess.run([sys.executable, '-u', '-m', 'improve.anchor_functional_morphology_v1',
                    stage, '--family', FAMILY, '--out', str(STUDY), '--task', task, '--device', 'cuda:1'],
                    stdout=stream, stderr=subprocess.STDOUT, check=True)
        counts = classify(stage)
        with (STUDY / (stage + '_assessment.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
            if stage == 'screen' and not (STUDY / 'SELECTION.json').exists():
                experiment.select(STUDY)
            filename = 'SCREEN_ASSESSMENT.json' if stage == 'screen' else 'HISTORICAL_ASSESSMENT.json'
            if (STUDY / filename).exists():
                raise ValueError('Preserve existing assessment')
            assessment = audit.assess(STUDY, stage)
        print('FUNCTIONAL_MORPHOLOGY_GATES', stage, assessment['performance_gate'],
              assessment['primary_mechanism_gate'], assessment['offset_null_gate'], flush=True)
        if not assessment['declared_gate']:
            base.write_json(STUDY / (stage.upper() + '_STOP_DECISION.json'), base.seal(dict(
                created_at_utc=now(), stage=stage, decision='Stop this frozen version before further fitting',
                assessment_sha256=base.digest(STUDY / filename), confirmed_innovations=0)))
            break
    base.write_json(target, base.seal(dict(created_at_utc=now(), stage=stage,
        assessment_sha256=base.digest(STUDY / filename), counts=counts,
        historical_gate=assessment['declared_gate'],
        partition_verification_pending=assessment['declared_gate'], confirmed_final_innovations=0)))
    print('FUNCTIONAL_MORPHOLOGY_QUEUE_COMPLETE', stage, assessment['declared_gate'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run'))
    args = parser.parse_args()
    globals()[args.command]()
