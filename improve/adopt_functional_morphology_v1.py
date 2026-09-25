"""Replay and adopt exact parent no-alignment screens without new fitting."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import torch

from improve import anchor_functional_morphology_v1 as experiment
from improve import tabular_foundation as base

ROOT = experiment.ROOT
PARENT = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916'
CHILD = ROOT / 'results/anchor_priority_functional_morphology_v1_20260916'


def compatible(old, new):
    if (old['family'] != 'morphological_alignment' or new['family'] != 'functional_morphology'
            or old['implementation_module'] != 'morphological_alignment_v1'
            or new['implementation_module'] != 'functional_morphology_v1'
            or new['kinds'] != ['full', 'shuffled_rna_targets']):
        raise ValueError('Unexpected source or destination family')
    for key in ('tasks', 'screen_seed', 'confirm_seeds', 'epochs', 'check_interval', 'lrs', 'weight_decay',
                'strong_weight', 'private_rng_offset', 'lnc_policy', 'nested_reference', 'lnc_absolute_tolerance'):
        if old[key] != new[key]:
            raise ValueError('Changed shared recipe: ' + key)
    c1, c2 = dict(old['core_parameters']), dict(new['core_parameters'])
    if c1.pop('alignment_weight') != .1 or c2.pop('alignment_weight') != 0.:
        raise ValueError('Expected the previously zero-multiplied parent arm')
    if c2.pop('parent_core_sha256') != old['source_sha256']['morphological_alignment_v1.py'] or c1 != c2:
        raise ValueError('Changed phenotype encoder/input recipe')
    if old['source_sha256']['morphological_alignment_v1.py'] != new['source_sha256']['morphological_alignment_v1.py']:
        raise ValueError('Parent encoder source differs')


def adopt(parent=PARENT, child=CHILD):
    parent, child = Path(parent).resolve(), Path(child).resolve()
    target = child / 'ADOPTION_AUDIT.json'
    if target.exists():
        raise ValueError('Preserve adoption audit')
    old, new = experiment.common.protocol(parent), experiment.common.protocol(child)
    compatible(old, new)
    ph, ch = base.digest(parent / 'protocol.json'), base.digest(child / 'protocol.json')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    for task in new['tasks']:
        source_array = parent / 'inputs' / task / 'arrays.npz'
        target_array = child / 'inputs' / task / 'arrays.npz'
        a1, a2 = experiment.common.load(source_array), experiment.common.load(target_array)
        if set(a1) != set(a2):
            raise ValueError('Changed input keys')
        for key in a1:
            np.testing.assert_array_equal(a1[key], a2[key])
        for lr in new['lrs']:
            src = parent / 'runs' / task / 'no_alignment' / ('lr' + str(lr)) / ('s' + str(new['screen_seed']))
            dest = child / 'runs' / task / 'full' / ('lr' + str(lr)) / ('s' + str(new['screen_seed']))
            if dest.exists():
                raise ValueError('Preserve destination record')
            previous = experiment.common.verify(src, ph)
            if previous.get('adopted_without_new_training', False):
                raise ValueError('Expected directly trained/scored parent record')
            state = torch.load(src / 'training_final.pt', map_location='cpu', weights_only=True)
            trace = base.read_json(src / 'trace.json')
            if task == 'MiDrug':
                if (not previous['trained_neural_model'] or previous['training_steps'] != 400
                        or len(trace) != 400 or state['epoch'] != 400
                        or any(row['alignment_multiplier'] != 0. for row in trace)):
                    raise ValueError('Parent is not the exact no-alignment training run')
            elif previous['trained_neural_model'] or trace or state is not None:
                raise ValueError('Invalid parent identity')
            model, xs, graphs, _ = experiment.setup({k: v for k, v in a2.items() if k in experiment.TRAIN_KEYS}, new, 'full', 'cuda:1')
            model.load_state_dict(torch.load(src / 'model.pt', map_location='cuda:1', weights_only=True))
            query = torch.as_tensor(a2['pairs_val'] + a2['pair_offset'][:, None], device='cuda:1')
            value = experiment.predict(model, xs, graphs, query, new).cpu().numpy()
            reverse = experiment.predict(model, xs, graphs, query.flip(1), new).cpu().numpy()[::-1]
            saved = experiment.common.load(src / 'scores.npz')
            for actual in (value, reverse):
                np.testing.assert_allclose(actual, saved['standalone'], atol=3e-5, rtol=3e-5)
            for score, expected in experiment.score_arrays(a2, saved['standalone'], previous['seed'], new['strong_weight']).items():
                np.testing.assert_array_equal(saved[score], expected)
            dest.mkdir(parents=True)
            for filename in ('model.pt', 'training_final.pt', 'trace.json', 'scores.npz'):
                shutil.copy2(src / filename, dest / filename)
                if base.digest(dest / filename) != previous['files'][filename]:
                    raise ValueError('Copied artifact changed')
            result = dict(previous, kind='full', protocol_sha256=ch,
                adopted_without_new_training=True, new_training_steps=0, new_training_seconds=0.,
                adoption_source_result=str(src / 'result.json'), adoption_source_result_sha256=base.digest(src / 'result.json'),
                adoption_parent_protocol_sha256=ph, adoption_input_arrays_identical=True,
                adoption_source_kind='no_alignment', source_trace_and_weights_preserved=True,
                adoption_prediction_max_abs=float(max(abs(value - saved['standalone']).max(), abs(reverse - saved['standalone']).max())))
            base.write_json(dest / 'result.json', result)
            experiment.common.verify(dest, ch)
            records.append(dict(task=task, lr=lr, source=str(src / 'result.json'), target=str(dest / 'result.json'),
                source_sha256=base.digest(src / 'result.json'), target_sha256=base.digest(dest / 'result.json'),
                trained_neural_model=result['trained_neural_model'], new_training_steps=0,
                copied_artifacts_identical=True, replay_max_abs=result['adoption_prediction_max_abs']))
            print('FUNCTIONAL_MORPHOLOGY_ADOPTED', task, lr, result['adoption_prediction_max_abs'], flush=True)
            del model, xs, graphs
            torch.cuda.empty_cache()
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=base.digest(__file__),
        parent_protocol_sha256=ph, child_protocol_sha256=ch, records=records, adopted_neural_fits=2,
        adopted_identity_records=2, new_neural_fits=0, original_test_source_access=False,
        source_observations_motivated_revision=True, confirmed_final_innovations=0))
    base.write_json(target, result)
    return result


if __name__ == '__main__':
    adopt()
