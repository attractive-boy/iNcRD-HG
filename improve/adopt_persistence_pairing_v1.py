"""Verify and copy the unchanged no-spectrum models for the pairing hypothesis."""
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil

import numpy as np
import torch

from improve import anchor_spectral_persistence_experiment_v1 as experiment
from improve import tabular_foundation as base


def adopt(out):
    out = Path(out).resolve()
    p = experiment.common.protocol(out)
    parent = experiment.ROOT / 'results/anchor_priority_spectral_persistence_v1_20260916'
    previous = experiment.common.protocol(parent)
    for key in ('epochs', 'check_interval', 'lrs', 'weight_decay', 'strong_weight', 'gnn_weight',
                'screen_seed', 'confirm_seeds', 'lnc_policy', 'core_parameters'):
        if p[key] != previous[key]:
            raise ValueError('Adoption recipe differs: ' + key)
    target = out / 'PARENT_ADOPTION_V1.json'
    if target.exists():
        raise ValueError('Preserve adoption evidence')
    ph, old_ph = base.digest(out / 'protocol.json'), base.digest(parent / 'protocol.json')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pending = []
    for task in p['tasks']:
        a = experiment.common.load(out / 'inputs' / task / 'arrays.npz')
        old_a = experiment.common.load(parent / 'inputs' / task / 'arrays.npz')
        if set(a) != set(old_a):
            raise ValueError('Adoption input keys differ')
        for key in a:
            np.testing.assert_array_equal(a[key], old_a[key])
        training = {key: value for key, value in a.items() if key in experiment.TRAIN_KEYS}
        pairs = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device='cuda:1')
        for lr in p['lrs']:
            source = parent / 'runs' / task / 'no_spectrum' / ('lr' + str(lr)) / 's666'
            dest = out / 'runs' / task / 'full' / ('lr' + str(lr)) / 's666'
            if dest.exists():
                raise ValueError('Preserve complete or partial adopted artifacts')
            r = experiment.common.verify(source, old_ph)
            scores = experiment.common.load(source / 'scores.npz')
            model, xs, graphs, _ = experiment.setup(training, p, 'full', 'cuda:1')
            model.load_state_dict(torch.load(source / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            value = experiment.predict(model, xs, graphs, pairs, p).cpu().numpy()
            np.testing.assert_array_equal(value, scores['standalone'])
            record = dict(r, kind='full', protocol_sha256=ph, effective_parent_kind='no_spectrum',
                adopted_without_new_training=True, new_training_steps=0,
                adoption_source_result=str(source / 'result.json'),
                adoption_source_result_sha256=base.digest(source / 'result.json'),
                adoption_input_arrays_identical=True, adoption_prediction_max_abs=0.,
                source_trace_and_weights_preserved=True)
            pending.append((source, dest, record))
            del model, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
    records = []
    for source, dest, record in pending:
        dest.mkdir(parents=True)
        for name, digest in record['files'].items():
            shutil.copy2(source / name, dest / name)
            assert base.digest(dest / name) == digest
        base.write_json(dest / 'result.json', record)
        records.append(dict(path=str(dest / 'result.json'), sha256=base.digest(dest / 'result.json'),
            task=record['task'], kind='full', lr=record['lr'], trained_neural_model=record['trained_neural_model']))
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=ph,
        parent_protocol_sha256=old_ph, new_neural_fits=0, adopted_neural_fits=2, adopted_identity_records=2,
        records=records, all_inputs_equal=True, all_predictions_bitwise_equal=True,
        previous_outcomes_seen=True, independent_screen=False, confirmed_final_innovations=0))
    base.write_json(target, result)
    print('PERSISTENCE_PAIRING_ADOPTED', result['sha256'], flush=True)
