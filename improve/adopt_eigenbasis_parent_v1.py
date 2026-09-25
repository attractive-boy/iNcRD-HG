"""Byte-preserving adoption of already completed SOAP and AdamW screening fits."""
import gc
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import torch

from improve import anchor_eigenbasis_experiment_v1 as experiment

base = experiment.common.base


def adopt(out):
    out = Path(out).resolve()
    manifest_path = out / 'PARENT_ADOPTION_V1.json'
    if manifest_path.exists():
        raise ValueError('Preserve completed adoption manifest')
    p = experiment.common.protocol(out)
    parent = Path(p['baseline_adoption']['parent'])
    old = experiment.common.protocol(parent)
    for key in ('epochs', 'check_interval', 'lrs', 'weight_decay', 'strong_weight',
                'gnn_weight', 'screen_seed', 'lnc_policy', 'core_parameters', 'optimizer_parameters'):
        if p[key] != old[key]:
            raise ValueError('Parent recipe is not exactly reusable: '+key)
    ph, old_ph = base.digest(out/'protocol.json'), base.digest(parent/'protocol.json')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    for task in p['tasks']:
        a = experiment.common.load(out / 'inputs' / task / 'arrays.npz')
        before = experiment.common.load(parent / 'inputs' / task / 'arrays.npz')
        if set(a) != set(before):
            raise ValueError('Input keys changed')
        for key in a:
            np.testing.assert_array_equal(a[key], before[key])
        training = {k: v for k, v in a.items() if k in experiment.TRAIN_KEYS}
        query = torch.as_tensor(a['pairs_val']+a['pair_offset'][:, None], device='cuda:1')
        for kind, old_kind in (('full', 'soap'), ('adamw', 'adamw')):
            for lr in p['lrs']:
                seed = p['screen_seed']
                relative = Path(task) / old_kind / ('lr'+str(lr)) / ('s'+str(seed))
                source = parent / 'runs' / relative
                previous = experiment.common.verify(source, old_ph)
                dest = out / 'runs' / task / kind / ('lr'+str(lr)) / ('s'+str(seed))
                if dest.exists():
                    raise ValueError('Preserve any existing adoption artifacts')
                model, xs, graphs, _ = experiment.setup(training, p, kind, 'cuda:1')
                model.load_state_dict(torch.load(source/'model.pt', map_location='cuda:1', weights_only=True))
                prediction = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
                old_scores = experiment.common.load(source/'scores.npz')
                np.testing.assert_array_equal(prediction, old_scores['standalone'])
                regenerated = experiment.score_arrays(a, prediction, seed, p['strong_weight'])
                for key in regenerated:
                    np.testing.assert_array_equal(regenerated[key], old_scores[key])
                dest.mkdir(parents=True)
                for name, digest in previous['files'].items():
                    shutil.copy2(source/name, dest/name)
                    if base.digest(dest/name) != digest:
                        raise ValueError('Copied source artifact changed')
                result = dict(previous, kind=kind, protocol_sha256=ph,
                    optimizer_kind=kind, effective_parent_optimizer_kind=old_kind,
                    adopted_without_new_training=True,
                    adopted_from=str(source/'result.json'), adopted_from_sha256=base.digest(source/'result.json'),
                    adoption_input_arrays_identical=True, adoption_prediction_max_abs=0.,
                    source_trace_and_weights_preserved=True)
                base.write_json(dest/'result.json', result)
                experiment.common.verify(dest, ph)
                records.append(dict(task=task, kind=kind, old_kind=old_kind, lr=lr, seed=seed,
                    source=str(source/'result.json'), source_sha256=base.digest(source/'result.json'),
                    target=str(dest/'result.json'), target_sha256=base.digest(dest/'result.json'),
                    neural_model=previous['trained_neural_model'], replay_max_abs=0.))
                del model, xs, graphs
                gc.collect(); torch.cuda.empty_cache()
    record = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(__file__), protocol_sha256=ph, parent_protocol_sha256=old_ph,
        records=records, new_neural_fits=0, adopted_neural_fits=4,
        adopted_identity_scoring_records=4, parent_results_seen_before_revision=True,
        preserves_parent_failure=True, exact_input_and_prediction_replay=True))
    base.write_json(manifest_path, record)
    print('EIGENBASIS_PARENT_ADOPTED', record['sha256'], flush=True)
