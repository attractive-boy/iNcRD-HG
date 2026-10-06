"""Complete missing metadata after the adoption writer's final seal lookup failed.

The original sealed code/protocol and all eight existing adoption records are
preserved. This script replays their predictions and only creates the missing
manifest, a recovery record, and the coordinator's previously unwritten seal.
"""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_eigenbasis_experiment_v1 as experiment
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    root = experiment.ROOT
    out = root/'results/anchor_priority_eigenbasis_adam_v1_20260916'
    core_out = root/'results/eigenbasis_adam_core_v1_20260916'
    p = experiment.common.protocol(out)
    experiment.common.protocol(core_out)
    parent = Path(p['baseline_adoption']['parent'])
    experiment.common.protocol(parent)
    manifest = out/'PARENT_ADOPTION_V1.json'
    recovery = HERE/'EIGENBASIS_REGISTRATION_RECOVERY_V1.json'
    queue_path = HERE/'EIGENBASIS_QUEUE_PROTOCOL_V1.json'
    if any(path.exists() for path in (manifest, recovery, queue_path)):
        raise ValueError('Preserve completed or partially recovered metadata')
    failed_log = HERE/'prepare_eigenbasis_v1.log'
    if "AttributeError: module 'improve.non_tabular_experiment' has no attribute 'seal'" not in failed_log.read_text():
        raise ValueError('Unexpected registration failure')
    before = {str(path):base.digest(path) for path in (out/'protocol.json', core_out/'protocol.json')}
    ph, parent_ph = base.digest(out/'protocol.json'), base.digest(parent/'protocol.json')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    expected = {(t, k, lr, 666) for t in p['tasks'] for k in ('full','adamw') for lr in p['lrs']}
    found = set()
    for task in p['tasks']:
        a = experiment.common.load(out/'inputs'/task/'arrays.npz')
        old_a = experiment.common.load(parent/'inputs'/task/'arrays.npz')
        if set(a) != set(old_a):
            raise ValueError('Changed input keys')
        for key in a:
            np.testing.assert_array_equal(a[key], old_a[key])
        training = {k:v for k,v in a.items() if k in experiment.TRAIN_KEYS}
        query = torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None], device='cuda:1')
        for kind, source_kind in (('full','soap'), ('adamw','adamw')):
            for lr in p['lrs']:
                source = parent/'runs'/task/source_kind/('lr'+str(lr))/'s666'
                dest = out/'runs'/task/kind/('lr'+str(lr))/'s666'
                old = experiment.common.verify(source, parent_ph)
                r = experiment.common.verify(dest, ph)
                before[str(dest/'result.json')] = base.digest(dest/'result.json')
                if (not r['adopted_without_new_training'] or r['adopted_from_sha256'] != base.digest(source/'result.json')
                        or r['files'] != old['files']):
                    raise ValueError('Adoption does not preserve source artifacts')
                model, xs, graphs, _ = experiment.setup(training, p, kind, 'cuda:1')
                model.load_state_dict(torch.load(dest/'model.pt', map_location='cuda:1', weights_only=True))
                value = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
                scores = experiment.common.load(dest/'scores.npz')
                np.testing.assert_array_equal(value, scores['standalone'])
                for key, actual in experiment.score_arrays(a, value, 666, p['strong_weight']).items():
                    np.testing.assert_array_equal(actual, scores[key])
                found.add((task,kind,lr,666))
                records.append(dict(task=task,kind=kind,old_kind=source_kind,lr=lr,seed=666,
                    source=str(source/'result.json'),source_sha256=base.digest(source/'result.json'),
                    target=str(dest/'result.json'),target_sha256=base.digest(dest/'result.json'),
                    neural_model=r['trained_neural_model'],replay_max_abs=0.))
                del model, xs, graphs
                gc.collect(); torch.cuda.empty_cache()
    if found != expected:
        raise ValueError('Incomplete adopted records')
    adopted = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(__file__), original_adoption_source_sha256=base.digest(root/'adopt_eigenbasis_parent_v1.py'),
        protocol_sha256=ph, parent_protocol_sha256=parent_ph, records=records,
        new_neural_fits=0, adopted_neural_fits=4, adopted_identity_scoring_records=4,
        parent_results_seen_before_revision=True, preserves_parent_failure=True,
        exact_input_and_prediction_replay=True, metadata_recovery=True))
    base.write_json(manifest, adopted)
    for path, digest in before.items():
        if base.digest(path) != digest:
            raise ValueError('Recovery changed a preserved file')
    report = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(__file__), original_error='Missing seal in non_tabular_experiment alias',
        failed_log_sha256=base.digest(failed_log), preserved_sha256=before,
        completed_manifest=str(manifest), manifest_sha256=base.digest(manifest),
        new_neural_fits=0, all_eight_replays_exact=True, source_and_protocol_unchanged=True))
    base.write_json(recovery, report)
    frozen = [core_out/'protocol.json', out/'protocol.json', manifest, recovery,
              HERE/'EIGENBASIS_SMOKE_V1.json', HERE/'EIGENBASIS_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family='eigenbasis_adam',
        source_sha256=base.digest(HERE/'run_eigenbasis_queue_v1.py'),
        input_sha256={str(path):base.digest(path) for path in frozen},device='cuda:1',
        stages=['screen','confirm'],stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen':2,'confirm_if_eligible':6},adopted_neural_fits=4,
        expected_identity_records={'screen':6,'confirm_if_eligible':6},adopted_identity_records=4,
        identity_records_count_as_neural_fits=False,independent_screen=False,confirmed_final_innovations=0))
    base.write_json(queue_path,queue)
    print('EIGENBASIS_REGISTRATION_COMPLETE',ph,queue['sha256'],flush=True)


if __name__ == '__main__':
    main()
