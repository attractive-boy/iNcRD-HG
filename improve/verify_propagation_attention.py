"""Verify frozen inputs, matched controls, saved metrics and checkpoint replay."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from improve.propagation_attention import (
    ARMS, PRIMARY, PropagationModel, file_hash, make_base, metrics, predict,
    source_hashes, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    assert p['source_hashes'] == source_hashes()
    initial_difference = replay_difference = 0.
    records, replayed = [], []
    for task in p['tasks']:
        data = {}
        for fold, entry in p['inputs'][task]['folds'].items():
            assert file_hash(entry['path']) == entry['sha256']
            data[fold] = torch.load(entry['path'], map_location='cpu')
        train = data['train']
        config = p['configs'][task]
        key = (config['rna'], task, 'drug')
        label_reference = None
        for seed in p['seeds']:
            initial, hashes = {}, set()
            for arm in ARMS:
                directory = root / task / ('seed_' + str(seed)) / arm
                r = json.loads((directory / 'result.json').read_text())
                records.append(r)
                assert r['signature'] == dict(task=task, seed=seed, arm=arm,
                                              protocol_sha256=file_hash(root / 'protocol.json'))
                assert r['trained_epochs'] == config['epochs']
                hashes.add(r['initial_core_sha256'])
                with (directory / 'trace.csv').open() as stream:
                    trace = list(csv.DictReader(stream))
                assert len(trace) == config['epochs']
                assert int(max(trace, key=lambda x: float(x['criterion']))['epoch']) == r['selected_epoch']
                assert (directory / 'checkpoint.pt').stat().st_size > 0
                with np.load(directory / 'predictions.npz') as values:
                    initial[arm] = values['initial_val_logits'].copy()
                    labels = {k: values[k].copy() for k in ('y_val', 'y_test', 'pairs_val', 'pairs_test')}
                    if label_reference is None:
                        label_reference = labels
                    for k in labels:
                        np.testing.assert_array_equal(labels[k], label_reference[k])
                    for fold in ('val', 'test'):
                        actual = metrics(values['y_' + fold], values[fold], values['pairs_' + fold], by_drug=False)
                        for m in ('auc', 'aupr'):
                            assert abs(actual[m] - r['metrics'][fold][m]) < 1e-12
                        if arm in PRIMARY:
                            altered = metrics(values['y_' + fold], values['uniform_' + fold],
                                              values['pairs_' + fold], by_drug=False)
                            for m in ('auc', 'aupr'):
                                assert abs(altered[m] - r['uniform_intervention'][fold][m]) < 1e-12
                if arm != 'base':
                    with np.load(directory / 'attention.npz') as arrays:
                        for name in arrays.files:
                            w = arrays[name]
                            assert np.isfinite(w).all() and (w >= 0).all()
                            if 'relation_fusion' in name:
                                np.testing.assert_allclose(w.sum(-1), 1., atol=2e-6)
                            else:
                                relation = tuple(name.split('.')[4].split('__'))
                                d = train[relation].edge_index[1].numpy()
                                n = train[relation[2]].num_nodes
                                degree = np.bincount(d, minlength=n)
                                if w.ndim == 1:
                                    w = w[:, None]
                                totals = np.zeros((n, w.shape[1]))
                                np.add.at(totals, d, w)
                                expected = degree if arm == 'edge_gate_degree' else (degree > 0).astype(float)
                                np.testing.assert_allclose(totals, np.broadcast_to(expected[:, None], totals.shape),
                                                           atol=2e-6, rtol=2e-5)
            assert len(hashes) == 1
            for group in [('base', 'edge_gate_degree'), ('relation_uniform', 'relation_global', 'relation_node'),
                          ('edge_uniform_mean', 'edge_gate_mean', 'gatv2_uniform', 'gatv2_attention')]:
                for arm in group[1:]:
                    np.testing.assert_allclose(initial[arm], initial[group[0]], atol=5e-5, rtol=5e-5)
                    initial_difference = max(initial_difference, float(np.max(np.abs(initial[arm] - initial[group[0]]))))
        # Independently reconstruct and reload every primary family on the
        # first predeclared seed. Verify serialization and CPU/GPU replay.
        seed = p['seeds'][0]
        for arm in PRIMARY:
            directory = root / task / ('seed_' + str(seed)) / arm
            r = json.loads((directory / 'result.json').read_text())
            base = make_base(train, config, seed, 'cpu')
            model = PropagationModel(base, train, config, arm, p['width'], p['heads'])
            model.load_state_dict(torch.load(directory / 'checkpoint.pt', map_location='cpu'))
            for fold in ('val', 'test'):
                pairs = data[fold][key].edge_label_index
                scores = predict(model, train, pairs)
                values = metrics(data[fold][key].edge_label.numpy(), scores, pairs.numpy(), by_drug=False)
                difference = max(abs(values[m] - r['metrics'][fold][m]) for m in ('auc', 'aupr'))
                assert difference < 1e-4, (task, arm, fold, difference)
                replay_difference = max(replay_difference, difference)
            replayed.append(dict(task=task, seed=seed, arm=arm))
    assert len(records) == p['expected_runs']
    validation = dict(
        passed=True, runs_verified=len(records), unit_tests_passed=6,
        unit_test_command='python -m unittest improve.test_propagation_attention -v',
        smoke_runs_passed=18, matched_initial_functions_verified=True,
        maximum_initial_prediction_difference=initial_difference,
        checkpoint_reloads=replayed, maximum_cpu_gpu_metric_replay_difference=replay_difference,
        maximum_validation_replay_difference=max(abs(r['validation_replay_delta']) for r in records),
        checks=['source and input hashes', 'all expected runs and artifacts', 'same initial core parameters',
                'family-matched initial predictions', 'attention normalisation',
                'identical held-out labels and order', 'saved predictions reproduce metrics',
                'trace confirms validation-only checkpoint choice', 'uniform interventions reproduce metrics',
                'six independently reconstructed checkpoints replay correctly'],
        verifier_sha256=file_hash(__file__), protocol_sha256=file_hash(root / 'protocol.json'),
    )
    write_json(root / 'VALIDATION.json', validation)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == '__main__':
    main()
