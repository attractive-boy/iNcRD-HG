"""Audit edge/path experiments independently of the training and summary loops."""
import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from improve.drug_condition_attention import file_hash, make_base, metrics, write_json
from improve.edge_path_attention import configure, jobs, predict, run_path, source_hashes
from improve.edge_path_modules import ARMS, PRIMARY, SIMILARITIES, EdgePathModel


def require(condition, message):
    if not condition:
        raise ValueError(message)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def pairs_set(indices):
    return set(map(tuple, indices.t().tolist()))


def verify_inputs(data, assets, config, task, matrices):
    train = data['train']
    key = (config['rna'], task, 'drug')
    labelled, positives = {}, {}
    for fold, graph in data.items():
        pairs, labels = graph[key].edge_label_index, graph[key].edge_label
        torch.testing.assert_close(assets['pairs'][fold], pairs, rtol=0, atol=0)
        require(set(labels.tolist()) == {0., 1.}, 'Invalid binary labels')
        labelled[fold] = pairs_set(pairs)
        positives[fold] = pairs_set(pairs[:, labels == 1])
        if fold != 'train':
            require(len(labelled[fold]) == labels.numel(), 'Repeated held-out pairs')
        for nt, features in train.x_dict.items():
            require(bool(torch.isfinite(features).all()), 'Nonfinite features')
            torch.testing.assert_close(graph[nt].x, features, rtol=0, atol=0)
    for a, b in itertools.combinations(labelled, 2):
        require(not labelled[a] & labelled[b], 'Overlapping labelled folds')
    require(pairs_set(train[key].edge_index) == positives['train'], 'Unexpected training edges')
    reverse = ('drug', 'rev_' + task, config['rna'])
    require(pairs_set(train[reverse].edge_index.flip(0)) == positives['train'], 'Reverse edges differ')
    known = set.union(*positives.values())
    for graph in data.values():
        require(not pairs_set(graph[key].edge_label_index[:, graph[key].edge_label == 0]) & known,
                'Negative label is a known positive')
    for relation, attributes in assets['attributes'].items():
        source, target = train[relation].edge_index.numpy()
        signed = matrices[relation][source, target].astype(np.float32).copy()
        signed[source == target] = 1.
        np.testing.assert_array_equal(attributes.numpy(), np.stack([signed, np.abs(signed)], -1))
    # Enumerate with Python sets, independently of build_assets/pack_mask.
    path_rows = 0
    for i, family in enumerate(assets['families']):
        first, second = tuple(family['first']), tuple(family['second'])
        require(first[0] == config['rna'] and first[2] == second[0] and second[2] == 'drug',
                'Invalid path family')
        left, right = defaultdict(set), defaultdict(set)
        for source, middle in train[first].edge_index.t().tolist():
            left[source].add(middle)
        for middle, drug in train[second].edge_index.t().tolist():
            right[drug].add(middle)
        for fold, pairs in assets['pairs'].items():
            pack = assets['packs'][fold][i]
            torch.testing.assert_close(pack['valid'].sum(1).float(), pack['counts'], rtol=0, atol=0)
            for row, (rna, drug) in enumerate(pairs.t().tolist()):
                expected = left[rna] & right[drug]
                if first[0] == first[2]:
                    expected = expected - {rna}
                actual = pack['ids'][row][pack['valid'][row]].tolist()
                require(actual == sorted(expected), 'Path pack does not match training-graph paths')
                path_rows += 1
    return path_rows


def verify_attention(directory, arm, train, assets):
    checked = 0
    if arm in ('base', 'path_count'):
        return checked
    for fold in ('val', 'test'):
        with np.load(directory / ('attention_' + fold + '.npz')) as arrays:
            expected = (2 * len(assets['attributes']) if arm.startswith('edge_')
                        else len(assets['families']))
            require(len(arrays.files) == expected, 'Missing attention arrays')
            for name in arrays.files:
                weights = arrays[name]
                require(np.isfinite(weights).all() and (weights >= 0).all(), 'Invalid attention weights')
                if arm.startswith('edge_'):
                    relation = tuple(name.rsplit('.', 1)[-1].split('__'))
                    edges = train[relation].edge_index.numpy()
                    require(weights.shape == (edges.shape[1],), 'Wrong edge-weight shape')
                    nonself = edges[0] != edges[1]
                    np.testing.assert_array_equal(weights[~nonself], np.ones((~nonself).sum()))
                    n = train[relation[2]].num_nodes
                    totals = np.bincount(edges[1, nonself], weights=weights[nonself], minlength=n)
                    degree = np.bincount(edges[1, nonself], minlength=n)
                    np.testing.assert_allclose(totals, degree, atol=2e-5, rtol=2e-6)
                else:
                    i = int(name.split('_')[-1])
                    pack = assets['packs'][fold][i]
                    valid = pack['valid'].numpy()
                    require(weights.shape == valid.shape, 'Wrong path-weight shape')
                    require((weights[~valid] == 0).all(), 'Padding received path weight')
                    np.testing.assert_allclose(weights.sum(1), valid.any(1).astype(float), atol=2e-6)
                checked += 1
    return checked


def check_metrics(labels, scores, pairs, expected):
    require(np.isfinite(scores).all(), 'Nonfinite predictions')
    actual = metrics(labels, scores, pairs)
    for name, value in actual.items():
        if value is None:
            require(expected[name] is None, 'Missing macro metric differs')
        else:
            require(abs(value - expected[name]) < 1e-12, 'Saved metric differs: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--net-dir', default='/data1/licheng/incrd-hg/incrd-run/2.construct_network')
    parser.add_argument('--replay-device', default='cpu')
    args = parser.parse_args()
    configure()
    root = Path(args.out_dir)
    protocol_path = root / 'protocol.json'
    p = json.loads(protocol_path.read_text())
    require(p['source_hashes'] == source_hashes(), 'Frozen training source changed')
    require(p['arms'] == list(ARMS) and p['primary_arms'] == list(PRIMARY), 'Experiment arms changed')
    require(file_hash(p['parent_protocol']) == p['parent_protocol_sha256'], 'Parent protocol changed')
    matrices = {}
    for relation, name in SIMILARITIES.items():
        path = Path(args.net_dir) / name
        require(file_hash(path) == p['matrix_hashes'][name], 'Similarity matrix changed')
        matrices[relation] = pd.read_csv(path, header=None).values
    protocol_hash = file_hash(protocol_path)
    records, initial, replayed = {}, {}, []
    paths_checked = attention_checked = 0
    max_initial_difference = max_replay_metric_difference = max_replay_logit_difference = 0.
    repeats_checked = 0
    for task in p['tasks']:
        config = p['configs'][task]
        key = (config['rna'], task, 'drug')
        data = {}
        for fold, entry in p['inputs'][task]['folds'].items():
            require(file_hash(entry['path']) == entry['sha256'], 'Frozen graph changed')
            data[fold] = torch.load(entry['path'], map_location='cpu')
        asset = p['assets'][task]
        require(file_hash(asset['path']) == asset['sha256'], 'Frozen assets changed')
        assets = torch.load(asset['path'], map_location='cpu')
        paths_checked += verify_inputs(data, assets, config, task, matrices)
        for _, job_task, seed, arm, repeat in jobs(p):
            if job_task != task:
                continue
            directory = run_path(root, task, seed, arm, repeat)
            result = json.loads((directory / 'result.json').read_text())
            signature = dict(task=task, seed=seed, arm=arm, repeat=repeat, protocol_sha256=protocol_hash)
            require(result['signature'] == signature, 'Run signature differs')
            require(json.loads((directory / 'config.json').read_text()) == signature, 'Run config differs')
            require(result['trained_epochs'] == config['epochs'], 'Training stopped early')
            require(result['validation_replay_delta'] == 0., 'Selected validation prediction changed')
            with (directory / 'trace.csv').open() as stream:
                trace = list(csv.DictReader(stream))
            require([int(row['epoch']) for row in trace] == list(range(1, config['epochs'] + 1)),
                    'Missing or unordered epochs')
            for row in trace:
                values = [float(row[n]) for n in ('loss', 'val_auc', 'val_aupr', 'criterion')]
                require(np.isfinite(values).all(), 'Invalid training trace')
                require(abs(values[3] - (values[1] + values[2]) / 2) < 1e-14, 'Selection criterion changed')
            best = max(trace, key=lambda row: float(row['criterion']))
            require(int(best['epoch']) == result['selected_epoch'], 'Checkpoint was not validation-selected')
            require(abs(float(best['criterion']) - sum(result['metrics']['val'][m] for m in ('auc', 'aupr')) / 2)
                    < 1e-14, 'Checkpoint metric differs from trace')
            state = torch.load(directory / 'checkpoint.pt', map_location='cpu')
            require(state_hash(state) == result['selected_state_sha256'], 'Checkpoint state changed')
            records[task, seed, arm, repeat] = result
            with np.load(directory / 'predictions.npz') as values:
                initial[task, seed, arm, repeat] = values['initial_val_logits'].copy()
                for fold in ('val', 'test'):
                    labels, pairs = data[fold][key].edge_label.numpy(), assets['pairs'][fold].numpy()
                    np.testing.assert_array_equal(values['y_' + fold], labels)
                    np.testing.assert_array_equal(values['pairs_' + fold], pairs)
                    check_metrics(labels, values[fold], pairs, result['metrics'][fold])
                    for mode, altered in result['interventions'].items():
                        check_metrics(labels, values[mode + '_' + fold], pairs, altered[fold])
                attention_checked += verify_attention(directory, arm, data['train'], assets)
                if repeat:
                    first_dir = run_path(root, task, seed, arm, 0)
                    first = records[task, seed, arm, 0]
                    require(first['selected_state_sha256'] == result['selected_state_sha256'], 'Repeated states differ')
                    require(first['device'] == result['device'], 'Repeat used a different device')
                    require(file_hash(first_dir / 'trace.csv') == file_hash(directory / 'trace.csv'), 'Repeated traces differ')
                    with np.load(first_dir / 'predictions.npz') as reference:
                        require(set(values.files) == set(reference.files), 'Repeated prediction keys differ')
                        for name in values.files:
                            np.testing.assert_array_equal(values[name], reference[name])
                    repeats_checked += 1
                # Every arm, first declared seed: reconstruct from code and
                # reload serialized state; other seeds still get all file audits.
                elif seed == p['seeds'][0]:
                    train = data['train'].clone().to(args.replay_device)
                    base = make_base(train, config, seed, args.replay_device)
                    model = EdgePathModel(base, data['train'], assets, config, arm, seed,
                                          p['width'], p['edge_width']).to(args.replay_device)
                    model.load_state_dict(state)
                    for fold in ('val', 'test'):
                        for mode in (None, *result['interventions']):
                            scores = predict(model, train, fold, mode)
                            name = fold if mode is None else mode + '_' + fold
                            logits_delta = float(np.max(np.abs(scores - values[name])))
                            expected = result['metrics'][fold] if mode is None else result['interventions'][mode][fold]
                            actual = metrics(values['y_' + fold], scores, values['pairs_' + fold], by_drug=False)
                            delta = max(abs(actual[m] - expected[m]) for m in ('auc', 'aupr'))
                            require(delta < 1e-4, 'CPU/GPU checkpoint metric replay differs')
                            max_replay_metric_difference = max(max_replay_metric_difference, delta)
                            max_replay_logit_difference = max(max_replay_logit_difference, logits_delta)
                    replayed.append(dict(task=task, seed=seed, arm=arm))
                    del model, base, train
            del state
            print('VERIFIED', task, seed, arm, repeat, flush=True)
        for seed in p['seeds']:
            hashes = {records[task, seed, arm, 0]['initial_core_sha256'] for arm in ARMS}
            require(len(hashes) == 1, 'Initial base parameters differ between arms')
            for arm in ARMS:
                if arm == 'edge_fixed':
                    continue
                a, b = initial[task, seed, arm, 0], initial[task, seed, 'base', 0]
                np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-6)
                max_initial_difference = max(max_initial_difference, float(np.max(np.abs(a - b))))
    require(len(records) == p['expected_runs'], 'Experiment is incomplete')
    validation = dict(
        passed=True, runs_verified=len(records), repeated_runs_exact=repeats_checked,
        path_rows_independently_enumerated=paths_checked, attention_arrays_verified=attention_checked,
        matched_initial_predictions_max_difference=max_initial_difference,
        checkpoint_reloads=replayed, replay_device=args.replay_device,
        maximum_replay_metric_difference=max_replay_metric_difference,
        maximum_replay_logit_difference=max_replay_logit_difference,
        checks=['source, matrix, input, asset and protocol hashes', 'disjoint labelled folds and training-only edges',
                'independent path enumeration and self-path exclusion', 'all checkpoint state hashes',
                'validation-only checkpoint selection including earliest ties', 'all saved metrics and interventions',
                'all edge/path weight constraints', 'same-device exact repeats',
                'common initial base weights and matched initial predictions', 'first-seed checkpoint replay for every arm'],
        verifier_sha256=file_hash(__file__), protocol_sha256=protocol_hash,
    )
    write_json(root / 'VALIDATION.json', validation)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == '__main__':
    main()
