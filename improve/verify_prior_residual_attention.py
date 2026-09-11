"""Verify saved predictions, weight constraints and matched reference controls."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from improve.attention_mechanisms import save_csv
from improve.drug_condition_attention import file_hash, make_base, metrics, set_seed, write_json
from improve.edge_path_attention import configure, predict, run_path
from improve.edge_path_modules import SIMILARITIES
from improve.prior_residual_attention import jobs, load_inputs, source_hashes
from improve.prior_residual_modules import ARMS, CONTROLS, NEW_ARMS, interventions, make_model
from improve.verify_edge_path_attention import check_metrics, require, state_hash, verify_inputs


def expected_strength(arm, edges, attributes, seed):
    """Numpy reconstruction, independent of the trainable convolution."""
    nonself = edges[0] != edges[1]
    values = attributes[nonself, 1].copy()
    if arm == 'bounded_uniform':
        return np.ones_like(values)
    if arm != 'prior_shuffled':
        return values
    canonical = np.sort(edges[:, nonself], axis=0).T
    _, first, inverse = np.unique(canonical, axis=0, return_index=True, return_inverse=True)
    order = np.random.RandomState(seed).permutation(len(first))
    return values[first[order]][inverse]


def check_weights(directory, arm, train, assets, state, seed, epsilon):
    if arm == 'base':
        return []
    diagnostics = []
    for fold in ('val', 'test'):
        with np.load(directory / ('attention_' + fold + '.npz')) as arrays:
            expected_names = {'convs.%d.%s' % (layer, '__'.join(rel))
                              for layer in (0, 1) for rel in assets['attributes']}
            require(set(arrays.files) == expected_names, 'Missing or unexpected edge weights')
            for layer in (0, 1):
                for i, (rel, attrs) in enumerate(assets['attributes'].items()):
                    name = 'convs.%d.%s' % (layer, '__'.join(rel))
                    edges = train[rel].edge_index.numpy()
                    nonself = edges[0] != edges[1]
                    n = train[rel[2]].num_nodes
                    w = arrays[name]
                    require(w.shape == (edges.shape[1],) and np.isfinite(w).all() and (w >= 0).all(), 'Invalid edge weights')
                    np.testing.assert_array_equal(w[~nonself], np.ones(int((~nonself).sum())))
                    degree = np.bincount(edges[1, nonself], minlength=n)
                    total = np.bincount(edges[1, nonself], weights=w[nonself], minlength=n)
                    np.testing.assert_allclose(total, degree, rtol=2e-6, atol=2e-5)
                    strength = expected_strength(arm, edges, attrs.numpy(), seed + 700003 + i)
                    denominator = np.bincount(edges[1, nonself], weights=strength, minlength=n)
                    prior = strength * degree[edges[1, nonself]] / denominator[edges[1, nonself]]
                    prefix = 'base.encoder.' + name + '.'
                    np.testing.assert_allclose(state[prefix + 'fixed_weights'].numpy()[nonself], prior,
                                               rtol=2e-6, atol=2e-6)
                    if arm in NEW_ARMS:
                        np.testing.assert_array_equal(state[prefix + 'prior_strength'].numpy(), strength)
                        np.testing.assert_array_equal(state[prefix + 'attributes'].numpy(), attrs.numpy())
                    if arm == 'edge_fixed':
                        np.testing.assert_allclose(w[nonself], prior, rtol=2e-6, atol=2e-6)
                    ratio = w[nonself] / prior
                    if arm in ('prior_bounded', 'bounded_uniform', 'prior_shuffled'):
                        lower, upper = (1 - epsilon) / (1 + epsilon), (1 + epsilon) / (1 - epsilon)
                        require(bool((ratio >= lower - 3e-6).all() and (ratio <= upper + 3e-6).all()), 'Bounded correction exceeded its normalized limit')
                    diagnostics.append(dict(fold=fold, layer=layer, relation='|'.join(rel),
                                            nonself_edges=int(nonself.sum()),
                                            mean_absolute_change=float(np.mean(np.abs(w[nonself] - prior))),
                                            ratio_min=float(ratio.min()), ratio_max=float(ratio.max())))
    return diagnostics


def verify(args):
    configure()
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    require(p['arms'] == list(ARMS) and p['epsilon'] == .25, 'Frozen experiment definition changed')
    require(p['source_hashes'] == source_hashes(), 'Frozen source changed')
    for name, digest in p['source_hashes'].items():
        require(file_hash(root / 'source_snapshot' / name) == digest, 'Source snapshot changed')
    data, assets = load_inputs(p)
    matrices = {}
    for relation, name in SIMILARITIES.items():
        path = Path(args.net_dir) / name
        require(file_hash(path) == p['matrix_hashes'][name], 'Similarity matrix changed')
        matrices[relation] = pd.read_csv(path, header=None).values
    for task in p['tasks']:
        verify_inputs(data[task], assets[task], p['configs'][task], task, matrices)
    protocol_hash = file_hash(root / 'protocol.json')
    records, initial, diagnostics, references = {}, {}, [], []
    repeated, replayed = 0, []
    max_logit_difference = max_metric_difference = 0.
    for _, task, seed, arm, repeat in jobs(p):
        cfg = p['configs'][task]
        key = (cfg['rna'], task, 'drug')
        directory = run_path(root, task, seed, arm, repeat)
        r = json.loads((directory / 'result.json').read_text())
        signature = dict(task=task, seed=seed, arm=arm, repeat=repeat, protocol_sha256=protocol_hash)
        require(r['signature'] == signature and json.loads((directory / 'config.json').read_text()) == signature, 'Run signature changed')
        require(r['trained_epochs'] == cfg['epochs'] and r['validation_replay_delta'] == 0., 'Incomplete training or checkpoint mismatch')
        with (directory / 'trace.csv').open() as stream:
            trace = list(csv.DictReader(stream))
        require([int(row['epoch']) for row in trace] == list(range(1, cfg['epochs'] + 1)), 'Missing training epochs')
        for row in trace:
            values = [float(row[k]) for k in ('loss', 'val_auc', 'val_aupr', 'criterion')]
            require(np.isfinite(values).all() and abs(values[3] - (values[1] + values[2]) / 2) < 1e-14, 'Invalid validation trajectory')
        best = max(trace, key=lambda row: float(row['criterion']))
        require(int(best['epoch']) == r['selected_epoch'], 'Checkpoint not selected by validation or earliest tie')
        require(abs(float(best['criterion']) - (r['metrics']['val']['auc'] + r['metrics']['val']['aupr']) / 2) < 1e-14, 'Selected metrics differ from trajectory')
        state = torch.load(directory / 'checkpoint.pt', map_location='cpu')
        require(state_hash(state) == r['selected_state_sha256'], 'Checkpoint hash changed')
        records[task, seed, arm, repeat] = r
        with np.load(directory / 'predictions.npz') as saved:
            initial[task, seed, arm, repeat] = saved['initial_val_logits'].copy()
            require(set(r['interventions']) == set(interventions(arm)), 'Missing mechanism interventions')
            for fold in ('val', 'test'):
                labels = data[task][fold][key].edge_label.numpy()
                pairs = assets[task]['pairs'][fold].numpy()
                np.testing.assert_array_equal(saved['y_' + fold], labels)
                np.testing.assert_array_equal(saved['pairs_' + fold], pairs)
                check_metrics(labels, saved[fold], pairs, r['metrics'][fold])
                for mode in interventions(arm):
                    check_metrics(labels, saved[mode + '_' + fold], pairs, r['interventions'][mode][fold])
            if repeat:
                first_dir = run_path(root, task, seed, arm, 0)
                first = records[task, seed, arm, 0]
                require(r['device'] == first['device'] and r['selected_state_sha256'] == first['selected_state_sha256'], 'Repeated state differs')
                require(file_hash(directory / 'trace.csv') == file_hash(first_dir / 'trace.csv'), 'Repeated trajectory differs')
                with np.load(first_dir / 'predictions.npz') as first_saved:
                    require(set(saved.files) == set(first_saved.files), 'Repeated prediction keys differ')
                    for name in saved.files:
                        np.testing.assert_array_equal(saved[name], first_saved[name])
                repeated += 1
            if not repeat and not p['smoke'] and arm in CONTROLS:
                reference_dir = run_path(Path(p['parent_protocol']).parent, task, seed, arm, 0)
                reference = json.loads((reference_dir / 'result.json').read_text())
                require(reference['device'] == r['device'], 'Reference ran on another device; exact comparison invalid')
                require(reference['selected_state_sha256'] == r['selected_state_sha256'], 'Matched historical control state differs')
                require(file_hash(reference_dir / 'trace.csv') == file_hash(directory / 'trace.csv'), 'Matched historical control trajectory differs')
                with np.load(reference_dir / 'predictions.npz') as ref:
                    require(set(saved.files) == set(ref.files), 'Reference prediction keys differ')
                    for name in saved.files:
                        np.testing.assert_array_equal(saved[name], ref[name])
                references.append(dict(task=task, seed=seed, arm=arm, exact=True))
            if not repeat and seed == p['seeds'][0]:
                train = data[task]['train'].clone().to(args.replay_device)
                base = make_base(train, cfg, seed, args.replay_device)
                set_seed(seed + p['branch_rng_offset'])
                model = make_model(base, data[task]['train'], assets[task], cfg, arm, seed, p).to(args.replay_device)
                model.load_state_dict(state)
                for mode in (None,) + interventions(arm):
                    for fold in ('val', 'test'):
                        name = fold if mode is None else mode + '_' + fold
                        actual = predict(model, train, fold, mode)
                        logit_delta = float(np.max(np.abs(actual - saved[name])))
                        require(logit_delta < 5e-4, 'Reloaded checkpoint does not reproduce saved logits')
                        max_logit_difference = max(max_logit_difference, logit_delta)
                        actual_metrics = metrics(saved['y_' + fold], actual, saved['pairs_' + fold])
                        expected = r['metrics'][fold] if mode is None else r['interventions'][mode][fold]
                        for metric in ('auc', 'aupr', 'per_drug_auc', 'per_drug_aupr'):
                            if expected[metric] is not None:
                                difference = abs(actual_metrics[metric] - expected[metric])
                                require(difference < 2e-4, 'Reloaded checkpoint metric differs materially')
                                max_metric_difference = max(max_metric_difference, difference)
                replayed.append(dict(task=task, seed=seed, arm=arm))
                del model, base, train
        for row in check_weights(directory, arm, data[task]['train'], assets[task], state, seed, p['epsilon']):
            diagnostics.append(dict(task=task, seed=seed, arm=arm, repeat=repeat, **row))
        print('VERIFIED', task, seed, arm, repeat, flush=True)
    for task, seed in itertools.product(p['tasks'], p['seeds']):
        require(len({records[task, seed, a, 0]['initial_core_sha256'] for a in p['arms']}) == 1, 'Initial baseline cores differ')
        scored = ('edge_attributes',) + NEW_ARMS
        require(len({json.dumps(records[task, seed, a, 0]['initial_scorer_sha256'], sort_keys=True) for a in scored}) == 1, 'Initial scorer parameters differ')
        require(len({records[task, seed, a, 0]['trainable_parameters'] for a in scored}) == 1, 'Scoring arms have different trainable capacity')
        for arm, reference in (('edge_attributes', 'base'), ('bounded_uniform', 'base'),
                               ('prior_bounded', 'edge_fixed'), ('prior_free', 'edge_fixed')):
            np.testing.assert_array_equal(initial[task, seed, arm, 0], initial[task, seed, reference, 0])
    save_csv(root / 'weight_diagnostics.csv', diagnostics)
    write_json(root / 'CONTROL_REPLAY.json', dict(passed=True, comparisons=references))
    result = dict(passed=True, runs_verified=len(records), repeated_runs_exact=repeated,
                  historical_controls_exact=len(references), edge_weight_arrays_verified=len(diagnostics),
                  checkpoint_replays=replayed, replay_device=args.replay_device,
                  maximum_replay_logit_difference=max_logit_difference,
                  maximum_replay_metric_difference=max_metric_difference,
                  protocol_sha256=protocol_hash, verifier_sha256=file_hash(__file__),
                  checks=['source and input hashes', 'disjoint labels and training-only evaluation graph',
                          'validation-only checkpoint selection', 'saved metrics and interventions',
                          'same core/scorer initialization and scorer capacity',
                          'fixed-weight initial predictions', 'degree and self-loop preservation',
                          'normalized correction bounds', 'shuffled prior reconstruction',
                          'same-device repeats', 'matched historical controls', 'first-seed CPU checkpoint reloads'])
    write_json(root / 'VALIDATION.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--net-dir', default='/data1/licheng/incrd-hg/incrd-run/2.construct_network')
    parser.add_argument('--replay-device', default='cpu')
    verify(parser.parse_args())


if __name__ == '__main__':
    main()
