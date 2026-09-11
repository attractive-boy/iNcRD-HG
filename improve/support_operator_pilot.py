"""Bounded internal-validation screen of the support-operator prototype.

New internal partition of original fitting rows; no outer/test labels loaded.
All arms use the same query batches, epoch cap and checkpoint-selection rule.
This screens the fixed prototype, not its best attainable tuned performance.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F

from improve import dual_pair_tuning as reference_runner
from improve import pair_exchange as parent
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import json_digest, sha256, write_json
from improve.network_feature_ablation import score
from improve.pair_exchange_modules import remove_target_pairs, sample_queries
from improve.support_operator_modules import SupportOperatorDecoder


TASKS = ('LncDrug', 'MiDrug')
MODES = {'prior_only': 'prior', 'uniform_support': 'uniform', 'adaptive_support': 'adaptive'}
OWN_FILES = ('support_operator_pilot.py', 'support_operator_modules.py', 'test_support_operator.py')


def sources():
    return dict(reference_runner.sources(), **{name: sha256(Path(__file__).with_name(name)) for name in OWN_FILES})


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if (out / 'protocol.json').exists() or (out / 'runs').exists():
        raise ValueError('Preserve existing experiment; choose another directory')
    ref = reference_runner.load_protocol(reference)
    validation = json.loads((reference / 'VALIDATION.json').read_text())
    if not validation['passed'] or validation['smoke']:
        raise ValueError('A verified full reference is required')
    prior_selection = json.loads((reference / 'selection.log').read_text())
    common = ['pca_query', 'prior_only', 'uniform_support', 'adaptive_support']
    arms = {t: list(common) for t in TASKS}
    arms['MiDrug'].insert(1, 'pca_locked')
    configurations = {t: {a: dict(lr=ref['training'][t]['lr'], weight_decay=0.) for a in arms[t]}
                      for t in TASKS}
    locked = prior_selection['MiDrug']['pca_query']['config']
    configurations['MiDrug']['pca_locked'] = dict(
        lr=ref['training']['MiDrug']['lr'] * locked['lr_multiplier'], weight_decay=locked['weight_decay'])
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=bool(args.smoke),
        reference_dir=str(reference), reference_protocol_sha256=ref['sha256'],
        reference_selection_sha256=sha256(reference / 'selection.log'),
        tasks=list(TASKS), arms=arms, primary='adaptive_support',
        seeds=[666] if args.smoke else [666, 0, 1], rank=16,
        configurations=configurations, query_fraction=.5, split_seed=20260912,
        split_fraction=.2, smoothing_window=2 if args.smoke else 5,
        training={t: dict(ref['training'][t], train_epochs=3 if args.smoke else ref['training'][t]['train_epochs'])
                  for t in TASKS},
        selection='Per run: first strict maximum trailing-five-epoch internal mean(AUC,AP); one checkpoint for both metrics; same epoch cap for every arm. No hyperparameter selection in this round.',
        definitions=dict(pca_query='Original PCA GNN/concat decoder with query-positive masking; original optimizer settings',
                         pca_locked='Same original model at the previously selected MiDrug optimizer setting; extra strong baseline',
                         prior_only='Target-label-free encoder; drug-conditioned prior with no case adaptation',
                         uniform_support='Same new decoder and solver; uniform source-drug retrieval',
                         adaptive_support='Same new decoder and solver; learned source-drug retrieval'),
        graph_policy='Target graph restricted to current fitting positives. pca arms mask query positives both ways during training. All operator arms remove the entire target and reverse relation; other-task/auxiliary relations remain frozen.',
        memory_policy='Only current fitting pairs and labels are passed as support; all queried rows including negatives are excluded inside decoder. No held-out label enters support.',
        optimizer='RMSprop alpha=.99; ordinary unweighted BCE; no clipping or schedule; query fraction .5 for every arm',
        screen='Candidate must improve mean AUC and AP versus every original-model baseline on both tasks, with >=2/3 paired positive mean-metric deltas per contrast. Mechanism additionally requires the same against prior_only and uniform_support.',
        scope='Fixed-prototype exploratory screen on one newly partitioned subset of previously used training data; not independent confirmation',
        no_outer_evaluation=True, no_test_evaluation=True,
        limitations=['No architecture/hyperparameter search for the new prototype.',
                     'Three initializations are not three independent splits.',
                     'Same internal holdout selects checkpoints and reports exploratory scores.',
                     'Input features/PCA and graph schema are inherited; existing circRNA extensions remain.',
                     'Total parameter counts are not matched effective capacities across decoder controls.',
                     'Sampled zero labels are not experimentally confirmed sensitivity.'],
        source_sha256=sources(), files={}, audit={})
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        dest = out / 'inputs' / task
        dest.mkdir(parents=True, exist_ok=True)
        source_file = reference / 'inputs' / task / 'outer.npz'
        # The archive also contains the historical outer holdout. Explicitly
        # access ONLY features, weights, and original fitting labels here.
        with np.load(source_file) as source:
            features = {k: source[k].copy() for k in source.files
                        if k.startswith(('x_', 'pca_x_', 'weight_'))}
            original_pairs = source['pairs_train'].copy()
            original_labels = source['y_train'].copy()
        fit, holdout = train_test_split(np.arange(len(original_labels)), test_size=p['split_fraction'],
                                      random_state=p['split_seed'], stratify=original_labels)
        fit, holdout = np.sort(fit), np.sort(holdout)
        arrays = dict(features)
        for name, indices in (('train', fit), ('val', holdout)):
            arrays['pairs_' + name] = original_pairs[:, indices]
            arrays['y_' + name] = original_labels[indices]
        parent.check_no_test(arrays)
        fit_set, val_set = set(map(tuple, arrays['pairs_train'].T)), set(map(tuple, arrays['pairs_val'].T))
        if len(fit_set) != len(fit) or len(val_set) != len(holdout) or fit_set.intersection(val_set):
            raise ValueError('Duplicate or overlapping query pairs')
        np.savez_compressed(dest / 'inner.npz', **arrays)
        np.savez_compressed(dest / 'indices.npz', fit=fit, holdout=holdout)
        shutil.copy2(reference / 'inputs' / task / 'train_graph.pth', dest / 'train_graph.pth')
        p['audit'][task] = dict(original_fitting_rows=len(original_labels), fitting_rows=len(fit),
                               internal_holdout_rows=len(holdout), fitting_positives=int(arrays['y_train'].sum()),
                               internal_holdout_positives=int(arrays['y_val'].sum()),
                               fitting_holdout_pair_overlap=0, source_archive_sha256=sha256(source_file))
        for path in dest.iterdir():
            p['files'][str(path.relative_to(out))] = sha256(path)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    p['sha256'] = json_digest(p)
    write_json(out / 'protocol.json', p)
    print('FROZEN', out, 'runs', sum(len(a) for a in arms.values()) * len(p['seeds']), flush=True)


def load_protocol(out):
    out = Path(out)
    p = json.loads((out / 'protocol.json').read_text())
    if p['sha256'] != json_digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('Protocol changed')
    if sources() != p['source_sha256']:
        raise ValueError('Frozen source changed')
    for name, digest in p['files'].items():
        if sha256(out / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return p


def build(arrays, graph, task, arm, seed, p, device):
    model, data, pairs, weights = parent.make_model(arrays, graph, task, 'pca', seed, device)
    if arm in MODES:
        torch.manual_seed(seed + 719331)
        model.decoder = SupportOperatorDecoder(p['training'][task]['channels'],
                                               p['training'][task]['rna'], rank=p['rank']).to(device)
    return model, data, pairs, weights


def encoder_graph(data, weights, arrays, task, arm, p, query_ids=None):
    if arm in MODES:
        held = arrays['pairs_train'][:, arrays['y_train'] == 1]
    elif query_ids is None:
        return data.edge_index_dict, weights
    else:
        held = arrays['pairs_train'][:, query_ids[arrays['y_train'][query_ids] == 1]]
    return remove_target_pairs(data.edge_index_dict, weights, task, p['training'][task]['rna'],
                               data['drug'].num_nodes, held)


def shuffled_labels(arrays, device):
    values = arrays['y_train'].copy()
    rng = np.random.default_rng(20260912)
    for drug in np.unique(arrays['pairs_train'][1]):
        ids = np.flatnonzero(arrays['pairs_train'][1] == drug)
        values[ids] = rng.permutation(values[ids])
    return torch.from_numpy(values).to(device)


@torch.no_grad()
def predict(model, data, pairs, weights, labels, arrays, task, arm, p, interventions=False):
    model.eval()
    edges, step_weights = encoder_graph(data, weights, arrays, task, arm, p)
    z = model.encoder(data.x_dict, edges, step_weights)
    if arm not in MODES:
        return {'val': model.decoder(z, pairs['val']).cpu().numpy()}, {}
    prediction, detail = model.decoder(z, pairs['val'], pairs['train'], labels,
                                       mode=MODES[arm], return_details=True)
    if detail['support_pairs'].shape[1] != len(labels):
        raise AssertionError('Fitting/holdout support overlap')
    result = {'val': prediction.cpu().numpy()}
    diagnostics = dict(ridge_median=float(detail['ridge'].median()),
                       ridge_min=float(detail['ridge'].min()), ridge_max=float(detail['ridge'].max()),
                       effective_support_median=float(detail['effective_support'].median()),
                       operator_prior_mean_abs=float((detail['operator'] - detail['prior']).abs().mean()))
    if interventions and arm == 'adaptive_support':
        result['no_support'] = model.decoder(z, pairs['val'], pairs['train'][:, :0], labels[:0]).cpu().numpy()
        result['uniform_weights'] = model.decoder(z, pairs['val'], pairs['train'], labels,
                                                 mode='uniform').cpu().numpy()
        result['shuffled_labels'] = model.decoder(z, pairs['val'], pairs['train'],
                                                  shuffled_labels(arrays, labels.device)).cpu().numpy()
    return result, diagnostics


def train(arrays, graph, task, arm, seed, p, device):
    started = time.monotonic()
    model, data, pairs, weights = build(arrays, graph, task, arm, seed, p, device)
    initial, encoder_initial = state_digest(model), state_digest(model.encoder)
    config = p['configurations'][task][arm]
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'], alpha=.99,
                                    weight_decay=config['weight_decay'])
    labels = torch.from_numpy(arrays['y_train']).to(device)
    best, state, saved, selected = -np.inf, None, None, None
    trace, ever_active = [], set()
    fixed_edges = encoder_graph(data, weights, arrays, task, arm, p) if arm in MODES else None
    for epoch in range(1, p['training'][task]['train_epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = sample_queries(arrays['y_train'], p['query_fraction'], (seed + 1) * 1000003 + epoch)
        edges, step_weights = fixed_edges if fixed_edges is not None else encoder_graph(
            data, weights, arrays, task, arm, p, ids)
        z = model.encoder(data.x_dict, edges, step_weights)
        queries = pairs['train'][:, ids]
        if arm in MODES:
            logits = model.decoder(z, queries, pairs['train'], labels, mode=MODES[arm])
        else:
            logits = model.decoder(z, queries)
        loss = F.binary_cross_entropy_with_logits(logits, labels[ids])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if epoch == 1:
            ever_active.update(n for n, value in model.named_parameters() if value.grad is not None)
        optimizer.step()
        prediction, _ = predict(model, data, pairs, weights, labels, arrays, task, arm, p)
        auc, ap = score(arrays['y_val'], prediction['val'])
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), queries=len(ids),
                          inner_auc=auc, inner_aupr=ap, criterion=(auc + ap) / 2))
        if epoch >= p['smoothing_window']:
            smooth = float(np.mean([r['criterion'] for r in trace[-p['smoothing_window']:]]))
            if smooth > best:
                best, selected, saved = smooth, epoch, prediction['val'].copy()
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0:
            print('EPOCH', task, arm, seed, epoch, 'inner', round((auc + ap) / 2, 6), flush=True)
    model.load_state_dict(state)
    prediction, diagnostics = predict(model, data, pairs, weights, labels, arrays, task, arm, p, True)
    np.testing.assert_allclose(prediction['val'], saved, atol=4e-5, rtol=4e-5)
    error = float(np.max(np.abs(prediction['val'] - saved)))
    prediction['val'] = saved
    auc, ap = score(arrays['y_val'], saved)
    result = dict(task=task, arm=arm, seed=seed, inner_auc=auc, inner_aupr=ap,
                  criterion=(auc + ap) / 2, selected_epoch=selected, selection_score=best,
                  trained_epochs=p['training'][task]['train_epochs'], config=config,
                  initial_state_sha256=initial, initial_encoder_sha256=encoder_initial,
                  params=sum(value.numel() for value in model.parameters()),
                  parameters_with_gradient=sum(value.numel() for n, value in model.named_parameters() if n in ever_active),
                  diagnostics=diagnostics, restore_max_abs=error, device=device,
                  elapsed_seconds=time.monotonic() - started, protocol_sha256=p['sha256'])
    return result, prediction, trace, state


def directory(out, task, arm, seed):
    return Path(out) / 'runs' / task / arm / ('s' + str(seed))


def run(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    for task in [args.task] if args.task else TASKS:
        with (out / (task + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            arrays, graph = parent.load_stage(out, p, task, 'inner')
            for seed in p['seeds']:
                for arm in p['arms'][task]:
                    dest = directory(out, task, arm, seed)
                    if (dest / 'result.json').exists():
                        parent.checked_result(dest, p)
                        print('SKIP', task, arm, seed, flush=True)
                        continue
                    dest.mkdir(parents=True, exist_ok=True)
                    print('START', task, arm, seed, flush=True)
                    result, prediction, trace, state = train(arrays, graph, task, arm, seed, p, args.device)
                    np.savez_compressed(dest / 'scores.npz', **prediction,
                                        pairs_val=arrays['pairs_val'], y_val=arrays['y_val'])
                    pd.DataFrame(trace).to_csv(dest / 'trace.csv', index=False)
                    torch.save(state, dest / 'checkpoint.pt')
                    result['artifacts_sha256'] = {name: sha256(dest / name)
                                                  for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                    write_json(dest / 'result.json', result)
                    print('DONE', task, arm, seed, 'AUC/AP %.6f/%.6f epoch %d %.1fs' % (
                        result['inner_auc'], result['inner_aupr'], result['selected_epoch'], result['elapsed_seconds']), flush=True)


def macro_ap(labels, logits, pairs):
    values = []
    for drug in np.unique(pairs[1]):
        keep = pairs[1] == drug
        if len(np.unique(labels[keep])) == 2:
            values.append(score(labels[keep], logits[keep])[1])
    return float(np.mean(values)) if values else None, len(values)


def report(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    rows, changes, slices, replay = [], [], [], []
    initial, operator_initial = {}, {}
    for task in TASKS:
        arrays, graph = parent.load_stage(out, p, task, 'inner')
        positive_counts = np.bincount(arrays['pairs_train'][1, arrays['y_train'] == 1],
                                      minlength=graph['drug'].num_nodes)
        for seed in p['seeds']:
            for arm in p['arms'][task]:
                dest = directory(out, task, arm, seed)
                r = parent.checked_result(dest, p)
                trace = pd.read_csv(dest / 'trace.csv')
                chosen = reference_runner.choose_epoch([trace], p['smoothing_window'])
                if chosen['epochs'] != r['selected_epoch']:
                    raise AssertionError('Checkpoint selection mismatch')
                if len(trace) != p['training'][task]['train_epochs']:
                    raise AssertionError('Epoch budget mismatch')
                with np.load(dest / 'scores.npz') as saved:
                    np.testing.assert_array_equal(saved['y_val'], arrays['y_val'])
                    np.testing.assert_array_equal(saved['pairs_val'], arrays['pairs_val'])
                    predictions = {k: saved[k].copy() for k in saved.files if k not in ('y_val', 'pairs_val')}
                auc, ap = score(arrays['y_val'], predictions['val'])
                np.testing.assert_allclose([auc, ap], [r['inner_auc'], r['inner_aupr']], atol=1e-12, rtol=0)
                eligible_ap, eligible_drugs = macro_ap(arrays['y_val'], predictions['val'], arrays['pairs_val'])
                r.update(macro_drug_ap=eligible_ap, eligible_drugs=eligible_drugs)
                key = (task, seed)
                if key in initial and initial[key] != r['initial_encoder_sha256']:
                    raise AssertionError('Encoder initialization differs')
                initial[key] = r['initial_encoder_sha256']
                if arm in MODES:
                    if key in operator_initial and operator_initial[key] != r['initial_state_sha256']:
                        raise AssertionError('Operator control initialization differs')
                    operator_initial[key] = r['initial_state_sha256']
                model, data, pairs, weights = build(arrays, graph, task, arm, seed, p, args.device)
                if state_digest(model) != r['initial_state_sha256']:
                    raise AssertionError('Initialization not reproducible')
                model.load_state_dict(torch.load(dest / 'checkpoint.pt', map_location='cpu'))
                labels = torch.from_numpy(arrays['y_train']).to(args.device)
                actual, diagnostics = predict(model, data, pairs, weights, labels, arrays, task, arm, p, True)
                if set(actual) != set(predictions):
                    raise AssertionError('Unexpected prediction arrays')
                errors = {}
                for name in predictions:
                    np.testing.assert_allclose(actual[name], predictions[name], atol=4e-5, rtol=4e-5)
                    errors[name] = float(np.max(np.abs(actual[name] - predictions[name])))
                    if name != 'val':
                        a, b = score(arrays['y_val'], predictions[name])
                        m, _ = macro_ap(arrays['y_val'], predictions[name], arrays['pairs_val'])
                        changes.append(dict(task=task, seed=seed, intervention=name,
                                            auc=a, ap=b, macro_drug_ap=m,
                                            delta_auc=a-auc, delta_ap=b-ap, delta_macro_drug_ap=m-eligible_ap))
                replay.append(dict(task=task, seed=seed, arm=arm, max_abs=errors))
                rows.append(r)
                counts = positive_counts[arrays['pairs_val'][1]]
                for label, keep in [('0', counts == 0), ('1-5', (counts >= 1) & (counts <= 5)),
                                     ('6-20', (counts >= 6) & (counts <= 20)), ('>20', counts > 20)]:
                    valid = len(np.unique(arrays['y_val'][keep])) == 2
                    a, b = score(arrays['y_val'][keep], predictions['val'][keep]) if valid else (None, None)
                    slices.append(dict(task=task, seed=seed, arm=arm, fitting_positive_count=label,
                                       rows=int(keep.sum()), positives=int(arrays['y_val'][keep].sum()), auc=a, ap=b))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'results.csv', index=False)
    pd.DataFrame(changes).to_csv(out / 'interventions.csv', index=False)
    pd.DataFrame(slices).to_csv(out / 'support_slices.csv', index=False)
    summary = frame.groupby(['task', 'arm'])[['inner_auc', 'inner_aupr', 'macro_drug_ap']].agg(['mean', 'std'])
    summary.to_csv(out / 'summary.csv')
    contrasts = []
    for task in TASKS:
        block = frame[frame.task == task]
        candidate = block[block.arm == p['primary']].set_index('seed').sort_index()
        for control in [a for a in p['arms'][task] if a != p['primary']]:
            reference = block[block.arm == control].set_index('seed').sort_index()
            delta_auc = candidate.inner_auc.to_numpy() - reference.inner_auc.to_numpy()
            delta_ap = candidate.inner_aupr.to_numpy() - reference.inner_aupr.to_numpy()
            delta_macro = candidate.macro_drug_ap.to_numpy() - reference.macro_drug_ap.to_numpy()
            contrasts.append(dict(task=task, control=control, delta_auc=float(delta_auc.mean()),
                                  delta_ap=float(delta_ap.mean()), delta_macro_ap=float(delta_macro.mean()),
                                  positive_auc_seeds=int((delta_auc > 0).sum()),
                                  positive_ap_seeds=int((delta_ap > 0).sum()),
                                  positive_criterion_seeds=int((delta_auc + delta_ap > 0).sum())))
    pd.DataFrame(contrasts).to_csv(out / 'contrasts.csv', index=False)
    def passed(group):
        return all(c['delta_auc'] > 0 and c['delta_ap'] > 0 and
                   c['positive_criterion_seeds'] >= (1 if p['smoke'] else 2) for c in group)
    baseline = [c for c in contrasts if c['control'].startswith('pca_')]
    controls = [c for c in contrasts if not c['control'].startswith('pca_')]
    validation = dict(passed=True, smoke=p['smoke'], runs=len(rows), checkpoint_replays=len(replay),
                      no_outer_evaluation=True, no_test_evaluation=True,
                      performance_screen_passed=passed(baseline), mechanism_screen_passed=passed(controls),
                      protocol_sha256=p['sha256'], replay=replay, contrasts=contrasts)
    write_json(out / 'VALIDATION.json', validation)
    print(summary.to_string(), flush=True)
    print('PERFORMANCE', validation['performance_screen_passed'], 'MECHANISM', validation['mechanism_screen_passed'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'report'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--out-dir', required=True)
        if name == 'prepare':
            cmd.add_argument('--reference-dir', default='improve/results/dual_pair_tuning_20260910')
            cmd.add_argument('--smoke', action='store_true')
        else:
            cmd.add_argument('--device', default='cuda:0')
        if name == 'run':
            cmd.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
