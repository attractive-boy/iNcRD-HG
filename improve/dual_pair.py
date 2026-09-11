"""Frozen follow-up: parallel local and transported pair representations.

Tune on an internal split of the existing training labels; refit for fixed
selected epochs and evaluate the existing outer validation once. The outer
validation has been reused historically and is exploratory, not independent.
No test split is opened by this runner.
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
import torch
from torch.nn import functional as F

from improve import pair_exchange as parent
from improve.expression_consensus import check_no_test, make_model, TASKS
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import json_digest, sha256, write_json
from improve.network_feature_ablation import score
from improve.pair_exchange_modules import remove_target_pairs, sample_queries
from improve.dual_pair_modules import DualPairModel


TUNED_ARMS = ('pca_query', 'dual_local', 'dual_transport', 'dual_pair')
ARMS = ('pca_default',) + TUNED_ARMS
DEFINITIONS = {
    'pca_default': 'Original PCA GNN/concat decoder, full fitting graph, fixed original lr and epoch budget',
    'pca_query': 'Original PCA GNN/concat decoder, half-row query sampling with query-positive edges removed both ways',
    'dual_local': 'Two parallel pointwise pair-state streams with a shared joint readout',
    'dual_transport': 'Two parallel coordinate-transport pair-state streams with the same readout and parameter count',
    'dual_pair': 'One pointwise interaction stream plus one coordinate-transport stream, jointly trained with a shared readout',
}
OWN_FILES = ('dual_pair.py', 'dual_pair_modules.py', 'test_dual_pair.py')


def sources():
    return dict(parent.own_sources(), **{n: sha256(Path(__file__).with_name(n)) for n in OWN_FILES + ('pair_state_modules.py', 'test_pair_state.py')})


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.parent_dir).resolve()
    if (out / 'protocol.json').exists():
        raise ValueError('Protocol exists; resume it or use another directory')
    prior = parent.load_config(reference)
    if not json.loads((reference / 'VALIDATION.json').read_text())['passed']:
        raise ValueError('Parent experiment has not passed verification')
    previous = Path(args.previous_dir).resolve()
    previous_validation = json.loads((previous / 'VALIDATION.json').read_text())
    if not previous_validation['passed'] or previous_validation['smoke']:
        raise ValueError('The preceding structural experiment must be fully verified')
    configurations = [dict(id='c'+str(i), lr_multiplier=lr, weight_decay=wd)
                      for i, (wd, lr) in enumerate((wd, lr) for wd in (0.,) for lr in (.3, 1., 3.))]
    if args.smoke:
        configurations = [dict(id='smoke', lr_multiplier=1., weight_decay=0.)]
    training = {t: dict(v, train_epochs=2 if args.smoke else v['train_epochs'])
                for t, v in prior['training'].items()}
    protocol = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), parent_dir=str(reference),
        parent_protocol_sha256=prior['sha256'], source_sha256=sources(),
        tasks=list(TASKS), tuned_arms=list(TUNED_ARMS), arms=list(ARMS), definitions=DEFINITIONS,
        primary='dual_pair', smoke=bool(args.smoke), configurations=configurations,
        previous_structure_dir=str(previous),
        previous_structure_validation_sha256=sha256(previous / 'VALIDATION.json'),
        design_uses_previous_outer_results=True,
        primary_controls=['pca_default','pca_query','dual_local','dual_transport'],
        mechanism_controls=['dual_local','dual_transport'], boundary_control=None,
        interventions=['branch_off','no_transport','zero_local'],
        tuning_seed=666, refit_seeds=[666] if args.smoke else [666, 0, 1, 2, 3],
        training=training, inner_split_seed=20260912, inner_fraction=.2,
        query_fraction=.5, pair_width=16, pair_depth=2,
        selection='Each tunable arm gets the same three learning-rate settings with fixed zero weight decay. Select config and first strict maximum inner mean(AUC,AUPR). Refit original training labels for fixed selected epochs; evaluate outer validation once. Fixed-default PCA is an additional non-tuned reference.',
        graph_policy='Same frozen node features and graph relations for every arm. Current-stage positives only in target/reverse support. Every query arm removes its query positives both ways. No explicit association boundary token in the extra pair branch. Pair transport uses existing lncRNA/miRNA and drug relations; the original backbone still includes all node/edge types.',
        pair_normalization='Incoming mean only inside the extra pair branch; the original GNN retains its original aggregation and scale',
        train_labels='Existing fixed positive/negative association labels, ordinary unweighted BCE; no resampling or new supervision',
        parameter_control='dual_local, dual_transport and dual_pair have identical allocated parameters and initialization; graph removal can make auxiliary-type parameters inactive in local-only controls',
        performance_screen='Positive mean outer AUC and AUPR versus BOTH pca_default and tuned pca_query, plus at least 4/5 paired mean-metric gains, on BOTH tasks',
        transport_screen='Positive mean AUC/AUPR and >=4/5 paired mean-metric gains against BOTH dual_local and dual_transport on BOTH tasks',
        boundary_screen='Not applicable; the unsupported explicit boundary token was removed',
        no_test_evaluation=True,
        limitations=['outer validation reused by earlier experiments; exploratory',
                     'initialization seeds are not independent data splits',
                     'pair/conditional message passing and graph diffusion have prior art',
                     'association prediction does not identify biological mechanisms',
                     'existing cleaned workspace graph includes circRNA extensions; not a pristine paper reproduction'],
        files={}, audit={})
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        dest = out / 'inputs' / task
        dest.mkdir(parents=True, exist_ok=True)
        with np.load(reference / 'inputs' / task / 'outer.npz') as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        check_no_test(arrays)
        fit, val = parent.inner_partition(arrays['y_train'], protocol['inner_split_seed'])
        inner = {k: v.copy() for k, v in arrays.items() if not k.startswith(('pairs_', 'y_'))}
        for fold, indices in (('train', fit), ('val', val)):
            inner['pairs_'+fold] = arrays['pairs_train'][:, indices]
            inner['y_'+fold] = arrays['y_train'][indices]
        check_no_test(inner)
        np.savez_compressed(dest / 'outer.npz', **arrays)
        np.savez_compressed(dest / 'inner.npz', **inner)
        np.savez_compressed(dest / 'inner_indices.npz', fit=fit, validation=val)
        shutil.copy2(reference / 'inputs' / task / 'train_graph.pth', dest / 'train_graph.pth')
        protocol['audit'][task] = dict(inner_fit_rows=len(fit), inner_validation_rows=len(val),
            outer_fit_rows=len(arrays['y_train']), outer_validation_rows=len(arrays['y_val']))
        for p in dest.iterdir():
            protocol['files'][str(p.relative_to(out))] = sha256(p)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in protocol['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    protocol['sha256'] = json_digest(protocol)
    write_json(out / 'protocol.json', protocol)
    print('FROZEN', out, 'tuning', len(TASKS)*len(TUNED_ARMS)*len(configurations),
          'refits', len(TASKS)*len(ARMS)*len(protocol['refit_seeds']), flush=True)


def load_protocol(out):
    out = Path(out)
    p = json.loads((out / 'protocol.json').read_text())
    if p['sha256'] != json_digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('Protocol changed')
    if sources() != p['source_sha256']:
        raise ValueError('Frozen training source changed')
    for name, digest in p['files'].items():
        if sha256(out / name) != digest:
            raise ValueError('Input changed: '+name)
    return p


def build(arrays, graph, task, arm, seed, protocol, device):
    if arm not in ARMS:
        raise ValueError(arm)
    base, data, pairs, weights = make_model(arrays, graph, task, 'pca', seed, device)
    torch.manual_seed(seed + 473299)
    settings = protocol['training'][task]
    model = DualPairModel(base, data, settings['channels'], settings['rna'], arm,
                           protocol['pair_width'], protocol['pair_depth']).to(device)
    return model, data, pairs, weights


@torch.no_grad()
def predict(model, data, pairs, weights, intervention=None):
    model.eval()
    pred, base, correction = model(data.x_dict, data.edge_index_dict, pairs, weights,
                                   return_components=True, intervention=intervention)
    return {'val': pred.cpu().numpy(), 'base': base.cpu().numpy(), 'correction': correction.cpu().numpy()}


def train(arrays, graph, task, arm, seed, config, stage, epochs, protocol, device):
    started = time.monotonic()
    model, data, pairs, weights = build(arrays, graph, task, arm, seed, protocol, device)
    initial, backbone = state_digest(model), state_digest(model.base)
    settings = protocol['training'][task]
    optimizer = torch.optim.RMSprop(model.parameters(), lr=settings['lr']*config['lr_multiplier'],
                                    alpha=.99, weight_decay=config['weight_decay'])
    labels = torch.from_numpy(arrays['y_train']).to(device)
    best, selected, state, best_prediction = -np.inf, None, None, None
    trace = []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        query = (np.arange(len(labels)) if arm == 'pca_default' else
                 sample_queries(arrays['y_train'], protocol['query_fraction'], (seed+1)*1000003+epoch))
        step_edges, step_weights = data.edge_index_dict, weights
        if arm != 'pca_default':
            held = arrays['pairs_train'][:, query[arrays['y_train'][query] == 1]]
            step_edges, step_weights = remove_target_pairs(step_edges, weights, task, settings['rna'],
                                                           data['drug'].num_nodes, held)
            if epoch == 1 and model.branch is not None:
                boundary = model.branch.boundary(step_edges, data['drug'].x)[settings['rna']]
                if torch.count_nonzero(boundary[torch.from_numpy(held[0]).to(device), torch.from_numpy(held[1]).to(device)]):
                    raise AssertionError('Query label entered pair boundary')
        indices = torch.from_numpy(arrays['pairs_train'][:, query]).to(device)
        logits = model(data.x_dict, step_edges, indices, step_weights)
        loss = F.binary_cross_entropy_with_logits(logits, labels[query])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        row = dict(epoch=epoch, loss=float(loss.detach()), queries=len(query))
        if stage == 'inner':
            output = predict(model, data, pairs['val'], weights)
            auc, ap = score(arrays['y_val'], output['val'])
            criterion = (auc+ap)/2
            row.update(inner_auc=auc, inner_aupr=ap, criterion=criterion)
            if criterion > best:
                best, selected, best_prediction = criterion, epoch, output['val'].copy()
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        trace.append(row)
        if epoch % max(25, epochs//2) == 0:
            print('EPOCH', task, arm, config['id'], seed, epoch, 'loss', row['loss'],
                  'inner' if stage == 'inner' else 'refit', row.get('criterion', ''), flush=True)
    if stage == 'outer':
        selected = epochs
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    output = predict(model, data, pairs['val'], weights)
    restore_error = 0.
    if best_prediction is not None:
        np.testing.assert_allclose(output['val'], best_prediction, atol=2e-4, rtol=2e-5)
        restore_error = float(np.max(np.abs(output['val']-best_prediction)))
    auc, ap = score(arrays['y_val'], output['val'])
    result = dict(task=task, arm=arm, seed=seed, config_id=config['id'], config=config,
        stage=stage, auc=auc, aupr=ap, criterion=(auc+ap)/2,
        selected_epoch=selected, trained_epochs=epochs,
        initial_state_sha256=initial, initial_backbone_sha256=backbone,
        params=sum(p.numel() for p in model.parameters()), restore_max_abs=restore_error,
        pair_abs_correction_mean=float(np.abs(output['correction']).mean()),
        elapsed_seconds=time.monotonic()-started, protocol_sha256=protocol['sha256'])
    return result, output, trace, state


def directory(out, task, arm, stage, identifier):
    return Path(out) / ('tuning' if stage == 'inner' else 'refits') / task / arm / str(identifier)


def checked_result(path, protocol):
    path = Path(path)
    result = json.loads((path / 'result.json').read_text())
    if result['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Wrong result protocol')
    for name, digest in result['artifacts_sha256'].items():
        if sha256(path / name) != digest:
            raise ValueError('Result artifact changed: '+str(path / name))
    return result


def select(args):
    out, p = Path(args.out_dir), load_protocol(args.out_dir)
    if (out / 'SELECTION.json').exists():
        raise ValueError('Selection already frozen')
    selection = dict(protocol_sha256=p['sha256'], chosen={}, tuning_results_sha256={})
    for task in TASKS:
        selection['chosen'][task] = {}
        for arm in TUNED_ARMS:
            records = []
            for config in p['configurations']:
                path = directory(out, task, arm, 'inner', config['id'])
                records.append(checked_result(path, p))
                selection['tuning_results_sha256'][str((path/'result.json').relative_to(out))] = sha256(path/'result.json')
            selected = max(records, key=lambda r: r['criterion'])
            selection['chosen'][task][arm] = dict(config=selected['config'], epochs=selected['selected_epoch'],
                                                  inner_auc=selected['auc'], inner_aupr=selected['aupr'])
        selection['chosen'][task]['pca_default'] = dict(
            config=dict(id='fixed_default', lr_multiplier=1., weight_decay=0.),
            epochs=p['training'][task]['train_epochs'])
    selection['sha256'] = json_digest(selection)
    write_json(out / 'SELECTION.json', selection)
    print(json.dumps(selection['chosen'], indent=2), flush=True)


def load_selection(out, protocol):
    out = Path(out)
    selection = json.loads((out / 'SELECTION.json').read_text())
    if selection['sha256'] != json_digest({k:v for k,v in selection.items() if k != 'sha256'}):
        raise ValueError('Selection changed')
    if selection['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Wrong selection protocol')
    for name, digest in selection['tuning_results_sha256'].items():
        if sha256(out / name) != digest:
            raise ValueError('A tuning result changed after selection')
    return selection


def run(args):
    out, p = Path(args.out_dir), load_protocol(args.out_dir)
    stage = args.stage
    selection = load_selection(out, p) if stage == 'outer' else None
    for task in [args.task] if args.task else TASKS:
        with (out / (task+'_'+stage+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            arrays, graph = parent.load_stage(out, p, task, stage)
            arms = [args.arm] if args.arm else (TUNED_ARMS if stage == 'inner' else ARMS)
            for arm in arms:
                if stage == 'inner' and arm == 'pca_default':
                    raise ValueError('Fixed-default reference is not tuned')
                jobs = [(p['tuning_seed'], c, p['training'][task]['train_epochs'], c['id']) for c in p['configurations']] if stage == 'inner' else [
                    (seed, selection['chosen'][task][arm]['config'], selection['chosen'][task][arm]['epochs'], 's'+str(seed))
                    for seed in p['refit_seeds']]
                for seed, config, epochs, identifier in jobs:
                    path = directory(out, task, arm, stage, identifier)
                    path.mkdir(parents=True, exist_ok=True)
                    if (path / 'result.json').exists():
                        checked_result(path, p)
                        print('SKIP', task, arm, stage, identifier, flush=True)
                        continue
                    print('START', task, arm, stage, identifier, flush=True)
                    result, output, trace, state = train(arrays, graph, task, arm, seed, config, stage, epochs, p, args.device)
                    np.savez_compressed(path / 'scores.npz', **output, y_val=arrays['y_val'], pairs_val=arrays['pairs_val'])
                    pd.DataFrame(trace).to_csv(path / 'trace.csv', index=False)
                    torch.save(state, path / 'checkpoint.pt')
                    result['artifacts_sha256'] = {n: sha256(path / n) for n in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                    write_json(path / 'result.json', result)
                    print('DONE', task, arm, identifier, '%.6f/%.6f epoch %d %.1fs' % (
                        result['auc'], result['aupr'], result['selected_epoch'], result['elapsed_seconds']), flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'select'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--parent-dir', default='improve/results/pair_exchange_20260909')
    parser.add_argument('--previous-dir', default='improve/results/pair_state_20260909')
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--stage', choices=('inner', 'outer'), default='inner')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    {'prepare':prepare, 'run':run, 'select':select}[args.command](args)


if __name__ == '__main__':
    main()
