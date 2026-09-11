"""Equal-budget inner tuning of a marginal/interaction pair-exchange model.

Outer validation labels never select a setting or checkpoint in this round.
The outer split has been used by earlier experiments, so these remain
exploratory results. The test split is neither loaded nor evaluated.
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

from improve.expression_consensus import all_sources, check_no_test, load_experiment, make_model, TASKS
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import json_digest, sha256, write_json
from improve.network_feature_ablation import score
from improve.pair_exchange_modules import (DecomposedDecoder, auxiliary_loss, exchange_pairs,
    exchange_support, positive_matrix, remove_target_pairs, restrict_training_graph,
    sample_exchanges, sample_queries)


ARMS = ('pca_full', 'pca_query', 'factor_bce', 'factor_pairwise', 'factor_exchange')
DEFINITIONS = {
    'pca_full': 'Original PCA GNN and concat decoder; full fitting graph; BCE',
    'pca_query': 'Same PCA GNN/decoder; query positives removed both ways; query BCE',
    'factor_bce': 'Query-absent graph; additive endpoint heads plus centered bilinear decoder; BCE',
    'factor_pairwise': 'factor_bce plus four ordinary pairwise ranking losses on sampled exchanges',
    'factor_exchange': 'factor_bce plus a joint four-pair exchange contrast canceling additive scores',
}
OWN_FILES = ('pair_exchange.py', 'pair_exchange_modules.py', 'test_pair_exchange.py')


def own_sources():
    return dict(all_sources(), **{name: sha256(Path(__file__).with_name(name)) for name in OWN_FILES})


def inner_partition(labels, seed=20260911):
    fit, validation = train_test_split(np.arange(len(labels)), test_size=.2,
                                      stratify=labels, random_state=seed)
    return np.sort(fit), np.sort(validation)


def prepare(args):
    out, parent_dir = Path(args.out_dir).resolve(), Path(args.parent_dir).resolve()
    if (out / 'protocol.json').exists():
        raise ValueError('A protocol already exists; choose a new output directory')
    parent = load_experiment(parent_dir)
    parent_validation = json.loads((parent_dir / 'VALIDATION.json').read_text())
    if not parent_validation['passed'] or parent_validation['smoke']:
        raise ValueError('A verified full parent experiment is required')
    configs = [dict(id='c'+str(i), lr_multiplier=lr, weight_decay=wd)
               for i, (wd, lr) in enumerate((wd, lr) for wd in (0., 1e-4) for lr in (.3, 1., 3.))]
    if args.smoke:
        configs = [dict(id='smoke', lr_multiplier=1., weight_decay=0.)]
    protocol = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), parent_dir=str(parent_dir),
        parent_protocol_sha256=parent['sha256'], source_sha256=own_sources(),
        tasks=list(TASKS), arms=list(ARMS), definitions=DEFINITIONS, primary='factor_exchange',
        smoke=bool(args.smoke), configurations=configs, tuning_seed=666,
        refit_seeds=[666] if args.smoke else [666, 0, 1, 2, 3],
        training={t: dict(c, train_epochs=2 if args.smoke else c['train_epochs']) for t, c in parent['training'].items()},
        inner_split_seed=20260911, inner_fraction=.2, query_fraction=.5,
        interaction_rank=32, auxiliary_weight=.1, auxiliary_ramp_fraction=.2,
        exchanges_per_epoch=2048,
        selection='Each arm gets all 6 learning-rate/weight-decay settings on an 80/20 split of original fitting labels; select one config and first strict maximum inner mean(AUC,AUPR); refit all original training labels for that fixed number of epochs, then evaluate outer validation once',
        ties='Configurations sorted by id; first highest inner criterion wins',
        graph_policy='Every stage graph contains exactly its fitting positives in the target relation; query arms additionally remove queried positives in both directions each epoch; other relations and PCA features remain frozen',
        swap_policy='Two fitting positives from the current query set; crossed pairs must not be fitting positives; crossed unobserved pairs are weak ranking alternatives, not verified biological negatives; no held-out-label filtering',
        equal_search_budget='Same configurations, tuning seed, maximum epochs and checkpoint rule for every arm; query arms have half as many supervised rows per epoch, with 5-seed refits after selection',
        no_test_evaluation=True,
        performance_screen='Positive paired mean outer-validation AUC AND AUPR on BOTH tasks versus tuned pca_full; at least 4/5 seeds improve mean(AUC,AUPR) in each task',
        mechanism_screen='Separately require positive mean macro-drug AUPR against BOTH factor_bce and factor_pairwise on both tasks, with at least 4/5 paired positive seeds per contrast and task',
        limitations=['previously reused outer validation split; not independent confirmation',
                     'five initialization seeds are not independent splits',
                     'unobserved crossed pairs may include undiscovered positives',
                     'the contrast cancels additive endpoint scores, not all observational confounding',
                     'components have prior art; no first-ever or causal claim'],
        files={}, audit={})
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        destination = out / 'inputs' / task
        destination.mkdir(parents=True, exist_ok=True)
        with np.load(parent_dir / 'inputs' / task / 'arrays.npz') as source:
            arrays = {k: source[k].copy() for k in source.files
                      if k.startswith(('x_', 'pca_x_', 'weight_')) or k in ('pairs_train', 'pairs_val', 'y_train', 'y_val')}
        check_no_test(arrays)
        fit, validation = inner_partition(arrays['y_train'], protocol['inner_split_seed'])
        inner = {k: v.copy() for k, v in arrays.items() if not k.startswith(('pairs_', 'y_'))}
        for fold, indices in (('train', fit), ('val', validation)):
            inner['pairs_'+fold] = arrays['pairs_train'][:, indices].copy()
            inner['y_'+fold] = arrays['y_train'][indices].copy()
        check_no_test(inner)
        np.savez_compressed(destination / 'outer.npz', **arrays)
        np.savez_compressed(destination / 'inner.npz', **inner)
        np.savez_compressed(destination / 'inner_indices.npz', fit=fit, validation=validation)
        shutil.copy2(parent_dir / 'inputs' / task / 'train_graph.pth', destination / 'train_graph.pth')
        nt = protocol['training'][task]['rna']
        protocol['audit'][task] = dict(
            outer_training_rows=len(arrays['y_train']), inner_fit_rows=len(fit), inner_validation_rows=len(validation),
            outer_validation_rows=len(arrays['y_val']),
            exchange_support=exchange_support(arrays['pairs_train'], arrays['y_train'], len(arrays['pca_x_'+nt]), len(arrays['x_drug'])))
        for path in destination.iterdir():
            protocol['files'][str(path.relative_to(out))] = sha256(path)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in protocol['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    protocol['sha256'] = json_digest(protocol)
    write_json(out / 'protocol.json', protocol)
    print('FROZEN', out, 'tuning', len(TASKS)*len(ARMS)*len(configs),
          'refits', len(TASKS)*len(ARMS)*len(protocol['refit_seeds']), flush=True)


def load_config(out):
    out = Path(out)
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['sha256'] != json_digest({k:v for k,v in protocol.items() if k != 'sha256'}):
        raise ValueError('Protocol changed')
    if own_sources() != protocol['source_sha256']:
        raise ValueError('Frozen training source changed')
    for name, digest in protocol['files'].items():
        if sha256(out / name) != digest:
            raise ValueError('Input changed: '+name)
    return protocol


def load_stage(out, protocol, task, stage):
    if stage not in ('inner', 'outer'):
        raise ValueError(stage)
    directory = Path(out) / 'inputs' / task
    # The tuning worker opens only inner.npz. The original outer validation
    # labels are not available to its training loop or checkpoint selector.
    with np.load(directory / (stage + '.npz')) as source:
        arrays = {k: source[k].copy() for k in source.files}
    check_no_test(arrays)
    graph = torch.load(directory / 'train_graph.pth', map_location='cpu')
    weights = {key: torch.from_numpy(arrays['weight_'+str(i)]) for i,key in enumerate(graph.edge_types)}
    graph, weights = restrict_training_graph(graph, weights, task, protocol['training'][task]['rna'],
                                            arrays['pairs_train'], arrays['y_train'])
    for i, key in enumerate(graph.edge_types):
        arrays['weight_'+str(i)] = weights[key].numpy().copy()
    target = (protocol['training'][task]['rna'], task, 'drug')
    if set(map(tuple, graph[target].edge_index.numpy().T)).intersection(map(tuple, arrays['pairs_val'].T)):
        raise ValueError('Held-out query is present in the target support graph')
    return arrays, graph


def build(arrays, graph, task, arm, seed, protocol, device):
    if arm not in ARMS:
        raise ValueError(arm)
    model, data, pairs, weights = make_model(arrays, graph, task, 'pca', seed, device)
    if arm.startswith('factor_'):
        cfg = protocol['training'][task]
        model.decoder = DecomposedDecoder(cfg['channels'], cfg['rna'],
            np.unique(arrays['pairs_train'][0]), np.unique(arrays['pairs_train'][1]),
            rank=protocol['interaction_rank']).to(device)
    return model, data, pairs, weights


@torch.no_grad()
def predict(model, data, pairs, weights):
    model.eval()
    z = model.encoder(data.x_dict, data.edge_index_dict, weights)
    if isinstance(model.decoder, DecomposedDecoder):
        marginal, interaction = model.decoder.components(z, pairs)
        return dict(val=(marginal+interaction).cpu().numpy(),
                    marginal=marginal.cpu().numpy(), interaction=interaction.cpu().numpy())
    return dict(val=model.decoder(z, pairs).cpu().numpy())


def train(arrays, graph, task, arm, seed, config, stage, epochs, protocol, device):
    started = time.monotonic()
    model, data, pairs, weights = build(arrays, graph, task, arm, seed, protocol, device)
    initial, backbone = state_digest(model), state_digest(model.encoder)
    settings = protocol['training'][task]
    optimizer = torch.optim.RMSprop(model.parameters(), lr=settings['lr']*config['lr_multiplier'],
                                    alpha=.99, weight_decay=config['weight_decay'])
    labels = torch.from_numpy(arrays['y_train']).to(device)
    known = positive_matrix(arrays['pairs_train'], arrays['y_train'],
                            data[settings['rna']].num_nodes, data['drug'].num_nodes)
    mode = arm[len('factor_'):] if arm in ('factor_pairwise','factor_exchange') else None
    best, state, selected, prediction = -np.inf, None, None, None
    trace = []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        query = (np.arange(len(labels)) if arm == 'pca_full' else
                 sample_queries(arrays['y_train'], protocol['query_fraction'], (seed+1)*1000003+epoch))
        query_positive = arrays['pairs_train'][:, query[arrays['y_train'][query] == 1]]
        edge_dict, step_weights = data.edge_index_dict, weights
        if arm != 'pca_full':
            edge_dict, step_weights = remove_target_pairs(edge_dict, weights, task, settings['rna'],
                                                         data['drug'].num_nodes, query_positive)
        exchanges = np.empty((4,0), dtype=np.int64)
        if mode is not None:
            exchanges = sample_exchanges(query_positive, known,
                min(protocol['exchanges_per_epoch'], 2*query_positive.shape[1]),
                (seed+1)*2000003+epoch)
        indices = np.concatenate([arrays['pairs_train'][:, query], exchange_pairs(exchanges)], axis=1)
        z = model.encoder(data.x_dict, edge_dict, step_weights)
        logits = model.decoder(z, torch.from_numpy(indices).to(device))
        supervised = F.binary_cross_entropy_with_logits(logits[:len(query)], labels[query])
        auxiliary = logits.new_zeros(()) if mode is None else auxiliary_loss(logits[len(query):].view(4,-1), mode)
        ramp = min(1., epoch/max(1., protocol['auxiliary_ramp_fraction']*settings['train_epochs']))
        loss = supervised + protocol['auxiliary_weight']*ramp*auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss')
        loss.backward()
        optimizer.step()
        row = dict(epoch=epoch, loss=float(loss.detach()), supervised=float(supervised.detach()),
                   auxiliary=float(auxiliary.detach()), queries=len(query), exchanges=exchanges.shape[1], ramp=ramp)
        if stage == 'inner':
            output = predict(model, data, pairs['val'], weights)
            auc, ap = score(arrays['y_val'], output['val'])
            criterion = (auc+ap)/2
            row.update(inner_auc=auc, inner_aupr=ap, criterion=criterion)
            if criterion > best:
                best, selected, prediction = criterion, epoch, output['val'].copy()
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        trace.append(row)
    if stage == 'outer':
        selected = epochs
        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    output = predict(model, data, pairs['val'], weights)
    error = 0.
    if prediction is not None:
        np.testing.assert_allclose(output['val'], prediction, atol=4e-5, rtol=4e-5)
        error = float(np.max(np.abs(output['val']-prediction)))
        output['val'] = prediction
    auc, ap = score(arrays['y_val'], output['val'])
    result = dict(task=task, arm=arm, seed=seed, config_id=config['id'], config=config,
        stage=stage, auc=auc, aupr=ap, criterion=(auc+ap)/2, selected_epoch=selected,
        trained_epochs=epochs, initial_state_sha256=initial, initial_encoder_sha256=backbone,
        params=sum(p.numel() for p in model.parameters()), restore_max_abs=error,
        elapsed_seconds=time.monotonic()-started, protocol_sha256=protocol['sha256'])
    return result, output, trace, state


def checked_result(directory, protocol):
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text())
    if result['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Result belongs to a different protocol')
    for name, digest in result['artifacts_sha256'].items():
        if sha256(directory / name) != digest:
            raise ValueError('Result artifact changed: '+str(directory/name))
    return result


def result_dir(out, task, arm, stage, identifier):
    return Path(out) / ('tuning' if stage == 'inner' else 'refits') / task / arm / str(identifier)


def load_selection(out, protocol):
    selection = json.loads((Path(out) / 'SELECTION.json').read_text())
    if selection['sha256'] != json_digest({k:v for k,v in selection.items() if k != 'sha256'}):
        raise ValueError('Selection changed')
    if selection['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Selection has a different protocol')
    for relative, digest in selection['tuning_results_sha256'].items():
        if sha256(Path(out)/relative) != digest:
            raise ValueError('A tuning result changed after selection')
    return selection


def run(args, stage):
    out = Path(args.out_dir)
    protocol = load_config(out)
    selection = load_selection(out, protocol) if stage == 'outer' else None
    for task in [args.task] if args.task else TASKS:
        arrays, graph = load_stage(out, protocol, task, stage)
        for arm in ARMS:
            jobs = [(c['id'], protocol['tuning_seed'], c, protocol['training'][task]['train_epochs'])
                    for c in protocol['configurations']]
            if stage == 'outer':
                choice = selection['choices'][task][arm]
                cfg = next(c for c in protocol['configurations'] if c['id'] == choice['config_id'])
                jobs = [('s'+str(seed), seed, cfg, choice['epochs']) for seed in protocol['refit_seeds']]
            for identifier, seed, config, epochs in jobs:
                directory = result_dir(out, task, arm, stage, identifier)
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / 'run.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if (directory / 'result.json').exists():
                        checked_result(directory, protocol)
                        continue
                    print('START', stage, task, arm, identifier, 'epochs', epochs, flush=True)
                    result, predictions, trace, state = train(arrays, graph, task, arm, seed, config,
                        stage, epochs, protocol, args.device)
                    if selection is not None:
                        result['selection_sha256'] = selection['sha256']
                    np.savez_compressed(directory/'scores.npz', **predictions,
                                        y_val=arrays['y_val'], pairs_val=arrays['pairs_val'])
                    pd.DataFrame(trace).to_csv(directory/'trace.csv', index=False)
                    torch.save(state, directory/'checkpoint.pt')
                    result['artifacts_sha256'] = {name:sha256(directory/name) for name in ('scores.npz','trace.csv','checkpoint.pt')}
                    write_json(directory/'result.json', result)
                    print('DONE', stage, task, arm, identifier, '%.6f/%.6f selected %d %.1fs' % (
                        result['auc'],result['aupr'],result['selected_epoch'],result['elapsed_seconds']), flush=True)


def pick_config(rows):
    """One metric and one tie rule; no outer result is accepted by this API."""
    if not rows or any(row['stage'] != 'inner' for row in rows):
        raise ValueError('Only completed inner results can select a configuration')
    ordered = sorted(rows, key=lambda row:row['config_id'])
    return max(ordered, key=lambda row:row['criterion'])


def select(args):
    out = Path(args.out_dir)
    protocol = load_config(out)
    if (out/'SELECTION.json').exists():
        load_selection(out, protocol)
        print('Selection already frozen', flush=True)
        return
    selection = dict(protocol_sha256=protocol['sha256'], choices={}, tuning_results_sha256={})
    all_rows = []
    for task in TASKS:
        selection['choices'][task] = {}
        for arm in ARMS:
            rows = []
            for config in protocol['configurations']:
                directory = result_dir(out, task, arm, 'inner', config['id'])
                result = checked_result(directory, protocol)
                with np.load(directory/'scores.npz') as saved:
                    auc, ap = score(saved['y_val'], saved['val'])
                np.testing.assert_allclose([auc,ap,(auc+ap)/2], [result['auc'],result['aupr'],result['criterion']], atol=1e-12, rtol=0)
                trace = pd.read_csv(directory/'trace.csv')
                if int(trace.loc[trace.criterion.idxmax(),'epoch']) != result['selected_epoch']:
                    raise ValueError('Inner checkpoint rule violated')
                rows.append(result)
                selection['tuning_results_sha256'][str((directory/'result.json').relative_to(out))] = sha256(directory/'result.json')
            winner = pick_config(rows)
            choice = dict(config_id=winner['config_id'], epochs=winner['selected_epoch'],
                          inner_auc=winner['auc'], inner_aupr=winner['aupr'], inner_criterion=winner['criterion'])
            selection['choices'][task][arm] = choice
            all_rows.extend(rows)
            print('SELECT',task,arm,choice,flush=True)
    selection['sha256'] = json_digest(selection)
    write_json(out/'SELECTION.json',selection)
    pd.DataFrame(all_rows).to_csv(out/'tuning_results.csv',index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','tune','select','refit'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--parent-dir', default='improve/results/expression_consensus_20260909')
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'select':
        select(args)
    else:
        run(args, 'inner' if args.command == 'tune' else 'outer')


if __name__ == '__main__':
    main()
