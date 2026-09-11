"""Matched two-task evaluation of complementary channel/path attention."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import copy
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from improve.attention_mechanisms import parameter_hash, save_csv
from improve.drug_condition_attention import file_hash, make_base, metrics, set_seed, write_json
from improve.edge_path_attention import configure, predict, run_path
from improve.smooth_path_attention import source_hashes as previous_source_hashes
from improve.complementary_attention_modules import ARMS, FUSIONS, PRIMARY, ComplementaryModel, interventions, routing_assets
from improve.verify_edge_path_attention import require


CONTRASTS = (
    ('channel_mlp', 'base'), ('channel_attention', 'base'), ('channel_attention', 'channel_mlp'),
    ('path_attention', 'base'), ('fusion_fixed', 'base'), ('fusion_fixed', 'channel_attention'),
    ('fusion_fixed', 'path_attention'), ('fusion_global', 'fusion_fixed'),
    ('fusion_gate', 'base'), ('fusion_gate', 'channel_attention'), ('fusion_gate', 'path_attention'),
    ('fusion_gate', 'fusion_fixed'), ('fusion_gate', 'fusion_global'),
    ('fusion_gate', 'fusion_gate_simple'), ('fusion_gate', 'fusion_no_null'),
    ('fusion_gate_simple', 'base'), ('fusion_no_null', 'base'),
)


def source_hashes():
    result = previous_source_hashes()
    for name in ('complementary_attention.py', 'complementary_attention_modules.py', 'test_complementary_attention.py'):
        result[name] = file_hash(Path(__file__).with_name(name))
    return result


def jobs(p):
    for group, (task, seed) in enumerate(itertools.product(p['tasks'], p['seeds'])):
        for arm in p['arms']:
            for repeat in range(2 if arm == 'base' or p['smoke'] else 1):
                yield group, task, seed, arm, repeat


def initialise(args):
    root, parent_path = Path(args.out_dir), Path(args.input_protocol)
    require(not (root / 'protocol.json').exists(), 'Plan exists; use run to resume')
    parent = json.loads(parent_path.read_text())
    validation_path = parent_path.with_name('VALIDATION.json')
    validation = json.loads(validation_path.read_text())
    require(validation['passed'] and validation['protocol_sha256'] == file_hash(parent_path), 'Unverified parent inputs')
    p = {name: copy.deepcopy(parent[name]) for name in
         ('tasks', 'configs', 'inputs', 'assets', 'width', 'alpha', 'branch_rng_offset', 'training_rng_offset')}
    p.update(arms=list(ARMS), primary_arms=list(PRIMARY), contrasts=[list(c) for c in CONTRASTS],
             seeds=[666] if args.smoke else [666, 0, 1, 2, 3], smoke=args.smoke,
             router_width=16, router_rng_offset=5000011, path_temperature=4.,
             parent_protocol=str(parent_path.resolve()), parent_protocol_sha256=file_hash(parent_path),
             parent_validation_sha256=file_hash(validation_path), source_hashes=source_hashes(),
             optimizer='RMSprop', loss='unweighted BCEWithLogitsLoss',
             deterministic_algorithms=True, cpu_threads=2, cublas_workspace_config=':4096:8',
             selection='one checkpoint maximizing validation AUC/AUPR mean; earliest exact tie',
             common_core='same seeded base in all arms; independently seeded channel/path experts match isolated controls',
             routing='three experts: zero residual, channel correction, T=4 path residual; path masked when no path exists',
             evidence='node embeddings plus train-graph log degrees excluding queried association, log path counts and family presence',
             scope='new project design combining known gating, null-expert and attention ideas; fixed-split exploratory study, no novelty claim',
             evaluation_graph='cleaned training graph in every phase',
             repeat_policy='two baseline runs per seed on the same device; smoke repeats every arm twice')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'inputs').mkdir(exist_ok=True)
    p['routing_assets'], p['references'] = {}, {}
    for task in p['tasks']:
        if args.smoke:
            p['configs'][task]['epochs'] = 3
        cfg = p['configs'][task]
        graph_entry = p['inputs'][task]['folds']['train']
        require(file_hash(graph_entry['path']) == graph_entry['sha256'], 'Training graph changed')
        train = torch.load(graph_entry['path'], map_location='cpu')
        entry = p['assets'][task]
        require(file_hash(entry['path']) == entry['sha256'], 'Frozen path assets changed')
        paths = torch.load(entry['path'], map_location='cpu')
        route = routing_assets(train, paths, cfg['rna'], task)
        destination = root / 'inputs' / (task + '_routing.pt')
        torch.save(route, destination)
        p['routing_assets'][task] = dict(path=str(destination.resolve()), sha256=file_hash(destination))
        reference_root = (Path(args.mi_reference) if task == 'MiDrug' else Path(args.lnc_reference)).resolve()
        ref_protocol = reference_root / 'protocol.json'
        ref_validation = reference_root / 'VALIDATION.json'
        ref_p, ref_v = json.loads(ref_protocol.read_text()), json.loads(ref_validation.read_text())
        require(ref_v['passed'] and ref_v['protocol_sha256'] == file_hash(ref_protocol), 'Reference experiment unverified')
        if not args.smoke:
            require(ref_p['configs'][task] == cfg, 'Control hyperparameters differ')
        p['references'][task] = dict(root=str(reference_root), protocol_sha256=file_hash(ref_protocol),
                                     validation_sha256=file_hash(ref_validation))
    p['expected_runs'] = len(list(jobs(p)))
    snapshot = root / 'source_snapshot'
    snapshot.mkdir()
    for name in p['source_hashes']:
        (snapshot / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_json(root / 'protocol.json', p)
    print('INITIALISED', root, 'runs', p['expected_runs'], flush=True)


def load_inputs(p):
    require(p['source_hashes'] == source_hashes(), 'Frozen source changed')
    require(file_hash(p['parent_protocol']) == p['parent_protocol_sha256'], 'Parent protocol changed')
    data, paths, routing = {}, {}, {}
    for task in p['tasks']:
        data[task] = {}
        for fold, entry in p['inputs'][task]['folds'].items():
            require(file_hash(entry['path']) == entry['sha256'], 'Frozen graph changed')
            data[task][fold] = torch.load(entry['path'], map_location='cpu')
        for source, target in (('assets', paths), ('routing_assets', routing)):
            entry = p[source][task]
            require(file_hash(entry['path']) == entry['sha256'], 'Frozen derived assets changed')
            target[task] = torch.load(entry['path'], map_location='cpu')
    return data, paths, routing


def make_model(base, paths, routing, cfg, arm, seed, p):
    return ComplementaryModel(base, paths, routing, cfg, arm, seed, p['width'],
                              p['branch_rng_offset'], p['router_rng_offset'])


def train_one(root, p, data, paths, routing, task, seed, arm, repeat, device):
    directory = run_path(root, task, seed, arm, repeat)
    signature = dict(task=task, seed=seed, arm=arm, repeat=repeat,
                     protocol_sha256=file_hash(root / 'protocol.json'))
    result_path = directory / 'result.json'
    if result_path.exists():
        require(json.loads(result_path.read_text())['signature'] == signature, 'Existing run has another protocol')
        print('SKIP', task, seed, arm, repeat, flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'config.json', signature)
    cfg = p['configs'][task]
    key = (cfg['rna'], task, 'drug')
    train = copy.deepcopy(data['train']).to(device)
    base = make_base(train, cfg, seed, device)
    initial_core = parameter_hash(base)
    model = make_model(base, paths, routing, cfg, arm, seed, p).to(device)
    expert_hashes = {name: parameter_hash(module) for name in ('channel', 'path', 'router')
                     if (module := getattr(model, name)) is not None}
    set_seed(seed + p['training_rng_offset'])
    initial = predict(model, train, 'val')
    labels = {fold: graph[key].edge_label.numpy() for fold, graph in data.items()}
    optimizer = torch.optim.RMSprop((v for v in model.parameters() if v.requires_grad),
                                    lr=cfg['lr'], alpha=p['alpha'])
    best, trace, started = -float('inf'), [], time.monotonic()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(train.x_dict, train.edge_index_dict, 'train')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, train[key].edge_label)
        require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        loss.backward()
        optimizer.step()
        val = predict(model, train, 'val')
        values = metrics(labels['val'], val, paths['pairs']['val'].numpy(), by_drug=False)
        criterion = (values['auc'] + values['aupr']) / 2
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), val_auc=values['auc'], val_aupr=values['aupr'], criterion=criterion))
        if criterion > best:
            best, best_epoch = criterion, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0 or epoch == cfg['epochs']:
            save_csv(directory / 'trace.csv', trace)
            print(task, seed, arm, repeat, epoch, 'val', values, 'seconds', round(time.monotonic()-started, 1), flush=True)
    model.load_state_dict(best_state)
    predictions, results, changed = {}, {}, {}
    for fold in ('val', 'test'):
        predictions[fold] = predict(model, train, fold)
        results[fold] = metrics(labels[fold], predictions[fold], paths['pairs'][fold].numpy())
        arrays = model.attention()
        if arrays:
            np.savez_compressed(directory / ('attention_' + fold + '.npz'), **arrays)
    require((results['val']['auc'] + results['val']['aupr']) / 2 == best, 'Selected validation prediction changed')
    for mode in interventions(arm):
        changed[mode] = {}
        for fold in ('val', 'test'):
            predictions[mode + '_' + fold] = predict(model, train, fold, mode)
            changed[mode][fold] = metrics(labels[fold], predictions[mode + '_' + fold], paths['pairs'][fold].numpy())
    np.savez_compressed(directory / 'predictions.npz', initial_val_logits=initial, **predictions,
                        **{'y_' + fold: labels[fold] for fold in ('val', 'test')},
                        **{'pairs_' + fold: paths['pairs'][fold].numpy() for fold in ('val', 'test')})
    torch.save(best_state, directory / 'checkpoint.pt')
    write_json(result_path, dict(signature=signature, metrics=results, interventions=changed,
        initial_core_sha256=initial_core, initial_experts_sha256=expert_hashes,
        selected_state_sha256=parameter_hash(model), selected_epoch=best_epoch,
        trained_epochs=cfg['epochs'], validation_replay_delta=0., device=device,
        elapsed_seconds=time.monotonic()-started,
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
    print('RESULT', task, seed, arm, repeat, json.dumps(results), flush=True)
    del optimizer, model, base, train, best_state
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def run(args):
    root = Path(args.out_dir)
    p = json.loads((root / 'protocol.json').read_text())
    data, paths, routing = load_inputs(p)
    require(0 <= args.shard_index < args.shard_count, 'Invalid shard')
    for group, task, seed, arm, repeat in jobs(p):
        if group % args.shard_count == args.shard_index:
            train_one(root, p, data[task], paths[task], routing[task], task, seed, arm, repeat, args.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--out-dir', required=True)
    init.add_argument('--input-protocol', default='improve/results/edge_path_attention_20260908/protocol.json')
    init.add_argument('--mi-reference', default='improve/results/smooth_path_attention_20260908')
    init.add_argument('--lnc-reference', default='improve/results/smooth_path_attention_lnc_20260908')
    init.add_argument('--smoke', action='store_true')
    worker = sub.add_parser('run')
    worker.add_argument('--out-dir', required=True)
    worker.add_argument('--device', default='cuda:0')
    worker.add_argument('--shard-index', type=int, default=0)
    worker.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    configure()
    {'init': initialise, 'run': run}[args.command](args)


if __name__ == '__main__':
    main()
