"""Train and compare a replacement backbone without changing historical runs.

prepare -> run screen -> select screen -> run confirm -> select confirm ->
run outer -> report. Two GPUs can run one task each. Only the final, fixed
epoch refits open the historical outer validation arrays. No test is loaded.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F

from improve.network_feature_ablation import ExperimentModel
from improve.pair_exchange_modules import restrict_training_graph, remove_target_pairs, sample_queries
from improve.pencil_backbone import HeteroPencil, SupportGraph
from improve.run import set_seed


TASKS = {'LncDrug': dict(rna='lncRNA', epochs=150, channels=64),
         'MiDrug': dict(rna='miRNA', epochs=450, channels=256)}
ARMS = ('base', 'pencil')
SOURCE_FILES = ('backbone_replacement.py', 'pencil_backbone.py',
                'test_pencil_backbone.py', 'network_feature_ablation.py',
                'pair_exchange_modules.py', 'model.py', 'run.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sources():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def configurations(task):
    old_lr = (2.5e-4, 5e-4) if task == 'LncDrug' else (1.5e-5, 3e-5)
    return {
        'base': [dict(id='c'+str(i), lr=lr, optimizer='rmsprop',
                      weight_decay=1e-4 if task == 'LncDrug' else 0., schedule='constant')
                 for i, lr in enumerate(old_lr)],
        'pencil': [dict(id='c'+str(i), lr=lr, optimizer='adamw', weight_decay=.01,
                        schedule='cosine', width=64, depth=2, heads=4, budget=34,
                        dropout=.1, use_counts=True)
                   for i, lr in enumerate((3e-4, 1e-3))],
    }


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory; completed experiments are immutable')
    prior = json.loads((reference / 'protocol.json').read_text())
    validation = json.loads((reference / 'VALIDATION.json').read_text())
    if not validation['passed'] or validation['smoke']:
        raise ValueError('The input reference must be a verified full experiment')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        reference_dir=str(reference), reference_protocol_sha256=digest(reference/'protocol.json'),
        tasks=TASKS, arms=list(ARMS), configurations={t: configurations(t) for t in TASKS},
        screening_seed=666, confirmation_seeds=[0] if args.smoke else [0, 1],
        refit_seeds=[666] if args.smoke else [666, 0, 1],
        training={t: dict(v, epochs=3 if args.smoke else v['epochs']) for t, v in TASKS.items()},
        batch_size=256, query_fraction=.5, smoothing_window=2 if args.smoke else 5,
        evaluation_sampling_seed=427190, source_sha256=sources(), files={},
        selection='Screen each arm on seed 666 using the earliest peak trailing-five inner mean(AUC,AP). Freeze one recipe per arm, then confirm seeds 0 and 1. Outer refit epochs are the median of these three inner-selected epochs, rounded to nearest integer.',
        optimization='One optimizer update per epoch for both arms. PENCIL minibatches accumulate exactly a mean BCE over the same query rows. Architecture-appropriate optimizers are tuned with the same two-recipe budget; this is a pipeline comparison.',
        masking='Restrict target and reverse relations to fitting positives; remove the entire query batch of positives both ways before computing any sampled context or degree. Other-task edges follow the existing single-task protocol.',
        feature_policy='Identical frozen PCA/existing node inputs and unit-weight graph relations. PENCIL derives relation counts and local adjacency from the visible graph; no extra labels, pretrained external model or learned global node IDs.',
        no_test_evaluation=True,
        limitations=['The inner and outer validation partitions were already used in historical development.',
            'Initialization seeds do not measure variation across data partitions.',
            'This is a small heterogeneous adaptation of PENCIL, not an official benchmark reproduction.',
            'A fixed sampled context is used per validation pair; different context orderings can change predictions.',
            'The existing graph contains circRNA extensions and other-task supervision.',
            'No final choice of recipes or seeds may use outer metrics.'])
    if args.smoke:
        for task in TASKS:
            for arm in ARMS:
                p['configurations'][task][arm] = p['configurations'][task][arm][:1]
    out.mkdir(parents=True)
    for task in TASKS:
        dest = out / 'inputs' / task
        dest.mkdir(parents=True)
        for name in ('inner.npz', 'outer.npz', 'train_graph.pth'):
            src = reference / 'inputs' / task / name
            expected = prior['files']['inputs/'+task+'/'+name]
            if digest(src) != expected:
                raise ValueError('Reference input has changed: '+str(src))
            # Outer is copied as bytes; its arrays are not opened here.
            shutil.copy2(src, dest / name)
            p['files'][str((dest / name).relative_to(out))] = digest(dest / name)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    p['sha256'] = json_digest(p)
    write_json(out / 'protocol.json', p)
    print('FROZEN', out, flush=True)


def protocol(out):
    out = Path(out)
    p = json.loads((out / 'protocol.json').read_text())
    if p['sha256'] != json_digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('Protocol digest mismatch')
    if p['source_sha256'] != sources():
        raise ValueError('Frozen implementation changed; use a new experiment')
    for name, sha in p['files'].items():
        if digest(out / name) != sha:
            raise ValueError('Frozen input changed: '+name)
    return p


def stage_inputs(out, task, stage):
    directory = Path(out) / 'inputs' / task
    with np.load(directory / ('outer.npz' if stage == 'outer' else 'inner.npz')) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if any('test' in key for key in arrays):
        raise ValueError('Test arrays are not allowed')
    for fold in ('train', 'val'):
        pairs, labels = arrays['pairs_'+fold], arrays['y_'+fold]
        if pairs.shape != (2, len(labels)) or set(np.unique(labels)) != {0, 1}:
            raise ValueError('Invalid association arrays')
        if len(set(map(tuple, pairs.T))) != len(labels):
            raise ValueError('Duplicated labeled pair')
    if set(map(tuple, arrays['pairs_train'].T)) & set(map(tuple, arrays['pairs_val'].T)):
        raise ValueError('Fitting and held-out pairs overlap')
    data = torch.load(directory / 'train_graph.pth', map_location='cpu', weights_only=False)
    for nt in data.node_types:
        key = ('pca_x_' if nt in ('lncRNA', 'miRNA') else 'x_') + nt
        data[nt].x = torch.from_numpy(arrays[key])
    weights = {key: torch.from_numpy(arrays['weight_'+str(i)]) for i, key in enumerate(data.edge_types)}
    data, weights = restrict_training_graph(data, weights, task, TASKS[task]['rna'],
                                           arrays['pairs_train'], arrays['y_train'])
    held = set(map(tuple, arrays['pairs_val'].T))
    for key, flip in [((TASKS[task]['rna'], task, 'drug'), False),
                       (('drug', 'rev_'+task, TASKS[task]['rna']), True)]:
        edges = data[key].edge_index.flip(0) if flip else data[key].edge_index
        if held & set(map(tuple, edges.numpy().T)):
            raise ValueError('Held-out query leaked into the support graph')
    return arrays, data, weights


def make_model(data, weights, task, arm, cfg, seed, device):
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = data.clone().to(device)
    weights = {key: value.to(device) for key, value in weights.items()}
    if arm == 'base':
        model = ExperimentModel(data, task).to(device)
        key = (TASKS[task]['rna'], task, 'drug')
        with torch.no_grad():
            model(data.x_dict, data.edge_index_dict, data[key].edge_label_index, weights)
    elif arm == 'pencil':
        kwargs = {k: cfg[k] for k in ('width', 'depth', 'heads', 'budget', 'dropout', 'use_counts')}
        model = HeteroPencil({nt: x.shape[1] for nt, x in data.x_dict.items()},
                            data.edge_types, TASKS[task]['rna'], **kwargs).to(device)
    else:
        raise ValueError(arm)
    return model, data, weights


def metrics(y, logits, pairs):
    auc, ap = roc_auc_score(y, logits), average_precision_score(y, logits)
    per_drug = []
    for drug in np.unique(pairs[1]):
        mask = pairs[1] == drug
        if len(np.unique(y[mask])) == 2:
            per_drug.append(average_precision_score(y[mask], logits[mask]))
    return dict(auc=float(auc), ap=float(ap), criterion=float((auc+ap)/2),
                macro_drug_ap=float(np.mean(per_drug)) if per_drug else None,
                macro_drugs=len(per_drug))


@torch.no_grad()
def predict(model, data, weights, pairs, context, sampled, batch_size, intervention=None):
    model.eval()
    if isinstance(model, HeteroPencil):
        output = [model(data.x_dict, context, sampled[i:i+batch_size], intervention).cpu().numpy()
                  for i in range(0, sampled.shape[0], batch_size)]
        return np.concatenate(output)
    idx = torch.as_tensor(pairs, device=next(model.parameters()).device)
    return model(data.x_dict, data.edge_index_dict, idx, weights).cpu().numpy()


def lr_at(cfg, epoch, horizon):
    if cfg['schedule'] == 'constant':
        return cfg['lr']
    warmup = max(1, int(.05*horizon))
    if epoch <= warmup:
        return cfg['lr'] * epoch / warmup
    progress = (epoch-warmup) / max(1, horizon-warmup)
    return cfg['lr'] * (.05 + .95*.5*(1+math.cos(math.pi*progress)))


def choose_checkpoint(trace, window):
    """The two metrics always come from one actual checkpoint."""
    if len(trace) < window:
        raise ValueError('Trace shorter than the checkpoint smoothing window')
    values = np.array([r['criterion'] for r in trace], dtype=np.float64)
    smooth = np.array([values[i-window+1:i+1].mean() for i in range(window-1, len(values))])
    index = window - 1 + int(np.argmax(smooth))
    return index+1, float(smooth.max())


def train(arrays, data, weights, task, arm, cfg, seed, stage, epochs, p, device):
    started = time.monotonic()
    model, data, weights = make_model(data, weights, task, arm, cfg, seed, device)
    optimizer_class = torch.optim.RMSprop if cfg['optimizer'] == 'rmsprop' else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    labels = torch.from_numpy(arrays['y_train']).to(device)
    context = SupportGraph(data.x_dict, data.edge_index_dict, weights) if arm == 'pencil' else None
    sampled = context.sample(arrays['pairs_val'], TASKS[task]['rna'], cfg['budget'],
                             p['evaluation_sampling_seed']) if context is not None else None
    best, selected, saved, state = -np.inf, None, None, None
    trace = []
    horizon = p['training'][task]['epochs']
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = lr_at(cfg, epoch, horizon)
        query = sample_queries(arrays['y_train'], p['query_fraction'], (seed+1)*1000003+epoch)
        held = arrays['pairs_train'][:, query[arrays['y_train'][query] == 1]]
        edges, step_weights = remove_target_pairs(data.edge_index_dict, weights, task, TASKS[task]['rna'],
                                                   data['drug'].num_nodes, held)
        loss_value = 0.
        if arm == 'pencil':
            support = SupportGraph(data.x_dict, edges, step_weights)
            sampled_train = support.sample(arrays['pairs_train'][:, query], TASKS[task]['rna'], cfg['budget'],
                                            (seed+1)*130003+epoch)
            for start in range(0, len(query), p['batch_size']):
                indices = query[start:start+p['batch_size']]
                logits = model(data.x_dict, support, sampled_train[start:start+p['batch_size']])
                loss = F.binary_cross_entropy_with_logits(logits, labels[indices], reduction='sum') / len(query)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite PENCIL loss')
                loss.backward()
                loss_value += float(loss.detach())
            del support
        else:
            idx = torch.as_tensor(arrays['pairs_train'][:, query], device=device)
            logits = model(data.x_dict, edges, idx, step_weights)
            loss = F.binary_cross_entropy_with_logits(logits, labels[query])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite baseline loss')
            loss.backward()
            loss_value = float(loss.detach())
        if not all(bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.grad is not None):
            raise FloatingPointError('Non-finite gradients')
        optimizer.step()
        row = dict(epoch=epoch, loss=loss_value, queries=len(query), lr=lr_at(cfg, epoch, horizon))
        if stage != 'outer':
            prediction = predict(model, data, weights, arrays['pairs_val'], context, sampled, p['batch_size'])
            row.update(metrics(arrays['y_val'], prediction, arrays['pairs_val']))
            trace.append(row)
            if epoch >= p['smoothing_window']:
                criterion = float(np.mean([r['criterion'] for r in trace[-p['smoothing_window']:]]))
                if criterion > best:
                    best, selected, saved = criterion, epoch, prediction.copy()
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            trace.append(row)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print('EPOCH', stage, task, arm, cfg['id'], seed, epoch, 'loss', round(loss_value, 5),
                  'inner', round(row.get('criterion', float('nan')), 6),
                  'seconds', round(time.monotonic()-started, 1), flush=True)
    if stage == 'outer':
        selected = epochs
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        expected, score = choose_checkpoint(trace, p['smoothing_window'])
        if expected != selected or abs(score-best) > 1e-12:
            raise AssertionError('Checkpoint selector drifted')
    model.load_state_dict(state)
    prediction = predict(model, data, weights, arrays['pairs_val'], context, sampled, p['batch_size'])
    error = 0.
    if saved is not None:
        np.testing.assert_allclose(prediction, saved, atol=5e-5, rtol=5e-5)
        error = float(np.max(np.abs(prediction-saved)))
    outputs = dict(val=prediction)
    if arm == 'pencil':
        for intervention in ('no_counts', 'no_graph'):
            outputs[intervention] = predict(model, data, weights, arrays['pairs_val'], context, sampled,
                                            p['batch_size'], intervention)
    result = dict(task=task, arm=arm, config=cfg, config_id=cfg['id'], seed=seed, stage=stage,
        **metrics(arrays['y_val'], prediction, arrays['pairs_val']),
        selected_epoch=selected, selection_score=best if stage != 'outer' else None,
        trained_epochs=epochs, params=sum(v.numel() for v in model.parameters()),
        trainable_params=sum(v.numel() for v in model.parameters() if v.requires_grad),
        restore_max_abs=error, elapsed_seconds=time.monotonic()-started,
        protocol_sha256=p['sha256'], device=device)
    return result, outputs, trace, state


def directory(out, stage, task, arm, cfg, seed):
    return Path(out) / stage / task / arm / cfg / ('s'+str(seed))


def checked_result(dest, p):
    r = json.loads((Path(dest) / 'result.json').read_text())
    if r['protocol_sha256'] != p['sha256']:
        raise ValueError('Result belongs to another experiment')
    for name, sha in r['artifacts_sha256'].items():
        if digest(Path(dest) / name) != sha:
            raise ValueError('Result artifact changed')
    return r


def decision(out, name, p):
    obj = json.loads((Path(out) / name).read_text())
    if obj['protocol_sha256'] != p['sha256'] or obj['sha256'] != json_digest({k: v for k, v in obj.items() if k != 'sha256'}):
        raise ValueError('Frozen decision changed')
    for name, sha in obj['results_sha256'].items():
        if digest(Path(out) / name) != sha:
            raise ValueError('Selection evidence changed')
    return obj


def jobs(out, p, stage):
    chosen = decision(out, 'PARAMETERS.json', p)['chosen'] if stage != 'screen' else None
    refits = decision(out, 'REFIT.json', p)['chosen'] if stage == 'outer' else None
    for task in p['tasks']:
        for arm in p['arms']:
            configs = p['configurations'][task][arm] if stage == 'screen' else [chosen[task][arm]]
            seeds = [p['screening_seed']] if stage == 'screen' else p['confirmation_seeds'] if stage == 'confirm' else p['refit_seeds']
            for cfg in configs:
                for seed in seeds:
                    epochs = refits[task][arm]['epochs'] if stage == 'outer' else p['training'][task]['epochs']
                    yield task, arm, cfg, seed, epochs


def run(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    # Resolve required frozen decisions before opening a stage's inputs.
    planned = list(jobs(out, p, args.stage))
    for task in ([args.task] if args.task else p['tasks']):
        with (out / (task+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            arrays, data, weights = stage_inputs(out, task, args.stage)
            for t, arm, cfg, seed, epochs in planned:
                if t != task or (args.arm and args.arm != arm):
                    continue
                dest = directory(out, args.stage, task, arm, cfg['id'], seed)
                if (dest / 'result.json').exists():
                    checked_result(dest, p)
                    print('SKIP', args.stage, task, arm, cfg['id'], seed, flush=True)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                print('START', args.stage, task, arm, cfg, seed, flush=True)
                r, outputs, trace, state = train(arrays, data, weights, task, arm, cfg, seed, args.stage,
                                                 epochs, p, args.device)
                np.savez_compressed(dest / 'scores.npz', **outputs, y_val=arrays['y_val'], pairs_val=arrays['pairs_val'])
                pd.DataFrame(trace).to_csv(dest / 'trace.csv', index=False)
                torch.save(state, dest / 'checkpoint.pt')
                r['artifacts_sha256'] = {name: digest(dest/name) for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                write_json(dest / 'result.json', r)
                print('DONE', task, arm, cfg['id'], seed, r['auc'], r['ap'], 'epoch', r['selected_epoch'], flush=True)


def select(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    filename = 'PARAMETERS.json' if args.stage == 'screen' else 'REFIT.json'
    if args.stage not in ('screen', 'confirm'):
        raise ValueError('Outer results cannot select a model')
    if (out / filename).exists():
        print(json.dumps(decision(out, filename, p)['chosen'], indent=2), flush=True)
        return
    obj = dict(protocol_sha256=p['sha256'], chosen={}, results_sha256={})
    selected = decision(out, 'PARAMETERS.json', p)['chosen'] if args.stage == 'confirm' else None
    for task in p['tasks']:
        obj['chosen'][task] = {}
        for arm in p['arms']:
            rows = []
            if args.stage == 'screen':
                locations = [directory(out, 'screen', task, arm, cfg['id'], p['screening_seed'])
                             for cfg in p['configurations'][task][arm]]
            else:
                cfg = selected[task][arm]
                locations = [directory(out, 'screen', task, arm, cfg['id'], p['screening_seed'])]
                locations += [directory(out, 'confirm', task, arm, cfg['id'], seed) for seed in p['confirmation_seeds']]
            for dest in locations:
                rows.append(checked_result(dest, p))
                obj['results_sha256'][str((dest / 'result.json').relative_to(out))] = digest(dest / 'result.json')
            if args.stage == 'screen':
                winner = sorted(rows, key=lambda r: (-r['selection_score'], r['config_id']))[0]
                obj['chosen'][task][arm] = winner['config']
            else:
                obj['chosen'][task][arm] = dict(config=selected[task][arm],
                    epochs=int(np.rint(np.median([r['selected_epoch'] for r in rows]))),
                    inner_auc=float(np.mean([r['auc'] for r in rows])),
                    inner_ap=float(np.mean([r['ap'] for r in rows])))
    obj['sha256'] = json_digest(obj)
    write_json(out / filename, obj)
    print(json.dumps(obj['chosen'], indent=2), flush=True)


def report(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    rows, audits = [], []
    for stage in ('screen', 'confirm', 'outer'):
        for task, arm, cfg, seed, epochs in jobs(out, p, stage):
            dest = directory(out, stage, task, arm, cfg['id'], seed)
            r = checked_result(dest, p)
            arrays, data, weights = stage_inputs(out, task, stage)
            with np.load(dest / 'scores.npz') as archive:
                saved = archive['val'].copy()
                np.testing.assert_array_equal(archive['y_val'], arrays['y_val'])
                np.testing.assert_array_equal(archive['pairs_val'], arrays['pairs_val'])
                for intervention in ('no_counts', 'no_graph'):
                    if intervention in archive:
                        for key, value in metrics(arrays['y_val'], archive[intervention], arrays['pairs_val']).items():
                            r[intervention+'_'+key] = value
            for key, value in metrics(arrays['y_val'], saved, arrays['pairs_val']).items():
                if value is not None:
                    np.testing.assert_allclose(r[key], value, atol=1e-12, rtol=0)
            trace = pd.read_csv(dest / 'trace.csv')
            np.testing.assert_array_equal(trace.epoch, np.arange(1, epochs+1))
            if stage == 'outer':
                if any(name in trace for name in ('auc', 'ap', 'criterion')) or r['selected_epoch'] != epochs:
                    raise AssertionError('Outer labels entered the training selection loop')
            else:
                epoch, smooth = choose_checkpoint(trace.to_dict('records'), p['smoothing_window'])
                if epoch != r['selected_epoch']:
                    raise AssertionError('Wrong validation checkpoint')
                np.testing.assert_allclose(smooth, r['selection_score'], atol=1e-12, rtol=0)
            model, gpu_data, gpu_weights = make_model(data, weights, task, arm, cfg, seed, args.device)
            model.load_state_dict(torch.load(dest / 'checkpoint.pt', map_location='cpu', weights_only=False))
            context = SupportGraph(gpu_data.x_dict, gpu_data.edge_index_dict, gpu_weights) if arm == 'pencil' else None
            sampled = context.sample(arrays['pairs_val'], TASKS[task]['rna'], cfg['budget'], p['evaluation_sampling_seed']) if context else None
            replay = predict(model, gpu_data, gpu_weights, arrays['pairs_val'], context, sampled, p['batch_size'])
            np.testing.assert_allclose(replay, saved, atol=5e-5, rtol=5e-5)
            audits.append(dict(path=str(dest.relative_to(out)), max_abs=float(np.max(np.abs(replay-saved)))))
            rows.append(r)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'results.csv', index=False)
    outer = frame[frame.stage == 'outer']
    outer.groupby(['task', 'arm'])[['auc', 'ap', 'macro_drug_ap']].agg(['mean', 'std']).to_csv(out / 'outer_summary.csv')
    deltas = []
    for task in p['tasks']:
        block = outer[outer.task == task].set_index(['arm', 'seed'])
        for metric in ('auc', 'ap', 'macro_drug_ap'):
            values = np.array([block.loc[('pencil', s), metric]-block.loc[('base', s), metric] for s in p['refit_seeds']])
            deltas.append(dict(task=task, metric=metric, mean_delta=float(values.mean()),
                std_delta=float(values.std(ddof=1)) if len(values)>1 else 0., positive_seeds=int((values>0).sum()), seeds=len(values)))
    pd.DataFrame(deltas).to_csv(out / 'paired_deltas.csv', index=False)
    write_json(out / 'VALIDATION.json', dict(passed=True, smoke=p['smoke'], completed_runs=len(rows),
        protocol_sha256=p['sha256'], no_test_evaluation=True, all_checkpoints_replayed=True, replays=audits))
    lines = ['# 骨干替换：异质 PENCIL 与原 GeneralConv', '',
        '参数由内部验证选定，三个初始化固定轮数重训后评价历史外层验证。AUC/AP 来自同一个模型。',
        '外层验证曾用于历史开发，以下是探索结果，不是全新独立测试。', '',
        '| 任务 | 架构 | AUC 均值 | AP 均值 | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for arm in p['arms']:
            b = outer[(outer.task == task) & (outer.arm == arm)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f |' % (task, arm, b.auc.mean(), b.ap.mean(), b.macro_drug_ap.mean()))
    lines += ['', '| 任务 | 指标 | PENCIL − base | 正增益种子 |', '|---|---|---:|---:|']
    for r in deltas:
        lines.append('| %s | %s | %+.6f | %d/%d |' % (r['task'], r['metric'], r['mean_delta'], r['positive_seeds'], r['seeds']))
    lines += ['', '完整试验配置、所有试次、逐轮轨迹、检查点、预测及独立推理核验均保存在本目录。',
        'PENCIL 是 2026 年预印本的紧凑异质适配；本实现加入关系类型和支持图计数，不宣称复现原论文成绩或提出新算法。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(outer.groupby(['task', 'arm'])[['auc', 'ap', 'macro_drug_ap']].mean().to_string(), flush=True)
    print('VERIFIED', len(rows), 'checkpoints', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'select', 'report'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--reference-dir', default='improve/results/additive_search_20260910')
    parser.add_argument('--stage', choices=('screen', 'confirm', 'outer'), default='screen')
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    {'prepare': prepare, 'run': run, 'select': select, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
