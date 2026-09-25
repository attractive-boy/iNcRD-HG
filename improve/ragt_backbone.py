"""Train and compare the structural-augmented heterogeneous graph transformer.

prepare -> run screen -> select screen -> run confirm -> select confirm ->
run outer -> report. The frozen inputs, query-masked support graph, cross-fitted
pair features and the inner/outer partitions are exactly those of the parent
backbone-replacement experiment. No test array is loaded.

The new arm `sahgt` combines relation-aware HGT attention with per-relation
degree encodings and the leak-free structural pair features used by the TabM
arm. It is compared, on identical outer rows and seeds, with the frozen `base`
GeneralConv arm and the TabM arm.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from improve import backbone_replacement as parent
from improve.graph_tabm_modules import crossfit_features, fit_transform
from improve.pair_exchange_modules import remove_target_pairs, sample_queries
from improve.ragt_modules import StructuralHGT


ARMS = ('sahgt',)
OWN = ('ragt_backbone.py', 'ragt_modules.py', 'test_ragt_backbone.py',
       'graph_tabm_modules.py')
WIDTH = {'LncDrug': 128, 'MiDrug': 256}


def sources():
    return dict(parent.sources(),
                **{name: parent.digest(Path(__file__).with_name(name)) for name in OWN})


def configurations(task):
    if task == 'LncDrug':
        return [dict(id='c'+str(i), lr=lr, weight_decay=wd, width=width, depth=1,
                     heads=4, dropout=drop, fusion='add', gate_init=gate)
                for i, (lr, wd, width, drop, gate) in enumerate([
                    (1e-3, 1e-4, 96, .1, -2.), (5e-4, 1e-4, 128, .1, -2.),
                    (1e-3, 1e-3, 128, .2, -2.), (2e-3, 1e-4, 128, .1, -2.)])]
    return [dict(id='c'+str(i), lr=lr, weight_decay=wd, width=width, depth=1,
                 heads=4, dropout=drop, fusion='add', gate_init=gate)
            for i, (lr, wd, width, drop, gate) in enumerate([
                (2e-3, 1e-4, 256, .1, -4.), (2e-3, 1e-3, 256, .1, -6.),
                (2e-3, 1e-4, 384, .1, -4.), (3e-3, 1e-4, 128, .1, -4.)])]


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory; completed experiments are immutable')
    ref = parent.protocol(reference)
    if ref['smoke']:
        raise ValueError('Use the full parent inputs')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        reference_dir=str(reference), reference_protocol_sha256=ref['sha256'],
        tasks=list(parent.TASKS), arms=list(ARMS),
        configurations={t: configurations(t) for t in parent.TASKS},
        screening_seed=666, confirmation_seeds=[0] if args.smoke else [0, 1],
        refit_seeds=[666] if args.smoke else [666, 0, 1],
        epochs={t: (3 if args.smoke else c['epochs']) for t, c in parent.TASKS.items()},
        batch_size=512, query_fraction=.5, folds=5, fold_seed=20260910,
        smoothing_window=2 if args.smoke else 5, source_sha256=sources(),
        selection='Two learning rates using seed 666 and the earliest best trailing-five inner mean(AUC,AP); freeze one recipe per task; confirm seeds 0/1; refit for the median selected inner epoch. Outer results never select a recipe.',
        optimization='AdamW, one optimizer update per epoch on a stratified query half. Architecture-appropriate optimizers are tuned with the same two-recipe budget; this is a pipeline comparison.',
        masking='Target and reverse relations are restricted to fitting positives; the entire query batch of positives is removed both ways before the graph encoder or the degree encoding runs. Other-task edges follow the existing single-task protocol.',
        feature_policy='Identical frozen PCA/drug/circRNA node inputs. The 36 structural pair features are the parent five-fold cross-fitted statistics excluding each row fold of target positives; the graph transformer sees only the current visible support graph. No learned global node IDs, external labels or pretrained model are added.',
        no_test_evaluation=True,
        limitations=['The inner and outer validation partitions were already used in historical development.',
            'Initialization seeds do not measure variation across data partitions.',
            'This is a compact adaptation of HGT with degree encodings, not an official benchmark reproduction.',
            'The existing graph contains circRNA extensions and other-task supervision.',
            'No final choice of recipe or seed may use outer metrics.'])
    if args.smoke:
        for task in p['tasks']:
            p['configurations'][task] = p['configurations'][task][:1]
    out.mkdir(parents=True)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    p['sha256'] = parent.json_digest(p)
    parent.write_json(out / 'protocol.json', p)
    print('FROZEN', out, flush=True)


def protocol(out):
    out = Path(out)
    p = json.loads((out / 'protocol.json').read_text())
    if p['sha256'] != parent.json_digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('Protocol digest mismatch')
    if p['source_sha256'] != sources():
        raise ValueError('Frozen implementation changed; use a new experiment')
    ref = parent.protocol(p['reference_dir'])
    if ref['sha256'] != p['reference_protocol_sha256']:
        raise ValueError('Parent input protocol changed')
    return p


def features(out, p, task, stage):
    """Cross-fitted structural pair features, cached per phase and task."""
    phase = 'outer' if stage == 'outer' else 'inner'
    path = Path(out) / 'features' / phase / task
    if not (path / 'manifest.json').exists():
        arrays, graph, _ = parent.stage_inputs(p['reference_dir'], task, phase)
        raw, val, names, folds = crossfit_features(
            graph, task, arrays['pairs_train'], arrays['y_train'], arrays['pairs_val'],
            p['folds'], p['fold_seed'])
        x, xv, transform = fit_transform(raw, val)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / 'arrays.npz', x_train=x, x_val=xv, raw_train=raw,
            raw_val=val, y_train=arrays['y_train'], y_val=arrays['y_val'],
            pairs_train=arrays['pairs_train'], pairs_val=arrays['pairs_val'], fold_ids=folds)
        with (path / 'transform.pkl').open('wb') as stream:
            pickle.dump(transform, stream, protocol=4)
        parent.write_json(path / 'manifest.json', dict(protocol_sha256=p['sha256'],
            names=names, files={name: parent.digest(path / name)
                                for name in ('arrays.npz', 'transform.pkl')}))
        print('FEATURES', phase, task, x.shape, xv.shape, flush=True)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['protocol_sha256'] != p['sha256']:
        raise ValueError('Feature cache protocol mismatch')
    for name, sha in manifest['files'].items():
        if parent.digest(path / name) != sha:
            raise ValueError('Feature cache changed')
    with np.load(path / 'arrays.npz') as archive:
        result = {key: archive[key].copy() for key in archive.files}
    result['feature_manifest_sha256'] = parent.digest(path / 'manifest.json')
    return result


def load_graph(p, task, stage):
    """Visible support graph and unit weights on the current device."""
    phase = 'outer' if stage == 'outer' else 'inner'
    arrays, data, weights = parent.stage_inputs(p['reference_dir'], task, phase)
    device = torch.device('cpu')
    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {key: data[key].edge_index.to(device) for key in data.edge_types}
    weight_dict = {key: value.to(device) for key, value in weights.items()}
    return x_dict, edge_index_dict, weight_dict, {nt: data[nt].x.shape[1] for nt in data.node_types}, list(data.edge_types)


def model_for(task, cfg, node_dims, edge_types, pair_dim, seed, device):
    parent.set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kwargs = dict(width=cfg['width'], depth=cfg['depth'], heads=cfg['heads'],
                  dropout=cfg['dropout'], fusion=cfg.get('fusion', 'concat'),
                  gate_init=cfg.get('gate_init', 0.))
    return StructuralHGT(node_dims, edge_types, parent.TASKS[task]['rna'], pair_dim,
                         **kwargs).to(device)


def train(a, graph, task, arm, cfg, seed, stage, epochs, p, device):
    started = time.monotonic()
    x_dict, edge_index_dict, weights, node_dims, edge_types = graph
    x_dict = {nt: value.to(device) for nt, value in x_dict.items()}
    edge_index_dict = {key: value.to(device) for key, value in edge_index_dict.items()}
    weights = {key: value.to(device) for key, value in weights.items()}
    feat = torch.from_numpy(a['x_train']).to(device)
    feat_val = torch.from_numpy(a['x_val']).to(device)
    pairs_train = torch.from_numpy(a['pairs_train']).to(device)
    pairs_val = torch.from_numpy(a['pairs_val']).to(device)
    labels = torch.from_numpy(a['y_train']).to(device)
    model = model_for(task, cfg, node_dims, edge_types, a['x_train'].shape[1], seed, device)
    model.eval()
    with torch.no_grad():
        model(x_dict, edge_index_dict, pairs_train[:, :2], feat[:2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    rna, n_drug = parent.TASKS[task]['rna'], x_dict['drug'].size(0)
    trace, best, selected, state, saved = [], -np.inf, None, None, None
    for epoch in range(1, epochs+1):
        query = sample_queries(a['y_train'], p['query_fraction'], (seed+1)*1000003+epoch)
        held = a['pairs_train'][:, query[a['y_train'][query] == 1]]
        edges, step_weights = remove_target_pairs(edge_index_dict, weights, task, rna, n_drug, held)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_dict, edges, pairs_train[:, query], feat[query])
        loss = F.binary_cross_entropy_with_logits(logits, labels[query])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite SA-HGT loss')
        loss.backward()
        if not all(bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.grad is not None):
            raise FloatingPointError('Non-finite gradients')
        optimizer.step()
        row = dict(epoch=epoch, loss=float(loss.detach()), queries=len(query))
        if stage != 'outer':
            model.eval()
            with torch.no_grad():
                prediction = model(x_dict, edge_index_dict, pairs_val, feat_val).cpu().numpy()
            row.update(parent.metrics(a['y_val'], prediction, a['pairs_val']))
            trace.append(row)
            if epoch >= p['smoothing_window']:
                criterion = float(np.mean([r['criterion'] for r in trace[-p['smoothing_window']:]]))
                if criterion > best:
                    best, selected, saved = criterion, epoch, prediction.copy()
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            trace.append(row)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print('EPOCH', stage, task, arm, cfg['id'], seed, epoch, 'loss', round(float(loss.detach()), 5),
                  'inner', round(row.get('criterion', float('nan')), 6),
                  'seconds', round(time.monotonic()-started, 1), flush=True)
    if stage == 'outer':
        selected = epochs
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        expected, score = parent.choose_checkpoint(trace, p['smoothing_window'])
        if expected != selected or abs(score-best) > 1e-12:
            raise AssertionError('Checkpoint selector drifted')
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        prediction = model(x_dict, edge_index_dict, pairs_val, feat_val).cpu().numpy()
    error = 0.
    if saved is not None:
        np.testing.assert_allclose(prediction, saved, atol=5e-5, rtol=5e-5)
        error = float(np.max(np.abs(prediction-saved)))
    result = dict(task=task, arm=arm, config=cfg, config_id=cfg['id'], seed=seed, stage=stage,
        **parent.metrics(a['y_val'], prediction, a['pairs_val']), selected_epoch=selected,
        selection_score=best if stage != 'outer' else None, trained_epochs=epochs,
        params=sum(v.numel() for v in model.parameters()),
        trainable_params=sum(v.numel() for v in model.parameters() if v.requires_grad),
        restore_max_abs=error, elapsed_seconds=time.monotonic()-started,
        protocol_sha256=p['sha256'], feature_manifest_sha256=a['feature_manifest_sha256'],
        device=device)
    return result, prediction, trace, state


def jobs(out, p, stage):
    selected = parent.decision(out, 'PARAMETERS.json', p)['chosen'] if stage != 'screen' else None
    refits = parent.decision(out, 'REFIT.json', p)['chosen'] if stage == 'outer' else None
    for task in p['tasks']:
        for arm in p['arms']:
            configs = p['configurations'][task] if stage == 'screen' else [selected[task][arm]]
            seeds = [p['screening_seed']] if stage == 'screen' else p['confirmation_seeds'] if stage == 'confirm' else p['refit_seeds']
            for cfg in configs:
                for seed in seeds:
                    epochs = refits[task][arm]['epochs'] if stage == 'outer' else p['epochs'][task]
                    yield task, arm, cfg, seed, epochs


def run(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    planned = list(jobs(out, p, args.stage))
    for task in ([args.task] if args.task else p['tasks']):
        with (out / (task+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            a = features(out, p, task, args.stage)
            graph = load_graph(p, task, args.stage)
            for t, arm, cfg, seed, epochs in planned:
                if t != task or (args.arm and args.arm != arm):
                    continue
                dest = parent.directory(out, args.stage, task, arm, cfg['id'], seed)
                if (dest / 'result.json').exists():
                    parent.checked_result(dest, p)
                    print('SKIP', args.stage, task, arm, cfg['id'], seed, flush=True)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                print('START', args.stage, task, arm, cfg, seed, flush=True)
                r, prediction, trace, state = train(a, graph, task, arm, cfg, seed, args.stage,
                                                    epochs, p, args.device)
                np.savez_compressed(dest / 'scores.npz', val=prediction, y_val=a['y_val'], pairs_val=a['pairs_val'])
                pd.DataFrame(trace).to_csv(dest / 'trace.csv', index=False)
                torch.save(state, dest / 'checkpoint.pt')
                r['artifacts_sha256'] = {name: parent.digest(dest / name)
                                         for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                parent.write_json(dest / 'result.json', r)
                print('DONE', task, arm, cfg['id'], seed, r['auc'], r['ap'], 'epoch', r['selected_epoch'], flush=True)


def select(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if args.stage not in ('screen', 'confirm'):
        raise ValueError('Outer results cannot select a model')
    filename = 'PARAMETERS.json' if args.stage == 'screen' else 'REFIT.json'
    if (out / filename).exists():
        print(json.dumps(parent.decision(out, filename, p)['chosen'], indent=2), flush=True)
        return
    selected = parent.decision(out, 'PARAMETERS.json', p)['chosen'] if args.stage == 'confirm' else None
    obj = dict(protocol_sha256=p['sha256'], chosen={}, results_sha256={})
    for task in p['tasks']:
        obj['chosen'][task] = {}
        for arm in p['arms']:
            if args.stage == 'screen':
                locations = [parent.directory(out, 'screen', task, arm, cfg['id'], p['screening_seed'])
                             for cfg in p['configurations'][task]]
            else:
                cfg = selected[task][arm]
                locations = [parent.directory(out, 'screen', task, arm, cfg['id'], p['screening_seed'])]
                locations += [parent.directory(out, 'confirm', task, arm, cfg['id'], s) for s in p['confirmation_seeds']]
            rows = [parent.checked_result(dest, p) for dest in locations]
            for dest in locations:
                obj['results_sha256'][str((dest / 'result.json').relative_to(out))] = parent.digest(dest / 'result.json')
            if args.stage == 'screen':
                winner = sorted(rows, key=lambda r: (-r['selection_score'], r['config_id']))[0]
                obj['chosen'][task][arm] = winner['config']
            else:
                obj['chosen'][task][arm] = dict(config=selected[task][arm],
                    epochs=int(np.rint(np.median([r['selected_epoch'] for r in rows]))),
                    inner_auc=float(np.mean([r['auc'] for r in rows])),
                    inner_ap=float(np.mean([r['ap'] for r in rows])))
    obj['sha256'] = parent.json_digest(obj)
    parent.write_json(out / filename, obj)
    print(json.dumps(obj['chosen'], indent=2), flush=True)


def _reference_outer(path):
    frame = pd.read_csv(path)
    return frame[(frame.stage == 'outer')].copy()


def report(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    rows, checks = [], []
    for stage in ('screen', 'confirm', 'outer'):
        for task, arm, cfg, seed, epochs in jobs(out, p, stage):
            dest = parent.directory(out, stage, task, arm, cfg['id'], seed)
            r = parent.checked_result(dest, p)
            a = features(out, p, task, stage)
            if r['feature_manifest_sha256'] != a['feature_manifest_sha256']:
                raise ValueError('Training features changed')
            with np.load(dest / 'scores.npz') as archive:
                saved = archive['val'].copy()
                np.testing.assert_array_equal(archive['y_val'], a['y_val'])
                np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
            for key, value in parent.metrics(a['y_val'], saved, a['pairs_val']).items():
                if value is not None:
                    np.testing.assert_allclose(r[key], value, atol=1e-12, rtol=0)
            trace = pd.read_csv(dest / 'trace.csv')
            np.testing.assert_array_equal(trace.epoch, np.arange(1, epochs+1))
            if stage != 'outer':
                epoch, smooth = parent.choose_checkpoint(trace.to_dict('records'), p['smoothing_window'])
                if epoch != r['selected_epoch']:
                    raise ValueError('Wrong checkpoint epoch')
                np.testing.assert_allclose(smooth, r['selection_score'], atol=1e-12, rtol=0)
            elif any(key in trace for key in ('auc', 'ap', 'criterion')):
                raise ValueError('Outer evaluation used during training')
            graph = load_graph(p, task, stage)
            model = model_for(task, cfg, graph[3], graph[4], a['x_train'].shape[1], seed, args.device)
            model.load_state_dict(torch.load(dest / 'checkpoint.pt', map_location='cpu', weights_only=False))
            x_dict = {nt: v.to(args.device) for nt, v in graph[0].items()}
            edge_index_dict = {k: v.to(args.device) for k, v in graph[1].items()}
            feat_val = torch.from_numpy(a['x_val']).to(args.device)
            pairs_val = torch.from_numpy(a['pairs_val']).to(args.device)
            model.eval()
            with torch.no_grad():
                replay = model(x_dict, edge_index_dict, pairs_val, feat_val).cpu().numpy()
            np.testing.assert_allclose(replay, saved, atol=5e-5, rtol=5e-5)
            checks.append(dict(path=str(dest.relative_to(out)), max_abs=float(np.max(np.abs(replay-saved)))))
            rows.append(r)
    pd.DataFrame(rows).to_csv(out / 'results.csv', index=False)
    outer = pd.DataFrame([r for r in rows if r['stage'] == 'outer'])
    outer.groupby(['task', 'arm'])[['auc', 'ap', 'macro_drug_ap']].agg(['mean', 'std']).to_csv(out / 'outer_summary.csv')

    deltas = []
    comparisons = {'base': Path(p['reference_dir']) / 'results.csv',
                   'tabm': Path('improve/results/graph_tabm_20260910/results.csv')}
    for name, path in comparisons.items():
        if not path.exists():
            continue
        reference = _reference_outer(path)
        if name == 'tabm':
            reference = reference[reference.arm == 'tabm']
        elif 'base' in set(reference.arm):
            reference = reference[reference.arm == 'base']
        for task in p['tasks']:
            ref_block = reference[reference.task == task].set_index('seed')
            our_block = outer[(outer.task == task)].set_index('seed')
            for metric in ('auc', 'ap', 'macro_drug_ap'):
                values = []
                for seed in p['refit_seeds']:
                    if seed in ref_block.index and seed in our_block.index:
                        values.append(float(our_block.loc[seed, metric] - ref_block.loc[seed, metric]))
                if not values:
                    continue
                values = np.array(values)
                deltas.append(dict(task=task, metric=metric, reference=name, mean_delta=float(values.mean()),
                    std_delta=float(values.std(ddof=1)) if len(values) > 1 else 0.,
                    positive_seeds=int((values > 0).sum()), seeds=len(values)))
    pd.DataFrame(deltas).to_csv(out / 'paired_deltas.csv', index=False)
    parent.write_json(out / 'VALIDATION.json', dict(passed=True, smoke=p['smoke'], completed_runs=len(rows),
        protocol_sha256=p['sha256'], no_test_evaluation=True, all_checkpoints_replayed=True, replays=checks))

    lines = ['# 结构增强异构图 Transformer 骨干替换', '',
        '参数由内部验证选定，三个初始化固定轮数重训后评价历史外层验证。AUC/AP 来自同一个模型。',
        '外层验证曾用于历史开发，以下是探索结果，不是全新独立测试。', '',
        '| 任务 | 架构 | AUC 均值 | AP 均值 | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        b = outer[outer.task == task]
        lines.append('| %s | sa-hgt | %.6f | %.6f | %.6f |' % (task, b.auc.mean(), b.ap.mean(), b.macro_drug_ap.mean()))
    lines += ['', '| 任务 | 指标 | 参照 | sa-hgt − 参照 | 正增益种子 |', '|---|---|---|---:|---:|']
    for r in deltas:
        lines.append('| %s | %s | %s | %+.6f | %d/%d |' % (r['task'], r['metric'], r['reference'],
                                                            r['mean_delta'], r['positive_seeds'], r['seeds']))
    lines += ['', '完整配置、逐轮轨迹、检查点、预测与独立推理核验均保存在本目录。',
        '这是 HGT 的紧凑适配（关系注意力 + 度编码 + 交叉拟合结构特征），不宣称复现原论文成绩。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(outer.groupby(['task', 'arm'])[['auc', 'ap', 'macro_drug_ap']].mean().to_string(), flush=True)
    print('VERIFIED', len(rows), 'checkpoints', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'select', 'report'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--reference-dir', default='improve/results/backbone_replacement_20260910')
    parser.add_argument('--stage', choices=('screen', 'confirm', 'outer'), default='screen')
    parser.add_argument('--task', choices=parent.TASKS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    {'prepare': prepare, 'run': run, 'select': select, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
