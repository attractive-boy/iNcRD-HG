"""Three independent feature experiments, shared with MLP/ExtraTrees/GNN.

prepare freezes inputs and the protocol; run executes a subset of its fixed
matrix; summarize checks completeness and reports only three paired contrasts.
Use a separate --smoke output directory for two-epoch integration checks.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier

from improve.first_round_inputs import (ARMS, COMPARISONS, FOLDS, prepare_task)
from improve.network_feature_ablation import ExperimentModel, score
from improve.network_feature_inputs import digest_array, extract_drugs
from improve.run import TASKS, set_seed


MODELS = ('gnn', 'mlp', 'extratrees')
SEEDS = (666, 0, 1, 2, 3)
SOURCE_FILES = (
    'first_round_features.py', 'first_round_inputs.py',
    'network_feature_ablation.py', 'network_feature_inputs.py',
    'model.py', 'run.py', 'hybrid.py', 'gbdt.py', 'feature_extra.py', 'features.py',
)
METRICS = ('val_auc', 'val_aupr', 'val_criterion', 'test_auc', 'test_aupr', 'test_criterion')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_hashes():
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in SOURCE_FILES}


def protocol_spec(smoke=False):
    return {
        'version': 1, 'smoke': bool(smoke), 'tasks': list(TASKS),
        'models': list(MODELS), 'arms': list(ARMS),
        'comparisons': {k: list(v) for k, v in COMPARISONS.items()},
        'seeds': [666] if smoke else list(SEEDS),
        'task_settings': {
            task: dict(config, train_epochs=2 if smoke else config['epochs'] - 1)
            for task, config in TASKS.items()},
        'selection': {
            'neural': 'one checkpoint maximizing (validation AUC + validation AUPR) / 2',
            'ties': 'first epoch; strict improvement only',
            'early_stopping': False,
            'extratrees': 'one fit with fixed hyperparameters; no search',
            'test': 'evaluate only after the checkpoint is fixed; never select seeds or arms on test',
        },
        'gnn': {'architecture': 'released two-layer GeneralConv and concat decoder',
                'optimizer': 'RMSprop', 'alpha': 0.99, 'weight_decay': 0.0,
                'loss': 'unweighted BCEWithLogits', 'dropedge': 0, 'mc': 1},
        'mlp': {'hidden': 64, 'activation': 'ReLU', 'optimizer': 'Adam',
                'lr': 0.001, 'weight_decay': 0.0001, 'batch': 'full training batch',
                'loss': 'unweighted BCEWithLogits'},
        'extratrees': {'n_estimators': 8 if smoke else 500,
                      'min_samples_leaf': 3, 'max_features': 'sqrt',
                      'criterion': 'gini', 'bootstrap': False, 'class_weight': None},
        'expression': {'components': 64, 'method': 'training-node StandardScaler + PCA',
                       'statistics': ['raw_mean', 'raw_std_ddof0', 'raw_zero_fraction'],
                       'output': 'replace RNA input with 64 scaled PCs + 3 scaled statistics'},
        'graph': 'the frozen TRAINING graph is used for all three folds and all models',
        'training_cleanup': 'once before all arms: remove training copies of evaluation pairs '
                            'and duplicate training pairs; keep evaluation rows and all remaining negatives',
        'shared_inputs': 'same node vectors and same existing-edge weights; simple models read '
                         'complete endpoint vectors plus 14 fixed graph statistics, GNN reads '
                         'node vectors and the full heterogeneous training graph',
        'pooling_control': 'masked vs unmasked from the same checkpoint, tokenizer, padding '
                           'length and embedding dimension; base is not its control',
        'screen': 'per task/model: mean paired validation criterion delta > 0 and at least '
                  '4/5 seeds positive; descriptive screening, not a significance test',
        'uncertainty': 'initialization variability on one existing split; no independent '
                       'split or external validation is claimed',
    }


def prepare(args):
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        protocol = load_protocol(out)
        if protocol['spec'] != protocol_spec(args.smoke):
            raise ValueError('Existing protocol differs; use a new output directory')
        verify_files(out, protocol)
        print('Already prepared and verified:', out, flush=True)
        return
    if (out / 'runs').exists() and any((out / 'runs').iterdir()):
        raise ValueError('Refusing to prepare over existing results')
    if not args.model_dir:
        raise ValueError('--model-dir is required to generate the matched pooling controls')
    model_dir = Path(args.model_dir).resolve()
    before = source_hashes()
    spec = protocol_spec(args.smoke)
    model_files = {
        str(p.relative_to(model_dir)): sha256(p)
        for p in sorted(model_dir.rglob('*')) if p.is_file()
    }
    if not model_files:
        raise ValueError('No local ChemBERTa checkpoint files found')
    pool_audit = extract_drugs(args.net_dir, model_dir, out, args.device)
    if pool_audit['padding_invariance_max_abs'] > 2e-5:
        raise ValueError('Real-model padding invariance failed')
    audits = {task: prepare_task(args.run_dir, args.net_dir, out, task) for task in spec['tasks']}
    if before != source_hashes():
        raise ValueError('Source changed during preparation')
    files = list((out / 'inputs').rglob('*')) + [out / 'drug_pooling.npz', out / 'drug_pooling_audit.json']
    inputs_sha = {str(p.relative_to(out)): sha256(p) for p in sorted(files) if p.is_file()}
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'spec': spec,
        'run_dir': str(Path(args.run_dir).resolve()), 'net_dir': str(Path(args.net_dir).resolve()),
        'source_sha256': before, 'input_files_sha256': inputs_sha,
        'pooling_model_files_sha256': model_files,
        'raw_network_files_sha256': {
            name: sha256(Path(args.net_dir) / name) for name in
            ('Drug.csv', 'drug_smiles.csv', 'lncRNA.csv', 'miRNA.csv', 'LncExp.csv',
             'MiExp.csv', 'drugSimMat.csv', 'LncSimMat_ken.csv', 'MiSimMat_ken.csv')},
        'versions': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'pandas', 'scikit-learn', 'torch', 'torch-geometric', 'transformers')},
        'split_sha256': {task: audit['released_split_fingerprints'] for task, audit in audits.items()},
    }
    protocol['protocol_sha256'] = json_digest(protocol)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    write_json(out / 'protocol.json', protocol)  # freeze after all inputs passed audit
    print('FROZEN', len(expected_tags(spec)), 'runs;', protocol['protocol_sha256'], flush=True)


def load_protocol(out):
    protocol = json.loads((Path(out) / 'protocol.json').read_text())
    recorded = protocol['protocol_sha256']
    body = {k: v for k, v in protocol.items() if k != 'protocol_sha256'}
    if json_digest(body) != recorded:
        raise ValueError('Frozen protocol has changed')
    if protocol['source_sha256'] != source_hashes():
        raise ValueError('Experiment code changed after preparation; use a new output directory')
    return protocol


def verify_files(out, protocol, prefix=None):
    for name, expected in protocol['input_files_sha256'].items():
        if prefix is not None and not name.startswith(prefix):
            continue
        if sha256(Path(out) / name) != expected:
            raise ValueError('Prepared input changed: ' + name)


def expected_tags(spec):
    return ['%s_%s_%s_s%d' % (task, model, arm, seed)
            for task in spec['tasks'] for model in spec['models']
            for arm in spec['arms'] for seed in spec['seeds']]


def metric_record(y, val, test):
    va, vp = score(y['val'], val)
    ta, tp = score(y['test'], test)
    return dict(val_auc=va, val_aupr=vp, val_criterion=(va + vp) / 2,
                test_auc=ta, test_aupr=tp, test_criterion=(ta + tp) / 2)


class PairMLP(torch.nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_features, hidden),
                                       torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).view(-1)


def run_neural(args, spec, task, model_name, seed, arrays, data):
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = spec['task_settings'][task]
    key = (config['rna'], task, 'drug')
    y = {f: arrays['y_' + f] for f in FOLDS}
    target = torch.as_tensor(y['train'], device=args.device)
    start = time.monotonic()
    if model_name == 'gnn':
        graph = data.clone()
        for nt in graph.node_types:
            graph[nt].x = torch.from_numpy(arrays['x_' + nt])
        graph = graph.to(args.device)
        weights = {k: torch.from_numpy(arrays['weight_' + str(i)]).to(args.device)
                   for i, k in enumerate(graph.edge_types)}
        indices = {f: torch.from_numpy(arrays['pairs_' + f]).to(args.device) for f in FOLDS}
        model = ExperimentModel(graph, task).to(args.device)

        def forward(fold):
            return model(graph.x_dict, graph.edge_index_dict, indices[fold], weights)

        with torch.no_grad():
            forward('train')  # materialize every lazy parameter before optimizer creation
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'], alpha=spec['gnn']['alpha'])
    else:
        X = {f: torch.from_numpy(arrays['phi_' + f]).to(args.device) for f in FOLDS}
        settings = spec['mlp']
        model = PairMLP(X['train'].shape[1], settings['hidden']).to(args.device)

        def forward(fold):
            return model(X[fold])

        optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    best, best_epoch, best_state = -np.inf, None, None
    trace = []
    for epoch in range(1, config['train_epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(forward('train'), target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pv = forward('val').cpu().numpy()
        auc, ap = score(y['val'], pv)
        criterion = (auc + ap) / 2
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), val_auc=auc,
                          val_aupr=ap, criterion=criterion))
        if criterion > best:
            best, best_epoch = criterion, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 100 == 0 or epoch == config['train_epochs']:
            print(task, model_name, seed, 'epoch', epoch, 'val %.5f/%.5f' % (auc, ap), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv, pt = [forward(f).cpu().numpy() for f in ('val', 'test')]
    metrics = metric_record(y, pv, pt)
    if abs(metrics['val_criterion'] - best) > 2e-3:
        raise AssertionError('Restored checkpoint does not reproduce validation selection')
    result = dict(metrics, selected_epoch=best_epoch, trained_epochs=config['train_epochs'],
                  selection='maximum_mean_validation_auc_aupr_first_tie',
                  selection_criterion=best, restore_criterion_abs_error=abs(metrics['val_criterion'] - best),
                  params=sum(p.numel() for p in model.parameters()), prediction_kind='logit',
                  elapsed_seconds=time.monotonic() - start)
    return result, pv, pt, trace, best_state


def run_trees(args, spec, seed, arrays):
    start = time.monotonic()
    model = ExtraTreesClassifier(**spec['extratrees'], n_jobs=args.cpu_threads, random_state=seed)
    model.fit(arrays['phi_train'], arrays['y_train'])
    pv, pt = [model.predict_proba(arrays['phi_' + f])[:, 1] for f in ('val', 'test')]
    metrics = metric_record({f: arrays['y_' + f] for f in FOLDS}, pv, pt)
    result = dict(metrics, selected_epoch=0, trained_epochs=0,
                  selection='fixed_hyperparameters_no_search', prediction_kind='probability',
                  elapsed_seconds=time.monotonic() - start)
    return result, pv, pt, [], model


def run(args):
    out = Path(args.out_dir).resolve()
    protocol = load_protocol(out)
    spec = protocol['spec']
    task = args.task
    if task not in spec['tasks']:
        raise ValueError('--task must be one of the frozen tasks')
    seeds = spec['seeds'] if args.seeds is None else args.seeds
    if not set(seeds).issubset(spec['seeds']) or len(set(seeds)) != len(seeds):
        raise ValueError('Run seeds must be a unique subset of the frozen seeds')
    for name in ('runs', 'scores', 'traces', 'checkpoints', 'locks'):
        (out / name).mkdir(exist_ok=True)
    verify_files(out, protocol, prefix='inputs/' + task + '/')
    graph = torch.load(out / 'inputs' / task / 'train_graph.pth', map_location='cpu')
    key = (TASKS[task]['rna'], task, 'drug')
    for arm in spec['arms']:
        with np.load(out / 'inputs' / task / (arm + '.npz'), allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        np.testing.assert_array_equal(graph[key].edge_label_index.numpy(), arrays['pairs_train'])
        np.testing.assert_array_equal(graph[key].edge_label.numpy(), arrays['y_train'])
        for model_name in args.models:
            for seed in seeds:
                tag = '%s_%s_%s_s%d' % (task, model_name, arm, seed)
                path = out / 'runs' / (tag + '.json')
                with (out / 'locks' / (tag + '.lock')).open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if path.exists():
                        result = json.loads(path.read_text())
                        if result['protocol_sha256'] != protocol['protocol_sha256']:
                            raise ValueError('Existing run belongs to another protocol: ' + tag)
                        for filename, digest in result['artifacts_sha256'].items():
                            if sha256(out / filename) != digest:
                                raise ValueError('Existing run artifact changed: ' + filename)
                        print('already complete', tag, flush=True)
                        continue
                    print('START', tag, flush=True)
                    if model_name == 'extratrees':
                        result, pv, pt, trace, state = run_trees(args, spec, seed, arrays)
                    else:
                        result, pv, pt, trace, state = run_neural(args, spec, task, model_name, seed, arrays, graph)
                    result.update(task=task, model=model_name, arm=arm, seed=seed,
                                  protocol_sha256=protocol['protocol_sha256'],
                                  device='cpu' if model_name == 'extratrees' else args.device,
                                  cpu_threads=args.cpu_threads,
                                  node_sha256={nt: digest_array(arrays['x_' + nt]) for nt in graph.node_types},
                                  pair_input_sha256={f: digest_array(arrays['phi_' + f]) for f in FOLDS},
                                  edge_weight_sha256={str(i): digest_array(arrays['weight_' + str(i)])
                                                      for i in range(len(graph.edge_types))})
                    prediction_path = out / 'scores' / (tag + '.npz')
                    np.savez_compressed(prediction_path, val=pv, test=pt,
                                        y_val=arrays['y_val'], y_test=arrays['y_test'],
                                        pairs_val=arrays['pairs_val'], pairs_test=arrays['pairs_test'])
                    artifacts = [prediction_path]
                    if trace:
                        trace_path = out / 'traces' / (tag + '.csv')
                        pd.DataFrame(trace).to_csv(trace_path, index=False)
                        artifacts.append(trace_path)
                    suffix = '.joblib' if model_name == 'extratrees' else '.pt'
                    checkpoint_path = out / 'checkpoints' / (tag + suffix)
                    if model_name == 'extratrees':
                        joblib.dump(state, checkpoint_path, compress=3)
                    else:
                        torch.save(state, checkpoint_path)
                    artifacts.append(checkpoint_path)
                    result['artifacts_sha256'] = {str(p.relative_to(out)): sha256(p) for p in artifacts}
                    write_json(path, result)  # last write is the completion marker
                    print('DONE', tag, 'val %.5f/%.5f test %.5f/%.5f' % (
                        result['val_auc'], result['val_aupr'], result['test_auc'], result['test_aupr']), flush=True)


def verify_results(out, protocol):
    spec = protocol['spec']
    expected = set(expected_tags(spec))
    found = {p.stem for p in (out / 'runs').glob('*.json')}
    if expected != found:
        raise ValueError('Run matrix incomplete or unexpected: missing=%s; extra=%s' %
                         (sorted(expected - found), sorted(found - expected)))
    verify_files(out, protocol)
    audits, labels = {}, {}
    for task in spec['tasks']:
        audits[task] = json.loads((out / 'inputs' / task / 'audit.json').read_text())
        with np.load(out / 'inputs' / task / 'base.npz', allow_pickle=False) as z:
            labels[task] = {k: z[k].copy() for k in ('y_val', 'y_test', 'pairs_val', 'pairs_test')}
    rows = []
    for tag in sorted(expected):
        result = json.loads((out / 'runs' / (tag + '.json')).read_text())
        if result['protocol_sha256'] != protocol['protocol_sha256']:
            raise ValueError('Mixed protocol in ' + tag)
        if tag != '%s_%s_%s_s%d' % (result['task'], result['model'], result['arm'], result['seed']):
            raise ValueError('Run identity mismatch: ' + tag)
        for filename, digest in result['artifacts_sha256'].items():
            if sha256(out / filename) != digest:
                raise ValueError('Run artifact changed: ' + filename)
        task, arm = result['task'], result['arm']
        arm_audit = audits[task]['arms'][arm]
        if result['node_sha256'] != arm_audit['node_sha256']:
            raise ValueError('Node inputs differ across models: ' + tag)
        if result['pair_input_sha256'] != arm_audit['pair_sha256']:
            raise ValueError('Pair inputs differ across models: ' + tag)
        expected_weights = {str(i): arm_audit['weight_sha256']['|'.join(k)]
                            for i, k in enumerate(audits[task]['edge_types'])}
        if result['edge_weight_sha256'] != expected_weights:
            raise ValueError('Edge weights differ across models: ' + tag)
        with np.load(out / 'scores' / (tag + '.npz'), allow_pickle=False) as z:
            for key, value in labels[task].items():
                np.testing.assert_array_equal(z[key], value)
            metrics = metric_record({'val': z['y_val'], 'test': z['y_test']}, z['val'], z['test'])
        for name, value in metrics.items():
            if abs(result[name] - value) > 1e-10:
                raise ValueError('Saved scores disagree with ' + name + ': ' + tag)
        if result['model'] != 'extratrees':
            trace = pd.read_csv(out / 'traces' / (tag + '.csv'))
            if len(trace) != spec['task_settings'][task]['train_epochs']:
                raise ValueError('Wrong fixed epoch budget: ' + tag)
            if not np.array_equal(trace['epoch'].values, np.arange(1, len(trace) + 1)):
                raise ValueError('Missing or repeated epochs: ' + tag)
            np.testing.assert_allclose(trace.criterion, (trace.val_auc + trace.val_aupr) / 2, atol=1e-12)
            selected = trace.loc[trace.criterion.idxmax()]
            if result['selected_epoch'] != int(selected.epoch):
                raise ValueError('Checkpoint selection did not use the fixed validation rule: ' + tag)
            if abs(result['selection_criterion'] - selected.criterion) > 1e-10:
                raise ValueError('Selected validation criterion mismatch: ' + tag)
        rows.append(result)
    return pd.DataFrame(rows)


def paired_summary(frame, spec):
    summary, paired, evidence = [], [], []
    for (task, model, arm), group in frame.groupby(['task', 'model', 'arm'], sort=False):
        row = dict(task=task, model=model, arm=arm, n_seeds=len(group))
        for metric in METRICS:
            row[metric + '_mean'] = float(group[metric].mean())
            row[metric + '_std'] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        summary.append(row)
    for task in spec['tasks']:
        for name, (control, treatment) in spec['comparisons'].items():
            decisions = {}
            for model in spec['models']:
                select = frame[(frame.task == task) & (frame.model == model)]
                a = select[select.arm == control].set_index('seed').loc[spec['seeds']]
                b = select[select.arm == treatment].set_index('seed').loc[spec['seeds']]
                row = dict(task=task, model=model, comparison=name, control=control,
                           treatment=treatment, n_seeds=len(a))
                for metric in METRICS:
                    values = b[metric] - a[metric]
                    row[metric + '_delta_mean'] = float(values.mean())
                    row[metric + '_delta_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    row[metric + '_positive_seeds'] = int((values > 0).sum())
                passed = (not spec['smoke'] and row['val_criterion_delta_mean'] > 0
                          and row['val_criterion_positive_seeds'] >= 4)
                row['validation_screen_pass'] = passed
                decisions[model] = passed
                paired.append(row)
            simple = decisions['mlp'] or decisions['extratrees']
            graph = decisions['gnn']
            if spec['smoke']:
                interpretation = '仅验证执行流程，不解释性能'
            elif simple and graph:
                interpretation = '简单模型与 GNN 均通过验证集筛查'
            elif simple:
                interpretation = '简单模型通过筛查；现有 GNN 未通过'
            elif graph:
                interpretation = '仅 GNN 通过筛查；跨模型信息增益未确认'
            else:
                interpretation = '本轮未确认稳定的验证集增益'
            evidence.append(dict(task=task, comparison=name, **decisions, interpretation=interpretation))
    return pd.DataFrame(summary), pd.DataFrame(paired), pd.DataFrame(evidence)


def write_report(out, protocol, paired, evidence):
    spec = protocol['spec']
    names = {'drug_pooling': '药物池化', 'expression': '表达压缩与统计量', 'continuous_edges': '连续边权'}
    title = '首轮三项特征实验：执行检查' if spec['smoke'] else '首轮三项特征实验'
    lines = [
        '# ' + title, '',
        '本轮只解释三组独立对照：同一 ChemBERTa 的 masked − unmasked 池化；'
        'PCA64 + 均值/标准差/零值比例 − 原表达输入；连续边权 − 二值边权。'
        'base 与两个池化臂使用不同维数的历史/本地模型，因此不把 masked − base 解释为池化效应。', '',
        '任务：LncDrug、MiDrug；模型：原两层 GNN、64 隐层 MLP、ExtraTrees；种子：'
        + ', '.join(map(str, spec['seeds'])) + '。原验证/测试样本保持不变。训练集只做一次去重和重叠清理，'
        '随后全部实验共用固定的训练样本、负例和拓扑；逐行清理记录见 `inputs/*/audit.json`。', '',
        'GNN/MLP 都训练固定轮数（LncDrug %d、MiDrug %d），按验证集 '
        '(AUC + AUPR) / 2 最大值选择一个检查点，同分取更早轮次。ExtraTrees 固定 %d 棵树，'
        '不搜索超参数。测试指标只在选模完成后计算。' % (
            spec['task_settings']['LncDrug']['train_epochs'], spec['task_settings']['MiDrug']['train_epochs'],
            spec['extratrees']['n_estimators']), '',
        '三类模型共享节点向量与现有边权。MLP/ExtraTrees 读取完整目标 RNA/药物向量及 14 个固定图统计，'
        'GNN 读取全部节点向量与异构训练图；两种结构表示不同，因此只将各模型内部的配对差值用于特征归因。'
        '图统计的训练行做五折交叉构造；表达变换与标准化只在训练节点/样本上拟合。', '',
        '验证和测试统一使用训练图。历史 GNN 结果曾在测试图加入验证正边，本轮结果应在本表内部比较。', '',
        '表中 Δ 为处理臂减对照臂，单位为原始 AUC/AUPR（+0.001 = +0.1 个百分点）。'
        '“正向种子”按验证集选模分数计算；完整均值、配对标准差和每个种子的结果见 CSV/XLSX。', '',
        '| 任务 | 实验 | 模型 | Δ 验证选模分数 | Δ 测试 AUC | Δ 测试 AUPR | 验证正向种子 |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for row in paired.to_dict('records'):
        lines.append('| %s | %s | %s | %+.5f | %+.5f | %+.5f | %d/%d |' % (
            row['task'], names[row['comparison']], row['model'], row['val_criterion_delta_mean'],
            row['test_auc_delta_mean'], row['test_aupr_delta_mean'],
            row['val_criterion_positive_seeds'], row['n_seeds']))
    lines.extend(['', '筛查规则预先固定为：验证集选模分数平均配对增益 > 0，且至少 4/5 个种子为正。'
                  '这只是进入下一阶段的描述性依据，不是显著性检验，也不根据测试成绩重新选择实验。', ''])
    for row in evidence.to_dict('records'):
        lines.append('- %s / %s：%s。' % (row['task'], names[row['comparison']], row['interpretation']))
    lines.extend(['', '表达臂同时改变压缩与统计量，不能单独归因于 PCA 或某个统计量。'
                  '五个种子的离散度只反映本划分上的初始化波动；当前数据已用于历史实验，'
                  '本轮不构成新划分或外部数据上的独立验证。', '',
                  '可复核文件：`protocol.json`、`inputs/*/audit.json`、`VALIDATION.json`、'
                  '`per_seed.csv`、`summary.csv`、`paired_deltas.csv`、`evidence.csv`、'
                  '`scores/`、`traces/`、`checkpoints/`、`source_snapshot/`。', ''])
    (out / 'REPORT.md').write_text('\n'.join(lines))


def summarize(args):
    out = Path(args.out_dir).resolve()
    protocol = load_protocol(out)
    frame = verify_results(out, protocol)
    summary, paired, evidence = paired_summary(frame, protocol['spec'])
    flat = frame.drop(columns=['artifacts_sha256', 'node_sha256', 'pair_input_sha256', 'edge_weight_sha256'])
    for name, table in (('per_seed', flat), ('summary', summary), ('paired_deltas', paired), ('evidence', evidence)):
        table.to_csv(out / (name + '.csv'), index=False)
    workbook = out / '首轮三项特征实验.xlsx'
    with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
        for sheet, table in (('配对增益', paired), ('筛查结论', evidence), ('均值与标准差', summary), ('逐种子结果', flat)):
            table.to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame([{'item': k, 'value': json.dumps(v, ensure_ascii=False)}
                      for k, v in protocol['spec'].items()]).to_excel(writer, sheet_name='固定协议', index=False)
        for sheet in writer.book:
            sheet.freeze_panes = 'D2'
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(
                    70, max(14, max(len(str(cell.value)) for cell in column) + 2))
    write_report(out, protocol, paired, evidence)
    write_json(out / 'VALIDATION.json', {
        'status': 'passed', 'protocol_sha256': protocol['protocol_sha256'],
        'expected_runs': len(expected_tags(protocol['spec'])), 'completed_runs': len(frame),
        'paired_contrasts': len(paired), 'models_per_contrast': 3,
        'checks': ['source and input file SHA256', 'complete fixed run matrix',
                   'shared node, pair and weight fingerprints across models',
                   'saved labels and pair IDs identical to the frozen split',
                   'metrics recomputed from saved predictions',
                   'checkpoint epoch is first maximum validation criterion',
                   'fixed epoch budget', 'all checkpoint/trace/prediction artifact SHA256'],
    })
    print(evidence.to_string(index=False), flush=True)
    print('VERIFIED', len(frame), 'runs;', out / 'REPORT.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'summarize'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--run-dir', default='/data1/licheng/incrd-hg/incrd-run')
    parser.add_argument('--net-dir', default='/data1/licheng/incrd-hg/incrd-run/2.construct_network')
    parser.add_argument('--model-dir')
    parser.add_argument('--smoke', action='store_true', help='prepare a separate two-epoch/one-seed protocol')
    parser.add_argument('--task', choices=tuple(TASKS))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--seeds', type=int, nargs='+', help='run a subset of already frozen seeds')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    if args.command == 'run' and not args.task:
        parser.error('run requires --task')
    if len(set(args.models)) != len(args.models):
        parser.error('--models must not contain duplicates')
    torch.set_num_threads(args.cpu_threads)
    {'prepare': prepare, 'run': run, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
