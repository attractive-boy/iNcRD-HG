"""Recheck two expression inputs on the original GNN, only at seed 666.

Uses the audited first-round training graph, labels, model, optimizer and
validation selection rule. Only RNA node inputs change; no MLP replaces GNN.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from improve.first_round_features import (FOLDS, METRICS, json_digest,
    load_protocol, metric_record, run_neural, sha256, source_hashes, verify_files,
    write_json)
from improve.network_feature_inputs import digest_array, expression_features


ARMS = ('expr_stats3', 'expr_ae64_stats')
TASKS = ('LncDrug', 'MiDrug')


def setup(args, task):
    ref, out = Path(args.reference_dir).resolve(), Path(args.out_dir).resolve()
    reference = load_protocol(ref)
    verify_files(ref, reference, prefix='inputs/' + task + '/')
    baseline_path = ref / 'runs' / (task + '_gnn_base_s666.json')
    baseline = json.loads(baseline_path.read_text())
    if baseline['protocol_sha256'] != reference['protocol_sha256']:
        raise ValueError('Baseline does not belong to the audited input protocol')
    for name, digest in baseline['artifacts_sha256'].items():
        if sha256(ref / name) != digest:
            raise ValueError('Baseline artifact changed: ' + name)
    source = dict(source_hashes(), gnn_expression_seed666=sha256(__file__))
    identity = {
        'task': task, 'seed': 666, 'model': 'original_gnn', 'arms': list(ARMS),
        'reference_protocol_sha256': reference['protocol_sha256'],
        'reference_baseline_sha256': sha256(baseline_path),
        'source_sha256': source,
        'definition': {
            'expr_stats3': 'replace both RNA types by standardized raw mean, std and zero fraction',
            'expr_ae64_stats': 'replace both RNA types by training-node AE64 plus the same three statistics',
        },
        'autoencoder': {'dims': [938, 256, 64, 256, 938], 'epochs': 300,
                        'seed': 0, 'optimizer': 'Adam', 'lr': 0.001,
                        'loss': 'training-node MSE; no validation/test selection', 'device': 'cpu'},
        'training': reference['spec']['task_settings'][task],
        'selection': reference['spec']['selection'],
    }
    dest = out / task
    dest.mkdir(parents=True, exist_ok=True)
    config_path = dest / 'config.json'
    graph = torch.load(ref / 'inputs' / task / 'train_graph.pth', map_location='cpu')
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config['identity'] != identity:
            raise ValueError('Changed protocol: use a new output directory')
        for name, digest in config['input_files_sha256'].items():
            if sha256(dest / name) != digest:
                raise ValueError('Cached candidate input changed: ' + name)
    else:
        for name, digest in reference['raw_network_files_sha256'].items():
            if sha256(Path(reference['net_dir']) / name) != digest:
                raise ValueError('Raw network input changed: ' + name)
        with np.load(ref / 'inputs' / task / 'base.npz', allow_pickle=False) as z:
            base = {k: z[k].copy() for k in z.files
                    if not k.startswith(('phi_', 'pair_scaler_'))}
        with np.load(ref / 'inputs' / task / 'expr_pca64_stats.npz', allow_pickle=False) as z:
            stats = {nt: z['x_' + nt][:, -3:].copy() for nt in ('lncRNA', 'miRNA')}
        key = (reference['spec']['task_settings'][task]['rna'], task, 'drug')
        ae, ae_stats, audit = expression_features(graph, reference['net_dir'], key, kind='ae')
        for nt in stats:
            np.testing.assert_allclose(stats[nt], ae_stats[nt], rtol=1e-6, atol=1e-6)
        candidates = {
            'expr_stats3': dict(base, **{'x_' + nt: x for nt, x in stats.items()}),
            'expr_ae64_stats': dict(base, **{'x_' + nt: x for nt, x in ae.items()}),
        }
        node_hashes, files = {}, {}
        for arm, arrays in candidates.items():
            for name, value in base.items():
                if name not in ('x_lncRNA', 'x_miRNA'):
                    np.testing.assert_array_equal(arrays[name], value)
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError('Non-finite candidate input')
            filename = arm + '.npz'
            np.savez_compressed(dest / filename, **arrays)
            files[filename] = sha256(dest / filename)
            node_hashes[arm] = {nt: digest_array(arrays['x_' + nt]) for nt in graph.node_types}
        config = dict(identity=identity, expression_ae_audit=audit,
                      input_files_sha256=files, node_sha256=node_hashes)
        config['sha256'] = json_digest(config)
        write_json(config_path, config)
        snapshot = dest / 'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in source_hashes():
            shutil.copy2(Path(__file__).parent / name, snapshot / name)
        shutil.copy2(__file__, snapshot / Path(__file__).name)
    if json_digest({k: v for k, v in config.items() if k != 'sha256'}) != config['sha256']:
        raise ValueError('Candidate configuration changed')
    return dest, reference['spec'], graph, baseline, config


def run(args):
    for task in [args.task] if args.task else TASKS:
        dest, spec, graph, baseline, config = setup(args, task)
        for arm in ARMS:
            target = dest / arm
            target.mkdir(exist_ok=True)
            result_path = target / 'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result['config_sha256'] != config['sha256']:
                    raise ValueError('Result belongs to another candidate configuration')
                for filename, digest in result['artifacts_sha256'].items():
                    if sha256(target / filename) != digest:
                        raise ValueError('Completed artifact changed: ' + filename)
                print('already complete', task, arm, flush=True)
                continue
            with np.load(dest / (arm + '.npz'), allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            print('START', task, arm, 'original GNN, seed 666', flush=True)
            result, val, test, trace, state = run_neural(args, spec, task, 'gnn', 666, arrays, graph)
            result.update(task=task, arm=arm, model='gnn', seed=666, config_sha256=config['sha256'],
                          node_dims={nt: arrays['x_' + nt].shape[1] for nt in graph.node_types},
                          delta={metric: result[metric] - baseline[metric] for metric in METRICS})
            np.savez_compressed(target / 'scores.npz', val=val, test=test,
                                y_val=arrays['y_val'], y_test=arrays['y_test'],
                                pairs_val=arrays['pairs_val'], pairs_test=arrays['pairs_test'])
            pd.DataFrame(trace).to_csv(target / 'trace.csv', index=False)
            torch.save(state, target / 'checkpoint.pt')
            result['artifacts_sha256'] = {name: sha256(target / name)
                                         for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
            write_json(result_path, result)
            print('DONE', task, arm, 'test %.6f/%.6f; delta %+.6f/%+.6f' % (
                result['test_auc'], result['test_aupr'],
                result['delta']['test_auc'], result['delta']['test_aupr']), flush=True)


def summarize(args):
    out = Path(args.out_dir).resolve()
    rows, baselines = [], {}
    for task in TASKS:
        dest, spec, graph, baseline, config = setup(args, task)
        baselines[task] = baseline
        for arm in ARMS:
            target = dest / arm
            result = json.loads((target / 'result.json').read_text())
            if result['config_sha256'] != config['sha256']:
                raise ValueError('Candidate config differs')
            for filename, digest in result['artifacts_sha256'].items():
                if sha256(target / filename) != digest:
                    raise ValueError('Candidate output differs: ' + filename)
            with np.load(target / 'scores.npz') as z, np.load(dest / (arm + '.npz')) as a:
                for name in ('y_val', 'y_test', 'pairs_val', 'pairs_test'):
                    np.testing.assert_array_equal(z[name], a[name])
                metrics = metric_record({'val': z['y_val'], 'test': z['y_test']}, z['val'], z['test'])
            for metric, value in metrics.items():
                np.testing.assert_allclose(result[metric], value, atol=1e-12, rtol=0)
                np.testing.assert_allclose(result['delta'][metric], value - baseline[metric], atol=1e-12, rtol=0)
            trace = pd.read_csv(target / 'trace.csv')
            if len(trace) != spec['task_settings'][task]['train_epochs']:
                raise ValueError('Wrong fixed epoch budget')
            if int(trace.loc[trace.criterion.idxmax(), 'epoch']) != result['selected_epoch']:
                raise ValueError('Wrong selected checkpoint')
            row = {k: result[k] for k in ('task', 'arm', 'seed', 'selected_epoch') + METRICS}
            row.update({k + '_delta': v for k, v in result['delta'].items()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'results.csv', index=False)
    lines = ['# 原 GNN 表达输入补测：seed 666', '',
             '保留原 GNN、当前冻结训练图、样本、训练参数和验证集选模规则。仅改变两种 RNA 节点的表达输入。'
             '对照直接引用已核验的首轮 seed 666 原 GNN；没有将 GNN 替换为 MLP。', '',
             '| 数据集 | 表达输入 | 测试 AUC | 测试 AUPR | Δ AUC | Δ AUPR |',
             '|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        base = baselines[task]
        lines.append('| %s | 原输入 | %.6f | %.6f | — | — |' % (task, base['test_auc'], base['test_aupr']))
        for row in rows:
            if row['task'] == task:
                lines.append('| %s | %s | %.6f | %.6f | %+.6f | %+.6f |' % (
                    task, row['arm'], row['test_auc'], row['test_aupr'], row['test_auc_delta'], row['test_aupr_delta']))
    verdicts = {}
    lines += ['', '通过条件为同一输入方案在两个数据集的测试 AUC、AUPR 上全部严格提高。', '']
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        passed = all(r['test_auc_delta'] > 0 and r['test_aupr_delta'] > 0 for r in selected)
        verdicts[arm] = passed
        lines.append('- %s：%s。' % (arm, '四项测试指标全部提升' if passed else '未满足四项测试指标全部提升'))
    lines += ['', '`expr_stats3` 为训练节点标准化的原表达均值、标准差、零值比例；'
              '`expr_ae64_stats` 为训练节点拟合的 AE64 加同样三个统计量。AE 固定种子 0、300 轮重建训练，'
              '不使用验证/测试数据选 AE。所有 GNN 固定 seed 666，以验证 AUC/AUPR 均值选择一个检查点，'
              '然后计算测试指标。', '', '完整指标及差值见 [results.csv](results.csv)。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    write_json(out / 'VALIDATION.json', {'status': 'passed', 'seed': 666, 'completed_runs': len(rows),
                                      'verdicts': verdicts, 'reference_baselines': 2,
                                      'checks': ['frozen source/input hashes', 'only RNA vectors change',
                                                 'reference artifact hashes', 'saved prediction metrics',
                                                 'validation-selected checkpoint', 'fixed epoch budgets']})
    print(frame[['task', 'arm', 'val_criterion_delta', 'test_auc_delta', 'test_aupr_delta']].to_string(index=False))
    print('VERIFIED', verdicts, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'summarize'))
    parser.add_argument('--reference-dir', default='improve/results/first_round_features_20260909')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    {'run': run, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
