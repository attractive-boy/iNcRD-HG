"""Descriptive mechanism diagnostics on frozen, validation-selected checkpoints."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from improve.drug_condition_attention import file_hash, metrics, write_json
from improve.edge_path_attention import run_path, source_hashes
from improve.edge_path_modules import ARMS, PRIMARY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    root = Path(args.out_dir)
    protocol_path = root / 'protocol.json'
    p = json.loads(protocol_path.read_text())
    validation = json.loads((root / 'VALIDATION.json').read_text())
    if (not validation['passed'] or validation['protocol_sha256'] != file_hash(protocol_path)
            or p['source_hashes'] != source_hashes()):
        raise ValueError('Run the verifier on the current complete experiment first')
    coverage, edge_rows, path_rows, strata, effects = [], [], [], [], []
    for task in p['tasks']:
        asset = p['assets'][task]
        if file_hash(asset['path']) != asset['sha256']:
            raise ValueError('Frozen assets changed')
        assets = torch.load(asset['path'], map_location='cpu')
        train = torch.load(p['inputs'][task]['folds']['train']['path'], map_location='cpu')
        masks = {}
        for fold, packs in assets['packs'].items():
            counts = np.stack([pack['counts'].numpy() for pack in packs], -1)
            present, multiple = (counts > 0).any(1), (counts > 1).any(1)
            masks[fold] = dict(no_path=~present, singleton_only=present & ~multiple,
                               multiple_paths=multiple, any_path=present)
            coverage.append(dict(task=task, fold=fold, pairs=len(counts),
                                 **{name: int(mask.sum()) for name, mask in masks[fold].items()},
                                 any_path_fraction=float(present.mean()),
                                 multiple_path_fraction=float(multiple.mean())))
        records = {}
        for seed, arm in itertools.product(p['seeds'], ARMS):
            directory = run_path(root, task, seed, arm)
            records[seed, arm] = json.loads((directory / 'result.json').read_text())
            if arm in ('base', 'path_count', 'path_uniform', 'path_attention'):
                with np.load(directory / 'predictions.npz') as prediction:
                    for fold in ('val', 'test'):
                        for name, mask in masks[fold].items():
                            labels = prediction['y_' + fold][mask]
                            if len(np.unique(labels)) != 2:
                                continue
                            values = metrics(labels, prediction[fold][mask],
                                             prediction['pairs_' + fold][:, mask])
                            strata.append(dict(task=task, seed=seed, arm=arm, fold=fold,
                                               stratum=name, n=int(mask.sum()), positives=int(labels.sum()), **values))
            if arm.startswith('edge_'):
                with np.load(directory / 'attention_val.npz') as arrays:
                    for name in arrays.files:
                        relation = tuple(name.rsplit('.', 1)[-1].split('__'))
                        edges = train[relation].edge_index.numpy()
                        nonself = edges[0] != edges[1]
                        weights = arrays[name][nonself]
                        destination = edges[1, nonself]
                        degree = np.bincount(destination, minlength=train[relation[2]].num_nodes)
                        probability = weights / degree[destination]
                        entropy = -np.bincount(destination,
                                              weights=probability * np.log(np.maximum(probability, 1e-30)),
                                              minlength=len(degree))
                        eligible = degree > 1
                        magnitude = assets['attributes'][relation].numpy()[nonself, 1]
                        corr = (float(spearmanr(weights, magnitude).statistic)
                                if np.ptp(weights) > 1e-12 and np.ptp(magnitude) > 1e-12 else None)
                        edge_rows.append(dict(task=task, seed=seed, arm=arm, layer=name,
                                              nonself_edges=len(weights), mean_abs_change=float(np.abs(weights - 1).mean()),
                                              weight_min=float(weights.min()), weight_max=float(weights.max()),
                                              near_uniform_fraction=float((np.abs(weights - 1) < 1e-3).mean()),
                                              normalised_entropy=float((entropy[eligible] / np.log(degree[eligible])).mean()),
                                              similarity_spearman=corr))
            if arm in ('path_uniform', 'path_attention'):
                for fold in ('val', 'test'):
                    with np.load(directory / ('attention_' + fold + '.npz')) as arrays:
                        for i, pack in enumerate(assets['packs'][fold]):
                            weights = arrays['path_' + str(i)]
                            counts = pack['counts'].numpy()
                            eligible = counts > 1
                            entropy = -(weights * np.log(np.maximum(weights, 1e-30))).sum(1)
                            valid = pack['valid'].numpy()
                            uniform = valid / np.maximum(counts[:, None], 1)
                            tv = np.abs(weights - uniform).sum(1) / 2
                            path_rows.append(dict(task=task, seed=seed, arm=arm, fold=fold, family=i,
                                                  middle=assets['families'][i]['middle'], eligible_pairs=int(eligible.sum()),
                                                  normalised_entropy=float((entropy[eligible] / np.log(counts[eligible])).mean()),
                                                  total_variation_from_uniform=float(tv[eligible].mean())))
        for arm, reference in p['contrasts']:
            for fold, metric in itertools.product(('val', 'test'), ('auc', 'aupr', 'per_drug_auc', 'per_drug_aupr')):
                delta = np.array([records[seed, arm]['metrics'][fold][metric] -
                                  records[seed, reference]['metrics'][fold][metric] for seed in p['seeds']])
                effects.append(dict(task=task, arm=arm, reference=reference, fold=fold, metric=metric,
                                    n=len(delta), mean=float(delta.mean()),
                                    std=float(delta.std(ddof=1)) if len(delta) > 1 else 0.,
                                    minimum=float(delta.min()), maximum=float(delta.max()), wins=int((delta > 0).sum())))
    for name, rows in [('path_coverage', coverage), ('edge_weight_diagnostics', edge_rows),
                       ('path_weight_diagnostics', path_rows), ('path_strata_per_seed', strata),
                       ('paired_effects', effects)]:
        pd.DataFrame(rows).to_csv(root / (name + '.csv'), index=False)
    lines = ['# 边属性与路径注意力：机制检查', '',
             '本分析读取已通过核验的验证集选定模型，不训练、不重选 checkpoint。', '',
             '## 路径覆盖', '',
             '“可选择”表示至少一类路径有两个或更多中间节点；只有一个中间节点时 softmax 与等权相同。', '',
             '| 任务 | 划分 | 样本数 | 至少一条路径 | 有可选择的路径 |',
             '|---|---|---:|---:|---:|']
    for row in coverage:
        lines.append('| %s | %s | %d | %d (%.1f%%) | %d (%.1f%%) |' %
                     (row['task'], row['fold'], row['pairs'], row['any_path'], 100 * row['any_path_fraction'],
                      row['multiple_paths'], 100 * row['multiple_path_fraction']))
    lines.extend(['', '## 多路径样本上的权重选择', '',
                  '熵除以 log(路径数)，1 表示等权。总变差为与等权分布的距离，0 表示相同。', '',
                  '| 任务 | 划分 | 中间 RNA | 平均归一化熵 | 平均总变差 |',
                  '|---|---|---|---:|---:|'])
    frame = pd.DataFrame(path_rows)
    for keys, group in frame[frame.arm == 'path_attention'].groupby(['task', 'fold', 'middle'], sort=False):
        lines.append('| %s | %s | %s | %.5f | %.5f |' %
                     (*keys, group.normalised_entropy.mean(), group.total_variation_from_uniform.mean()))
    lines.extend(['', '## 解释范围', '',
                  '- 路径覆盖由输入图决定，不等于模型在这些样本上的有效贡献。',
                  '- 按有无路径分层的 AUC/AUPR 见 path_strata_per_seed.csv；分层指标不能直接相加恢复总体指标。',
                  '- edge_weight_diagnostics.csv 记录各层权重偏离等权的程度及其与真实相似度的描述性相关。',
                  '- 不同的注意力权重不等于性能提升；须同时阅读独立训练的对照和等权干预结果。',
                  '- paired_effects.csv 的标准差描述同一划分上的初始化种子波动，不能解释为新划分的不确定性。', ''])
    (root / 'MECHANISM_CHECKS.md').write_text('\n'.join(lines))
    write_json(root / 'analysis_provenance.json', dict(
        protocol_sha256=file_hash(protocol_path), validation_sha256=file_hash(root / 'VALIDATION.json'),
        analyzer_sha256=file_hash(__file__), seeds=p['seeds'], repeats_used=[0],
        checkpoint_selection='unchanged: validation AUC/AUPR mean',
        scope='descriptive diagnostics and predefined paired contrasts; fixed split'))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
