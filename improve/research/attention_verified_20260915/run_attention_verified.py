"""Verified attention screen: does any path-attention variant improve MiDrug?

Reuses the frozen edge-path assets and the SmoothPathModel arms, but runs an
extended seed set (10 seeds) with a paired, validation-selected protocol, so a
claim of improvement has seed-level support.  Test metrics never touch model
selection.

Run with /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from improve.attention_mechanisms import parameter_hash, save_csv
from improve.drug_condition_attention import (file_hash, make_base, metrics,
                                              set_seed, write_json)
from improve.edge_path_attention import configure, predict
from improve.smooth_path_attention import ARMS, SmoothPathModel

PARENT = Path('improve/results/edge_path_attention_20260908/protocol.json')
REFERENCE = 'base'
CONTRASTS = (('path_uniform', 'base'), ('path_attention', 'base'),
             ('path_attention', 'path_uniform'), ('temperature4', 'base'),
             ('temperature4', 'path_uniform'), ('temperature4', 'path_attention'),
             ('shrink25', 'base'), ('shrink25', 'path_uniform'),
             ('shrink25', 'temperature4'))


def load_parent(task):
    p = json.loads(PARENT.read_text())
    data = {}
    for fold, entry in p['inputs'][task]['folds'].items():
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('Frozen graph changed: ' + entry['path'])
        data[fold] = torch.load(entry['path'], map_location='cpu')
    asset = p['assets'][task]
    if file_hash(asset['path']) != asset['sha256']:
        raise ValueError('Frozen assets changed')
    return p, data, torch.load(asset['path'], map_location='cpu')


def directory(root, task, seed, arm):
    return root / task / ('seed_' + str(seed)) / arm


def train_one(root, parent, data, assets, task, seed, arm, device):
    out = directory(root, task, seed, arm)
    result_path = out / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    config = parent['configs'][task]
    key = (config['rna'], task, 'drug')
    train = copy.deepcopy(data['train']).to(device)
    base = make_base(train, config, seed, device)
    set_seed(seed + parent['branch_rng_offset'])
    model = SmoothPathModel(base, assets, config, arm, parent['width']).to(device)
    set_seed(seed + parent['training_rng_offset'])
    labels = {f: g[key].edge_label.cpu().numpy() for f, g in data.items()}
    optimizer = torch.optim.RMSprop(
        (v for v in model.parameters() if v.requires_grad),
        lr=config['lr'], alpha=parent['alpha'])
    best, started = -float('inf'), time.monotonic()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        score = model(train.x_dict, train.edge_index_dict, 'train')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            score, train[key].edge_label)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = predict(model, train, 'val')
        values = metrics(labels['val'], validation, assets['pairs']['val'].numpy(),
                         by_drug=False)
        criterion = (values['auc'] + values['aupr']) / 2
        if criterion > best:
            best, best_epoch = criterion, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    results = {}
    with torch.no_grad():
        for fold in ('val', 'test'):
            prediction = predict(model, train, fold)
            results[fold] = metrics(labels[fold], prediction,
                                    assets['pairs'][fold].numpy())
    record = dict(task=task, seed=seed, arm=arm, selected_epoch=best_epoch,
                  state_sha256=parameter_hash(model), metrics=results,
                  elapsed_seconds=time.monotonic() - started)
    write_json(result_path, record)
    print('RESULT', task, seed, arm,
          'val %.5f/%.5f test %.5f/%.5f' % (
              results['val']['auc'], results['val']['aupr'],
              results['test']['auc'], results['test']['aupr']), flush=True)
    del optimizer, model, base, train, best_state
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return record


def summarise(root, parent, seeds, arms, task):
    records, rows = {}, []
    for seed in seeds:
        for arm in arms:
            record = json.loads((directory(root, task, seed, arm) / 'result.json').read_text())
            records[seed, arm] = record
            row = dict(seed=seed, arm=arm, selected_epoch=record['selected_epoch'])
            for fold, values in record['metrics'].items():
                row.update({fold + '_' + k: v for k, v in values.items()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / ('per_seed_' + task + '.csv'), index=False)
    comparisons = []
    for arm, reference in CONTRASTS:
        if arm not in arms or reference not in arms:
            continue
        row = dict(arm=arm, reference=reference, n=len(seeds))
        for fold in ('val', 'test'):
            for metric in ('auc', 'aupr'):
                delta = np.array([records[s, arm]['metrics'][fold][metric] -
                                  records[s, reference]['metrics'][fold][metric]
                                  for s in seeds])
                row[fold + '_' + metric + '_delta'] = float(delta.mean())
                row[fold + '_' + metric + '_wins'] = int((delta > 0).sum())
                row[fold + '_' + metric + '_std'] = float(delta.std(ddof=1))
        comparisons.append(row)
    save_csv(root / ('comparisons_' + task + '.csv'), comparisons)
    return frame, comparisons


def report(root, frame, comparisons, seeds, arms, task):
    summary = frame.groupby('arm', sort=False).agg(
        n=('seed', 'count'), val_auc=('val_auc', 'mean'),
        test_auc=('test_auc', 'mean'), test_aupr=('test_aupr', 'mean'),
        test_per_drug_auc=('test_per_drug_auc', 'mean')).reset_index()
    summary.to_csv(root / ('summary_' + task + '.csv'), index=False)
    best_by_val = summary.loc[((summary.val_auc)).idxmax(), 'arm']
    lines = ['# 扩展种子集：注意力是否可验证地提升 %s（固定划分）' % task, '',
             '资源：`improve/results/edge_path_attention_20260908` 的冻结图与路径资产；',
             '选模只用验证集 AUC/AUPR 均值，测试指标不参与选点。',
             '种子：%s。' % seeds, '',
             '| 方案 | n | Val AUC | Test AUC | Test AUPR | 逐药物 Test AUC |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary.to_dict('records'):
        lines.append('| %s | %d | %.6f | %.6f | %.6f | %.6f |' %
                     (row['arm'], row['n'], row['val_auc'], row['test_auc'],
                      row['test_aupr'], row['test_per_drug_auc']))
    lines.extend(['', '## 配对对照', '',
                  '| 方案 | 对照 | ΔVal AUC | ΔTest AUC | AUC 提升种子 | ΔTest AUPR | AUPR 提升种子 |',
                  '|---|---|---:|---:|---:|---:|---:|'])
    for row in comparisons:
        lines.append('| %s | %s | %+.6f | %+.6f | %d/%d | %+.6f | %d/%d |' %
                     (row['arm'], row['reference'], row['val_auc_delta'],
                      row['test_auc_delta'], row['test_auc_wins'], row['n'],
                      row['test_aupr_delta'], row['test_aupr_wins'], row['n']))
    lines.extend(['', '## 读法与边界', '',
                  '- 判据：测试 AUC/AUPR 同时多数种子提升，并同时超过 `path_uniform`（无注意力对照），',
                  '  才把收益归给"注意力选择"而不是"新增路径分支"。',
                  '- 固定划分、10 个初始化种子；跨划分泛化未在本轮验证。',
                  '- 验证集最优方案：`%s`（仅作参考，最终看配对对照）。' % best_by_val, ''])
    (root / ('REPORT_' + task + '.md')).write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--task', default='MiDrug')
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[666, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--reports-only', action='store_true')
    args = ap.parse_args()
    configure()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    parent, data, assets = load_parent(args.task)
    for arm in args.arms:
        if arm not in ARMS:
            raise ValueError('Unknown arm: ' + arm)
    if not args.reports_only:
        for seed in args.seeds:
            for arm in args.arms:
                train_one(root, parent, data, assets, args.task, seed, arm, args.device)
    frame, comparisons = summarise(root, parent, args.seeds, args.arms, args.task)
    report(root, frame, comparisons, args.seeds, args.arms, args.task)


if __name__ == '__main__':
    main()
