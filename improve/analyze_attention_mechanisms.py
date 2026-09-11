"""Inference-only interventions and neighbour-support stability for the screen.

No checkpoint is selected or tuned here. Each checkpoint was selected by the
frozen training protocol. Interventions measure dependence of a jointly trained
model on its added branch; they are not biological causal interventions.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from improve.attention_mechanisms import (
    PRIMARY, MechanismModel, file_hash, make_base, metrics, norm, rms, save_csv,
    source_hashes, write_json,
)
from improve.drug_condition_attention import rna_neighbour_masks


@torch.no_grad()
def intervene(model, train, rna_type):
    encoded = model.base.encoder(train.x_dict, train.edge_index_dict)
    z, first = encoded if model.arm.startswith('layer_') else (encoded, None)
    branch = model.branch
    normal, weights = branch(train.x_dict, first, z)
    outcomes = {'learned': normal, 'branch_off': z}
    if model.arm == 'channel_attention':
        uniform = dict(z)
        for nt in model.node_types:
            uniform[nt] = z[nt] * weights['channel_' + nt].mean(-1, keepdim=True)
        outcomes['channel_mean_gain'] = uniform
    elif model.arm == 'layer_attention':
        previous = branch.mode
        branch.mode = 'layer_uniform'
        outcomes['layer_uniform'], _ = branch(train.x_dict, first, z)
        branch.mode = previous
    elif model.arm == 'neighbour_topk':
        contexts = []
        for i, rel in enumerate(branch.relations):
            support = weights['neighbour_' + '|'.join(rel)] > 0
            uniform = support.to(z[rna_type].dtype) / support.sum(-1, keepdim=True).clamp_min(1)
            contexts.append(uniform @ branch.values[i](norm(z[rel[0]])))
        out = dict(z)
        out[rna_type] = z[rna_type] + rms(z[rna_type]) * branch.head(torch.cat(contexts, -1))
        outcomes['selected_neighbours_uniform'] = out
        previous = branch.mode
        branch.mode = 'neighbour_randomk'
        outcomes['random_neighbours_same_k'], _ = branch(train.x_dict, first, z)
        branch.mode = previous
    return outcomes


def support_stability(root, protocol):
    rows = []
    for task, config in protocol['configs'].items():
        train = torch.load(protocol['inputs'][task]['folds']['train']['path'], map_location='cpu')
        relations, masks = rna_neighbour_masks(train, config['rna'])
        for arm in ('neighbour_topk', 'neighbour_randomk'):
            arrays = []
            for seed in protocol['seeds']:
                path = root / task / ('seed_' + str(seed)) / arm / 'attention.npz'
                with np.load(path) as data:
                    arrays.append({k: data[k] > 0 for k in data.files})
            for rel, mask in zip(relations, masks):
                name = 'neighbour_' + '|'.join(rel)
                # Degree <= k gives trivial agreement: all available neighbours
                # are retained. Exclude those nodes from selection stability.
                eligible = mask.sum(-1).numpy() > protocol['top_k']
                jaccard = []
                for a, b in itertools.combinations(arrays, 2):
                    left, right = a[name][eligible], b[name][eligible]
                    intersection = (left & right).sum(-1)
                    union = (left | right).sum(-1)
                    if len(union):
                        jaccard.append(float(np.mean(intersection / np.maximum(union, 1))))
                rows.append(dict(task=task, arm=arm, relation='|'.join(rel),
                                 eligible_nodes=int(eligible.sum()), all_nodes=len(eligible),
                                 seed_pairs=len(jaccard),
                                 mean_support_jaccard=float(np.mean(jaccard)) if jaccard else None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.out_dir)
    protocol = json.loads((root / 'protocol.json').read_text())
    if source_hashes() != protocol['source_hashes']:
        raise ValueError('Training code has changed')
    rows, diagnostics = [], []
    for task, config in protocol['configs'].items():
        key = (config['rna'], task, 'drug')
        data = {}
        for fold, entry in protocol['inputs'][task]['folds'].items():
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('Frozen input changed')
            data[fold] = torch.load(entry['path'], map_location='cpu')
        train = data['train'].to(args.device)
        for seed, arm in itertools.product(protocol['seeds'], PRIMARY):
            directory = root / task / ('seed_' + str(seed)) / arm
            record = json.loads((directory / 'result.json').read_text())
            if record['signature']['protocol_sha256'] != file_hash(root / 'protocol.json'):
                raise ValueError('Mismatched checkpoint protocol')
            base = make_base(train, config, seed, args.device)
            model = MechanismModel(base, data['train'].cpu(), config, arm,
                                   protocol['width'], protocol['top_k'], seed).to(args.device)
            # data['train'] and train can share an object; restore requested
            # device after constructing the masks from CPU graph metadata.
            train = data['train'].to(args.device)
            model.load_state_dict(torch.load(directory / 'checkpoint.pt', map_location=args.device))
            model.eval()
            outcomes = intervene(model, train, config['rna'])
            with torch.no_grad():
                for fold in ('val', 'test'):
                    pairs = data[fold][key].edge_label_index.to(args.device)
                    y = data[fold][key].edge_label.cpu().numpy()
                    learned = None
                    for intervention, embeddings in outcomes.items():
                        scores = model.base.decoder(embeddings, pairs).cpu().numpy()
                        values = metrics(y, scores, pairs.cpu().numpy())
                        if intervention == 'learned':
                            learned = scores
                        row = dict(task=task, seed=seed, arm=arm, fold=fold, intervention=intervention,
                                   mean_abs_logit_change=float(np.abs(scores - learned).mean()))
                        row.update(values)
                        rows.append(row)
                        if intervention == 'learned':
                            replay = max(abs(values[m] - record['metrics'][fold][m]) for m in ('auc', 'aupr'))
                            if replay > 1e-4:
                                raise ValueError('Large checkpoint replay difference: ' + str(directory))
            diagnostics.append(dict(task=task, seed=seed, arm=arm, **record['diagnostics']))
            print('ANALYSED', task, seed, arm, flush=True)
            del model, base, outcomes
    save_csv(root / 'interventions_per_seed.csv', rows)
    aggregates = []
    for task, arm, fold in itertools.product(protocol['tasks'], PRIMARY, ('val', 'test')):
        subset = [r for r in rows if r['task'] == task and r['arm'] == arm and r['fold'] == fold]
        reference = {r['seed']: r for r in subset if r['intervention'] == 'learned'}
        for intervention in dict.fromkeys(r['intervention'] for r in subset):
            group = [r for r in subset if r['intervention'] == intervention]
            agg = dict(task=task, arm=arm, fold=fold, intervention=intervention, n=len(group))
            for metric in ('auc', 'aupr', 'per_drug_auc', 'per_drug_aupr', 'mean_abs_logit_change'):
                agg[metric + '_mean'] = float(np.mean([r[metric] for r in group]))
            for metric in ('auc', 'aupr'):
                delta = [r[metric] - reference[r['seed']][metric] for r in group]
                agg[metric + '_delta_from_learned'] = float(np.mean(delta))
                agg[metric + '_decreases'] = int(sum(v < 0 for v in delta))
            aggregates.append(agg)
    stability = support_stability(root, protocol)
    save_csv(root / 'interventions_summary.csv', aggregates)
    save_csv(root / 'neighbour_stability.csv', stability)
    write_json(root / 'mechanism_diagnostics.json', diagnostics)
    write_json(root / 'analysis_provenance.json', dict(
        script_sha256=file_hash(__file__), protocol_sha256=file_hash(root / 'protocol.json'),
        device=args.device, selection='uses every preselected primary-arm checkpoint; no tuning',
        interventions=['branch removal', 'nonselective channel/layer weights',
                       'uniform weights on selected neighbours', 'random neighbours at the same k'],
        support_stability='Jaccard across seed pairs, restricted to nodes with degree > k',
        limitation='Inference interventions on jointly trained models measure reliance, not biological causality.',
    ))
    report = ['# 注意力依赖与稳定性检查', '',
              '全部使用验证集已选定的 checkpoint；没有重新训练、调参或重选模型。',
              '下表变化均为干预后减去同一模型使用学习权重时的指标。负值表示干预使性能下降。', '',
              '| Task | Arm | Intervention | Δ test AUC | Δ test AUPR | AUPR decreases |',
              '|---|---|---|---:|---:|---:|']
    for r in aggregates:
        if r['fold'] == 'test' and r['intervention'] != 'learned':
            report.append('| %s | %s | %s | %+.5f | %+.5f | %d/%d |' %
                          (r['task'], r['arm'], r['intervention'], r['auc_delta_from_learned'],
                           r['aupr_delta_from_learned'], r['aupr_decreases'], r['n']))
    report.extend(['', '## 邻居选择跨种子稳定性', '',
                   '只统计邻居数大于 k=8 的节点，排除“全部邻居都被选中”造成的平凡一致。', '',
                   '| Task | Arm | Relation | Eligible nodes | Mean Jaccard |',
                   '|---|---|---|---:|---:|'])
    for r in stability:
        value = 'N/A' if r['mean_support_jaccard'] is None else '%.4f' % r['mean_support_jaccard']
        report.append('| %s | %s | %s | %d | %s |' %
                      (r['task'], r['arm'], r['relation'].replace('|', '\\|'), r['eligible_nodes'], value))
    report.extend(['', '## 解释', '',
                   '- 关闭分支保留联合训练后的编码器和解码器；它不等同于重新训练原模型。',
                   '- 通道干预保留每个节点的平均增益，层间干预改为三层等权。',
                   '- 稀疏邻居分别检验选中集合内部的权重和邻居集合本身。',
                   '- 支持集稳定或干预造成下降，并不保证模型优于原基线，也不构成生物因果证据。',
                   '- 不比较不同种子的隐藏通道编号，因为它们没有天然对齐。', ''])
    (root / 'MECHANISM_CHECKS.md').write_text('\n'.join(report))
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
