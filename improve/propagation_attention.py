"""Attention inside propagation: node-wise relations, degree gates and GATv2.

Uses the previous round's immutable cleaned inputs. The original relation
convolutions/decoder are materialised first, then altered in place in the FX
encoder. No previous source or result file is changed.
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric
from torch_geometric.nn import GeneralConv, GATv2Conv
from torch_geometric.utils import softmax

from improve.attention_mechanisms import norm, parameter_hash, save_csv
from improve.drug_condition_attention import (
    METRICS, file_hash, make_base, metrics, set_seed, write_json,
)


ARMS = ('base', 'relation_uniform', 'relation_global', 'relation_node',
        'edge_uniform_mean', 'edge_gate_mean', 'edge_gate_degree',
        'gatv2_uniform', 'gatv2_attention')
PRIMARY = ('relation_node', 'edge_gate_degree', 'gatv2_attention')
CONTRASTS = (
    ('relation_uniform', 'base'), ('relation_global', 'relation_uniform'),
    ('relation_node', 'base'), ('relation_node', 'relation_uniform'),
    ('relation_node', 'relation_global'),
    ('edge_uniform_mean', 'base'), ('edge_gate_mean', 'edge_uniform_mean'),
    ('edge_gate_degree', 'base'), ('edge_gate_degree', 'edge_gate_mean'),
    ('gatv2_uniform', 'edge_uniform_mean'), ('gatv2_attention', 'base'),
    ('gatv2_attention', 'gatv2_uniform'),
)


def source_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ('propagation_attention.py', 'attention_mechanisms.py',
             'drug_condition_attention.py', 'model.py')}


class RelationFusion(nn.Module):
    def __init__(self, hidden, mode, width=16):
        super().__init__()
        self.mode, self.force_uniform = mode, False
        self.scorer = nn.Sequential(nn.Linear(hidden, width), nn.Tanh(),
                                    nn.Linear(width, 1, bias=False))
        nn.init.zeros_(self.scorer[-1].weight)
        if mode == 'relation_uniform':
            self.scorer.requires_grad_(False)
        self.last_weights = None

    def forward(self, messages):
        stacked = torch.stack(messages, dim=1)
        if self.mode == 'relation_uniform' or self.force_uniform:
            weights = stacked.new_full(stacked.shape[:2], 1 / len(messages))
        else:
            scores = self.scorer(norm(stacked)).squeeze(-1)
            if self.mode == 'relation_global':
                scores = scores.mean(0, keepdim=True).expand_as(scores)
            weights = scores.softmax(-1)
        self.last_weights = weights.detach()
        return (stacked * weights.unsqueeze(-1)).sum(1)


def install_relation_fusion(encoder, node_types, hidden, mode, width):
    """Replace only fusion nodes; preserve every original relation convolution."""
    graph = copy.deepcopy(encoder.graph)
    output = next(n for n in graph.nodes if n.op == 'output')
    for layer in (0, 1):
        for nt in node_types:
            messages = [n for n in graph.nodes if n.op == 'call_module' and
                        str(n.target).startswith('convs.' + str(layer) + '.') and
                        str(n.target).split('.')[-1].split('__')[-1] == nt]
            if not messages:
                raise ValueError('No incoming relations for ' + nt)
            if layer == 0:
                anchor = next(n for n in graph.nodes if n.op == 'call_function' and
                              n.target == F.leaky_relu and n.name == 'leaky_relu__' + nt)
                original = anchor.args[0]
            else:
                anchor, original = output, output.args[0][nt]
            # Capture users before insertion: for a single relation the old
            # fusion is also an input of the replacement, and must not cycle.
            users = list(original.users)
            name = 'relation_fusion_' + str(layer) + '_' + nt
            encoder.add_module(name, RelationFusion(hidden, mode, width))
            with graph.inserting_before(anchor):
                replacement = graph.call_module(name, args=(tuple(messages),))
            for user in users:
                user.replace_input_with(original, replacement)
    graph.owning_module = encoder
    graph.eliminate_dead_code()
    graph.lint()
    encoder.graph = graph
    encoder.recompile()


class WeightedMessages(GeneralConv):
    def message(self, x_i, x_j, edge_index_i, size_i, edge_attr):
        message = super().message(x_i, x_j, edge_index_i, size_i, None)
        return message if edge_attr is None else message * edge_attr.reshape(-1, 1, 1)


def gate_weights(scores, destination, n_target, preserve_degree):
    positive = scores.sigmoid()
    total = positive.new_zeros(n_target).index_add(0, destination, positive)
    if preserve_degree:
        degree = torch.bincount(destination, minlength=n_target).to(positive.dtype)
        positive = positive * degree[destination]
    return positive / total[destination].clamp_min(torch.finfo(positive.dtype).tiny)


class EdgeGateConv(nn.Module):
    def __init__(self, original, mode, width=16):
        super().__init__()
        dims = (original.lin_msg.in_channels, original.lin_self.in_channels)
        self.core = WeightedMessages(dims, original.out_channels)
        self.core.load_state_dict(original.state_dict())
        self.source = nn.Linear(dims[0], width, bias=False)
        self.target = nn.Linear(dims[1], width, bias=False)
        self.score = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self.mode, self.force_uniform = mode, False
        if mode == 'edge_uniform_mean':
            for layer in (self.source, self.target, self.score):
                layer.requires_grad_(False)
        self.last_weights = self.last_destination = None
        self.last_n_target = 0

    def forward(self, x, edge_index, edge_attr=None, size=None):
        source, target = x if isinstance(x, (tuple, list)) else (x, x)
        s, d = edge_index
        if self.mode == 'edge_uniform_mean' or self.force_uniform:
            scores = source.new_zeros(len(s))
        else:
            hidden = self.source(norm(source))[s] + self.target(norm(target))[d]
            scores = self.score(F.leaky_relu(hidden, 0.2)).squeeze(-1)
        weights = gate_weights(scores, d, target.size(0), self.mode == 'edge_gate_degree')
        self.last_weights, self.last_destination = weights.detach(), d.detach()
        self.last_n_target = target.size(0)
        return self.core((source, target), edge_index, edge_attr=weights, size=size)


class ControlledGATv2(GATv2Conv):
    def __init__(self, *args, uniform=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.uniform, self.force_uniform, self.last_weights = uniform, False, None
        if uniform:
            self.att.requires_grad_(False)
            self.lin_r.requires_grad_(False)

    def edge_update(self, x_j, x_i, edge_attr, index, ptr, dim_size):
        if self.uniform or self.force_uniform:
            alpha = softmax(x_j.new_zeros(x_j.shape[:2]), index, ptr, dim_size)
        else:
            alpha = super().edge_update(x_j, x_i, edge_attr, index, ptr, dim_size)
        self.last_weights = alpha.detach()
        return alpha


class GATv2WithRoot(nn.Module):
    def __init__(self, original, mode, heads=4):
        super().__init__()
        hidden = original.out_channels
        if hidden % heads:
            raise ValueError('Hidden width must be divisible by heads')
        dims = (original.lin_msg.in_channels, original.lin_self.in_channels)
        self.gat = ControlledGATv2(dims, hidden // heads, heads=heads,
                                   concat=True, dropout=0., add_self_loops=False,
                                   uniform=mode == 'gatv2_uniform')
        # Shared source value and root initialisations make both GATv2 arms
        # initially equivalent to uniform-mean GeneralConv, up to roundoff.
        self.gat.lin_l.load_state_dict(original.lin_msg.state_dict())
        self.gat.lin_r.load_state_dict(original.lin_self.state_dict())
        nn.init.zeros_(self.gat.att)
        nn.init.zeros_(self.gat.bias)
        self.root = copy.deepcopy(original.lin_self)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        source, target = x if isinstance(x, (tuple, list)) else (x, x)
        return self.gat((source, target), edge_index) + self.root(target)


class PropagationModel(nn.Module):
    def __init__(self, base, data, config, arm, width=16, heads=4):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.base, self.arm = base, arm
        if arm.startswith('relation_'):
            install_relation_fusion(base.encoder, data.node_types, config['hidden'], arm, width)
        elif arm.startswith(('edge_', 'gatv2_')):
            modules = [(name, module) for name, module in base.encoder.named_modules()
                       if isinstance(module, GeneralConv)]
            for name, original in modules:
                parent, child = name.rsplit('.', 1)
                replacement = (EdgeGateConv(original, arm, width) if arm.startswith('edge_')
                               else GATv2WithRoot(original, arm, heads))
                setattr(base.encoder.get_submodule(parent), child, replacement)

    def forward(self, x, edges, pairs):
        return self.base(x, edges, pairs)

    def uniform(self, enabled):
        for module in self.modules():
            if isinstance(module, (RelationFusion, EdgeGateConv, ControlledGATv2)):
                module.force_uniform = enabled

    def attention(self):
        arrays, statistics = {}, {}
        for name, module in self.named_modules():
            if isinstance(module, (RelationFusion, EdgeGateConv, ControlledGATv2)):
                weights = module.last_weights
                if weights is None:
                    continue
                arrays[name] = weights.cpu().numpy()
                info = dict(shape=list(weights.shape), mean=float(weights.mean()),
                            std=float(weights.std()) if weights.numel() > 1 else 0.)
                if isinstance(module, RelationFusion):
                    info.update(mean_relation_weights=weights.mean(0).cpu().tolist(),
                                node_variation=float(weights.std(0).mean()),
                                normalisation_error=float((weights.sum(-1) - 1).abs().max()))
                elif isinstance(module, EdgeGateConv):
                    d = module.last_destination
                    degree = torch.bincount(d, minlength=module.last_n_target).to(weights.dtype)
                    sums = weights.new_zeros(module.last_n_target).index_add(0, d, weights)
                    expected = degree if module.mode == 'edge_gate_degree' else (degree > 0).to(weights.dtype)
                    info['normalisation_error'] = float((sums - expected).abs().max())
                statistics[name] = info
        return arrays, statistics


def initialise(args):
    root = Path(args.out_dir)
    if (root / 'protocol.json').exists():
        raise ValueError('Plan already exists; use run to resume')
    parent_path = Path(args.input_protocol)
    parent = json.loads(parent_path.read_text())
    for audit in parent['inputs'].values():
        for entry in audit['folds'].values():
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('Parent input changed')
    configs = copy.deepcopy(parent['configs'])
    if args.smoke:
        for cfg in configs.values():
            cfg['epochs'] = 3
    protocol = dict(
        tasks=list(configs), configs=configs, inputs=parent['inputs'],
        parent_protocol=str(parent_path.resolve()), parent_protocol_sha256=file_hash(parent_path),
        seeds=[666] if args.smoke else [666, 0, 1, 2, 3], arms=list(ARMS),
        primary_arms=list(PRIMARY), contrasts=[list(x) for x in CONTRASTS],
        width=16, heads=4, source_hashes=source_hashes(), smoke=args.smoke,
        optimizer='RMSprop', alpha=0.99, loss='unweighted BCEWithLogitsLoss',
        selection='one checkpoint maximising mean validation AUC/AUPR; earliest exact tie',
        evaluation_graph='same immutable cleaned training graph as previous round',
        initialisation='shared original core parameters; relation arms start at relation mean; degree gate starts at sum; GATv2 and mean gates start at neighbour mean',
        relation_control='same per-relation GeneralConv; uniform, globally shared and per-node relation weights',
        relation_scope='all incoming relation messages including the original self-node terms',
        gate_control='sigmoid edge scores normalised to sum 1 or incoming degree; uniform mean control',
        gatv2_control='4 heads, concatenated original width, original root projection, no added self loops, zero attention-vector initialisation; same transforms with uniform edge weights',
        intervention='after selection, replace learned weights with family-matched uniform weights; no retraining or selection',
        scope='fixed-split exploratory screen with original per-task schedules; no architecture-specific hyperparameter search',
        versions=dict(torch=torch.__version__, torch_geometric=torch_geometric.__version__),
    )
    protocol['expected_runs'] = len(protocol['tasks']) * len(protocol['seeds']) * len(ARMS)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'protocol.json', protocol)
    print('INITIALISED', root, protocol['expected_runs'], flush=True)


@torch.no_grad()
def predict(model, train, pairs):
    model.eval()
    return model(train.x_dict, train.edge_index_dict, pairs).cpu().numpy()


def train_one(root, protocol, data, task, seed, arm, device):
    directory = root / task / ('seed_' + str(seed)) / arm
    signature = dict(task=task, seed=seed, arm=arm, protocol_sha256=file_hash(root / 'protocol.json'))
    result_path = directory / 'result.json'
    if result_path.exists():
        if json.loads(result_path.read_text())['signature'] != signature:
            raise ValueError('Incompatible existing run')
        print('SKIP', task, seed, arm, flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'config.json', signature)
    config = protocol['configs'][task]
    key = (config['rna'], task, 'drug')
    train = copy.deepcopy(data['train']).to(device)
    base = make_base(train, config, seed, device)
    core_hash = parameter_hash(base)
    set_seed(seed + 2000003)
    model = PropagationModel(base, data['train'], config, arm,
                             protocol['width'], protocol['heads']).to(device)
    pairs = {fold: g[key].edge_label_index.to(device) for fold, g in data.items()}
    labels = {fold: g[key].edge_label.cpu().numpy() for fold, g in data.items()}
    initial = predict(model, train, pairs['val'][:, :64])
    optimizer = torch.optim.RMSprop((p for p in model.parameters() if p.requires_grad),
                                    lr=config['lr'], alpha=protocol['alpha'])
    best, trace, started = -float('inf'), [], time.monotonic()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(train.x_dict, train.edge_index_dict, pairs['train'])
        loss = F.binary_cross_entropy_with_logits(scores, train[key].edge_label)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        optimizer.step()
        validation = predict(model, train, pairs['val'])
        values = metrics(labels['val'], validation, pairs['val'].cpu().numpy(), by_drug=False)
        criterion = (values['auc'] + values['aupr']) / 2
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), val_auc=values['auc'],
                          val_aupr=values['aupr'], criterion=criterion))
        if criterion > best:
            best, best_epoch = criterion, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0 or epoch == config['epochs']:
            print(task, arm, 'seed', seed, 'epoch', epoch, 'val %.5f/%.5f' %
                  (values['auc'], values['aupr']), 'elapsed %.1fs' % (time.monotonic() - started), flush=True)
            save_csv(directory / 'trace.csv', trace)
    model.load_state_dict(best_state)
    predictions, results, interventions = {}, {}, {}
    for fold in ('val', 'test'):
        predictions[fold] = predict(model, train, pairs[fold])
        results[fold] = metrics(labels[fold], predictions[fold], pairs[fold].cpu().numpy())
    replay_delta = (results['val']['auc'] + results['val']['aupr']) / 2 - best
    if abs(replay_delta) > 5e-5:
        raise ValueError('Selected checkpoint replay mismatch')
    arrays, statistics = model.attention()
    if arrays:
        np.savez_compressed(directory / 'attention.npz', **arrays)
    if arm in PRIMARY:
        model.uniform(True)
        try:
            for fold in ('val', 'test'):
                predictions['uniform_' + fold] = predict(model, train, pairs[fold])
                interventions[fold] = metrics(labels[fold], predictions['uniform_' + fold],
                                              pairs[fold].cpu().numpy())
        finally:
            model.uniform(False)
    np.savez_compressed(directory / 'predictions.npz', initial_val_logits=initial, **predictions,
                        **{'y_' + f: labels[f] for f in ('val', 'test')},
                        **{'pairs_' + f: pairs[f].cpu().numpy() for f in ('val', 'test')})
    torch.save(best_state, directory / 'checkpoint.pt')
    record = dict(signature=signature, initial_core_sha256=core_hash, metrics=results,
                  uniform_intervention=interventions, selected_epoch=best_epoch,
                  trained_epochs=config['epochs'], validation_replay_delta=replay_delta,
                  attention_statistics=statistics,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  elapsed_seconds=time.monotonic() - started, device=device)
    write_json(result_path, record)
    print('RESULT', task, arm, seed, json.dumps(results), flush=True)
    del optimizer, model, base, train, best_state
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def run(args):
    root = Path(args.out_dir)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['source_hashes'] != source_hashes():
        raise ValueError('Source changed after plan freeze')
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard')
    torch.set_num_threads(args.threads)
    inputs = {}
    for task, audit in protocol['inputs'].items():
        inputs[task] = {}
        for fold, entry in audit['folds'].items():
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('Frozen input changed')
            inputs[task][fold] = torch.load(entry['path'], map_location='cpu')
    jobs = itertools.product(protocol['tasks'], protocol['seeds'], ARMS)
    for i, (task, seed, arm) in enumerate(jobs):
        if i % args.shard_count == args.shard_index:
            train_one(root, protocol, inputs[task], task, seed, arm, args.device)


def summarize(args):
    root = Path(args.out_dir)
    protocol = json.loads((root / 'protocol.json').read_text())
    records, rows, interventions = {}, [], []
    for task, seed, arm in itertools.product(protocol['tasks'], protocol['seeds'], ARMS):
        directory = root / task / ('seed_' + str(seed)) / arm
        r = json.loads((directory / 'result.json').read_text())
        if r['signature'] != dict(task=task, seed=seed, arm=arm, protocol_sha256=file_hash(root / 'protocol.json')):
            raise ValueError('Mixed experiment protocols')
        records[task, seed, arm] = r
        row = dict(task=task, seed=seed, arm=arm, selected_epoch=r['selected_epoch'],
                   trainable_parameters=r['trainable_parameters'])
        for fold, values in r['metrics'].items():
            row.update({fold + '_' + k: v for k, v in values.items()})
        rows.append(row)
        if arm in PRIMARY:
            intervention = dict(task=task, seed=seed, arm=arm)
            for fold, values in r['uniform_intervention'].items():
                for metric in METRICS:
                    intervention[fold + '_' + metric + '_delta'] = values[metric] - r['metrics'][fold][metric]
            interventions.append(intervention)
    for task, seed in itertools.product(protocol['tasks'], protocol['seeds']):
        if len({records[task, seed, a]['initial_core_sha256'] for a in ARMS}) != 1:
            raise ValueError('Initial core parameters differ')
    aggregates, comparisons, intervention_summary = [], [], []
    for task, arm in itertools.product(protocol['tasks'], ARMS):
        group = [r for r in rows if r['task'] == task and r['arm'] == arm]
        agg = dict(task=task, arm=arm, n=len(group))
        for fold, metric in itertools.product(('val', 'test'), METRICS):
            key = fold + '_' + metric
            values = [r[key] for r in group]
            agg[key + '_mean'] = float(np.mean(values))
            agg[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        aggregates.append(agg)
    for task, (arm, reference) in itertools.product(protocol['tasks'], CONTRASTS):
        row = dict(task=task, arm=arm, reference=reference, n=len(protocol['seeds']))
        for fold, metric in itertools.product(('val', 'test'), METRICS):
            delta = [records[task, s, arm]['metrics'][fold][metric] -
                     records[task, s, reference]['metrics'][fold][metric] for s in protocol['seeds']]
            key = fold + '_' + metric
            row[key + '_delta'] = float(np.mean(delta))
            row[key + '_wins'] = sum(d > 0 for d in delta)
        comparisons.append(row)
    for task, arm in itertools.product(protocol['tasks'], PRIMARY):
        group = [r for r in interventions if r['task'] == task and r['arm'] == arm]
        row = dict(task=task, arm=arm, n=len(group))
        for fold, metric in itertools.product(('val', 'test'), METRICS):
            key = fold + '_' + metric + '_delta'
            row[key] = float(np.mean([r[key] for r in group]))
            row[key + '_decreases'] = sum(r[key] < 0 for r in group)
        intervention_summary.append(row)
    for name, table in (('per_seed', rows), ('summary', aggregates), ('comparisons', comparisons),
                         ('interventions_per_seed', interventions), ('interventions_summary', intervention_summary)):
        save_csv(root / (name + '.csv'), table)
    report = ['# 传播内部注意力实验', '',
              f"完成 {len(rows)}/{protocol['expected_runs']} 次训练；每组 {len(protocol['seeds'])} 个种子。", '',
              '复用上一轮清理后的固定输入。所有评估使用训练图；仅用验证集 AUC/AUPR 均值选 checkpoint。',
              '原模型关系卷积和解码器初始参数配对一致。关系注意力从关系均值开始；保度门控从邻居求和开始；GATv2 与均值门控从邻居均值开始。', '',
              '| Task | Arm | Val AUC | Val AUPR | Test AUC | Test AUPR | Per-drug test AUC |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        report.append('| %s | %s | %.5f | %.5f | %.5f | %.5f | %.5f |' %
                      (r['task'], r['arm'], r['val_auc_mean'], r['val_aupr_mean'],
                       r['test_auc_mean'], r['test_aupr_mean'], r['test_per_drug_auc_mean']))
    report.extend(['', '## 配对对照', '',
                   '| Task | Comparison | Δ test AUC | AUC wins | Δ test AUPR | AUPR wins |',
                   '|---|---|---:|---:|---:|---:|'])
    for r in comparisons:
        report.append('| %s | %s − %s | %+.5f | %d/%d | %+.5f | %d/%d |' %
                      (r['task'], r['arm'], r['reference'], r['test_auc_delta'], r['test_auc_wins'],
                       r['n'], r['test_aupr_delta'], r['test_aupr_wins'], r['n']))
    report.extend(['', '## 已训练模型的等权干预', '',
                   '以下变化为等权干预后减去学习权重时的分数；负值表示模型依赖学习到的权重。', '',
                   '| Task | Arm | Δ test AUC | Δ test AUPR | AUPR decreases |',
                   '|---|---|---:|---:|---:|'])
    for r in intervention_summary:
        report.append('| %s | %s | %+.5f | %+.5f | %d/%d |' %
                      (r['task'], r['arm'], r['test_auc_delta'], r['test_aupr_delta'],
                       r['test_aupr_delta_decreases'], r['n']))
    report.extend(['', '## 验证集筛选', ''])
    for task in protocol['tasks']:
        candidates = [r for r in aggregates if r['task'] == task and r['arm'] in ('base',) + PRIMARY]
        best = max(candidates, key=lambda r: (r['val_auc_mean'] + r['val_aupr_mean']) / 2)
        report.append('- %s：原模型和三个主要候选中，验证集均值选择 `%s`。' % (task, best['arm']))
    report.extend(['', '## 限定', '',
                   '- 固定划分探索；种子只衡量优化波动，不能替代新的数据划分。',
                   '- 所有方案沿用原任务训练日程，未做架构专属超参数搜索；负结果只适用于本轮配置。',
                   '- 逐节点关系注意力需超过全局共享权重与均匀融合对照；GATv2 需超过相同变换的均匀聚合对照。',
                   '- 保度门控让各目标节点的边权总和等于入度；均值门控的边权总和为 1。自节点项保持原模型方式。',
                   '- GATv2 使用 4 头并拼接到原隐藏宽度，保留原自节点投影，不额外增加自环。',
                   '- GATv2 及其等权对照具有相同变换，新增输出偏置也配对；冻结不被等权对照使用的打分参数。',
                   '- 关系消息包括自节点项；没有为零入度关系额外添加筛选规则。',
                   '- 训练正例同时参与消息传递，遵循原协议；留出目标边不进入训练图。',
                   '- 等权干预使用联合训练后的同一个模型，度量预测依赖，不构成生物因果证据。',
                   '- 本轮未测试边属性感知注意力或路径实例注意力。', ''])
    (root / 'REPORT.md').write_text('\n'.join(report))
    print('\n'.join(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--out-dir', required=True)
    init.add_argument('--input-protocol', default='improve/results/attention_mechanisms_20260907/protocol.json')
    init.add_argument('--smoke', action='store_true')
    worker = sub.add_parser('run')
    worker.add_argument('--out-dir', required=True)
    worker.add_argument('--device', default='cuda:0')
    worker.add_argument('--threads', type=int, default=4)
    worker.add_argument('--shard-index', type=int, default=0)
    worker.add_argument('--shard-count', type=int, default=1)
    summary = sub.add_parser('summarize')
    summary.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    {'init': initialise, 'run': run, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
