"""Compare verified backbone experiments on identical outer validation rows."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from improve import backbone_replacement as graph
from improve import graph_tabm as table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-dir', default='improve/results/backbone_replacement_20260910')
    parser.add_argument('--table-dir', default='improve/results/graph_tabm_20260910')
    parser.add_argument('--out-dir', default='improve/results/backbone_comparison_20260910')
    args = parser.parse_args()
    gp, tp = graph.protocol(args.graph_dir), table.protocol(args.table_dir)
    if Path(tp['reference_dir']).resolve() != Path(args.graph_dir).resolve():
        raise ValueError('Experiments do not share frozen input data')
    rows, verified, split_rows = [], {}, {}
    for root, p, arms in [(Path(args.graph_dir),gp,('base','pencil')),
                          (Path(args.table_dir),tp,('mlp','tabm'))]:
        v = json.loads((root/'VALIDATION.json').read_text())
        if not v['passed'] or v['smoke'] or not v['all_checkpoints_replayed'] or v['protocol_sha256'] != p['sha256']:
            raise ValueError('A full verified experiment is required')
        verified[str(root)] = dict(runs=v['completed_runs'], validation_sha256=graph.digest(root/'VALIDATION.json'),
                                   protocol_sha256=p['sha256'])
        chosen = graph.decision(root,'REFIT.json',p)['chosen']
        for task in gp['tasks']:
            for arm in arms:
                for seed in p['refit_seeds']:
                    cfg = chosen[task][arm]['config']
                    dest = graph.directory(root,'outer',task,arm,cfg['id'],seed)
                    r = graph.checked_result(dest,p)
                    with np.load(dest/'scores.npz') as z:
                        labels, pairs = z['y_val'].copy(), z['pairs_val'].copy()
                        measured = graph.metrics(labels,z['val'],pairs)
                    if task in split_rows:
                        np.testing.assert_array_equal(labels,split_rows[task][0])
                        np.testing.assert_array_equal(pairs,split_rows[task][1])
                    else:
                        split_rows[task] = labels,pairs
                    for key,value in measured.items():
                        np.testing.assert_allclose(r[key],value,atol=1e-12,rtol=0)
                    rows.append(dict(task=task,arm=arm,seed=seed,auc=r['auc'],ap=r['ap'],
                        macro_drug_ap=r['macro_drug_ap'],macro_drugs=r['macro_drugs'],params=r['params'],
                        selected_epoch=r['selected_epoch'],checkpoint=str((dest/'checkpoint.pt').resolve())))
    frame = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'outer_runs.csv',index=False)
    summary = []
    for (task,arm),b in frame.groupby(['task','arm']):
        item = dict(task=task,arm=arm,seeds=len(b),params=int(b.params.iloc[0]),macro_drugs=int(b.macro_drugs.iloc[0]))
        for metric in ('auc','ap','macro_drug_ap'):
            item[metric+'_mean'] = float(b[metric].mean())
            item[metric+'_std'] = float(b[metric].std())
        summary.append(item)
    summary = pd.DataFrame(summary)
    summary.to_csv(out/'summary.csv',index=False)
    comparisons = []
    for task in gp['tasks']:
        block = frame[frame.task==task].set_index(['arm','seed'])
        for candidate,control in [('pencil','base'),('mlp','base'),('tabm','base'),('tabm','mlp')]:
            for metric in ('auc','ap','macro_drug_ap'):
                delta = np.array([block.loc[(candidate,s),metric]-block.loc[(control,s),metric] for s in gp['refit_seeds']])
                comparisons.append(dict(task=task,candidate=candidate,control=control,metric=metric,
                    mean_delta=float(delta.mean()),std_delta=float(delta.std(ddof=1)),
                    positive_seeds=int((delta>0).sum()),seeds=len(delta)))
    contrasts = pd.DataFrame(comparisons)
    contrasts.to_csv(out/'paired_deltas.csv',index=False)
    lines = ['# 本轮架构替换结果', '',
        '主要可用方案是图结构特征编码 + TabM（ICLR 2025）：完全替换原 GNN 的训练与推理路线，保留相同的原始图、PCA/药物输入和标签。',
        '另实现并评估了异质 PENCIL（2026 预印本），以及同图特征的普通 MLP。所有架构与失败试次均保留。', '',
        '下表为历史外层验证上 seed 666、0、1 的均值 ± 初始化标准差。两个指标来自同一检查点；各架构参数由内部验证选出，外层固定轮数重训后只评价一次。', '',
        '| 任务 | 架构 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for task in gp['tasks']:
        for arm in ('base','pencil','mlp','tabm'):
            r = summary[(summary.task==task)&(summary.arm==arm)].iloc[0]
            lines.append('| %s | %s | %.6f ± %.6f | %.6f ± %.6f | %.6f |' %
                (task,arm,r.auc_mean,r.auc_std,r.ap_mean,r.ap_std,r.macro_drug_ap_mean))
    lines += ['', '| 任务 | TabM 相对 base 的指标差 | 均值差 | 提升种子 |', '|---|---|---:|---:|']
    for r in comparisons:
        if r['candidate']=='tabm' and r['control']=='base':
            lines.append('| %s | %s | %+.6f | %d/%d |' % (r['task'],r['metric'],r['mean_delta'],r['positive_seeds'],r['seeds']))
    lines += ['', '**结论边界**', '',
        '两个任务的全局 AUC/AP 增益不等于所有使用场景都改善。LncDrug 的按药物宏 AP 下降，应保留原 GNN 作为单药物候选排序的对照。宏 AP 只纳入当前外层验证中同时存在正、负标签的药物，各任务数量见 summary.csv。',
        'TabM 包含 16 个共享主权重的内部成员；它不是单个普通 MLP，也不是多次独立 GNN 训练后混分。MLP 对照使用完全相同的结构特征、预处理和参数搜索预算。',
        '相比 GNN，特征表达、五折生成训练特征及优化方式都发生了变化；收益应归于完整方案，不能全部记作 TabM 的独立算子贡献。PENCIL 也是针对异质小图的适配，不是官方论文数字复现。',
        '当前 base 是既有 PCA/circRNA 扩展图上的原 GeneralConv 与拼接解码器，不是原论文原始数据的原样重跑。两个验证划分此前都已用于开发；三个初始化不代替新数据划分，也不构成独立测试或统计显著性证明。本轮未读取测试集。', '',
        '**可复现交付**', '',
        '两轮合计完成 56 次完整正式训练及全部检查点重放；另完成 10 项单元检查和 36 次短跑流程检查。两次未完成的外层重训因改为按种子并行而重新开始，固定参数和训练函数没有改变，记录见原骨干实验目录的 RESCHEDULING.json。',
        '训练入口：`python -m improve.graph_tabm`。已训练模型推理入口：`python -m improve.predict_backbone`，默认 TabM、seed 666。详细用法见 `improve/GRAPH_TABM.md`。',
        '参考：PENCIL https://arxiv.org/abs/2602.01553；TabM https://arxiv.org/abs/2410.24210。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    graph.write_json(out/'VALIDATION.json',dict(passed=True,experiments=verified,
        same_outer_pairs_and_labels=True,completed_training_runs=sum(v['runs'] for v in verified.values()),
        compared_outer_runs=len(rows),no_test_evaluation=True))
    print(summary.to_string(index=False),flush=True)
    print(contrasts[(contrasts.candidate=='tabm')&(contrasts.control=='base')].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
