"""Compare normalized encoders to both original controls under the frozen gate."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    verified=json.loads((HERE/'verification.json').read_text())
    assert verified['status']=='passed' and len(verified['records'])==24
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    runs={name:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for name,c in plan['controls'].items()}
    for name in plan['configs']:
        runs[name]=[json.loads((HERE/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for name,rs in runs.items():
        comparisons={}
        for cname in plan['controls']:
            comparisons[cname]={}
            for metric in ['auc','ap']:
                delta=[r['inner_'+metric]-c['inner_'+metric] for r,c in zip(rs,runs[cname])]
                comparisons[cname][metric]=dict(mean_delta=float(np.mean(delta)),fold_deltas=delta)
        rows.append(dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),
            mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),comparisons=comparisons,
            selection_score=min(comparisons[c][m]['mean_delta'] for c in comparisons for m in ['auc','ap'])))
        flat += [dict(config=name,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch']}) for f,r in enumerate(rs)]
    best=max((r for r in rows if r['config'] in plan['configs']),key=lambda r:r['selection_score'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    paired=[]
    for name,comparator in [('all05','target05'),('all10','target10'),('target05_rec10','target05'),('all05_rec10','all05')]:
        a=next(r for r in rows if r['config']==name);b=next(r for r in rows if r['config']==comparator)
        paired.append(dict(config=name,comparator=comparator,auc_delta=a['mean_inner_auc']-b['mean_inner_auc'],ap_delta=a['mean_inner_ap']-b['mean_inner_ap']))
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,paired_comparisons=paired,
        promote_to_outer_validation=promote,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))})
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results'
    protection=[]
    for r in protected['records']:
        ds,f=r['dataset'],r['fold']
        assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,
        prior_full_replay_sha256=sha256(original),source_unchanged=True),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | "
        f"{r['comparisons']['control']['auc']['mean_delta']:+.9f} | {r['comparisons']['control']['ap']['mean_delta']:+.9f} | "
        f"{r['comparisons']['unmasked10']['auc']['mean_delta']:+.9f} | {r['comparisons']['unmasked10']['ap']['mean_delta']:+.9f} |" for r in rows)
    report=f'''# 接收节点度归一化编码器：内部实验

**最终双指标 +0.02 尚未证实。** 完成 18 个新模型训练，全部 checkpoint 及 6 个原对照通过重放。
仅原训练内部三折、seed42；所有新候选原验证/测试均未评估。

| 配置 | 内部 AUC | 内部 AP | 对纯 M123 ΔAUC | ΔAP | 对 unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

预先固定选择规则为相对两个对照的四项平均增益的最小值。
选中 {best['config']}，分数 {best['selection_score']:+.9f}；达到内部双 +0.002 推进门槛：{promote}。
内部推进门槛不替代原测试 AUC≥0.9733929274、AP≥0.9336833737 的最终要求。

本轮 all05 相对纯 M123 的均值小幅提高，但第 1 折两项都下降；相对 unmasked10
的两项均值都较低。两种 0.5 次幂配置加重建后都没有超过 unmasked10，因此未找到
可推进的新候选。首折较好不能代表三折一致改善，也不能证明归一化在其他配置下无效。

原 GeneralConv 的邻居求和改为按接收节点度数的 0.5/1 次幂缩放，分别只作用于
关联关系或全部关系。自身投影和关系间平均不变；M3 图统计保留原定义。
两个附加配置将 0.5 次幂归一化与原特征重建权重 10 组合。
这不同于只降低 M2 池中高度节点的权重：本轮改变了两层编码器中的消息传播。

M1 在删掉当前查询消息后重新计算接收度分母；M2 中间节点的修正使用各自的接收度。
每个新 checkpoint 对至少 6 个训练正例/2 个负例显式删除记录并重建图，
检查端点、四个路径池、M3 和最终 logit；含最低关联度端点，覆盖 1→0 的边界情况。
所有模型选轮、float32 损失组成与重建权重爬升均核验；移除辅助解码器不改变关联推理。
初始测试中独立 CUDA 计算产生 5.96e-8 差异，标签翻转核验采用数值容差；
对应参数、度缩放、QIS/EASE 输入必须精确相等。没有为此改变训练模型。

所有输入仅 miRNA/drug，未引入 lncRNA 或外部数据。发布路径和 PLMF 的 10 个受保护
checkpoint、图、划分、源码哈希与此前完整重放一致；本次归一化只在研究目录中实现。
既有纯 miRNA M123+BPR 原测试均值仍为 0.9575892678/0.9233078960。
完整度修正推导、消融设置与复现说明见 README.md，逐折完整精度见 selection.json。
后续原 SMILES 分子表示方向及已测化学相似图的区别见 NEXT_EXPERIMENT.md。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn='progress: matrix completion and strict audit',current_turn='progress: 18 degree-aware encoder fits with complete query-deletion replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,
        processes_expected_live=[],next_action='No promotion. Original-SMILES molecular descriptors have been prepared separately; freeze molecular-encoder / atom-bag / original-ChemBERTa controls before new training. See NEXT_EXPERIMENT.md.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';progress=p.read_text().split('最新完成')[0]
    progress+=f'''最新完成 [18 次度归一化编码器实验](midrug_degree_encoder_20261006/REPORT.md)和全部核验。
内部选中 {best['config']}，AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}，
相对纯 M123 与 unmasked10 的双指标最小增益 {best['selection_score']:+.9f}。推进原验证：{promote}。

此前 [低秩补全混合](midrug_completion_20261006/REPORT.md)只有小幅增益，
[重建+BPR 组合](midrug_reconstruction_ranking_20261006/REPORT.md)未超过重建单独对照。
没有新候选进入原验证或测试。PLMF 的 10 个 checkpoint、冻结输入和发布源码哈希复核通过。
'''
    p.write_text(progress)
    print(json.dumps(dict(best=best,promote=promote,new_runs=18),indent=2))


if __name__=='__main__':main()
