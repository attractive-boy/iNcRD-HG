"""Apply the frozen paired-comparator rule and archive the completed experiment."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text());audit=json.loads((HERE/'verification.json').read_text())
    assert audit['status']=='passed' and len(audit['records'])==24
    assert all(r['status']=='passed' for r in audit['records'])
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    runs={name:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for name,c in plan['controls'].items()}
    for name in plan['configs']:runs[name]=[json.loads((HERE/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for name,rs in runs.items():
        comparisons={}
        for ref in plan['comparators'].get(name,['control','unmasked10']):
            comparisons[ref]={}
            for metric in ['auc','ap']:
                delta=[r['inner_'+metric]-b['inner_'+metric] for r,b in zip(rs,runs[ref])]
                comparisons[ref][metric]=dict(mean_delta=float(np.mean(delta)),fold_deltas=delta)
        rows.append(dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),
            mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),comparisons=comparisons,
            selection_score=min(comparisons[c][m]['mean_delta'] for c in comparisons for m in ['auc','ap'])))
        flat += [dict(config=name,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch']}) for f,r in enumerate(rs)]
    best=max((r for r in rows if r['config'] in plan['configs']),key=lambda r:r['selection_score'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,
        promote_to_outer_validation=promote,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))})
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
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
    matched=[]
    for r in rows:
        for c,v in r['comparisons'].items():
            if c in ['control','unmasked10']:continue
            matched.append(f"| {r['config']} | {c} | {v['auc']['mean_delta']:+.9f} | {v['ap']['mean_delta']:+.9f} |")
    new=[r for r in audit['records'] if not r['reused_control']]
    max_error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    report=f'''# 全局特征路由：内部实验

**原测试 AUC/AP 均绝对 +0.02 尚未证实。** 本轮 18 次新训练与 6 个复用对照全部重放通过。
只有原训练内部三折、seed42；没有评估新候选的原验证/测试。

| 配置 | 内部 AUC | 内部 AP | 对纯 M123 ΔAUC | ΔAP | 对 unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

| 候选 | 匹配特征变换对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(matched)}

按训练前规则，选中 {best['config']}，对所有规定对照的双指标最小平均增益
{best['selection_score']:+.9f}。内部 +0.002 推进门槛：{promote}。
最终目标仍为原测试三种子 AUC≥0.9733929274、AP≥0.9336833737，不能用内部结果代替。

## 方法和依据

小型残差特征变换先于原两层编码器。local 仅用自身 MLP，pool 汇聚两类各自均值，
route 用每类型 8 个 token 汇聚原节点特征，再以节点相关权重读取共 16 个 token。
所有模式分配同结构模块；不活动路径冻结，未宣称有效容量或计算量相同。
原特征残差输出从零初始化，保留相同 M123 初始参数；另比较原无掩码重建权重 0/10。
原相似边、M2/M3 统计及图划分均保持原定义。

该设计受 HDIGRL/RGDAGMAE 关于全局上下文的摘要启发，是独立适配，不是完整复现。
来源和已核实内容见 ../midrug_literature_20261006/FOLLOWUP_NOTES.md。
没有用论文中的数据、预训练权重或额外生物注释。

## 核验

特征变换只依赖原始节点特征，不读图边、标签、度数或查询集合。查询删边不会改变
该变换，随后原 M1 的完整消息和删除消息统一使用变换后的输入，M2 修正继续成立。
初始检查比较 9 个非零特征残差下的 float64 最终输出与梯度；覆盖正例、度一变零和非边。
六配置均检验零初始化一致性、节点重排等变性、真实图训练梯度和显式查询删除。
每个训练后模型再检查 6 个正例/2 个负例的 M1/M2/M3 及最终 logit；最大 logit 差
{max_error:.3g}。重放预测、标签/索引、选轮、float32 损失和重建爬升均一致。
去掉辅助重建解码器不改变推理；留出标签翻转不改变初始预测。

使用原冻结节点全集的无标签属性属于传导设置；不声称具有新节点归纳泛化证据。
PLMF 的 10 个 checkpoint、图、划分与发布源码哈希和此前完整重放一致。
既有纯 miRNA M123+BPR 原测试均值仍为 0.9575892678 / 0.9233078960。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn='progress: exact attentive M123, 18 fits and 30 checkpoint replays, no promotion',
        current_turn='progress: newer literature, query-invariant global feature transform, 18 fits and 24 checkpoint replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Freeze original-validation confirmation if promoted; otherwise use paired controls to determine the next mechanism.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';progress=p.read_text().split('最新完成')[0]
    progress+=f'''最新完成 [18 次全局特征变换实验](midrug_global_features_20261006/REPORT.md)，包括局部 MLP、均匀全局池和可学习路由。
内部选中 {best['config']}，AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；
对规定对照的双指标最小增益 {best['selection_score']:+.9f}。推进原验证：{promote}。

此前 [18 次精确编码器注意力实验](midrug_exact_attention_20261006/REPORT.md)已完成 30 个 checkpoint 复核，
最佳 target_rec10 内部为 0.954021318 / 0.914244961，比 unmasked10 仅 +0.000308836 / +0.000154379，未推进。
追加论文与适用边界见 [文献笔记](midrug_literature_20261006/FOLLOWUP_NOTES.md)。
没有新候选进入原验证或测试；PLMF 的 10 个 checkpoint、冻结输入和发布源码哈希复核通过。
'''
    p.write_text(progress)
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        verification_sha256=sha256(HERE/'verification.json'),selection_sha256=sha256(HERE/'selection.json'),
        report_sha256=sha256(HERE/'REPORT.md'),plmf_hash_recheck_sha256=sha256(HERE/'plmf_hash_recheck.json'))
    (HERE/'archive_manifest.json').write_text(json.dumps(archive,indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=18,max_deletion_logit_error=max_error),indent=2))


if __name__=='__main__':main()
