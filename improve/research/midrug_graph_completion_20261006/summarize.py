"""Apply frozen matched-control selection; preserve protected PLMF artifacts."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());audit=json.loads((HERE/'verification.json').read_text())
    assert audit['status']=='passed' and len(audit['records'])==24
    assert all(r['status']=='passed' for r in audit['records'])
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    for n in plan['configs']:runs[n]=[json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for name,rs in runs.items():
        comparisons={}
        for ref in plan['comparators'].get(name,['control','unmasked10']):
            comparisons[ref]={}
            for metric in ['auc','ap']:
                d=[r['inner_'+metric]-b['inner_'+metric] for r,b in zip(rs,runs[ref])]
                comparisons[ref][metric]=dict(mean_delta=float(np.mean(d)),fold_deltas=d)
        rows.append(dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),
            mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),comparisons=comparisons,
            selection_score=min(comparisons[c][m]['mean_delta'] for c in comparisons for m in ['auc','ap'])))
        flat += [dict(config=name,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch']}) for f,r in enumerate(rs)]
    best=max((r for r in rows if r['config'] in plan['eligible_candidates']),key=lambda r:r['selection_score'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,
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
        if r['config'] not in plan['eligible_candidates']:continue
        suffix='_rec10' if r['config'].endswith('_rec10') else ''
        for mode in ['direct','uniform']:
            c=mode+suffix;v=r['comparisons'][c]
            matched.append(f"| {r['config']} | {c} | {v['auc']['mean_delta']:+.9f} | {v['ap']['mean_delta']:+.9f} |")
    new=[r for r in audit['records'] if not r['reused_control']]
    max_error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    coverage=json.loads((HERE/'coverage.json').read_text())
    cov_table='\n'.join(f"| {r['fold']} | {r['queries']} | {r['with_observed_path']} | {r['with_one_unknown_path']} | {r['newly_covered']} |" for r in coverage['records'])
    report=f'''# 第一层图预测器与精确软补全：内部实验

**原测试双指标绝对 +0.02 尚未证实。** 本轮18次训练、6个复用对照全部重放通过。
只使用原训练内部三折、seed42；没有评估新候选的原验证或测试。

| 配置 | 内部 AUC | 内部 AP | 对纯 M123 ΔAUC | ΔAP | 对 unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

| 软补全候选 | 匹配对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(matched)}

按冻结规则，候选 {best['config']} 对所有规定对照的双指标最小平均增益为
{best['selection_score']:+.9f}。内部 +.002 推进门槛：{promote}。
direct和uniform是对照，不在可推进候选中；没有事后把对照改称软补全成功。
最终仍需原测试三种子 AUC≥.9733929274、AP≥.9336833737，PLMF 两数据集不降。

## 方法与文献范围

上一轮原特征预测器的soft_rec10未超过匹配uniform_rec10，本轮引入共享第一层图
表示，检验训练图上下文是否提供更有效的权重。NCNC和2025年后续方法的核读记录
见../midrug_literature_20261006/COMPLETION_READING_NOTES.md；本轮仍是独立适配，
不是官方递归NCN复现。旧矩形包含排除恒等式继续复用，不另计全新方法家族。

每个模式都使用原M123的第一层GeneralConv表示，接LayerNorm、Linear128、
LeakyReLU、Linear32、缩放内积和特征偏置；0.1倍辅助BCE与直接logit均使用
查询删边后的端点表示。补全概率停止梯度，预测器及共享第一层通过直接/辅助损失学习。
原M1/M2/M3与图保持原实现；分支另算一次第一层以免改变发布源码。

删除(r,d)只改变第一层的两个端点。对直接预测和辅助监督，在激活前精确扣除
该边消息。对补全路径，左侧缺边概率使用修正miRNA端点与全体药物，右侧缺边
概率使用全体miRNA与修正药物端点；中间缺边的两个中间节点均未改变。
因此不能直接把包含查询消息的全局概率矩阵用于左/右缺边。

两条已观测正边构成的权重由A^T A、A A^T求得，再扣除重复端点贡献，分别与
修正后的预测行/列点积。中间缺边使用矩形包含排除。训练观测位置K排除已知
采样零标签和正例；i!=r、j!=d同时隔离查询在A/K中的变化。详细公式见README。
三个log(1+加权路径数)与直接查询logit进入4→64→1零初始化残差头。

direct只用同一图预测器的直接logit；uniform未知边权重为0.5；soft使用上述
查询修正的预测权重。三组共享相同辅助监督，分别配原重建0/10。软权重不是
真实关联或校准概率；没有硬伪标签或新增图边。

## 覆盖与检查

| 内部折 | 查询数 | 有三条观测边的路径 | 有两观测边加一未观测边的路径 | 新覆盖 |
|---|---:|---:|---:|---:|
{cov_table}

覆盖统计没有读取留出标签，不用于模型选择；可计算路径不证明缺失边真实存在。
原查询在A或K的变化均被i!=r、j!=d的排除规则隔离。即使删除记录使U[r,d]
重新非零，也无法进入任何保留路径。矩阵包含排除用float64计算再转回模型dtype。

初始核验包含12个加权矩形和梯度枚举、36个三模式查询、12个任意行/列修正对照、12个float64完整模型
预测/参数梯度删图案例、六配置真实训练步和非零分支梯度、原初始化一致以及
留出标签翻转。每个训练后checkpoint重做M1/M2/M3、独立路径枚举与最终预测
显式删记录检查；直接调用原PyG GeneralConv计算删图第一层，再枚举其软路径，最大最终logit差{max_error:.3g}。所有保存概率、指标、标签、
配对、索引、选轮及三项训练损失的float32计算重放一致。去掉重建解码器不改变推理。

没有lncRNA、外部数据、外部注释或新权重。PLMF10个checkpoint、图、划分和
发布源码哈希与此前完整重放一致。既有纯miRNA M123+BPR原测试均值仍为
0.9575892678 / 0.9233078960；本轮内部结果不作为最终测试达标证据。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn='progress: feature-only soft completion completed and audited',
        current_turn='progress: graph-conditioned completion with exact first-layer and probability-row/column repair, 18 fits and 24 checkpoint replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Freeze original-validation confirmation if promoted; otherwise use paired controls and coverage to assess the next mechanism.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';progress=p.read_text().split('最新完成')[0]
    progress+=f'''最新完成 [18次图条件软补全](midrug_graph_completion_20261006/REPORT.md)，精确修正第一层和概率行/列，比较直接图预测器、常数补全及学习权重。
内部候选 {best['config']}，AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；
对规定对照的双指标最小增益 {best['selection_score']:+.9f}，推进原验证：{promote}。
此前 [原特征软补全](midrug_soft_paths_20261006/REPORT.md)、[组间频率](midrug_class_prior_20261006/REPORT.md) 和 [路径矩](midrug_path_moments_20261006/REPORT.md) 均已审计，未通过内部门槛。
没有新候选进入原验证/测试，PLMF10个checkpoint、冻结输入和发布源码哈希复核通过。
'''
    p.write_text(progress)
    paths=['verification.json','selection.json','REPORT.md','plmf_hash_recheck.json','results/plan.json',
        'initial_checks.json','coverage.json','README.md','predecessor_review.json']
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths})
    (HERE/'archive_manifest.json').write_text(json.dumps(archive,indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=18,max_deletion_logit_error=max_error),indent=2))

if __name__=='__main__':main()
