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
    report=f'''# 两观测边与一条软缺边：内部实验

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

NCNC（ICLR2024，arXiv:2302.00890v4）式10以及已核读的2025年后续方法，
提供了在已知邻居附近补全的思路。本轮不是NCNC原样复现：使用原特征MLP、
rank32缩放内积与特征偏置预测软权重；它由同一训练集0.1倍BCE辅助损失学习，
没有外部数据或预训练权重。每组都接收该预测器的直接logit，保证配对特征监督相同。

令A为训练正边，K为训练观测位置，U=(1-K)*sigmoid(L)。路径r—j—i—d只允许
恰好一条边来自U、另两条来自A，且i!=r、j!=d。缺失左/中/右边分别汇总，
使用log(1+加权路径数)，与直接logit共同进入4→64→1零初始化残差头。
uniform使用U=(1-K)*0.5；direct把三个路径输入置零。均搭配原重建0/10。
补全权重停止梯度，预测器经直接logit和辅助BCE学习；M123编码器的图不加伪边。

旧rectangle_modules.py已有三个潜在配对的乘法。本轮使用同一包含排除恒等式，
但固定两条观测正边且只补一条未观测边，是受限结构消融，不是全新方法家族。
旧硬伪边方法会改图，本轮不会；已观测采样零标签不会作为待补边。
权重不等于真实关联，也不声称为校准的存在概率。

## 覆盖与检查

| 内部折 | 查询数 | 有三条观测边的路径 | 有两观测边加一未观测边的路径 | 新覆盖 |
|---|---:|---:|---:|---:|
{cov_table}

覆盖统计没有读取留出标签，不用于模型选择；可计算路径不证明缺失边真实存在。
原查询在A或K的变化均被i!=r、j!=d的排除规则隔离。即使删除记录使U[r,d]
重新非零，也无法进入任何保留路径。矩阵包含排除用float64计算再转回模型dtype。

初始核验包含12个加权矩形和梯度枚举、36个三模式查询、12个float64完整模型
预测/参数梯度删图案例、六配置真实训练步和非零分支梯度、原初始化一致以及
留出标签翻转。每个训练后checkpoint重做M1/M2/M3、独立路径枚举与最终预测
显式删记录检查，最大最终logit差{max_error:.3g}。所有保存概率、指标、标签、
配对、索引、选轮及三项训练损失的float32计算重放一致。去掉重建解码器不改变推理。

没有lncRNA、外部数据、外部注释或新权重。PLMF10个checkpoint、图、划分和
发布源码哈希与此前完整重放一致。既有纯miRNA M123+BPR原测试均值仍为
0.9575892678 / 0.9233078960；本轮内部结果不作为最终测试达标证据。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn='progress: path moments and class-frequency experiments completed and audited',
        current_turn='progress: restricted soft path completion, exact query/gradient checks, 18 fits and 24 checkpoint replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Freeze original-validation confirmation if promoted; otherwise use paired controls and coverage to assess the next mechanism.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';progress=p.read_text().split('最新完成')[0]
    progress+=f'''最新完成 [18次受限软路径补全](midrug_soft_paths_20261006/REPORT.md)，比较直接特征预测、常数补全和学习权重补全。
内部候选 {best['config']}，AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；
对规定对照的双指标最小增益 {best['selection_score']:+.9f}，推进原验证：{promote}。
此前 [组间频率](midrug_class_prior_20261006/REPORT.md) 和 [路径矩](midrug_path_moments_20261006/REPORT.md) 均已审计，未通过内部门槛。
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
