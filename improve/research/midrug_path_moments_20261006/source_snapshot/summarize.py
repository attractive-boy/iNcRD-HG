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
    report=f'''# 同一路径中间节点联合矩：内部实验

**原测试 AUC/AP 均绝对 +0.02 尚未证实。** 18 次新训练与 6 个复用对照全部重放通过。
本轮仅原训练内部三折、seed42；没有评估新候选的原验证或测试。

| 配置 | 内部 AUC | 内部 AP | 对纯 M123 ΔAUC | ΔAP | 对 unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

| 候选 | 匹配矩对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(matched)}

按训练前规则，选中 {best['config']}，对所有规定对照的双指标最小平均增益
{best['selection_score']:+.9f}。内部 +0.002 推进门槛：{promote}。
最终仍需原测试三种子 AUC≥0.9733929274、AP≥0.9336833737，PLMF 两数据集不降。

## 方法与解释

在真实简单路径 r—j—i—d 上，显式计算两个中间节点的 rank16 投影乘积。
independent 使用匹配边际均值的乘积；joint 使用同一路径上乘积的平均值；
covariance 使用联合矩减边际均值乘积。三者均使用路径数/(路径数+1) 可靠性缩放、
相同 LayerNorm/MLP 头，残差末层零初始化；无路径时残差严格为零。
另外分别比较原无掩码重建权重 0/10，主干原 M123 参数与初始预测保持一致。

新摘要显式保留中间节点的配对；不同于旧端点双线性项或分别池化的路径注意力。
固定中间状态的小例子验证：边际均值相同而联合矩相反。它不证明原整个神经网络
无法间接学习这类信息，也不构成原创性或论文复现主张。
README.md 给出公式和既有实验区别，predecessor_review.json 记录阅读源码哈希。
路径覆盖率另存 coverage.json，未参与配置选择。

## 精确性与数值处理

两个查询端点都从路径中排除，所有中间状态都使用原 M1 的精确删边修正。
线性投影后的变化在联合乘积前施加，而非用包含查询边时的状态近似。
矩计算采用 float64 累积，再转回原模型 dtype。初版 float32 协方差消减误差没有
通过检查，已修正计算而没有放宽原检查容差。独立参考逐条枚举实际路径，并以
中心化乘积求协方差，避免与实现共用相同消减公式。

12 个 float64 最终输出/梯度删图案例通过；六配置真实训练步、非零分支梯度和
初始主干一致性通过。每个训练后 checkpoint 再次验证 M1/M2/M3、路径联合特征和
最终 logit 的显式删记录一致；最大最终 logit 差 {max_error:.3g}。
所有概率/标签/索引、选轮、float32 损失及重建爬升重放一致。去掉重建解码器不改变推理。
留出标签翻转不改变非零分支预测；没有 lncRNA、外部数据、外部注释或新预训练权重。

PLMF 10 个 checkpoint、图、划分及发布源码哈希与此前完整重放一致。
既有纯 miRNA M123+BPR 原测试均值仍为 0.9575892678 / 0.9233078960。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn='progress: observed-context experiment and common-ranking-error bound',
        current_turn='progress: exact joint path moments, independent enumeration and gradient checks, 18 fits and 24 checkpoint replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Freeze original-validation confirmation if promoted; otherwise use paired controls to determine the next mechanism.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';progress=p.read_text().split('最新完成')[0]
    progress+=f'''最新完成 [18 次路径联合矩实验](midrug_path_moments_20261006/REPORT.md)，比较匹配边际乘积、同路径联合矩与协方差。
内部选中 {best['config']}，AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；
对规定对照的双指标最小增益 {best['selection_score']:+.9f}。推进原验证：{promote}。

此前 [训练观测消息](midrug_observed_context_20261006/REPORT.md)、
[全局特征路由](midrug_global_features_20261006/REPORT.md)和
[精确编码器注意力](midrug_exact_attention_20261006/REPORT.md)均未通过内部推进门槛。
没有新候选进入原验证或测试；PLMF 的 10 个 checkpoint、冻结输入和发布源码哈希复核通过。
'''
    p.write_text(progress)
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        verification_sha256=sha256(HERE/'verification.json'),selection_sha256=sha256(HERE/'selection.json'),
        report_sha256=sha256(HERE/'REPORT.md'),plmf_hash_recheck_sha256=sha256(HERE/'plmf_hash_recheck.json'),
        coverage_sha256=sha256(HERE/'coverage.json'),plan_sha256=sha256(HERE/'results/plan.json'),
        initial_checks_sha256=sha256(HERE/'initial_checks.json'),
        derivation_sha256=sha256(HERE/'README.md'),predecessor_review_sha256=sha256(HERE/'predecessor_review.json'))
    (HERE/'archive_manifest.json').write_text(json.dumps(archive,indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=18,max_deletion_logit_error=max_error),indent=2))


if __name__=='__main__':main()
