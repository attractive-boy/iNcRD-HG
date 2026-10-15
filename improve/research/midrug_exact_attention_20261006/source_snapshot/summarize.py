"""Apply the predeclared dual-metric, matched-uniform-control selection rule."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text());audit=json.loads((HERE/'verification.json').read_text())
    assert audit['status']=='passed' and len(audit['records'])==30
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    runs={name:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for name,c in plan['controls'].items()}
    for name in plan['configs']:runs[name]=[json.loads((HERE/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for name,rs in runs.items():
        refs=['control','unmasked10']
        if name in plan['matched_uniform_controls']:
            matched=plan['matched_uniform_controls'][name]
            if matched not in refs:refs.append(matched)
        comparisons={}
        for ref in refs:
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
    intervention=[]
    for name in plan['configs']:
        record=[next(r for r in audit['records'] if r['tag']==f'{name}_f{f}_s42' and not r['reused_control']) for f in range(3)]
        intervention.append(dict(config=name,
            auc=float(np.mean([r['gate_diagnostics']['uniform_intervention_auc'] for r in record])),
            ap=float(np.mean([r['gate_diagnostics']['uniform_intervention_ap'] for r in record])),
            max_prediction_changes=[r['gate_diagnostics']['max_prediction_change'] for r in record]))
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,uniform_interventions=intervention,
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
    itable='\n'.join(f"| {r['config']} | {r['auc']:.9f} | {r['ap']:.9f} |" for r in intervention)
    new=[r for r in audit['records'] if not r['reused_control']]
    max_error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    report=f'''# 编码器注意力与精确查询删边：内部实验

**原测试 AUC/AP 均绝对 +0.02 尚未证实。** 本轮 18 次新训练与 12 个复用均匀权重对照
全部重放通过；新模型原验证/测试均未评估。仅原训练内部三折、seed42。

| 配置 | 内部 AUC | 内部 AP | 对纯 M123 ΔAUC | ΔAP | 对 unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

按训练前的双指标规则，选中 {best['config']}，最小平均增益 {best['selection_score']:+.9f}。
对照包含纯 M123、unmasked10 以及配置对应的均匀编码器。内部 +0.002 推进门槛：{promote}。
这不是最终 +0.02 的验收；原测试仍需 AUC≥0.9733929274、AP≥0.9336833737。

## 方法与核验

保留原两层消息/自身投影与 M123 融合头。新增原始节点特征驱动的 width16 边门，
权重范围 [0.1,0.9]，按接收门质量归一化并保留 degree^(1-alpha) 尺度。
alpha=0/.5、目标关系/全部关系、重建权重 0/10 的六项配置在训练前固定。
均匀权重时分别恢复原编码器或上一轮度归一化编码器，初始数值核验通过。

查询删除同步更新第一层分母及消息、第二层所有受影响边的注意力，再更新 M2 中间节点池。
事件索引按查询内容缓存；没有使用包含查询边时的注意力权重替代真实删边后的权重。
M3 和训练内标准化保持原定义。门函数不同于早期原模型上的传播注意力脚本，
后者没有集成本轮 M123 精确查询删除；本轮没有调用其数据或测试执行器。

NumPy 原型通过 240 组前向对照。PyTorch float64 又比较 12 个非均匀注意力下的
最终输出及参数梯度，覆盖关系作用范围、alpha、度数归零和非边。
真实首折完整训练步含重建的峰值显存约 3.51 GiB。
每个新 checkpoint 对 6 个正例/2 个负例显式删记录并重建图，检查 M1、四个 M2 池、M3
与最终 logit；最大最终 logit 删边差 {max_error:.3g}。
预测、选轮、精确 float32 损失与重建权重爬升均核验，去掉辅助重建解码器不改变推理。

## 所选 checkpoint 的均匀化诊断

| 配置 | 强制均匀门后的内部 AUC | AP |
|---|---:|---:|
{itable}

上表是在选定权重上替换门，不是单独训练的对照，未参与选型。
正式归因仍以匹配的均匀编码器对照为准；不能仅凭某次替换降低分数就声称泛化收益。
逐关系门范围及波动、逐折结果保存于 verification.json / selection.json。

没有 lncRNA 或外部数据，发布路径未修改。PLMF 10 个 checkpoint、图、划分及原源码
哈希与此前完整重放一致。既有纯 miRNA M123+BPR 原测试均值仍为 0.9575892678/0.9233078960。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn='progress: molecular branch controls and NumPy attention derivation',
        current_turn='progress: exact attentive M123 implementation, autograd and memory checks, 18 fits and 30 checkpoint replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Freeze and confirm the selected config on original validation if the prespecified internal gate passes; otherwise use the paired results to choose a different hypothesis.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';progress=p.read_text().split('最新完成')[0]
    progress+=f'''最新完成 [18 次编码器注意力实验](midrug_exact_attention_20261006/REPORT.md)，包含精确查询删边与梯度核验。
内部选中 {best['config']}，AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；
相对规定对照的双指标最小增益 {best['selection_score']:+.9f}。推进原验证：{promote}。

此前 [分子表示分支](midrug_molecular_encoder_20261006/REPORT.md)和
[度归一化编码器](midrug_degree_encoder_20261006/REPORT.md)均未超过单独重建。
没有新候选进入原验证或测试。PLMF 的 10 个 checkpoint、冻结输入和发布源码哈希复核通过。
'''
    p.write_text(progress)
    print(json.dumps(dict(best=best,promote=promote,new_fits=18,max_deletion_logit_error=max_error),indent=2))


if __name__=='__main__':main()
