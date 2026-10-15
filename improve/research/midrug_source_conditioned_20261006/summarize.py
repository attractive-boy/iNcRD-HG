"""Frozen true-source/uniform comparisons and protected PLMF evidence."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==18 and all(r['status']=='passed' for r in v['records'])
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    for n in plan['configs']:runs[n]=[json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for n,rs in runs.items():
        comparisons={ref:{metric:dict(mean_delta=float(np.mean([r['inner_'+metric]-b['inner_'+metric] for r,b in zip(rs,runs[ref])])),
            fold_deltas=[r['inner_'+metric]-b['inner_'+metric] for r,b in zip(rs,runs[ref])]) for metric in ['auc','ap']}
            for ref in plan['comparators'].get(n,['control','unmasked10'])}
        rows.append(dict(config=n,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),
            comparisons=comparisons,selection_score=min(c[m]['mean_delta'] for c in comparisons.values() for m in ['auc','ap'])))
        flat.extend(dict(config=n,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch']}) for f,r in enumerate(rs))
    best=max((r for r in rows if r['config'] in plan['eligible_candidates']),key=lambda r:r['selection_score'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))},no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
    for r in protected['records']:
        ds,f=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256'];assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,source_unchanged=True,prior_full_replay_sha256=sha256(original)),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['comparisons']['control']['auc']['mean_delta']:+.9f} | {r['comparisons']['control']['ap']['mean_delta']:+.9f} | {r['comparisons']['unmasked10']['auc']['mean_delta']:+.9f} | {r['comparisons']['unmasked10']['ap']['mean_delta']:+.9f} |" for r in rows)
    matched=[]
    for r in rows:
        if r['config'] not in plan['eligible_candidates']:continue
        ref=r['config'].replace('source','uniform');c=r['comparisons'][ref]
        matched.append(f"| {r['config']} | {ref} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |")
    new=[r for r in v['records'] if not r['reused_control']]
    error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions']);branch=max(x['max_error'] for r in new for x in r['branch_deletion'])
    batch=max(r['batch_max_logit_error'] for r in new);decoder=max(r['without_reconstruction_error'] for r in new)
    report=f'''# 全图药物源条件传播：内部迁移检验

**原测试AUC/AP同时绝对+0.02尚未证实。** 完成12次新拟合、12个新检查点重放。
另6个旧对照复用此前完整重放证据，当前再次核对源码/权重/数据/概率哈希及
原始配对/标签/索引/概率指标。这不算6次新的GPU重放。没有新候选进入原验证或测试。

| 配置 | 内部AUC | 内部AP | 对纯M123 ΔAUC | ΔAP | 对unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

| 候选 | 等范数均匀边界对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(matched)}

按冻结规则，候选{best['config']}对三个规定对照的双指标最小平均增益
{best['selection_score']:+.9f}，是否通过+.002内部推进门槛：{promote}。
原测试仍需seed42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF两集不下降。

本轮将本地PLMF已有三层源条件方法移到当前纯miRNA图，未声称发明新家族。
全部原miRNA/drug节点、训练正边及M2的零对角同类型关系参与传播，不截邻居。
source在查询药物放边界，uniform等L2范数地散布到所有药物，参数/运算预算相同。
初始miRNA为0，三层精确删边的第二/第三层状态经零初始化投影加入原lin1激活前。
原M1/M2/M3、主干初始化、对应重建初始化、主干优化器不变。
分支lr.003来自历史PLMF配方并预先冻结，不在本次MiDrug结果上搜索学习率。

历史BioPathNet实验的其他节点、背景注释、预训练父输出和折组遮边未导入；
本次不是BioPathNet六层DistMult/PNA的官方复现，也不以文献其他任务成绩作提升证据。

独立float64物理删图的dense单源传播验证层2/3输出和全部参数梯度，包括有向同类型
图、重复关联权重、度一端点和零查询。真实数据独立删图分支、非零投影下完整M123
删图预测、有限训练梯度、原初始化一致和留出标签翻转检查通过。
训练后12个检查点的最大分支删图误差{branch:.3g}、完整logit误差{error:.3g}、
批次/逐查询logit误差{batch:.3g}、移除重建解码器概率误差{decoder:.3g}。
新传播分支属于推理模型，不能移除后声称预测等价；仅重建解码器可移除。

只用原miRNA/drug输入，无lncRNA、外部数据集、序列、注释或新权重。
PLMF10个受保护检查点、冻结图/划分和发布源码哈希与此前完整重放一致。
已验证纯miRNA M123+BPR原测试均值仍为.9575892678/.9233078960，目标仍未完成。
'''
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        ds=json.loads(diag.read_text());assert ds['status']=='passed' and ds['selection_sha256']==sha256(HERE/'selection.json')
        report+='\n冻结选择后的[分支诊断](COMPONENT_DIAGNOSTICS.md)比较已训练模型、置零投影及切换边界；不用于重新选配置，置零分支不等于撤销它对主干训练的影响。\n'
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',previous_turn_classification='progress:12 pair perturbation fits and sensitivity/source review',
        current_turn='progress:12 full-graph source/uniform fits, exact nonlinear deletion/gradient verification and12 new checkpoint replays',best=best,promote_to_outer_validation=promote,
        no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Freeze outer-validation confirmation only if promoted; otherwise use the matched source/uniform evidence and component diagnostics to identify a distinct next hypothesis. Do not claim this complete-graph3-layer transfer is a new family.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[全图药物源条件传播](midrug_source_conditioned_20261006/REPORT.md)12次新拟合和12个新检查点重放。
候选{best['config']}：内部AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；对规定对照双指标最小平均增益{best['selection_score']:+.9f}，推进原验证：{promote}。
此前[配对表示扰动](midrug_pair_perturbation_20261006/REPORT.md)及其敏感性诊断保留。
没有新候选原验证/测试成绩；PLMF10个检查点、冻结输入和发布源码保持原哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','plmf_hash_recheck.json','results/plan.json','initial_checks.json','README.md','predecessor_review.json']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md']
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},artifact_hashes={p:sha256(HERE/p) for p in paths}),indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=12,new_checkpoint_replays=12,reused_controls=6,max_deletion_logit_error=error),indent=2))

if __name__=='__main__':main()
