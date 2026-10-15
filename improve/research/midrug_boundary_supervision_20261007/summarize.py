"""Apply the frozen dual-metric gate and verify protected PLMF artifacts."""
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
    sp=HERE/'selection.json'
    if sp.exists():assert json.loads(sp.read_text())==selection
    else:sp.write_text(json.dumps(selection,indent=2))
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
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} |" for r in rows)
    paired=[]
    for r in rows:
        if r['config'] not in plan['eligible_candidates']:continue
        for ref,c in r['comparisons'].items():paired.append(f"| {r['config']} | {ref} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |")
    new=[r for r in v['records'] if not r['reused_control']]
    error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    batch=max(r['batch_max_logit_error'] for r in new);decoder=max(r['without_reconstruction_error'] for r in new)
    report=f'''# M123配对边界辅助监督：内部检验

**原测试AUC/AP同时绝对+0.02尚未证实。** 本轮12次新拟合、12个新检查点重放。
另6个纯M123/unmasked10旧对照复用既有完整重放，重新检查哈希、配对标签及指标；
不把这6个核查算作新GPU重放。没有新候选原验证或测试成绩。

| 配置 | 内部AUC | 内部AP |
|---|---:|---:|
{table}

| 候选 | 规定对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(paired)}

按冻结规则选{best['config']}；对所有规定对照双指标最小平均增益为
{best['selection_score']:+.9f}，通过+.002内部推进门槛：{promote}。
最终仍需原MiDrug测试seed42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF两集不降。

## 实现范围与匹配条件

训练从原初始化开始，仅用原训练配对的查询删除后lin3输入。
每轮训练表示的标准化、加对角项协方差和正确中心偏移LDA方向产生边界距离。
boundary在每个观测类中选距离最低的四分之一；random按类无放回选同样数量。
两臂都计算几何和两种索引，在训练轨迹不同之后仍保证每轮样本/类别预算相同。
这是类内固定预算策略，不是前期全局25%+5NN诊断，也不是BES原样复现。

选中表示与两个停止梯度的类别原型计算余弦交叉熵，温度.2、最大权重.05，
30–60轮爬升；原重建权重0/10、前50轮爬升。mean/scale/原型/选样均停止梯度，
辅助梯度只直接进入选中配对表示，再更新原共享参数。不能把动态停止梯度训练
规则说成所有统计量的完整目标梯度，也不声称潜变量可识别性或生物因果分类。
原M1/M2/M3、优化器、初始化、输入、图和划分保留；没有增加推理模型参数。

独立NumPy几何/损失/显式梯度、固定上下文有限差分、未选配对零辅助梯度、
类预算/缩放平移不变性、常量特征有限、零权重BCE等价、真实训练BN一次及
留出标签翻转检查通过。12个训练后checkpoint核对预测/指标/选轮/损失组成、
逐轮选样数量、完整查询删记录和无hook残留；最大完整logit删记录误差{error:.3g}，
批次误差{batch:.3g}，移除重建解码器概率误差{decoder:.3g}。

没有lncRNA、外部数据、序列、注释或新预训练权重。
PLMF10个受保护检查点、冻结图/划分及发布源码哈希与此前完整重放一致。
已验证纯miRNA M123+BPR原测试均值仍为.9575892678/.9233078960，本轮内部成绩
不替代原测试验收。双指标+.02目标仍未完成。
'''
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        ds=json.loads(diag.read_text());assert ds['status']=='passed' and ds['selection_sha256']==sha256(sp)
        selected=next(r for r in ds['means'] if r['config']==best['config'])
        report+=f"\n冻结选择后的[训练信号诊断](COMPONENT_DIAGNOSTICS.md)：{best['config']}在eval模式训练配对上的边界原型损失为{selected['boundary_auxiliary']:.6f}；.05加权辅助梯度/BCE梯度范数比为{selected['boundary_gradient']['weighted_norm_ratio']:.6f}，余弦为{selected['boundary_gradient']['cosine_to_clean']:.6f}。这是lin3前表示的梯度诊断，不更新参数、不重新选型，也不将当前梯度当成历史因果效应。\n"

    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn_classification='progress:conditioned-expression fits and source/boundary diagnostics',
        current_turn='progress:12 class-budget-matched boundary/random fits, independent prototype gradients and12 new checkpoint replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='If promoted, freeze original-validation confirmation. Otherwise review the original-expression column-token hypothesis in post_result_review.json and implement a matched, query-deletion-verified test if distinct from existing code. Do not import historical annotation MIL inputs or automatically search boundary temperatures/weights/fractions.')
    if diag.exists():state['component_diagnostics']=dict(status='passed',selection_unchanged=True,means=ds['means'])
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[配对边界辅助监督](midrug_boundary_supervision_20261007/REPORT.md)12次新拟合和12个新检查点重放。
候选{best['config']}：内部AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；对规定对照双指标最小平均增益{best['selection_score']:+.9f}，推进原验证：{promote}。
此前[药物条件化表达传播](midrug_conditioned_expression_20261006/REPORT.md)及近期文献/边界预检保留。
没有新候选原验证/测试成绩；PLMF10个检查点、冻结输入和发布源码保持原哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json','initial_checks.json','README.md','predecessor_review.json','inner_summary.csv','check.log','post_result_review.json']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies=plan['source_hashes']),indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=12,new_checkpoint_replays=12,reused_controls=6,max_deletion_logit_error=error),indent=2))

if __name__=='__main__':main()
