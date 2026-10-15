"""Apply the frozen dual-metric gate and verify protected PLMF artifacts."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==24 and all(r['status']=='passed' for r in v['records'])
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
    error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions']);branch=max(x['max_error'] for r in new for x in r['branch_deletion'])
    batch=max(r['batch_max_logit_error'] for r in new);decoder=max(r['without_reconstruction_error'] for r in new)
    report=f'''# 源条件传播中的训练零观测：内部组合检验

**原测试AUC/AP同时绝对+0.02尚未证实。** 完成12次新拟合、12个新检查点重放。
另12个旧对照复用此前完整重放证据，本轮再次核对源码/权重/数据/概率哈希及
配对/标签/索引/指标，不算12次新的GPU重放。没有新候选原验证或测试成绩。

| 配置 | 内部AUC | 内部AP |
|---|---:|---:|
{table}

| 零观测候选 | 规定对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(paired)}

按冻结规则，候选{best['config']}对四个规定对照的双指标最小平均增益
{best['selection_score']:+.9f}；通过+.002内部推进门槛：{promote}。
原测试仍需seed42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF两集不下降。

本轮组合既有完整药物源条件传播与原训练正/零观测消息，未声称新家族。
新增关系零初始化，原source/主干/重建参数及初始预测一致。全部原节点参加，
第二/三层查询miRNA状态经投影进入原lin1。原M1/M2/M3不变。
额外正例关系为参数/初始化/优化预算匹配对照；其边数和拓扑不同，不能把差异
单独解释为符号语义。零记录仅是抽样观测状态，不是耐药/敏感/拮抗关系。

三层精确删记录同时删除A和C的查询值，包括零标签查询。独立float64 dense单源
删图核对输出及全部参数梯度，涵盖重叠关系、重复权重、度一节点、有向同类图。
真实图非零新增权重完整模型删图和留出标签翻转检查通过。
12个训练后检查点最大分支删图误差{branch:.3g}，完整logit误差{error:.3g}，
批次/单查询logit误差{batch:.3g}，移除重建解码器概率误差{decoder:.3g}。
源传播和观测消息均属于推理模型；只有重建解码器可等价移除。

不使用lncRNA、外部数据集、序列、注释或新权重。
PLMF10个受保护检查点、冻结图/划分和发布源码与此前完整重放哈希一致。
既有纯miRNA M123+BPR原测试均值仍为.9575892678/.9233078960，目标未完成。
'''
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        ds=json.loads(diag.read_text());assert ds['status']=='passed' and ds['selection_sha256']==sha256(sp)
        report+='\n冻结选择后的[分支诊断](COMPONENT_DIAGNOSTICS.md)比较完整预测、置零额外观测变换和置零源投影；不重新选配置，也不把干预等同于撤销联合训练。\n'
        selected_diag=next(r for r in ds['means'] if r['config']==best['config'])
        effect={k:selected_diag['full'][k]-selected_diag['zero_observation'][k] for k in ['auc','ap']}
        report+=f"\n已训练{best['config']}相对置零C消息的干预，内部AUC/AP差为{effect['auc']:+.9f}/{effect['ap']:+.9f}，平均概率变化{selected_diag['observation_mean_abs_probability_shift']:.6f}。额外消息确实参与预测，但相对冻结训练对照的双指标收益仍未通过门槛。\n"
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn_classification='progress:full-source experiment archived and component effects verified',
        current_turn='progress:12 source-conditioned observation fits, exact dual-relation deletion and12 new checkpoint replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='If promoted, freeze original-validation confirmation. Otherwise use matched observation and component evidence to choose a distinct next hypothesis; do not repeat a relation/depth grid automatically.')
    if diag.exists():
        state['component_diagnostics']=dict(status='passed',selection_unchanged=True,means=ds['means'])
        state['component_findings']=dict(selected_model_full_minus_zero_observation=effect,
            interpretation='Additional observation messages affect predictions; this intervention cannot undo joint training and does not override the failed paired gate.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[源条件观测关系组合](midrug_source_observed_20261006/REPORT.md)12次新拟合和12个新检查点重放。
候选{best['config']}：内部AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；对规定对照双指标最小平均增益{best['selection_score']:+.9f}，推进原验证：{promote}。
此前[完整源传播](midrug_source_conditioned_20261006/REPORT.md)及其分支诊断已经归档。
没有新候选原验证/测试成绩；PLMF10个检查点、冻结输入和发布源码保持原哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    literature=json.loads((HERE/'literature_manifest.json').read_text())
    for n,h in literature['files'].items():assert sha256(Path(literature['source_directory'])/n)==h
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json','initial_checks.json','README.md','predecessor_review.json','literature_manifest.json']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies=plan['source_hashes']),indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=12,new_checkpoint_replays=12,reused_controls=12,max_deletion_logit_error=error),indent=2))

if __name__=='__main__':main()
