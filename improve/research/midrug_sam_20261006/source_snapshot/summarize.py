"""Apply the frozen SAM-vs-random comparisons and archive honest outcomes."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==18
    assert all(r['status']=='passed' for r in v['records'])
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
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    protected_path=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(protected_path.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protected_records=[]
    for r in protected['records']:
        ds,f=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protected_records.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protected_records,source_unchanged=True,prior_full_replay_sha256=sha256(protected_path)),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['comparisons']['control']['auc']['mean_delta']:+.9f} | {r['comparisons']['control']['ap']['mean_delta']:+.9f} | {r['comparisons']['unmasked10']['auc']['mean_delta']:+.9f} | {r['comparisons']['unmasked10']['ap']['mean_delta']:+.9f} |" for r in rows)
    matched=[]
    for r in rows:
        if r['config'] not in plan['eligible_candidates']:continue
        n=r['config'].replace('sam','random');c=r['comparisons'][n]
        matched.append(f"| {r['config']} | {n} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |")
    new=[r for r in v['records'] if not r['reused_control']]
    max_error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    max_plain=max(r['plain_m123_inference_error'] for r in new)
    report=f'''# SAM 与随机参数扰动的内部对照

**原测试AUC/AP同时绝对+0.02仍未证实。** 本轮12次新拟合及12个新检查点重放通过；
6个原对照复用此前完整重放证据，再次核对源码/权重/概率哈希与原始概率指标。
复用核验不计为新GPU重放。没有新候选进入原验证或测试。

| 配置 | 内部AUC | 内部AP | 对纯M123 ΔAUC | ΔAP | 对unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

| 候选 | 等半径随机方向对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(matched)}

预先规定只有sam/sam_rec10可推进，需同时超过纯M123、unmasked10和匹配随机方向，
每个对照三折均值AUC/AP增益均≥.002。候选{best['config']}，最低增益
{best['selection_score']:+.9f}，推进原验证：{promote}。没有扩大半径或追加参数搜索。

这是相同前向架构的优化方式实验：完整两次反传的标准SAM与独立高斯扰动，
全局参数L2半径固定.05。两次计算共用重建节点，每轮一次RMSprop更新，BN累计一次。
模型恢复原参数后更新，预测时不使用扰动。重建仍为原始特征、权重0/10。

近期论文E2A已核读，但它使用条件VAE和表示扰动做OOD图分类，本轮不是其复现。
GraphSAM在每轮首步重算扰动梯度；当前每轮只有一个全批次步，因此没有使用其近似。
详细证据和适用限制见[文献记录](../midrug_literature_20261006/OPTIMIZATION_READING_NOTES.md)。

验证覆盖二次函数独立解、异常恢复、零半径数值对照、实际模型独立梯度重算、
BN计数、留出标签翻转、检查点指标/选轮/两次loss构成与半径/方向。
12个训练后模型做M1/M2/M3显式查询删除；最大logit误差{max_error:.3g}。
移除训练扰动和重建解码器、使用原M123推理，最大概率差{max_plain:.3g}。
浮点累加误差与精确恢复的区分见README及初始检查，不宣称CUDA训练逐位确定。

只加载独立miRNA/drug白名单图和原训练内部划分，无lncRNA、外部数据集、序列、
注释或新权重。PLMF10个受保护检查点、冻结图/划分和发布源码保持原哈希。
已验证纯miRNA M123+BPR原测试均值仍为AUC .9575892678、AP .9233078960；
最终目标仍为seed42/43/44测试均值AUC≥.9733929274、AP≥.9336833737，PLMF不下降。
'''
    diagnosis_path=HERE/'sensitivity_diagnostics.json'
    diagnosis=None
    if diagnosis_path.exists():
        diagnosis=json.loads(diagnosis_path.read_text())
        assert diagnosis['status']=='passed' and diagnosis['selection_sha256']==sha256(HERE/'selection.json')
        dm={r['config']:r for r in diagnosis['means']}
        report+=f'''
冻结选择后的训练损失诊断：相同参数半径.05、相同关联BCE下，一步梯度方向扰动的
平均损失增加从纯M123的{dm['control']['gradient_direction_excess']:.6f}降至
SAM的{dm['sam']['gradient_direction_excess']:.6f}；重建对照为
{dm['unmasked10']['gradient_direction_excess']:.6f}，SAM加重建为
{dm['sam_rec10']['gradient_direction_excess']:.6f}。
这表明本轮确实降低了这一特定方向敏感性，但没有产生足够的内部AUC/AP增益。
不将该数值称为Hessian、精确最大sharpness或泛化保证；详见
[训练敏感性记录](SENSITIVITY_DIAGNOSTICS.md)。selection.json在诊断前后哈希不变。
'''
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        current_turn='progress: 6 differentiable-completion fits audited, literature methods reviewed, 12 SAM/random fits and checkpoint replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='If promoted freeze outer-validation confirmation. Otherwise parameter sensitivity was reduced without sufficient dual-metric gain: do not revisit this grid; review pair-representation perturbation and existing related implementations before any new frozen experiment.')
    if diagnosis is not None:state['post_selection_sensitivity']=dict(status='passed',selection_changed=False,artifact_sha256=sha256(diagnosis_path))
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[标准SAM与随机方向对照](midrug_sam_20261006/REPORT.md)的12次新拟合与12次新检查点重放。
候选{best['config']}：内部AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}，
对指定对照双指标最小平均增益{best['selection_score']:+.9f}，推进原验证：{promote}。
此前[完整补全梯度](midrug_differentiable_completion_20261006/REPORT.md)六次拟合及审计已完成，未通过内部门槛。
[近期训练方法文献](midrug_literature_20261006/OPTIMIZATION_READING_NOTES.md)核读E2A与GraphSAM，并明确迁移限制。
未测试新候选；PLMF10个受保护检查点、冻结输入和发布源码哈希保持不变。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','plmf_hash_recheck.json','results/plan.json','initial_checks.json','README.md','predecessor_review.json','cuda_repeated_backward_diagnostic.json','cpu_repeated_backward_diagnostic.json']
    if diagnosis is not None:paths+=['sensitivity_diagnostics.json','SENSITIVITY_DIAGNOSTICS.md']
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},artifact_hashes={p:sha256(HERE/p) for p in paths})
    (HERE/'archive_manifest.json').write_text(json.dumps(archive,indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=12,new_checkpoint_replays=12,reused_controls=6,max_deletion_logit_error=max_error),indent=2))

if __name__=='__main__':main()
