"""Freeze the sole ensemble gate, then diagnose diversity and protect PLMF."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,RELEASE,sha256,load_inner,probability_mean

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==27 and v['new_checkpoint_replays']==21 and v['immutable_reused_control_checks']==6
    for p,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(p)==h
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    members=[[runs['unmasked10'][f]]+[json.loads((HERE/'results'/f'member{j}_f{f}_r42.json').read_text()) for j in range(1,8)] for f in range(3)]
    runs['mean_individual_members']=[{f'inner_{m}':float(np.mean([r[f'inner_{m}'] for r in rs])) for m in ['auc','ap']} for rs in members]
    runs['ensemble8']=[json.loads((HERE/'results'/f'ensemble8_f{f}_r42.json').read_text()) for f in range(3)]
    rows=[dict(config=n,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs]))) for n,rs in runs.items()]
    best=next(r for r in rows if r['config']=='ensemble8')
    best['comparisons']={n:{m:dict(mean_delta=float(np.mean([r['inner_'+m]-b['inner_'+m] for r,b in zip(runs['ensemble8'],runs[n])])),
        fold_deltas=[r['inner_'+m]-b['inner_'+m] for r,b in zip(runs['ensemble8'],runs[n])]) for m in ['auc','ap']} for n in plan['comparators']}
    best['selection_score']=min(c[m]['mean_delta'] for c in best['comparisons'].values() for m in ['auc','ap'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_r42.json'))},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    sp=HERE/'selection.json'
    if sp.exists():assert json.loads(sp.read_text())==selection
    else:sp.write_text(json.dumps(selection,indent=2))
    # Selection is now immutable. Diagnostics below cannot choose members or weights.
    diagnostics=[]
    for fold,rs in enumerate(members):
        data=load_inner(fold);path=HERE/'results'/f'ensemble8_f{fold}_r42_inner.npz'
        assert sha256(path)==runs['ensemble8'][fold]['probability_sha256']
        with np.load(path) as z:p=z['member_probabilities'];ensemble=z['probability']
        np.testing.assert_array_equal(ensemble,probability_mean(p));y=data['val_y'];residual=p.astype(np.float64)-y[None,:]
        cross=np.corrcoef(residual)[np.triu_indices(8,1)]
        meansq_members=float(np.mean(residual**2));meansq_ensemble=float(np.mean((ensemble-y)**2))
        variance=float(np.mean((p.astype(np.float64)-ensemble[None,:])**2))
        assert abs(meansq_members-meansq_ensemble-variance)<1e-12
        diagnostic=dict(fold=fold,mean_residual_correlation=float(cross.mean()),min_residual_correlation=float(cross.min()),max_residual_correlation=float(cross.max()),
            mean_member_probability_sd=float(np.std(p.astype(np.float64),axis=0).mean()),
            mean_member_brier=meansq_members,ensemble_brier=meansq_ensemble,probability_variance=variance,
            fraction_threshold_disagreement=float(((p.min(0)<.5)&(p.max(0)>=.5)).mean()),
            individual_auc=[r['inner_auc'] for r in rs],individual_ap=[r['inner_ap'] for r in rs],
            best_epochs=[r['best_epoch'] for r in rs],epochs_run=[r['epochs_run'] for r in rs])
        diagnostics.append(diagnostic)
    numeric=['mean_residual_correlation','mean_member_probability_sd','mean_member_brier','ensemble_brier','probability_variance','fraction_threshold_disagreement']
    means={k:float(np.mean([d[k] for d in diagnostics])) for k in numeric}
    (HERE/'diversity_diagnostics.json').write_text(json.dumps(dict(status='passed',selection_sha256=sha256(sp),folds=diagnostics,means=means,no_further_selection=True),indent=2))
    flat=[dict(config=n,fold=f,inner_auc=r['inner_auc'],inner_ap=r['inner_ap']) for n,rs in runs.items() for f,r in enumerate(rs)]
    flat += [dict(config=f'member{j}',fold=f,inner_auc=r['inner_auc'],inner_ap=r['inner_ap']) for f,rs in enumerate(members) for j,r in enumerate(rs)]
    with (HERE/'inner_summary.csv').open('w') as out:
        w=csv.DictWriter(out,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
    for r in protected['records']:
        ds,f=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256'];assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,source_unchanged=True,prior_full_replay_sha256=sha256(original)),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} |" for r in rows)
    paired='\n'.join(f"| {ref} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |" for ref,c in best['comparisons'].items())
    new=[r for r in v['records'] if not r['reused_control']];deleted=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    report=f'''# 独立初始化八成员概率集成：内部检验

**原测试AUC/AP同时绝对+.02尚未证实。** 新增21次训练及检查点重放；每折成员0
复用原unmasked10/seed42，另核验纯M123对照，共6个既有完整重放记录。
固定三内折、root seed42，无新候选原验证/测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| ensemble8的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选ensemble8的最小规定增益{best['selection_score']:+.9f}；相对每个规定对照，
三个内折平均AUC/AP均须≥+.002，推进原验证：{promote}。没有筛选种子、成员子集、
集成大小或权重。mean_individual_members只是8个单模型指标的平均，不是另一预测器。
最终仍需原测试seeds42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF两集不降。

## 方法和核验

每成员完整原M123加无掩码重构10，RMSprop及800/300/200选轮日程同既有成员0。
前50轮重构线性爬升。每成员独立初始化与优化；重构私有采样20261006+fold一致。
种子由root和成员序号固定生成，未来42/43/44三组24个种子不重叠。
对8个float32概率作float64等权平均；完整模型集成约8倍训练/推理成本。

预检原初始化、私有采样、留出标签翻转、查询批次独立及种子差异通过。
21个新模型独立重放保存预测，核对损失分解、最早最大均值选轮、BN计数及权重哈希。
完整M1/M2/M3查询删边最大logit误差{deleted:.3g}；去掉辅助重构模块后预测等价。
最终保存集成逐查询与标量概率求和完全一致，改变查询顺序或子集不影响预测。
历史旧种子集成已有小幅/无增益结果，旧混合RNA及评估口径不直接比较，未导入旧加载器。

## 冻结选择后的误差多样性

三个折内先算再平均：成员概率标准差{means['mean_member_probability_sd']:.6f}；
成员残差Pearson相关{means['mean_residual_correlation']:.6f}；阈值.5判断有分歧的样本
比例{means['fraction_threshold_disagreement']:.4%}。平均单成员Brier
{means['mean_member_brier']:.9f}，集成Brier{means['ensemble_brier']:.9f}，两者之差等于
成员概率方差{means['probability_variance']:.9f}（逐折验证恒等式）。这解释了平方误差的
平均收益，不能推出AUC/AP的普遍上限，也不能代替双指标验收。

PLMF的10个既有模型、冻结图/划分和发布源码哈希与先前完整重放一致；本轮未重新训练
或重放PLMF。全程只有原miRNA/drug，无lncRNA、外部数据或新预训练权重。
'''
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',best=best,
        promote_to_outer_validation=promote,new_fits=21,new_checkpoint_replays=21,reused_control_checks=6,
        no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],diversity=means,
        next_action='Use frozen diversity evidence and review recent miRNA-drug methods against prior experiments; no retrospective seed/weight sweep.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[独立初始化八成员集成](midrug_independent_ensemble_20261007/REPORT.md)21次新训练及模型重放。
内部AUC/AP为{best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}，推进原验证：{promote}。
没有新增原验证/测试成绩；PLMF十个模型、冻结输入和发布源码保持哈希。
近期[文献核查](midrug_representation_literature_20261007/READING_NOTES.md)及既有负结果保留。
''')
    scope_path=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(scope_path.read_text())
    scope.update(active_directory=str(HERE),latest_continuation=str(HERE/'CONTINUATION.json'),next_action=state['next_action'])
    scope_path.write_text(json.dumps(scope,indent=2))
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json',
        'initial_checks.json','README.md','inner_summary.csv','check.log','launch.log','audit.log','diversity_diagnostics.json']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'replay_predictions').glob('*.npz'))
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'audit_schema_fix').iterdir())
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes']})
    (HERE/'archive_manifest.json').write_text(json.dumps(archive,indent=2))
    count=0
    for p,h in archive['source_hashes'].items():assert sha256(HERE/p)==h and sha256(snap/p)==h;count+=2
    for p,h in archive['artifact_hashes'].items():assert sha256(HERE/p)==h;count+=1
    for p,h in archive['frozen_dependencies'].items():assert sha256(p)==h;count+=1
    (HERE/'archive_recheck.json').write_text(json.dumps(dict(status='passed',hash_checks=count,new_checkpoint_replays=21,
        reused_control_checks=6,protected_plmf_records=10,selection_unchanged=True,
        archive_sha256=sha256(HERE/'archive_manifest.json'),selection_sha256=sha256(sp),no_lncrna=True,
        outer_validation_evaluated=False,test_evaluated=False),indent=2))
    scope['latest_archive_check']=str(HERE/'archive_recheck.json');scope_path.write_text(json.dumps(scope,indent=2))
    print(json.dumps(dict(best=best,promote=promote,diversity=means),indent=2))

if __name__=='__main__':main()
