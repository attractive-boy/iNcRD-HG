"""Combine both audited stages and preserve the original dual-metric target."""
import json,csv,shutil
import numpy as np
from common import HERE,CLEAN,RELEASE,sha256,load_inner
from model import CONFIGS
from error_audit import summarize_errors

def main():
    stages=[HERE,HERE/'phase2'];verified=[]
    for stage in stages:
        v=json.loads((stage/'verification.json').read_text());assert v['status']=='passed' and len(v['records'])==15
        verified += [r for r in v['records'] if not r['reused_control']]
    assert len(verified)==24
    controls=[json.loads((CLEAN/'results'/f'control_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[dict(config='control',stage='reused',mean_inner_auc=float(np.mean([r['inner_auc'] for r in controls])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in controls])),mean_auc_gain=0.,mean_ap_gain=0.,selection_score=0.)];flat=[];locations={}
    for stage in stages:
        plan=json.loads((stage/'results/plan.json').read_text())
        for path,h in plan['source_hashes'].items():assert sha256(path)==h
        for name,cfg in plan['configs'].items():
            runs=[json.loads((stage/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
            da=[r['inner_auc']-b['inner_auc'] for r,b in zip(runs,controls)];dp=[r['inner_ap']-b['inner_ap'] for r,b in zip(runs,controls)]
            rows.append(dict(config=name,stage='phase1' if stage==HERE else 'phase2',settings=cfg,
                mean_inner_auc=float(np.mean([r['inner_auc'] for r in runs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in runs])),
                mean_auc_gain=float(np.mean(da)),mean_ap_gain=float(np.mean(dp)),fold_auc_gains=da,fold_ap_gains=dp,selection_score=min(float(np.mean(da)),float(np.mean(dp)))))
            flat += [{k:r[k] for k in ['config','fold','inner_auc','inner_ap','best_epoch']} for r in runs];locations[name]=stage
    best=max(rows,key=lambda x:x['selection_score']);promote=best['config']!='control' and best['selection_score']>=.002
    masking=[]
    for masked,unmasked in [('masked01','unmasked01'),('masked3','unmasked3'),('masked10','unmasked10')]:
        m=next(r for r in rows if r['config']==masked);u=next(r for r in rows if r['config']==unmasked)
        masking.append(dict(masked=masked,unmasked=unmasked,mean_auc_difference=m['mean_inner_auc']-u['mean_inner_auc'],mean_ap_difference=m['mean_inner_ap']-u['mean_inner_ap']))
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,masking_comparisons=masking,promote_to_outer_validation=promote,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,
        plan_hashes={str(s.relative_to(HERE))+'/results/plan.json':sha256(s/'results/plan.json') for s in stages},
        result_hashes={str(p.relative_to(HERE)):sha256(p) for s in stages for p in sorted((s/'results').glob('*_s42.json'))})
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    # The error decomposition is descriptive, computed after configuration selection.
    er=[]
    for f in range(3):
        for name,root in [('control',CLEAN),(best['config'],locations.get(best['config'],CLEAN))]:
            path=root/'results'/f'{name}_f{f}_s42_inner.npz'
            with np.load(path) as z:r=summarize_errors(load_inner(f),z['probability'])
            er.append(dict(config=name,fold=f,prediction_sha256=sha256(path),**r))
    (HERE/'selected_error_audit.json').write_text(json.dumps(dict(records=er,scope='original-training inner folds only; descriptive, not causal or final-test evidence'),indent=2))
    ec=[r for r in er if r['config']=='control'];eb=[r for r in er if r['config']==best['config']]
    within0=float(np.mean([r['conditional']['within_drug']['auc'] for r in ec]));within1=float(np.mean([r['conditional']['within_drug']['auc'] for r in eb]))
    # Protection check: exact checkpoints, frozen inputs and original source remain unchanged.
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results'
    protection=[]
    for r in protected['records']:
        ds,f=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,prior_full_replay_sha256=sha256(original),source_unchanged=True),indent=2))
    table='\n'.join(f"| {r['config']} | {r['stage']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['mean_auc_gain']:+.9f} | {r['mean_ap_gain']:+.9f} |" for r in rows)
    report='''# 原特征重建正则：两阶段实验结果

**最终 AUC/AP 均绝对 +0.02 的目标未达到。** 本轮完成 24 个新增训练运行，
并重放全部 checkpoint；原纯 miRNA M123 三个对照也通过重放。
两阶段都只使用原训练集内部三折，没有新候选进入原验证或测试。

| 配置 | 阶段 | 内部 AUC | 内部 AP | ΔAUC | ΔAP |
|---|---|---:|---:|---:|---:|
'''+table+f'''

选择 {best['config']}，内部平均 ΔAUC={best['mean_auc_gain']:+.9f}、ΔAP={best['mean_ap_gain']:+.9f}。
'''+('达到开发推进门槛，下一步需冻结配置后进入原验证确认。' if promote else '未同时达到预声明的双指标内部 +0.002 推进门槛，不进入原验证或测试。')+'''

第一阶段中，masked1 在三折的两项指标都改善，且为被试最大权重，
因此第二阶段预先固定权重 3/10 及相应未遮蔽对照后再运行。
更强的重建正则有小幅增益；第二阶段权重 3/10 的未遮蔽版本，两项平均指标
均高于相应遮蔽版本。第一阶段权重 0.1 的两项比较方向并不一致。
本轮证据更支持“辅助特征重建有小幅增益”，不能归因为掩码机制。
这是同一训练种子的三折开发结果，没有据此宣称统计显著或达到最终测试目标。

## 核验和模型边界

仅使用 605 个 miRNA 的 938 维原表达与 216 个药物的 768 维原特征。
保留 M1/M2/M3，辅助解码器和 token 只在训练中使用。
各模型移除这些辅助参数后重新推理，关联预测数值等价。
重建目标标准化只依赖原节点属性，同类相似图保持冻结；
这是现有图上的转导正则，不是完全未知节点属性的重建试验。

24 个新 checkpoint 的预测、指标、选轮、损失构成/权重 warmup 均核对通过。
每个新模型另有 6 条正例的 M1/M2/M3/最终 logit 显式删边检查。
初始检查确认主干参数相同、输入没有被辅助遮蔽改写、内部留出标签翻转不改变重建，
并确认共享编码器、token 和解码器均收到有限非零辅助梯度。
重建 MSE 的下降另存为诊断，不用来宣称关联预测收益。

PLMF 的 10 个 checkpoint、冻结图/划分及原源码哈希与此前完整重推理一致。
MDR/MDS 的保护值仍为 0.9483092901/0.9436701562 和 0.9508978837/0.9490119744。
没有修改发布实现。

## 剩余错误与后续

'''+f'''纯 miRNA 对照的平均同药物内 AUC 为 {within0:.6f}，本轮最佳为 {within1:.6f}。
该指标在同一药物的 miRNA 之间排序；它与全局 AUC 不同，不能直接互相替代。
error_audit.json / selected_error_audit.json 中，错误主要涉及训练正例度数为 2–15
的 miRNA，而非只有完全无关联的节点。这是错误位置描述，不是病因或性能上限证明。

已知纯 miRNA M123+BPR 的原测试三种子均值仍为 0.9575892678/0.9233078960，
低于最终要求 0.9733929274/0.9336833737。本轮没有新增测试成绩。
下一步可检验重建正则与既有同药物排序目标是否互补，须包含 BPR 单独对照；
具体约束见 NEXT_EXPERIMENT.md，尚未执行。

方法与复现见 README.md；第二阶段位于 phase2/。完整精度见 selection.json。
'''
    (HERE/'REPORT.md').write_text(report)
    for stage in stages:
        snap=stage/'source_snapshot';snap.mkdir(exist_ok=True)
        for p in stage.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,previous_turn='progress: literature, OCN and nested residual experiments',
        current_turn='progress: 24 masked/unmasked reconstruction fits, full checkpoint/deletion replays, pure-miRNA error decomposition, PLMF protection checks',
        best_inner=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Test reconstruction regularization with previously established same-drug BPR under the same inner-fold protocol; include BCE, BPR-only and reconstruction-only controls and require evidence beyond the stronger BPR control.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    progress='''# MiDrug 专用 miRNA 模型：当前进度

**严格仅 miRNA 与药物；AUC/AP 同时绝对 +0.02 尚未完成。**

| 项目 | AUC | AP |
|---|---:|---:|
| 原始 iNcRD-HG 测试三种子均值 | 0.9533929274 | 0.9136833737 |
| 必须达到 | 0.9733929274 | 0.9336833737 |
| 既有纯 miRNA M123+BPR 测试均值 | 0.9575892678 | 0.9233078960 |
| 需保住 MDR | 0.9483092901 | 0.9436701562 |
| 需保住 MDS | 0.9508978837 | 0.9490119744 |

最新完成 [24 次原特征重建正则训练](midrug_masked_reconstruction_20261006/REPORT.md)及全部重放。
'''+f"内部最佳 {best['config']}：AUC {best['mean_inner_auc']:.9f}，AP {best['mean_inner_ap']:.9f}；\n相对同折优化过的 M123 对照增益为 {best['mean_auc_gain']:+.9f}/{best['mean_ap_gain']:+.9f}。\n"+'''
未达到内部双指标推进门槛，没有新候选进入原验证或测试。
未遮蔽重建略好于相应遮蔽版本，不能将收益归因为掩码。
PLMF 保护的 10 个 checkpoint、冻结输入和原源码哈希复核通过。

此前 [文献检索](midrug_literature_20261006/RESEARCH_NOTES.md)、
[OCN 消融](midrug_ocn_20261006/REPORT.md)和
[嵌套折外实验](midrug_crossfit_20261006/REPORT.md)均保留完整审计。
后续检验重建与既有同药物 BPR 是否互补；仍然只在训练内部开发。
'''
    (HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md').write_text(progress)
    print(json.dumps(dict(best=best,promote=promote,within_drug_auc_before=within0,within_drug_auc_selected=within1,new_runs=24,goal_status='active'),indent=2),flush=True)

if __name__=='__main__':main()
