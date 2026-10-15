"""Freeze the declared gate, protect PLMF and archive the six-fit study."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==12
    assert v['new_checkpoint_replays']==6 and v['immutable_reused_control_checks']==6
    for path,digest in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(path)==digest
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    for name in plan['configs']:
        runs[name]=[json.loads((HERE/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for n,rs in runs.items():
        comparisons={ref:{metric:dict(mean_delta=float(np.mean([r['inner_'+metric]-b['inner_'+metric] for r,b in zip(rs,runs[ref])])),
            fold_deltas=[r['inner_'+metric]-b['inner_'+metric] for r,b in zip(rs,runs[ref])]) for metric in ['auc','ap']}
            for ref in plan['comparators'].get(n,['control','unmasked10'])}
        rows.append(dict(config=n,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),
                         comparisons=comparisons,selection_score=min(c[m]['mean_delta'] for c in comparisons.values() for m in ['auc','ap'])))
        flat.extend(dict(config=n,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch']}) for f,r in enumerate(rs))
    candidate=next(r for r in rows if r['config']=='independent_rec10')
    promote=candidate['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=candidate,promote_to_outer_validation=promote,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
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
    paired='\n'.join(f"| {ref} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |" for ref,c in candidate['comparisons'].items())
    new=[r for r in v['records'] if not r['reused_control']]
    deletion=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    oracle=max(r['head']['member_oracle_error'] for r in new)
    member_delete=max(x['errors']['all_member_logits'] for r in new for x in r['explicit_deletions'])
    report=f'''# 原 M1/M2/M3 的多成员融合头：内部检验

**原测试 AUC/AP 同时绝对 +0.02 尚未证实。** 六次新拟合、六个新检查点重放；
另六个纯 M123/unmasked10 对照复用既有完整重放，并核对保存概率/指标及哈希。
仅原训练内部三折，seed42；没有新候选原验证或测试。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| independent_rec10 的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选 independent_rec10 的最小规定平均增益 {candidate['selection_score']:+.9f}；
是否同时超过所有规定对照 .002：{promote}。
最终仍须原测试 seed42/43/44 均值 AUC≥.9733929274、AP≥.9336833737，PLMF 两集不降。

## 实现与可归因范围

保留原图编码器、M1/M2/M3 表示及融合顺序，联合训练 16 成员的共享矩阵预测头。
沿用原 512/256 隐层和 LeakyReLU；首层输入缩放为随机符号，其他缩放初始为 1，
成员偏置和最终输出头独立。共享主矩阵与原通道、重建模块初始参数保持相同。
单成员单位缩放检查恢复原预测。该方法属于既有 TabM 家族的本地适配；
未调用历史混合 RNA 图手工统计的 GraphTabM 数据流程。

independent_rec10 对各成员分别计算 BCE 后平均；mixture_rec10 对平均概率计算
一次 BCE，是等参数/初始化/输入的目标对照。两个模型推理均平均概率。
兼容接口用稳定的 logsumexp 返回混合概率的 log-odds，不进行概率裁剪。
原重建权重 10、优化器、选轮与内部划分按 README 和冻结 plan 执行。

## 验证

float64 逐成员物化矩阵核对前向、输入与全部头参数梯度；独立概率公式核对两种
目标及梯度，并检查极端 logits。原输入精确相等、单成员初始预测复现；
真实完整更新中原共享主干及成员均有非零梯度，BN 只更新一次，留出标签翻转不变。
第一次 GPU 融合向量逐位相等检查未通过，失败日志保留；最终采用既有真实图
删除检查容差，同时记录相同模型重复计算误差，未放宽 float64 参考容差。

每个保存模型重放概率/指标/选轮/损失及成员矩阵参考；完整删除逐一核对每个成员，
另核对混合后单分数。最大头参考误差 {oracle:.3g}、成员删除误差 {member_delete:.3g}、
混合 logit 删除误差 {deletion:.3g}。去除重建解码器保持预测等价。

只有原 miRNA/drug 属性与训练标签，无 lncRNA、外部数据、注释或新权重。
PLMF 10 个受保护检查点、冻结图/划分与发布源码均保持既有完整重放的哈希。
已验证纯 miRNA M123+BPR 原测试均值仍为 .9575892678/.9233078960，目标未完成。
'''
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
               best=candidate,promote_to_outer_validation=promote,new_fits=6,new_checkpoint_replays=6,
               reused_control_checks=6,new_candidate_test_evaluations=0,no_lncrna=True,processes_expected_live=[],
               next_action='Use frozen gate and individual-versus-ensemble diagnostics before another model; no retrospective member selection or automatic member-count sweep.')
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        ds=json.loads(diag.read_text());assert ds['status']=='passed' and ds['selection_sha256']==sha256(sp)
        row=next(r for r in ds['means'] if r['config']=='independent_rec10')
        state['component_diagnostics']=dict(status='passed',selection_unchanged=True,means=ds['means'])
        report+=f"\n冻结后的[成员诊断](COMPONENT_DIAGNOSTICS.md)：independent_rec10 平均个体 AUC/AP 为 {row['val']['mean_individual']['auc']:.9f}/{row['val']['mean_individual']['ap']:.9f}，平均成员概率标准差 {row['val']['mean_member_probability_std']:.6g}；未按留出成绩挑选任何成员子集。\n"
    (HERE/'REPORT.md').write_text(report);(HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[原 M123 多成员融合头](midrug_ensemble_fusion_20261007/REPORT.md)六次新拟合及重放。
independent_rec10 内部 AUC {candidate['mean_inner_auc']:.9f}、AP {candidate['mean_inner_ap']:.9f}；
对规定对照最小平均增益 {candidate['selection_score']:+.9f}，推进原验证：{promote}。
前轮[原表达列查询/键归一化](midrug_cell_qknorm_20261007/REPORT.md)的完整证据保留。
没有新的原验证/测试成绩；PLMF 10 个检查点及冻结输入、发布实现保持相同哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json','initial_checks.json','README.md','inner_summary.csv','check.log','check_initial_failure.log','launch.log','audit.log']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md','diagnose.log']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes']}),indent=2))
    print(json.dumps(dict(best=candidate,promote=promote,new_fits=6,new_checkpoint_replays=6,reused_controls=6,max_deletion_logit_error=deletion),indent=2))

if __name__=='__main__':main()
