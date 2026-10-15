"""Freeze the declared gate, protect PLMF and archive the six-fit study."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==18
    assert v['new_checkpoint_replays']==6 and v['immutable_reused_control_checks']==12
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
    candidate=next(r for r in rows if r['config']=='drug_rec10')
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
    oracle=max(r['branch']['oracle_error'] for r in new)
    report=f'''# 原表达列的查询/键归一化：内部检验

**原测试 AUC/AP 同时绝对 +0.02 尚未证实。** 本轮六次新拟合、六个新检查点重放；
另 12 个旧对照复用已完成的重放证据并检查哈希、保存概率及配对标签。
仅原训练内部三折，seed42；没有新候选原验证或测试。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

dot_ 前缀是前轮原始缩放点积；本轮两个无前缀的列注意力臂使用固定范数查询/键。
control 为纯 M123，unmasked10 为原特征重建正则，均来自已审核的不可变对照。

| drug_rec10 的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选 drug_rec10 的最小规定平均增益 {candidate['selection_score']:+.9f}，
是否同时超过所有规定对照 .002：{promote}。旧轮未通过的结论没有被改写。
最终门槛仍为原测试 seed42/43/44 均值 AUC≥.9733929274、AP≥.9336833737。

## 机制与核验

前轮训练后查询范数收缩，注意力接近 938 列均匀平均。新实验只把注意力得分
改为 sqrt(32) 乘查询/键的余弦相似度，epsilon=1e-8；mean 臂先平均查询再归一化。
原 token、全部表达列、值向量、模型维度、主干、优化器与重建设置保持一致。
该机制去除向量范数对得分的直接影响，不保证方向多样性或分类收益。

独立逐列/配对 float64 参考核对全部参数梯度、列置换与分块；检查正比例缩放
不变性、零向量有限性和合成查询敏感性。真实非零投影训练核对主干初始化、
分支梯度、BN 一次、留出标签翻转和完整物理删除。
六个检查点均核对保存预测、选轮、损失、分支参考和去除重建解码器等价。
最大逐列参考误差 {oracle:.3g}，完整删查询 logit 误差 {deletion:.3g}。

所有输入限于原 miRNA/drug 属性及原训练标签；不引入 lncRNA、注释或新权重。
PLMF 10 个受保护检查点、冻结图/划分及发布源码与既有完整重放保持相同哈希。
已验证纯 miRNA M123+BPR 原测试均值仍为 .9575892678/.9233078960，目标未完成。
'''
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
               best=candidate,promote_to_outer_validation=promote,new_fits=6,new_checkpoint_replays=6,
               reused_control_checks=12,new_candidate_test_evaluations=0,no_lncrna=True,processes_expected_live=[],
               next_action='Use frozen gate and post-selection conditioning/column-alignment diagnostics; no automatic attention parameter sweep.')
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        ds=json.loads(diag.read_text());assert ds['status']=='passed' and ds['selection_sha256']==sha256(sp)
        row=next(r for r in ds['means'] if r['config']=='drug_rec10')
        state['component_diagnostics']=dict(status='passed',selection_unchanged=True,means=ds['means'])
        report+=f"\n冻结后的[分支诊断](COMPONENT_DIAGNOSTICS.md)：drug_rec10 平均有效列数 {row['effective_columns']:.2f}/938；切换真实/平均查询、仅反转背景列的平均概率变化依次为 {row['swapped_mean_abs_probability_shift']:.6g}、{row['misalignment_mean_abs_probability_shift']:.6g}。这些不是重训对照，不撤销联合训练，也不用于重新选型。\n"
    (HERE/'REPORT.md').write_text(report);(HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[原表达列查询/键归一化](midrug_cell_qknorm_20261007/REPORT.md)六次新拟合及重放。
drug_rec10 内部 AUC {candidate['mean_inner_auc']:.9f}、AP {candidate['mean_inner_ap']:.9f}；
对规定对照最小平均增益 {candidate['selection_score']:+.9f}，推进原验证：{promote}。
前轮[原表达列点积注意力](midrug_cell_tokens_20261007/REPORT.md)的完整证据保留。
没有新的原验证/测试成绩；PLMF 10 个检查点及冻结输入、发布实现保持相同哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json','initial_checks.json','README.md','shrinkage_precheck.json','inner_summary.csv','check.log','precheck.log','launch.log','audit.log']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md','diagnose.log']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes']}),indent=2))
    print(json.dumps(dict(best=candidate,promote=promote,new_fits=6,new_checkpoint_replays=6,reused_controls=12,max_deletion_logit_error=deletion),indent=2))

if __name__=='__main__':main()
