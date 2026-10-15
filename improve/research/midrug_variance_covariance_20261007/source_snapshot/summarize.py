"""Freeze the endpoint variance/covariance gate and archive with PLMF protection."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==15 and v['new_checkpoint_replays']==9
    assert v['immutable_reused_control_checks']==6
    for p,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(p)==h
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    for name in plan['configs']:runs[name]=[json.loads((HERE/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for name,rs in runs.items():
        comparisons={ref:{m:dict(mean_delta=float(np.mean([r['inner_'+m]-b['inner_'+m] for r,b in zip(rs,runs[ref])])),
            fold_deltas=[r['inner_'+m]-b['inner_'+m] for r,b in zip(rs,runs[ref])]) for m in ['auc','ap']}
            for ref in plan['comparators'].get(name,['control','unmasked10'])}
        rows.append(dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),
            comparisons=comparisons,selection_score=min(c[m]['mean_delta'] for c in comparisons.values() for m in ['auc','ap'])))
        flat.extend(dict(config=name,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch','epochs_run']}) for f,r in enumerate(rs))
    best=next(r for r in rows if r['config']=='varcov');promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    sp=HERE/'selection.json'
    if sp.exists():assert json.loads(sp.read_text())==selection
    else:sp.write_text(json.dumps(selection,indent=2))
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
    report=f'''# M1端点方差—协方差正则：内部检验

**原测试AUC/AP同时绝对+.02尚未证实。** 九次新拟合及模型重放；六个既有纯M123/
unmasked10对照复用完整重放并核对原记录、权重、配对/标签及保存概率。
只有原训练内部三折seed42，没有新候选原验证/测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| varcov的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选varcov最小规定平均增益{best['selection_score']:+.9f}；全部对照的平均AUC/AP
均须提高≥.002，推进原验证：{promote}。不事后改选var或cov。最终仍需原测试
seed42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF MDR/MDS不下降。

## 方法与边界

原M123与无掩码特征重构保持相同推理结构，没有新增推理参数。训练前向捕获两侧
M1查询删边表示；分别逐行无参数LayerNorm后，计算方差下限损失V与非对角协方差
平方和/D损失K，两端等权平均。varcov/var/cov依次采用两项、仅V、仅K。

BCE+.2V+.2K+10重构；正则权重30–60步爬升、重构前50步爬升。原RMSprop lr1e-4、
alpha.99，重构wd1e-4，set_to_none=True；原划分、初始化和私有采样种子保留。
这只是NodeJEPA启发、采用VICReg公式的单视图监督适配，没有复现其教师、预测器、
多视图不变性或SIGReg。NodeJEPA公开代码按D²归一化协方差，本地按D；.2不是
数值等价移植。逐行LayerNorm使精确单位协方差不可达；常数表示处梯度可能为零。
低有效秩不自动证明性能瓶颈，改善几何性质也不保证关联排序提高。

## 核验

独立NumPy矩公式、float64梯度检查、行/列置换、逐行常量平移、常数表示有限值
检查通过。三折所有臂的原始权重/预测匹配；正则对共享编码器有有限非零梯度；
BN每步仅一次，留出标签翻转不改预测或训练正则。全部新模型通过选轮、逐步损失
组成、完整物理删边和批次不变性检查；最大删边logit误差{deleted:.3g}。
训练后正则重新用float64与独立矩公式核对；移除重构与训练hook后原预测等价。
初检曾对重复CUDA稀疏前向要求逐位相同，观察到3.81e-6舍入差；保留失败日志，
改用既有2e-5容差后全项通过，没有修改训练计算。

全程只有原miRNA/drug，无lncRNA、外部数据、额外注释或新权重。PLMF十个受保护
模型、冻结图/划分和发布源码与此前完整重放保持相同哈希。
'''
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',best=best,
        promote_to_outer_validation=promote,new_fits=9,new_checkpoint_replays=9,reused_control_checks=6,
        no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Use frozen endpoint geometry diagnostics to select next hypothesis; no retrospective var/cov promotion or regularization-weight sweep.')
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        d=json.loads(diag.read_text());assert d['status']=='passed' and d['selection_sha256']==sha256(sp)
        state['component_diagnostics']=dict(status='passed',means=d['means'],selection_unchanged=True)
        report+='\n冻结后的[端点几何诊断](COMPONENT_DIAGNOSTICS.md)已经完成，不用于重新选型。\n'
    (HERE/'REPORT.md').write_text(report);(HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[M1端点方差—协方差正则](midrug_variance_covariance_20261007/REPORT.md)九次新训练及模型重放。
varcov内部AUC/AP为{best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}，推进原验证：{promote}。
前轮[跨支持度组排名](midrug_regime_ranking_20261007/REPORT.md)和本轮[近期文献核查](midrug_representation_literature_20261007/READING_NOTES.md)保留。
没有新增原验证/测试成绩；PLMF十个模型、冻结输入和发布源码保持哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json',
        'initial_checks.json','README.md','inner_summary.csv','check.log','check_initial_float_failure.log','launch.log','audit.log']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md','diagnose.log']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes']}),indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=9,new_checkpoint_replays=9),indent=2))


if __name__=='__main__':main()
