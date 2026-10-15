"""Freeze the comparison-set ranking gate and archive with PLMF protection."""
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
    best=next(r for r in rows if r['config']=='cross');promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
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
    report=f'''# 跨支持度组排名：内部检验

**原测试AUC/AP同时绝对+.02尚未证实。** 九次新拟合及模型重放；六个既有纯M123/
unmasked10对照复用完整重放并核对原记录、权重、配对/标签及保存概率。
只有原训练内部三折seed42，没有新候选原验证/测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| cross的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选cross最小规定平均增益{best['selection_score']:+.9f}；全部对照的平均AUC/AP
均须提高≥.002，推进原验证：{promote}。不事后改选all或within。最终仍需原测试
seed42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF MDR/MDS不下降。

## 方法与可归因范围

原M123和无掩码特征重构推理结构不变，没有新推理参数或新增数据。只改变训练
正负分数比较集合：all为全部，cross为两端查询删边支持度组不同，within为相同。
四组来自当前训练查询两端度的中位数，不以留出标签分组。每个集合的全部有效
比较按其数量归一化，损失mean softplus(s_neg-s_pos)，温度1，无抽样或硬例选择。

BCE+.1排名+10重构，排名30–60步爬升、重构前50步爬升；原RMSprop lr1e-4、
alpha.99，重构wd1e-4，清梯度set_to_none=True。原训练划分、标签、初始化及
私有重构采样种子保留。分组不进入推理。未观测零标签不解释为实验真阴性。

已有同药物BPR及重构组合；历史全局SmoothAP与双目标实验亦已尝试，未调用其
旧模型/混合RNA数据入口。本轮是同一排名家族中对比较集合的受控检验，不是
全新方法家族。上一轮观察跨组错误较多不证明这些错误可通过目标重加权消除。

## 核验

独立NumPy枚举损失/梯度、float64 gradcheck、分块/平移不变性、正负梯度方向、
无有效比较的零梯度检查通过。cross+within比较数等于all，按数量加权的两个
损失重构all；训练后各模型再次用float64分数核对该恒等式。真实训练分组与
前轮预检一致，BN每步只更新一次，翻转留出标签不改变排名损失或预测。
九个模型的保存概率/指标/选轮/逐步损失、固定组阈值/比较数、完整物理删边和
批次不变性均通过；最大完整logit删边误差{deleted:.3g}。移除重构解码器预测等价。

全程只有原miRNA/drug，无lncRNA、外部数据、额外注释或新权重。PLMF十个受保护
模型、冻结图/划分和发布源码与此前完整重放保持相同哈希。
'''
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',best=best,
        promote_to_outer_validation=promote,new_fits=9,new_checkpoint_replays=9,reused_control_checks=6,
        no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Use frozen comparison-set diagnostics to choose a distinct next hypothesis; no retrospective all/within promotion or ranking-weight/temperature sweep.')
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        d=json.loads(diag.read_text());assert d['status']=='passed' and d['selection_sha256']==sha256(sp)
        state['component_diagnostics']=dict(status='passed',means=d['means'],selection_unchanged=True)
        report+='\n冻结后的[比较集合和梯度诊断](COMPONENT_DIAGNOSTICS.md)已经完成，不用于重新选型。\n'
    (HERE/'REPORT.md').write_text(report);(HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[跨支持度组排名](midrug_regime_ranking_20261007/REPORT.md)九次新训练及模型重放。
cross内部AUC/AP为{best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}，推进原验证：{promote}。
前轮[固定参数平均](midrug_weight_average_20261007/REPORT.md)、[近期文献核查](midrug_finetune_literature_20261007/READING_NOTES.md)保留。
没有新增原验证/测试成绩；PLMF十个模型、冻结输入和发布源码保持哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json',
        'initial_checks.json','README.md','inner_summary.csv','check.log','launch.log','audit.log']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md','diagnose.log']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes']}),indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=9,new_checkpoint_replays=9),indent=2))


if __name__=='__main__':main()
