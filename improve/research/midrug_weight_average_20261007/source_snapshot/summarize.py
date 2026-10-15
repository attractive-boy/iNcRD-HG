"""Freeze the averaging gate, hash-protect PLMF, and archive the evidence."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from common import HERE, RELEASE, sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==12 and v['new_training_trajectories']==3
    assert v['new_checkpoint_replays']==6 and v['immutable_reused_control_checks']==6
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
        flat.extend(dict(config=name,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch','epochs_run']},average_count=r.get('average_count')) for f,r in enumerate(rs))
    best=next(r for r in rows if r['config']=='mean');promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
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
    report=f'''# 固定参数平均的内部检验

**原测试AUC/AP同时绝对+0.02尚未证实。** 三条训练轨迹，每条同时生成raw与mean，
共六个选中模型及重放；另六个既有对照复用已验证重放并核查哈希及保存概率。
仅原训练内部三折，seed42；没有新增原验证/测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| mean 的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

mean的最小规定平均增益{best['selection_score']:+.9f}；所有对照/指标均须≥.002。
推进原验证：{promote}。原测试目标仍为seed42/43/44均值AUC≥.9733929274、
AP≥.9336833737，PLMF MDR/MDS不下降。

## 方法与可归因范围

受Sapkota等[Parameter Averaging in Link Prediction](https://doi.org/10.1145/3731443.3771365)
（K-CAP2025，arXiv2510.25361v1）§3启发，固定从第100步起纳入所有参数快照。
float64累加等权平均，只用于独立评估；原RMSprop更新轨迹、采样和停止规则不受
mean成绩影响。每种模型均以内部(AUC+AP)/2独立选轮。没有按验证分数决定成员，
不使用论文ASWA，也未采用作者SWA学习率日程；是本地固定平均适配。
历史coldstart.py已有top10参数平均，未调用旧数据入口，不声称首次使用该家族。

保存选中和最终累积和、快照计数；所选参数与所存总和/计数逐项精确一致。
BN位于固定EASE输入，复制当轮缓冲；独立BN重算统计验证，整数计数为所选轮次。
初始检查覆盖七个快照显式平均、起点单快照恒等、评估前后参数/缓冲/梯度/
优化器/辅助采样器均不变。六个模型完成保存概率/选轮、物理删边、批次不变性、
去除重构解码器检查；最大完整logit删除误差{deleted:.3g}。

本次独立重训未逐步复现历史unmasked10轨迹，原始“相同seed应有相同停止轮次”
审计假设失败，日志保留。差异在第100步启用平均前已经出现；数值差异的具体
来源未被因果定位，不能宣称精确重现旧训练。审计保存完整共享前缀及平均启用
前的差异，不放宽模型概率重放或累积均值的检查。预定关键对照是本次同一轨迹
raw，且mean仍须超过旧unmasked10与纯M123；没有改变训练代码、门槛或结果。

只使用原miRNA/drug数据，无lncRNA、外部数据/注释/新权重。PLMF十个受保护
检查点、冻结输入和发布源码保持既有完整重放时的哈希。
'''
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',best=best,
        promote_to_outer_validation=promote,new_training_trajectories=3,new_checkpoint_replays=6,reused_control_checks=6,
        no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Read frozen diagnostics before choosing another hypothesis; no retrospective averaging-window or validation-gated member search.')
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        d=json.loads(diag.read_text());assert d['status']=='passed' and d['selection_sha256']==sha256(sp)
        state['component_diagnostics']=dict(status='passed',means=d['means'],selection_unchanged=True)
        report+='\n冻结后的[训练/留出和重构诊断](COMPONENT_DIAGNOSTICS.md)已完成，不按诊断重新选择平均区间。\n'
    (HERE/'REPORT.md').write_text(report);(HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[固定参数平均](midrug_weight_average_20261007/REPORT.md)三条轨迹、六个模型重放。
mean内部AUC/AP为{best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}，推进原验证：{promote}。
[近期文献核查](midrug_finetune_literature_20261007/READING_NOTES.md)及
[重构微调修正版](midrug_reconstruction_finetune_v2_20261007/REPORT.md)完整保留。
没有新增原验证/测试成绩；PLMF十个模型与冻结输入/发布源码保持哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json',
           'initial_checks.json','README.md','inner_summary.csv','check.log','launch.log','audit.log','audit_historical_trajectory_failure.log']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md','diagnose.log']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes']}),indent=2))
    print(json.dumps(dict(best=best,promote=promote,training_trajectories=3,selected_models=6),indent=2))


if __name__=='__main__':main()
