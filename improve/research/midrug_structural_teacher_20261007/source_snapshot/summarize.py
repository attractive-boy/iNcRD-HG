"""Freeze the structured context teacher gate and archive with PLMF protection."""
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
    best=next(r for r in rows if r['config']=='block_ema');promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
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
    report=f'''# 结构遮蔽与可见上下文教师：内部检验

**原测试AUC/AP同时绝对+.02尚未证实。** 九次新拟合及模型重放；六个既有纯M123/
unmasked10对照复用完整重放，并核对原记录、权重、配对/标签及保存概率。
仅原训练内部三折seed42，没有新候选原验证/测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| block_ema的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选block_ema最小规定平均增益{best['selection_score']:+.9f}；全部对照的平均
AUC/AP均须提高≥.002，推进原验证：{promote}。不事后改选random_ema或block_current。
最终仍需原测试seed42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF两集不降。

## 方法和边界

原M123推理与10倍无掩码属性重构保留。额外辅助视图每步遮住302个miRNA与108个
药物的原属性，代之以学习token。原共享编码器得到上下文状态；只汇聚可见来源的
同型/跨型邻居均值，结合四个度描述符，按类型用516→128→256预测器预测遮蔽目标。
教师清洁输入层2输出先按类型全节点中心化再L2归一化，目标停止梯度；仅遮蔽目标
参与1−cosine损失，两类等权。block_ema/random_ema教师动量.99，block_current每次
更新后直接复制学生。原重构前50步爬升；潜在损失权重.1在31–60步爬升。

结构遮蔽以原同类型相似图BFS形成一半集合，再用互补集合形成下一视图；随机对照
对同一批可用节点随机二分。每两步两臂逐节点遮蔽次数完全匹配，奇数个miRNA时
共省略一个节点。具体边内比例/跨边比例与曝光数见mask_manifest.json。初版单BFS
曾使部分节点曝光为0/800，在任何正式训练前依据采样统计修订，原源码和掩码保留。
本地不是NodeJEPA的k-hop日程，也不是旧GraphMAE2全节点平方cosine教师的简单重命名。
旧四类型模型只作源码查重，没有导入；不能宣称教师学习是项目首次尝试。

原RMSprop lr1e-4/alpha.99，主干wd0、两类辅助模块wd1e-4，set_to_none=True。
潜在损失关闭时其预测器/token不更新，教师仍每步跟踪学生；教师不进入优化器。
额外模块仅训练使用，不能把增加辅助前向称为等算力对照原模型。

## 核验

独立NumPy中心化/cosine公式及float64梯度检查、零目标边界、独立稠密可见邻居池、
遮蔽属性改变不变性及严格零输入梯度通过。真实更新核对EMA/复制递推、教师无梯度、
关闭辅助时不更新、BN一次、原初始化相同、留出标签翻转不变。
初检修复了PyG FX整体深复制丢失ModuleDict的问题，改为逐关系复制；失败日志保留，
所有正式训练使用修复后冻结源码。

九个训练模型重放预测/指标、选轮、损失组成、固定掩码哈希、教师计数及current精确
复制；完整M1/M2/M3物理删边最大logit误差{deleted:.3g}，推理批次不变。
移除重构与上下文教师模块后，原M123预测等价。教师递推的数值单步检查来自预检，
不把计数器单独当作所有历史更新权重的重放。

全程只有原miRNA/drug，无lncRNA、外部数据、额外注释或新权重。PLMF十个受保护
模型、冻结图/划分和发布源码与此前完整重放保持相同哈希。
'''
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',best=best,
        promote_to_outer_validation=promote,new_fits=9,new_checkpoint_replays=9,reused_control_checks=6,
        no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Use frozen masked-context and teacher diagnostics to decide next hypothesis; no retrospective control promotion or momentum/mask/weight sweep.')
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        d=json.loads(diag.read_text());assert d['status']=='passed' and d['selection_sha256']==sha256(sp)
        state['component_diagnostics']=dict(status='passed',means=d['means'],selection_unchanged=True)
        report+='\n冻结后的[教师与上下文诊断](COMPONENT_DIAGNOSTICS.md)已经完成，不用于重新选型。\n'
    (HERE/'REPORT.md').write_text(report);(HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[结构遮蔽与可见上下文教师](midrug_structural_teacher_20261007/REPORT.md)九次新训练及模型重放。
block_ema内部AUC/AP为{best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}，推进原验证：{promote}。
前轮[跨支持度组排名](midrug_regime_ranking_20261007/REPORT.md)和本轮[近期文献核查](midrug_representation_literature_20261007/READING_NOTES.md)保留。
没有新增原验证/测试成绩；PLMF十个模型、冻结输入和发布源码保持哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json',
        'initial_checks.json','README.md','inner_summary.csv','check.log','launch.log','audit.log','masks.npz','mask_manifest.json','prepare.log']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md','diagnose.log']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes']}),indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=9,new_checkpoint_replays=9),indent=2))


if __name__=='__main__':main()
