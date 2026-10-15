"""Apply the frozen dual-metric gate and verify protected PLMF artifacts."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==18 and all(r['status']=='passed' for r in v['records'])
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
    error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    batch=max(r['batch_max_logit_error'] for r in new);decoder=max(r['without_reconstruction_error'] for r in new)
    merged=max(r['merged_block_error'] for r in new if r['merged_block_error'] is not None)
    report=f'''# 原版药物Transformer的最后一层适配：内部检验

**原测试AUC/AP同时绝对+0.02尚未证实。** 完成12次新拟合和12个新检查点重放。
另6个旧对照复用此前完整重放证据，再核对源码/权重/输入/概率哈希及配对标签指标；
这不算6次新的GPU重放。没有新候选原验证或测试成绩。

| 配置 | 内部AUC | 内部AP |
|---|---:|---:|
{table}

| token候选 | 规定对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(paired)}

按冻结规则，候选{best['config']}对三个规定对照的双指标最小平均增益
{best['selection_score']:+.9f}；通过+.002内部推进门槛：{promote}。
最终仍要求原测试seed42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF两集不降。

本地原版ChemBERTa-zinc-base-v1完全离线加载，重新编码原216个SMILES后，
归一化特征与原输入最大误差1.08e-6。所有原始权重冻结，只训练最后一层Q/V的
rank8 LoRA，共24576参数；pooled对照在原固定池化后向量上训练同参数量适配器。
参数数量/优化设置匹配，不声称计算量或表达能力等价。没有换模型或下载新权重。

前五层缓存不含标签。最后一层按有效长度裁去尾部padding，非padding均值仍保留
BOS/EOS；1个长药物继续按原规则截断到512。不是恢复该分子全部结构的实验。
两臂都以原特征为锚，只加入归一化的学习变化，因此初始预测保持原M123。
原M1/M2/M3、关联/同类型图、原重建目标及初始化均保留。
主干/重建沿用RMSprop，新适配参数用AdamW；这不是LoRA新方法声明。

独立合并矩阵核对LoRA输出及梯度；真实模型检查初始特征/主干/预测一致、有限非零
适配梯度、冻结原始权重零变化、特征上下文复用/清理及完整M123显式删记录。
所有12个训练后检查点重新核对原始冻结参数、预测/指标/选轮/损失组成、批次不变、
重建解码器移除；token模型另核对LoRA合并权重及完整512位与长度分桶的结果。
最大完整删记录logit误差{error:.3g}、批次误差{batch:.3g}、合并Transformer误差{merged:.3g}，
移除重建解码器概率误差{decoder:.3g}。适配模块属于推理模型，不能等价移除。

无lncRNA、外部数据集、额外RNA序列、注释或新预训练权重。
PLMF10个受保护检查点、冻结图/划分及发布源码与此前完整重放哈希一致。
已验证纯miRNA M123+BPR原测试均值仍为.9575892678/.9233078960，目标未完成。
'''
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        ds=json.loads(diag.read_text());assert ds['status']=='passed' and ds['selection_sha256']==sha256(sp)
        selected=next(r for r in ds['means'] if r['config']==best['config'])
        report+=f"\n冻结选型后的[适配诊断](COMPONENT_DIAGNOSTICS.md)：{best['config']}去除学习变化后的内部AUC/AP为{selected['zero_adaptation']['auc']:.9f}/{selected['zero_adaptation']['ap']:.9f}。该干预不撤销原主干的联合训练，不用于重新选择候选。\n"
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn_classification='progress:12 observed-source fits/replays archived',
        current_turn='progress:offline original-ChemBERTa input identity,12 attention/pooled adaptation fits and12 new replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='If promoted, freeze original-validation confirmation. Otherwise use the matched token/pooled and component evidence before selecting a distinct next hypothesis; do not start a LoRA rank/layer grid automatically.')
    if diag.exists():state['component_diagnostics']=dict(status='passed',selection_unchanged=True,means=ds['means'])
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[原版药物Transformer任务适配](midrug_chemberta_adaptation_20261006/REPORT.md)12次新拟合和12个新检查点重放。
候选{best['config']}：内部AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；对规定对照双指标最小平均增益{best['selection_score']:+.9f}，推进原验证：{promote}。
此前[源条件观测组合](midrug_source_observed_20261006/REPORT.md)及其分支诊断保留。
没有新候选原验证/测试成绩；PLMF10个检查点、冻结输入和发布源码保持原哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json','initial_checks.json','README.md','predecessor_review.json','input_manifest.json','token_cache.pt','original_last_layer.pt']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies=plan['source_hashes']),indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=12,new_checkpoint_replays=12,reused_controls=6,max_deletion_logit_error=error),indent=2))

if __name__=='__main__':main()
