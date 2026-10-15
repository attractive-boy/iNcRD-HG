import json,csv
from pathlib import Path
import numpy as np
from common import HERE,BASES,PREV,RELEASE,sha256

def main():
 plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text());assert v['status']=='passed' and len(v['records'])==9
 for p,h in {**plan['source_hashes'],**plan['evidence_hashes'],**plan['input_hashes']}.items():assert sha256(p)==h
 runs={n:[json.loads(Path(cs[str(f)]['path']).read_text()) for f in range(3)] for n,cs in {**plan['controls'],**plan['extra_controls']}.items()}
 for n in plan['configs']:runs[n]=[json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)]
 rows=[dict(config=n,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),selected_rounds=[r.get('best_round',r.get('best_epoch')) for r in rs]) for n,rs in runs.items()]
 comparisons={n:{m:dict(mean_delta=float(np.mean([a['inner_'+m]-b['inner_'+m] for a,b in zip(runs['oof_raw'],runs[n])])),fold_deltas=[a['inner_'+m]-b['inner_'+m] for a,b in zip(runs['oof_raw'],runs[n])]) for m in ['auc','ap']} for n in plan['comparators']}
 minimum=min(c[m]['mean_delta'] for c in comparisons.values() for m in ['auc','ap']);promote=minimum>=plan['promotion_minimum_both_gains_over_each_control']
 selection=dict(status='frozen_inner_selection',rows=rows,candidate='oof_raw',comparisons=comparisons,minimum_gain=minimum,promote_to_outer_validation=promote,plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 sp=HERE/'selection.json';assert not sp.exists();sp.write_text(json.dumps(selection,indent=2))
 with (HERE/'inner_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text());frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
 for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
 for r in protected['records']:
  ds,f=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256'];assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256'];assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256'];protection.append(r)
 (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,source_unchanged=True,prior_full_replay_sha256=sha256(original),new_gpu_replay=False),indent=2))
 table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['selected_rounds']} |" for r in rows)
 gains='\n'.join(f"| {n} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |" for n,c in comparisons.items())
 error=max(x['full_logit_error'] for r in v['records'] for x in r['physical_deletions']);tree=max(r['independent_tree_error'] for r in v['records']);inp=json.loads((HERE/'input_manifest.json').read_text())
 report=f'''# 折外与训练内offset的原始坐标树修正

**原测试AUC/AP双+.02尚未证实。** 本轮9次新树拟合、9个选择模型重放及9个最终树检查。
复用18个旧固定300轮M123+unmasked10基模型，并新重放这18个基模型；没有新基模型训练。
另复用12个对照记录（control、已选unmasked10、full300、前轮raw），不将它们全算作新训练。
没有新原验证/测试。

| 配置 | 内部AUC | 内部AP | 选择轮数（旧神经对照为训练轮） |
|---|---:|---:|---|
{table}

唯一候选oof_raw相对规定对照的三折配对均值差：

| 对照 | ΔAUC | ΔAP |
|---|---:|---:|
{gains}

最小平均增益{minimum:+.9f}；每项AUC/AP都须≥+.002的冻结推进门槛：{promote}。
不事后选择对照，不改树超参/融合权重，不拿较弱full300替代已选unmasked10或前轮raw。

## 干预和差异范围

oof_raw与insample_raw使用相同8+938+768维树输入和相同完整300轮推理基模型；
只有训练DMatrix base_margin来自OOF/完整模型训练预测的区别。oof_pca32使用每侧PCA32。
所有树参数及训练预算沿用前轮，推理始终sigmoid(full300_logit+tree_delta)，含stage0。
源基模型固定300轮、不用被预测折选轮，原始索引独立重建、OOF恰好一次覆盖。

OOF来自较稀疏的4/5训练图，推理来自完整训练图；树的精确拓扑计数始终用完整当前
内部训练图并删除自身查询。因而不能把效果全归为去掉单样本标签影响，也不能将
整个树修正后的训练成绩称为折外泛化成绩。源OOF平均训练BCE为
{np.mean([g['oof_bce'] for g in inp['gaps']]):.6f}，完整模型训练/内部留出BCE分别为
{np.mean([g['in_sample_bce'] for g in inp['gaps']]):.6f}/{np.mean([g['full_validation_bce'] for g in inp['gaps']]):.6f}。

## 核验

18个旧基模型新重放、全部源hash/固定轮次/索引/覆盖通过；9个树的实际offset、
stage0共同推理margin、匹配臂逐位相同输入、留出标签翻转、独立逐树参考和float32
阈值边界核对通过。九个保存预测重放逐位一致，独立树margin误差最大{tree:.3g}。
全部所查正负查询删记录的树输入逐位相等，完整M123+树logit误差最大{error:.3g}。
没有修改此前任何冻结实验。本轮直接复用已修正的规范JSON数组参考，无新增容差修正。

仅原miRNA/drug，无lncRNA、外部数据、序列/注释或新预训练权重。PLMF十个受保护
检查点和冻结输入/发布源码哈希保持，本轮没有新PLMF GPU重放。
最终仍须原MiDrug测试seeds42/43/44均值AUC≥.9733929274189016、AP≥.933683373732677。
已验证纯miRNA M123+BPR原测试均值仍为.9575892678/.9233078960；目标未完成。
'''
 (HERE/'REPORT.md').write_text(report)
 state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',previous_turn_classification='progress:9 raw feature residual fits, audits and fixed-model diagnostics',current_turn='progress:18 old bases newly replayed,9 matched OOF tree fits and selected/final audits',promote_to_outer_validation=promote,selection_sha256=sha256(sp),no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],next_action='If promoted freeze original-validation confirmation. If not, examine source-margin and residual alignment without training or rescoring interventions; do not tune trees or promote controls. Original dual+.02 goal remains active.')
 (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
 paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive_manifest.json','archive_recheck.json','summarize.log']]
 (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',artifact_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted(paths)},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes'],**plan['input_hashes']}),indent=2))
 active=HERE.parent/'ACTIVE_SCOPE.json';s=json.loads(active.read_text());s.update(active_directory=str(HERE),latest_continuation=str(HERE/'CONTINUATION.json'),next_action=state['next_action'],latest_archive_check=str(HERE/'archive_recheck.json'),latest_experiment_archive_check=str(HERE/'archive_recheck.json'));active.write_text(json.dumps(s,indent=2))
 progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0];best=next(r for r in rows if r['config']=='oof_raw')
 progress.write_text(prefix+f"最新完成[OOF树修正](midrug_oof_tree_20261007/REPORT.md)：复用18个基模型并重放，9次新树拟合及重放。oof_raw内部AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}，推进：{promote}。原验证/测试未使用，PLMF哈希保持。\n")
 print(json.dumps(selection,indent=2))
if __name__=='__main__':main()
