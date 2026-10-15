import json,csv
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256

def main():
 plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text());assert v['status']=='passed' and len(v['records'])==9
 for p,h in {**plan['source_hashes'],**plan['evidence_hashes'],**plan['input_hashes']}.items():assert sha256(p)==h
 runs={n:[json.loads(Path(cs[str(f)]['path']).read_text()) for f in range(3)] for n,cs in plan['controls'].items()}
 for n in plan['configs']:runs[n]=[json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)]
 rows=[dict(config=n,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),selected_rounds=[r.get('best_round',r.get('best_epoch')) for r in rs]) for n,rs in runs.items()]
 comparisons={n:{m:dict(mean_delta=float(np.mean([a['inner_'+m]-b['inner_'+m] for a,b in zip(runs['raw'],runs[n])])),fold_deltas=[a['inner_'+m]-b['inner_'+m] for a,b in zip(runs['raw'],runs[n])]) for m in ['auc','ap']} for n in plan['comparators']}
 minimum=min(c[m]['mean_delta'] for c in comparisons.values() for m in ['auc','ap']);promote=minimum>=plan['promotion_minimum_both_gains_over_each_control']
 selection=dict(status='frozen_inner_selection',rows=rows,candidate='raw',comparisons=comparisons,minimum_gain=minimum,promote_to_outer_validation=promote,plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 sp=HERE/'selection.json';assert not sp.exists();sp.write_text(json.dumps(selection,indent=2))
 with (HERE/'inner_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text());frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
 for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
 for r in protected['records']:
  ds,f=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256'];assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256'];assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256'];protection.append(r)
 (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,source_unchanged=True,prior_full_replay_sha256=sha256(original),new_gpu_replay=False),indent=2))
 table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['selected_rounds']} |" for r in rows)
 gains='\n'.join(f"| {n} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |" for n,c in comparisons.items())
 error=max(x['full_logit_error'] for r in v['records'] for x in r['physical_deletions']);oracle=max(r['independent_tree_error'] for r in v['records'])
 report=f'''# 原始特征坐标的固定M123树修正

**原测试AUC/AP双+.02尚未证实。** 9次新拟合、9个选择模型重放、9个最终树检查完成；
复用6个旧对照并核查输入/概率/模型哈希及保存指标。没有新原验证/测试。

| 配置 | 内部AUC | 内部AP | 选择轮数（对照为原神经训练轮） |
|---|---:|---:|---|
{table}

仅raw可推进，相对规定对照的三折配对均值差：

| 对照 | ΔAUC | ΔAP |
|---|---:|---:|
{gains}

最小平均增益{minimum:+.9f}；各对照双指标均≥+.002的冻结门槛：{promote}。
不事后选pca32/structure，不调整树深/学习率/轮数或融合系数。

完整原M123作为冻结base margin；树只加logit修正，stage0精确退回原margin。
raw包含全部938+768原特征，pca32每侧32维，structure仅8维精确计数。三者预算相同，
不声称维数、容量或计算量相同。来源检查点在同一内部折选过，属于自适应开发。

首次预检在[前版目录](../midrug_raw_residual_20261007/precheck.log)发现硬树阈值放大浮点
M3差异，删查询logit差约.0581。保留失败，不放宽容差；本版树输入改为int64精确
拓扑计数，原完整M3/EASE仍在M123中。独立二部矩阵3/5次幂与解析计数逐位一致，
全部所查真实正负删除的树特征逐位一致。九个完整预测删查询最大logit误差{error:.3g}。
保存模型概率重放逐位一致，独立Python逐树遍历最大margin误差{oracle:.3g}。

近期[FlexBIP文献/作者代码范围核查](../midrug_decoupling_review_20261007/REPORT.md)没有
提供足够依据直接重做神经融合。本轮属于经典提升树的信息边界检验，不是该论文复现。
只用原miRNA/drug；无lncRNA、外部数据、注释或新权重。PLMF十个受保护检查点及原
冻结数据/源码保持哈希，本轮没有新PLMF GPU重放。

最终要求仍为原MiDrug测试seeds42/43/44均值AUC≥.9733929274189016、AP≥.933683373732677。
已确认纯miRNA M123+BPR原测试均值仍为.9575892678/.9233078960；目标未完成。
'''
 (HERE/'REPORT.md').write_text(report)
 state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',previous_turn_classification='progress:9 drug-token fits and9 fixed-model diagnoses',
  current_turn='progress:primary/source review, caught float hard-split deletion failure, exact-integer fix,9 residual fits and9 replays',promote_to_outer_validation=promote,
  selection_sha256=sha256(sp),no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],next_action='If promoted freeze original-validation confirmation; otherwise use fixed-tree feature utilization to determine which information actually contributed, without tuning or promoting controls. Original dual+.02 goal remains active.')
 (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
 paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive_manifest.json','archive_recheck.json','summarize.log']]
 (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',artifact_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted(paths)},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes'],**plan['input_hashes']}),indent=2))
 active=HERE.parent/'ACTIVE_SCOPE.json';s=json.loads(active.read_text());s.update(active_directory=str(HERE),latest_continuation=str(HERE/'CONTINUATION.json'),latest_research_review=str(HERE.parent/'midrug_decoupling_review_20261007/REPORT.md'),next_action=state['next_action'],latest_archive_check=str(HERE/'archive_recheck.json'),latest_experiment_archive_check=str(HERE/'archive_recheck.json'));active.write_text(json.dumps(s,indent=2))
 progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0];best=next(r for r in rows if r['config']=='raw')
 progress.write_text(prefix+f"最新完成[原始坐标树修正](midrug_raw_residual_v2_20261007/REPORT.md)：9次拟合和重放。raw内部AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}，推进：{promote}。原验证/测试未使用，PLMF哈希保持。首次硬阈值删查询失败已保留，并用精确整数计数修正。\n")
 print(json.dumps(selection,indent=2))
if __name__=='__main__':main()
