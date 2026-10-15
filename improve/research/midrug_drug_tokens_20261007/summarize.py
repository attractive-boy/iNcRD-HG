"""Apply fixed gate, protect PLMF and archive all sources/results."""
import json,csv
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256

def main():
 plan=json.loads((HERE/'results/plan.json').read_text());audit=json.loads((HERE/'verification.json').read_text())
 assert audit['status']=='passed' and len(audit['records'])==9
 for p,h in {**plan['source_hashes'],**plan['evidence_hashes'],**plan['input_hashes']}.items():assert sha256(p)==h
 runs={n:[json.loads(Path(cs[str(f)]['path']).read_text()) for f in range(3)] for n,cs in plan['controls'].items()}
 for n in plan['configs']:runs[n]=[json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)]
 rows=[dict(config=n,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),best_epochs=[r['best_epoch'] for r in rs]) for n,rs in runs.items()]
 comparisons={n:{m:dict(mean_delta=float(np.mean([a['inner_'+m]-b['inner_'+m] for a,b in zip(runs['mirna'],runs[n])])),fold_deltas=[a['inner_'+m]-b['inner_'+m] for a,b in zip(runs['mirna'],runs[n])]) for m in ['auc','ap']} for n in plan['comparators']}
 minimum=min(c[m]['mean_delta'] for c in comparisons.values() for m in ['auc','ap']);promote=minimum>=plan['promotion_minimum_both_gains_over_each_control']
 selection=dict(status='frozen_inner_selection',rows=rows,candidate='mirna',comparisons=comparisons,minimum_gain=minimum,promote_to_outer_validation=promote,
  plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 target=HERE/'selection.json';assert not target.exists();target.write_text(json.dumps(selection,indent=2))
 with (HERE/'inner_summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
 for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
 frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
 for r in protected['records']:
  ds,f=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
  assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256'];assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
  protection.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
 (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,source_unchanged=True,prior_full_replay_sha256=sha256(original)),indent=2))
 table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['best_epochs']} |" for r in rows)
 gains='\n'.join(f"| {n} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |" for n,c in comparisons.items())
 error=max(x['errors']['full_logit'] for r in audit['records'] for x in r['physical_deletions'])
 replay=max(r['prediction_error'] for r in audit['records']);oracle=max(r['branch']['oracle_error'] for r in audit['records'])
 report=f'''# 原药物token条件汇总：内部检验

**原MiDrug测试AUC/AP双+.02尚未达到。** 本轮9次新拟合、9个选择模型重放，9个最终状态
检查及选择/最终优化器与mask RNG审计全部通过。复用6个旧对照，原验证/测试未使用。

| 配置 | 内部AUC | 内部AP | 选择轮数 |
|---|---:|---:|---|
{table}

仅mirna可推进，其相对规定对照的三折配对均值差：

| 对照 | ΔAUC | ΔAP |
|---|---:|---:|
{gains}

最小双指标增益{minimum:+.9f}；冻结门槛每个对照的AUC/AP均≥+.002；推进：{promote}。
不事后将population/pooled作为候选，不据此扫描宽度、头数或温度。

本轮使用原ChemBERTa最终token，由原miRNA表达query做注意力，保留全局乘性交互。
两个对照保留相同全局交互和参数：共享query、均值汇总，见[冻结设计](README.md)。
三组均加原unmasked10重建；与先前药物query汇总表达列及ChemBERTa均值LoRA不同。
仅迁移近期论文的局部交互想法，不是论文完整复现；[文献核验](READING_NOTES.md)。

所有权重和原输入哈希复核；原512截断保留（1种药物），新汇总排除特殊ID和padding。
预检查独立float64全参数梯度、padding污染、token同步置换和分块均通过。
训练后最大概率重放误差{replay:.3g}，独立token参考误差{oracle:.3g}，
完整M1/M2/M3物理删查询logit误差{error:.3g}。重建解码器移除后推理等价。
仅原miRNA/drug，不引入外部数据、序列、lncRNA或新权重。token不是已验证化学子结构。

PLMF10个受保护检查点、原冻结图/划分和发布源码哈希不变，本轮没有新PLMF GPU重放。
已确认纯miRNA M123+BPR原测试均值仍为AUC .9575892678、AP .9233078960；
最终门槛为AUC .9733929274、AP .9336833737。负结果不能证明目标不可能。
'''
 (HERE/'REPORT.md').write_text(report)
 state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',current_turn='progress:9 original ChemBERTa token attention fits,9 selected replays,9 final inspections and optimizer/RNG audits',
  promote_to_outer_validation=promote,selection_sha256=sha256(target),no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
  next_action='If promoted freeze original-validation confirmation; otherwise inspect attention utilization without retuning or promoting controls. Original dual+.02 requirement remains active.')
 (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
 paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive_manifest.json','archive_recheck.json','summarize.log']]
 (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',artifact_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted(paths)},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes'],**plan['input_hashes']}),indent=2))
 active=HERE.parent/'ACTIVE_SCOPE.json';s=json.loads(active.read_text());s.update(active_directory=str(HERE),latest_continuation=str(HERE/'CONTINUATION.json'),latest_research_review=str(HERE/'READING_NOTES.md'),next_action=state['next_action'],latest_archive_check=str(HERE/'archive_recheck.json'),latest_experiment_archive_check=str(HERE/'archive_recheck.json'));active.write_text(json.dumps(s,indent=2))
 progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
 best=next(r for r in rows if r['config']=='mirna')
 progress.write_text(prefix+f"最新完成[原药物token条件汇总](midrug_drug_tokens_20261007/REPORT.md)：9次新拟合与重放全部通过；mirna内部AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}，推进：{promote}。原验证/测试未使用，PLMF哈希保持。\n")
 print(json.dumps(selection,indent=2))
if __name__=='__main__':main()
