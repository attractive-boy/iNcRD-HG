"""Apply four paired-control gates and archive the training-only experiment."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==24 and all(r['status']=='passed' for r in v['records'])
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
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
    for r in protected['records']:
        ds,f=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,source_unchanged=True,prior_full_replay_sha256=sha256(original)),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['comparisons']['control']['auc']['mean_delta']:+.9f} | {r['comparisons']['control']['ap']['mean_delta']:+.9f} | {r['comparisons']['unmasked10']['auc']['mean_delta']:+.9f} | {r['comparisons']['unmasked10']['ap']['mean_delta']:+.9f} |" for r in rows)
    matched=[]
    for r in rows:
        if r['config'] not in plan['eligible_candidates']:continue
        suffix='_rec10' if r['config'].endswith('_rec10') else ''
        for n in ['random'+suffix,'weight_sam'+suffix]:
            c=r['comparisons'][n];matched.append(f"| {r['config']} | {n} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |")
    new=[r for r in v['records'] if not r['reused_control']]
    error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions']);plain=max(r['plain_m123_inference_error'] for r in new)
    report=f'''# 配对表示扰动的纯miRNA内部结果

**原测试AUC/AP同时绝对+0.02尚未证实。** 已完成12次新拟合和12个新检查点重放。
12个旧对照复用已完成重放的证据，并重新核对源码/检查点/原始概率/输入哈希、
概率指标和配对标签索引。这些复用核验不算新的GPU检查点重放。
没有新候选进入原验证或测试。

| 配置 | 内部AUC | 内部AP | 对纯M123 ΔAUC | ΔAP | 对unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

| 表示扰动候选 | 匹配对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(matched)}

候选{best['config']}对四个规定对照的双指标最小平均增益为
{best['selection_score']:+.9f}，是否通过+.002内部推进门槛：{promote}。
该门槛没有替代最终测试的双指标+.02；随机扰动配置不作为事后新增候选。

扰动位置为M1/M2/M3输出拼接、非线性融合头之前。尺度只由当次训练配对表示的
列标准差决定，固定标准化RMS半径.05，比较逐配对梯度方向与等半径高斯方向。
两种方向对零梯度配对采用相同的不扰动规则。方向和尺度停止梯度，原H保留梯度。
损失是权重和为1的clean/perturbed BCE，前30轮普通BCE，再30轮升至各一半；
原始特征重建0/10、训练预算、RMSprop配方及原推理结构保留。
不是E2A条件VAE/能量方法的原样复现，未执行历史扩散增强的其他图加载器。

验证包含float64独立椭球公式与线性化最优性、尺度等变、固定扰动有限差分、
零半径/零混合权重退化、融合头等价、BN每轮一次、无原地参数扰动、留出标签翻转。
12个训练后模型的损失组成、选轮、半径/方向、原始概率和M1/M2/M3显式删图通过。
最大删图logit差{error:.3g}；移除训练增强与重建解码器后原M123推理最大概率差{plain:.3g}。

只用原miRNA/drug白名单输入，无lncRNA、外部数据、注释、序列或新预训练权重。
PLMF10个检查点、冻结输入和发布源码哈希与此前完整重放一致。
已验证纯miRNA M123+BPR原测试均值仍为.9575892678/.9233078960；
最终要求seed42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF两集不下降。
'''
    diag=HERE/'sensitivity_diagnostics.json'
    if diag.exists():
        ds=json.loads(diag.read_text());assert ds['status']=='passed' and ds['selection_sha256']==sha256(HERE/'selection.json')
        dm={r['config']:r for r in ds['means']}
        report+=f'''
冻结选择后的配对表示敏感性：同一标准化RMS半径.05下，一阶梯度方向扰动使
纯M123关联训练BCE增加{dm['control']['adversarial_excess']:.6f}，adv为
{dm['adv']['adversarial_excess']:.6f}；unmasked10为{dm['unmasked10']['adversarial_excess']:.6f}，
adv_rec10为{dm['adv_rec10']['adversarial_excess']:.6f}。
这是固定表示尺度和有限半径的一步诊断，不是精确最坏损失或泛化保证。
详见[训练诊断](SENSITIVITY_DIAGNOSTICS.md)，不用于更改配置选择。
'''
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',previous_turn_classification='progress:12 SAM fits and sensitivity diagnosis',
        current_turn='progress:12 pair-representation perturbation fits and new checkpoint replays,12 immutable control checks',best=best,promote_to_outer_validation=promote,
        no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Freeze outer-validation confirmation only if promoted; otherwise use paired representation-vs-parameter results, not further radius retuning, to identify a materially different information or representation mechanism.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[配对表示扰动](midrug_pair_perturbation_20261006/REPORT.md)12次新拟合和12个新检查点重放。
候选{best['config']}：内部AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；
对四个指定对照双指标最小平均增益{best['selection_score']:+.9f}，推进原验证：{promote}。
上一轮[参数SAM及敏感性诊断](midrug_sam_20261006/REPORT.md)保留。
没有新候选原验证/测试结果；PLMF10个受保护检查点、冻结输入和发布源码保持原哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','plmf_hash_recheck.json','results/plan.json','initial_checks.json','README.md','predecessor_review.json']
    if diag.exists():paths+=['sensitivity_diagnostics.json','SENSITIVITY_DIAGNOSTICS.md']
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},artifact_hashes={p:sha256(HERE/p) for p in paths}),indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=12,new_checkpoint_replays=12,reused_controls=12,max_deletion_logit_error=error),indent=2))

if __name__=='__main__':main()
