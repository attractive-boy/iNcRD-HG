"""Apply frozen matched-control selection; preserve protected PLMF artifacts."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());audit=json.loads((HERE/'verification.json').read_text())
    assert audit['status']=='passed' and len(audit['records'])==30
    assert all(r['status']=='passed' for r in audit['records'])
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    for n in plan['configs']:runs[n]=[json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for name,rs in runs.items():
        comparisons={}
        for ref in plan['comparators'].get(name,['control','unmasked10']):
            comparisons[ref]={}
            for metric in ['auc','ap']:
                d=[r['inner_'+metric]-b['inner_'+metric] for r,b in zip(rs,runs[ref])]
                comparisons[ref][metric]=dict(mean_delta=float(np.mean(d)),fold_deltas=d)
        rows.append(dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),
            mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),comparisons=comparisons,
            selection_score=min(comparisons[c][m]['mean_delta'] for c in comparisons for m in ['auc','ap'])))
        flat += [dict(config=name,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch']}) for f,r in enumerate(rs)]
    best=max((r for r in rows if r['config'] in plan['eligible_candidates']),key=lambda r:r['selection_score'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))})
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results'
    protection=[]
    for r in protected['records']:
        ds,f=r['dataset'],r['fold']
        assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,
        prior_full_replay_sha256=sha256(original),source_unchanged=True),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | "
        f"{r['comparisons']['control']['auc']['mean_delta']:+.9f} | {r['comparisons']['control']['ap']['mean_delta']:+.9f} | "
        f"{r['comparisons']['unmasked10']['auc']['mean_delta']:+.9f} | {r['comparisons']['unmasked10']['ap']['mean_delta']:+.9f} |" for r in rows)
    matched=[]
    for r in rows:
        if r['config'] not in plan['eligible_candidates']:continue
        suffix='_rec10' if r['config'].endswith('_rec10') else ''
        for mode in ['soft','direct','uniform']:
            c=mode+suffix;v=r['comparisons'][c]
            matched.append(f"| {r['config']} | {c} | {v['auc']['mean_delta']:+.9f} | {v['ap']['mean_delta']:+.9f} |")
    new=[r for r in audit['records'] if not r['reused_control']]
    assert len(new)==6
    max_error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    max_switch=max(r['gradient_switch_forward_error'] for r in new)
    report=f'''# 补全概率完整梯度：内部对照

**原测试双指标绝对 +0.02 尚未证实。** 本轮6次新训练和6个新检查点重放通过；
24个对照复用此前完整重放，当前再次核对冻结源码、检查点、原始概率及其指标/标签/索引。
这些复用核验不计为24次新的GPU检查点重放。没有访问新候选的原验证或测试。

| 配置 | 内部 AUC | 内部 AP | 对纯 M123 ΔAUC | ΔAP | 对 unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

| 完整梯度候选 | 匹配旧对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(matched)}

按冻结规则，候选 {best['config']} 对所有规定对照的双指标最小平均增益为
{best['selection_score']:+.9f}。内部 +.002 推进门槛：{promote}。
最终要求仍为原测试seed42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF两集不降。

## 仅改变梯度路径

前向仍是上一轮的第一层图条件预测器、精确查询删除、预测行/列修正和
两观测边加一未观测边的路径汇总。参数、初始值、原M123、辅助BCE0.1、
重建0/10及训练预算保持对应设置。仅去掉补全概率与路径特征的no_grad，
使关联预测损失能通过这些权重训练预测器与共享第一层。
旧代码和结果保持原哈希；直接复用已归档路径函数的__wrapped__，不重写算子。
外部推理no_grad继续生效，没有在推理中强制启用梯度。

因此这是训练机制消融，不是新架构或NCNC官方实现的原样复现。软概率不宣称校准。
没有lncRNA、外部数据、序列、注释或新权重；仅原miRNA/drug及当次训练关系。

## 验证

非零残差下的前向与停止梯度版本一致。微型float64图中分别验证正边、度一端点、
已观测零标签和未观测查询：物理删图后直接调用PyG GeneralConv，再逐条枚举
带梯度的路径，与解析修正后的最终预测和全部参数梯度一致。
路径单独损失对预测器的梯度非零，原停止梯度版本的路径张量不带梯度。
真实完整训练步、参数有限性和留出标签翻转检查通过。

六个训练后检查点再次核对概率/指标/选轮/三项float32损失和M1/M2/M3最终删图预测；
最大最终logit差{max_error:.3g}。把同一训练后权重切回停止梯度版本，最大推理概率差
{max_switch:.3g}。独立PyG删图路径特征复核通过；移除重建解码器不改变推理。

PLMF10个检查点、冻结图和划分、发布源码哈希与此前完整重放一致。
已验证的纯miRNA M123+BPR原测试均值仍为0.9575892678 / 0.9233078960。
此前固定概率混合证书不涵盖本轮新训练，不把它当作本轮或原测试的上限。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn='progress: two soft-completion experiments and certified bound on fixed probability mixtures',
        current_turn='progress: same-forward full completion gradients, 6 fits and new checkpoint replays, 24 immutable control checks',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Freeze original-validation confirmation if promoted; otherwise use the paired gradient-only result to select the next mechanism.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';progress=p.read_text().split('最新完成')[0]
    progress+=f'''最新完成 [完整补全梯度](midrug_differentiable_completion_20261006/REPORT.md) 的6次新训练，前向保持一致，只改变补全概率的梯度路径。
候选 {best['config']}，内部AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；
对规定对照的最小双指标增益 {best['selection_score']:+.9f}，推进原验证：{promote}。
24个旧对照以冻结检查点/原始概率哈希和此前完整重放证据复用。
之前的[图条件补全](midrug_graph_completion_20261006/REPORT.md)及其[固定概率组合上限](midrug_graph_completion_20261006/MIXTURE_BOUND.md)保留。
没有新候选进入原验证/测试；PLMF10个检查点和冻结输入、发布源码保持原哈希。
'''
    p.write_text(progress)
    paths=['verification.json','selection.json','REPORT.md','plmf_hash_recheck.json','results/plan.json','initial_checks.json','README.md','predecessor_review.json']
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths})
    (HERE/'archive_manifest.json').write_text(json.dumps(archive,indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=6,new_checkpoint_replays=6,reused_controls=24,max_deletion_logit_error=max_error,max_gradient_switch_prediction_error=max_switch),indent=2))

if __name__=='__main__':main()
