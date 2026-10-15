"""Freeze selection, preserve PLMF inputs/models, and archive this finite experiment."""
import json,csv
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,CLEAN,sha256,verify_plan

def write(p,value):p.write_text(json.dumps(value,indent=2))

def main():
    plan=verify_plan();verification=json.loads((HERE/'verification.json').read_text());assert verification['status']=='passed'
    runs=[json.loads((HERE/'results'/f'f{fold}_s42.json').read_text()) for fold in range(3)]
    scores={k:[r['inner_metrics'][k] for r in runs] for k in plan['readouts']}
    for name in plan['controls']:
        scores[name]=[next(r['metrics'] for r in verification['reused_controls'] if r['name']==name and r['fold']==fold) for fold in range(3)]
    rows=[dict(config=k,**{m:float(np.mean([r[m] for r in rs])) for m in ['auc','ap','bce','brier']}) for k,rs in scores.items()]
    comparisons={k:{m:dict(mean_delta=float(np.mean([a[m]-b[m] for a,b in zip(scores['sandwich'],scores[k])])),
        fold_deltas=[a[m]-b[m] for a,b in zip(scores['sandwich'],scores[k])]) for m in ['auc','ap']} for k in plan['comparators']}
    minimum=min(r[m]['mean_delta'] for r in comparisons.values() for m in ['auc','ap'])
    promote=minimum>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',candidate='sandwich',rows=rows,comparisons=comparisons,minimum_gain=minimum,
        promote_to_outer_validation=promote,plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={f'f{f}_s42.json':sha256(HERE/'results'/f'f{f}_s42.json') for f in range(3)},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    assert not (HERE/'selection.json').exists();write(HERE/'selection.json',selection)
    with (HERE/'inner_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    deps={**plan['source_hashes'],**plan['evidence_hashes'],str(CLEAN/'graph.pt'):plan['graph_sha256'],str(CLEAN/'inner_data.npz'):plan['data_sha256']}
    for cs in plan['controls'].values():
        for r in cs.values():
            p=Path(r['path']);deps.update({str(p):r['metadata_sha256'],str(p.with_suffix('.pt')):r['checkpoint_sha256'],str(p.with_name(p.stem+'_inner.npz')):r['probability_sha256']})
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text());deps[str(original)]=sha256(original)
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h;deps[str(RELEASE/p)]=h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results'
    protection=[]
    for r in protected['records']:
        ds,fold=r['dataset'],r['fold']
        paths={str(refs/f'incrd_m123_{ds}_f{fold}_s42.pt'):r['checkpoint_sha256'],str(frozen/f'{ds}_fold{fold}.npz'):r['split_sha256'],
            str(frozen/f'{ds}_graph.pt'):manifest['datasets'][ds]['graph_sha256']}
        for p,h in paths.items():assert sha256(p)==h
        deps.update(paths);protection.append(dict(dataset=ds,fold=fold,hashes=paths))
    write(HERE/'plmf_hash_recheck.json',dict(status='passed',records=protection,new_gpu_replays=0,prior_gpu_replay_sha256=sha256(original)))
    table='\n'.join(f"| {r['config']} | {r['auc']:.9f} | {r['ap']:.9f} | {r['bce']:.6f} |" for r in rows)
    paired='\n'.join(f"| {name} | {r['auc']['mean_delta']:+.9f} | {r['ap']['mean_delta']:+.9f} |" for name,r in comparisons.items())
    (HERE/'REPORT.md').write_text(f'''# 末层条件不确定性：内部实验结果

原测试 AUC/AP 同时绝对 +.02 尚未证实。本轮完成三个固定内层折的三次凸拟合，
三次冻结特征与拟合状态重放，12 个读出重放，六个历史对照文件核对。
仅使用 miRNA 和 drug；没有新原验证/测试评估。

| 配置 | 内部 AUC | 内部 AP | 内部 BCE |
|---|---:|---:|---:|
{table}

| Gaussian sandwich 相对对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选 sandwich 的最小规定增益为 {minimum:+.9f}，通过推进门槛：{promote}。
该门槛为相对全部五个对照的 AUC/AP 均值各 +.002，不是最终 +.02 验收。

## 方法及限制

从相同 unmasked10 提取固定 256 维末层输入，以训练 RMS 缩放并添加截距。
标准正态先验精度固定为 1（含截距），Newton 从零拟合至梯度无穷范数 ≤1e-9。
三个折实际更新次数：{[len(r['convergence'])-1 for r in runs]}。
原神经网络末层在本轮指定目标下的梯度范数：{[r['original_gradient_inf'] for r in runs]}。
先拟合驻点，避免直接把早停权重的局部曲率称作该目标的 MAP 后验。

四种读出共享同一 theta。Laplace 用精确末层 Hessian；sandwich 用居中的训练样本
梯度二阶矩及固定 alpha=1 的有限样本因子。后者采用额外的高斯矩匹配，随后做
64 点 Gauss–Hermite 积分，不是有限样本 Dirichlet 积分或完整 Ribbon 复现。
128 点核查最大概率差 {max(max(r['quadrature_error'].values()) for r in runs):.3g}。
constant 使用训练平均方差，其 AUC/AP 与 point 一致，符合单调变换预期。

独立自动微分核对 Hessian/梯度、微小加权重拟合核对影响函数、Dirichlet Monte Carlo
核对有限样本协方差、独立积分核对概率均通过。重放重新提取特征，并用独立求解
验证 Hessian 逆、sandwich、驻点、顺序及分批不变性。模型参数和 BN 全部冻结。
真实正负查询还物理删除后重建图，核对末层特征和全部四种读出。
预检查的模块导入失败及 GPU 浮点输出逐位相等断言失败已保留，均在方案冻结及拟合
之前修复。留出标签翻转要求权重/输入通道逐位相等，浮点输出按 1e-5 容差核查。

结果以固定已学习特征和训练图为条件，忽略编码器、选轮和共享实体依赖带来的
不确定性；不声称具有完整网络后验或覆盖保证。BCE 改善也不能代替排序指标验收。
源检查点选轮及前期开发已反复使用这些内层留出，本轮属于开发证据，不是独立确认。
来源见[文献核读](../midrug_curvature_literature_20261007/READING_NOTES.md)。

PLMF 十个旧检查点、原输入及发布源代码哈希核查通过，本轮新 PLMF GPU 重放为 0。
最终仍须原测试三种子均值 AUC≥.9733929274189016、AP≥.933683373732677，且 PLMF 两集不下降。
''')
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',new_fits=3,new_readout_replays=12,
        new_candidate_test_evaluations=0,promote_to_outer_validation=promote,processes_expected_live=[],
        next_action='Use the fixed mean-estimate and covariance comparison to identify a distinct bottleneck; do not retrospectively tune prior precision, alpha, smoothing scale or select controls. Original dual +.02 goal remains active.')
    write(HERE/'CONTINUATION.json',state)
    artifacts=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive_manifest.json','archive_recheck.json','summarize.log']]
    archive=dict(status='archived',artifact_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted(artifacts)},frozen_dependencies=deps)
    write(HERE/'archive_manifest.json',archive)
    count=0
    for p,h in archive['artifact_hashes'].items():assert sha256(HERE/p)==h;count+=1
    for p,h in deps.items():assert sha256(p)==h;count+=1
    write(HERE/'archive_recheck.json',dict(status='passed',hash_checks=count,archive_sha256=sha256(HERE/'archive_manifest.json'),selection_sha256=sha256(HERE/'selection.json')))
    scopefile=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(scopefile.read_text())
    scope.update(active_directory=str(HERE),latest_continuation=str(HERE/'CONTINUATION.json'),latest_archive_check=str(HERE/'archive_recheck.json'),
        latest_experiment_archive_check=str(HERE/'archive_recheck.json'),latest_research_review=str(HERE.parent/'midrug_curvature_literature_20261007/READING_NOTES.md'),
        latest_research_archive_check=str(HERE.parent/'midrug_curvature_literature_20261007/review_recheck.json'),next_action=state['next_action'])
    write(scopefile,scope)
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'最新完成[末层条件不确定性](midrug_last_layer_uncertainty_20261007/REPORT.md)：3 次固定凸拟合、12 个读出重放。推进原验证：{promote}。没有新原验证/测试结果；PLMF 冻结产物未变。\n')
    print(json.dumps(selection,indent=2))

if __name__=='__main__':main()
