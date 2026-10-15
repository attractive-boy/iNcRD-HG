"""Freeze selection and archive experiment, dependencies and protected PLMF state."""
import json,csv
from pathlib import Path
import numpy as np
from common import HERE,CLEAN,RELEASE,sha256,verify_plan,write

def main():
    plan=verify_plan();v=json.loads((HERE/'verification.json').read_text());assert v['status']=='passed'
    fitted=json.loads((HERE/'results/fits_frozen.json').read_text());rows=[];comparisons={};minimum=None;promote=False;standalone=[]
    if fitted['all_converged']:
        results=[json.loads((HERE/'results'/f'f{fold}_evaluation.json').read_text()) for fold in range(3)]
        scores={name:[r['inner_metrics'][name] for r in results] for name in plan['readouts']}
        for name in plan['controls']:
            scores[name]=[next(r['metrics'] for r in v['reused_controls'] if r['name']==name and r['fold']==f) for f in range(3)]
        rows=[dict(config=name,**{m:float(np.mean([r[m] for r in scores[name]])) for m in ['auc','ap','bce','brier']}) for name in scores]
        standalone=[dict(config=name,**{m:float(np.mean([r['block_standalone'][name][m] for r in results])) for m in ['auc','ap']}) for name in plan['readouts']]
        comparisons={name:{m:dict(mean_delta=float(np.mean([a[m]-b[m] for a,b in zip(scores['learned'],scores[name])])),
            fold_deltas=[a[m]-b[m] for a,b in zip(scores['learned'],scores[name])]) for m in ['auc','ap']} for name in plan['comparators']}
        minimum=min(r[m]['mean_delta'] for r in comparisons.values() for m in ['auc','ap'])
        promote=minimum>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',candidate='learned',all_converged=fitted['all_converged'],rows=rows,standalone=standalone,
        comparisons=comparisons,minimum_gain=minimum,promote_to_outer_validation=promote,plan_sha256=sha256(HERE/'results/plan.json'),
        verification_sha256=sha256(HERE/'verification.json'),fits_frozen_sha256=sha256(HERE/'results/fits_frozen.json'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    assert not (HERE/'selection.json').exists();write(HERE/'selection.json',selection)
    if rows:
        with (HERE/'inner_summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    deps={**plan['source_hashes'],**plan['evidence_hashes']}
    for cs in plan['controls'].values():
        for r in cs.values():
            p=Path(r['path']);deps.update({str(p):r['metadata_sha256'],str(p.with_suffix('.pt')):r['checkpoint_sha256'],str(p.with_name(p.stem+'_inner.npz')):r['probability_sha256']})
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text());deps[str(original)]=sha256(original)
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h;deps[str(RELEASE/p)]=h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
    for r in protected['records']:
        ds,fold=r['dataset'],r['fold']
        paths={str(refs/f'incrd_m123_{ds}_f{fold}_s42.pt'):r['checkpoint_sha256'],str(frozen/f'{ds}_fold{fold}.npz'):r['split_sha256'],
            str(frozen/f'{ds}_graph.pt'):manifest['datasets'][ds]['graph_sha256']}
        for p,h in paths.items():assert sha256(p)==h
        deps.update(paths);protection.append(dict(dataset=ds,fold=fold,hashes=paths))
    write(HERE/'plmf_hash_recheck.json',dict(status='passed',records=protection,new_gpu_replays=0,prior_gpu_replay_sha256=sha256(original)))
    table='\n'.join(f"| {r['config']} | {r['auc']:.9f} | {r['ap']:.9f} |" for r in rows)
    standalone_table='\n'.join(f"| {r['config']} | {r['auc']:.9f} | {r['ap']:.9f} |" for r in standalone)
    comparison_table='\n'.join(f"| {name} | {r['auc']['mean_delta']:+.9f} | {r['ap']['mean_delta']:+.9f} |" for name,r in comparisons.items())
    convergence=[dict(fold=r['fold'],chosen=r['chosen'],sweeps={k:len(s['history'])-1 for k,s in r['starts'].items()},
        elbo={k:s['final_elbo'] for k,s in r['starts'].items()}) for r in fitted['records']]
    (HERE/'REPORT.md').write_text(f'''# 共轭二部分块模型：固定内层实验

原测试 AUC/AP 同时绝对 +0.02 尚未达到。本轮完成 6 次 CAVI 拟合、6 次固定
分组闭式拟合，全部参数与预测复核；没有新神经训练或原验证/测试评估。
仅使用原 miRNA–药物数据，未引入 lncRNA、外部注释或新预训练权重。

| 配置（分块修正与原 unmasked10 融合） | 内层 AUC | 内层 AP |
|---|---:|---:|
{table}

| learned 相对各对照 | ΔAUC | ΔAP |
|---|---:|---:|
{comparison_table}

所有 CAVI 收敛：{fitted['all_converged']}。候选通过冻结推进门槛：{promote}。
推进要求对全部四个规定对照的两个平均指标均至少 +.002，不替代用户 +.02 目标。

## 机制与边界

这是经典 categorical/Beta/Dirichlet 共轭模型的有限检验，不是新提出的模型族，
也不是 TGSBM 复现。TGSBM 重叠群组与原特征固定分组此前均已有失败证据。
本轮仅对当前训练已观测正负配对建模，未知配对不算负例；分组数16/8，
Beta(.5,.5)、Dirichlet 每类别1，固定 .1 居中 log-odds 修正，不调参。
每折两个初始化按训练 ELBO 选取，所有拟合状态和选择先冻结，再访问内层指标。
两固定分组对照用相同正负标签、相同先验、相同融合权重。

单独分块预测（仅诊断，不是额外候选）：

| 配置 | 内层 AUC | 内层 AP |
|---|---:|---:|
{standalone_table}

收敛及选择：`{json.dumps(convergence)}`。
成员概率与 Beta 参数是训练所得模型参数，不是逐查询重新计算的图统计量。
它们不满足训练观测删除后的后验不变性；所有留出查询未参与其拟合。
潜在分组不等于已验证 miRNA 家族或药物类别。概率以采样基准的观测机制为条件，
不能直接解释为生物总体患病率或关联发生率。变分近似和局部最优仍是限制。

独立逐项计数/更新、数值积分 ELBO、小模型穷举边际证据下界均通过。
初次预检查发现当前 SciPy 的 Beta entropy 用默认精度积分，误差约9.2e-9；
随后直接提高积分精度仍遇端点奇异误差；已在冻结前改用处理代数/对数端点奇异的
加权积分，保留三个失败日志，没有改训练目标或先验。
正式复核用逐观测统计和优化后的独立 ELBO 公式，并重拟合6个状态；
另核对概率、查询次序/分批、潜类别重编号和独立排序指标。

原神经概率直接复用冻结产物，新增神经 GPU 重放为0；PLMF 十个检查点、
输入与发布代码哈希核查通过，新增 PLMF GPU 重放为0。
这些内层折已反复用于开发，不能视为独立验证。仍需原测试三种子均值
AUC≥.9733929274189016、AP≥.933683373732677，且 PLMF 两数据集不下降。
论文核读及与旧实验的差别见 [READING_NOTES.md](READING_NOTES.md)。
''')
    next_action='Use the frozen observed-entry latent-block comparison as bounded evidence; do not tune group counts, priors, initialization smoothing, fusion weight or select controls after seeing these metrics. Original dual +.02 goal remains active.'
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',new_iterative_fits=6,new_closed_form_fits=6,
        new_iterative_fit_replays=6,new_closed_form_state_checks=6,new_candidate_test_evaluations=0,promote_to_outer_validation=promote,
        processes_expected_live=[],next_action=next_action)
    write(HERE/'CONTINUATION.json',state)
    artifacts=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive_manifest.json','archive_recheck.json','summarize.log']]
    archive=dict(status='archived',artifact_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted(artifacts)},frozen_dependencies=deps)
    write(HERE/'archive_manifest.json',archive);count=0
    for p,h in archive['artifact_hashes'].items():assert sha256(HERE/p)==h;count+=1
    for p,h in deps.items():assert sha256(p)==h;count+=1
    write(HERE/'archive_recheck.json',dict(status='passed',hash_checks=count,archive_sha256=sha256(HERE/'archive_manifest.json'),selection_sha256=sha256(HERE/'selection.json')))
    sf=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(sf.read_text());scope.update(active_directory=str(HERE),latest_continuation=str(HERE/'CONTINUATION.json'),
        latest_archive_check=str(HERE/'archive_recheck.json'),latest_experiment_archive_check=str(HERE/'archive_recheck.json'),
        latest_research_review=str(HERE/'READING_NOTES.md'),latest_research_continuation=str(HERE/'CONTINUATION.json'),
        latest_research_archive_check=str(HERE/'archive_recheck.json'),next_action=next_action);write(sf,scope)
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'最新完成[共轭二部分块模型](midrug_conjugate_blocks_20261007/REPORT.md)：6 次 CAVI 与 6 次闭式拟合及复核。推进原验证：{promote}。没有新原验证/测试；PLMF 冻结产物未变。\n')
    print(json.dumps(dict(rows=rows,minimum_gain=minimum,promote=promote,archive_hash_checks=count),indent=2),flush=True)

if __name__=='__main__':main()
