"""Apply frozen matched-control selection; preserve protected PLMF artifacts."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());audit=json.loads((HERE/'verification.json').read_text())
    assert audit['status']=='passed' and len(audit['records'])==18
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
        c='shuffled'+('_rec10' if r['config'].endswith('_rec10') else '')
        v=r['comparisons'][c];matched.append(f"| {r['config']} | {c} | {v['auc']['mean_delta']:+.9f} | {v['ap']['mean_delta']:+.9f} |")
    new=[r for r in audit['records'] if not r['reused_control']]
    max_error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    report=f'''# 原特征分组与双向组间频率：内部实验

**原测试双指标绝对 +0.02 尚未证实。** 本轮12次训练、6个复用对照全部重放通过。
只使用原训练内部三折、seed42；没有评估新候选的原验证或测试。

| 配置 | 内部 AUC | 内部 AP | 对纯 M123 ΔAUC | ΔAP | 对 unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

| 特征分组 | 同大小随机分组对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(matched)}

按冻结规则，候选 {best['config']} 的所有规定对照双指标最小平均增益为
{best['selection_score']:+.9f}。内部 +.002 推进门槛：{promote}。
随机分组是机制对照，不在可推进候选中；没有事后改选随机分组。
最终仍需原测试三种子 AUC≥.9733929274、AP≥.9336833737，PLMF 两数据集不降。

## 方法与适用边界

CGLE（2025，arXiv:2511.06982v1）式8/13给出组间条件边频率与预测器融合。
本地采用原表达谱和 ChemBERTa 按行 L2 归一化后独立 KMeans16/8，固定 seed20261006、
n_init10；不做目标关联图聚合，也不引入真实生物类别、序列或外部权重。
特征簇不是生物学类别。对照固定置换成员、保持两侧组大小。

当前训练正边的组间计数按查询精确扣除，两个方向均以目标组大小比例为基准，
使用 Dirichlet10 平滑，再转成 log 富集度。它不是具体配对的校准存在概率。
两维特征经2→64→1残差头加入原 M123，末层零初始化；没有三边路径也可计算。
另行比较有无原无掩码重建。这是独立二部图适配，不是CGLE原样复现。

不同于此前PLMF的生物family/sequence交叉随机效应，本轮不使用那些数据或标签。
前一轮路径矩检验观察路径内中间节点的联合特征；本轮检验跨配对共享的组频率。
详细原文核读和未优先采用NCNC/IECNC/CELP的理由，见
../midrug_literature_20261006/COMPLETION_READING_NOTES.md。

## 检查

16个微型查询与独立逐边计数一致；4个float64最终预测/参数梯度显式删图案例通过。
四配置均恢复原主干初始参数与预测，通过真实训练步、非零分支梯度和留出标签翻转。
训练后每个checkpoint重复验证M1/M2/M3、组频率和最终预测删记录一致，
最大最终logit差{max_error:.3g}。概率、配对、标签、索引、指标、选轮和训练损失重放一致。
删除辅助重建解码器不改变推理。组频率表属于每次输入图预处理，没有写入参数。

PLMF10个checkpoint、冻结图和划分、发布源码哈希与既有完整重放一致。
既有纯miRNA M123+BPR原测试均值仍为0.9575892678 / 0.9233078960；本轮内部结果
不替代最终验收。没有使用lncRNA，也没有新增候选的原测试评估。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn='progress: exact joint path moments',
        current_turn='progress: recent literature methods audit, exact group frequencies, 12 fits and 18 checkpoint replays',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Freeze original-validation confirmation if promoted; otherwise inspect paired results before selecting another distinct mechanism.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';progress=p.read_text().split('最新完成')[0]
    progress+=f'''最新完成 [12次组间频率实验](midrug_class_prior_20261006/REPORT.md)，核读CGLE/NCNC/IECNC/CELP方法并落实纯miRNA适配。
内部候选 {best['config']}，AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；
对规定对照的双指标最小增益 {best['selection_score']:+.9f}，推进原验证：{promote}。
此前 [18次路径矩实验](midrug_path_moments_20261006/REPORT.md) 已审计，未通过内部门槛。
没有新候选进入原验证/测试，PLMF10个checkpoint、冻结输入和发布源码哈希复核通过。
'''
    p.write_text(progress)
    paths=['verification.json','selection.json','REPORT.md','plmf_hash_recheck.json','results/plan.json',
        'initial_checks.json','groups.npz','groups.json','README.md','predecessor_review.json']
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths})
    (HERE/'archive_manifest.json').write_text(json.dumps(archive,indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=12,max_deletion_logit_error=max_error),indent=2))

if __name__=='__main__':main()
