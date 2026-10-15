"""Predeclared fixed blends and paired comparisons; no test evaluation."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE, RELEASE, sha256


def main():
    plan = json.loads((HERE/'results/plan.json').read_text())
    verified = json.loads((HERE/'verification.json').read_text())
    assert verified['status'] == 'passed' and len(verified['records']) == 24
    for p,h in plan['source_hashes'].items():
        assert sha256(p) == h
    controls = {name:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)]
                for name,c in plan['controls'].items()}
    rows = []
    flat = []
    def collect(name,kind,runs,weight=None):
        row = dict(config=name,kind=kind,weight=weight,
                   mean_inner_auc=float(np.mean([r['inner_auc'] for r in runs])),
                   mean_inner_ap=float(np.mean([r['inner_ap'] for r in runs])))
        gains = {}
        for control,cruns in controls.items():
            gains[control] = {}
            for m in ['auc','ap']:
                delta = [r['inner_'+m]-b['inner_'+m] for r,b in zip(runs,cruns)]
                gains[control][m] = dict(mean_delta=float(np.mean(delta)),fold_deltas=delta)
        row['comparisons'] = gains
        row['selection_score'] = min(gains[c][m]['mean_delta'] for c in gains for m in ['auc','ap'])
        rows.append(row)
        for f,r in enumerate(runs):
            flat.append(dict(config=name,kind=kind,weight=weight,fold=f,
                             inner_auc=r['inner_auc'],inner_ap=r['inner_ap']))
        return row
    for name,runs in controls.items():
        collect(name,'control',runs)
    blends = HERE/'blends'
    blends.mkdir(exist_ok=True)
    for name in plan['configs']:
        runs = [json.loads((HERE/'results'/f'{name}_f{f}.json').read_text()) for f in range(3)]
        collect(name,'standalone',runs)
        for weight in plan['mixture_weights']:
            combined = []
            for f in range(3):
                control_path = Path(plan['controls']['unmasked10'][str(f)]['path'])
                with np.load(control_path.with_name(control_path.stem+'_inner.npz')) as z:
                    y,pairs,indices = z['y'],z['pairs'],z['indices']
                    baseline = logit(np.clip(z['probability'].astype(float),1e-6,1-1e-6))
                with np.load(HERE/'results'/f'{name}_f{f}_inner.npz') as z:
                    score = (1-weight)*baseline+weight*z['score']
                result = dict(inner_auc=float(roc_auc_score(y,score)),inner_ap=float(average_precision_score(y,score)))
                combined.append(result)
                np.savez_compressed(blends/f'{name}_w{weight}_f{f}.npz',y=y,pairs=pairs,indices=indices,score=score)
            collect(name,'blend',combined,weight)
    best = max((r for r in rows if r['kind']=='blend'),key=lambda r:r['selection_score'])
    selected_converged = all(r['converged'] for r in verified['records'] if r['tag'].startswith(best['config']+'_f'))
    promote = selected_converged and best['selection_score'] >= plan['promotion_minimum_both_gains_over_each_control']
    robust_differences = []
    for r in rows:
        if r['kind']=='standalone' and r['config'].endswith('_robust'):
            comparator = next(v for v in rows if v['kind']=='standalone' and v['config']==r['config'].replace('_robust','_logistic'))
            robust_differences.append(dict(robust=r['config'],comparator=comparator['config'],
                auc_delta=r['mean_inner_auc']-comparator['mean_inner_auc'],ap_delta=r['mean_inner_ap']-comparator['mean_inner_ap']))
    selection = dict(status='frozen_inner_selection',rows=rows,best_blend=best,
        robust_comparisons=robust_differences,promote_to_outer_validation=promote,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted((HERE/'results').glob('*_f*.json'))},
        blend_hashes={p.name:sha256(p) for p in sorted(blends.glob('*.npz'))})
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    # Exact preservation of the previously replayed PLMF artifacts.
    original = HERE.parent/'midrug_internal_20261006/plmf_protection.json'
    protected = json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():
        assert sha256(RELEASE/p) == h
    frozen = RELEASE/'data/frozen'
    manifest = json.loads((frozen/'manifest.json').read_text())
    refs = HERE.parent/'baselines_m123_20261004/results'
    protection = []
    for r in protected['records']:
        ds,f = r['dataset'],r['fold']
        assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt') == r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz') == r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt') == manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],
            split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,
        prior_full_replay_sha256=sha256(original),source_unchanged=True),indent=2))
    table = '\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} |"
                      for r in rows if r['kind']!='blend')
    btable = '\n'.join(f"| {r['config']} | {r['weight']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | "
        f"{r['comparisons']['unmasked10']['auc']['mean_delta']:+.9f} | {r['comparisons']['unmasked10']['ap']['mean_delta']:+.9f} |"
        for r in rows if r['kind']=='blend')
    converged = sum(r['converged'] for r in verified['records'])
    report = f'''# 原训练观测上的低秩矩阵补全

**原测试双指标 +0.02 目标尚未达到。** 本轮 24 个矩阵拟合已完成并重放，
其中 {converged}/24 达到预固定训练收敛阈值。固定三折来自原训练集；没有读取原验证或测试。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| 与 unmasked10 组合的分支 | 分支权重 | 内部 AUC | 内部 AP | 相对 unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|
{btable}

按预声明的两项对照、双指标最小平均增益选择，最佳组合是
{best['config']}，权重 {best['weight']}，选择分数 {best['selection_score']:+.9f}。
推进原验证：{promote}。内部 +0.002 只是开发门槛，不替代最终 +0.02 测试要求。

这次测试的四组稀疏误差版本均低于匹配的普通 logistic 版本；权重 0.05 的相似图平滑
也均低于各自无平滑版本。因此不能把最佳混合的小幅收益归因于稳健误差或图平滑。
最佳混合相对 unmasked10 的三折 AUC 均小幅提高，但第 0 折 AP 下降；
这不是跨种子稳定性或统计显著性证据，也不能证明低秩方法的性能上限。

## 方法与证据范围

这是受到 IRPCA（JCIM 2025）摘要启发的独立凸优化适配，不是原论文完整复现。
只使用原 miRNA/drug 标签、同类相似图和节点集合，保留原划分；未增加序列、注释或 lncRNA。
直接拟合核范数约束的观测项 logistic 目标；未观测配对不进入监督损失。
相似图平滑的权重 0/0.05、核范数比例 0.03/0.1、无误差/稀疏误差 .25 均先固定。
与此前固定低秩神经分支不同，这里是凸矩阵优化，rank 由奇异值阈值决定，且只按训练收敛停止。
完整稳健/普通成对差存于 selection.json。误差修正不是标签错误的生物学判定。

混合只使用内部留出预测和预固定 0.1/0.25 权重，不用矩阵的训练内分数训练 M123。
分支 logit 用原训练先验对齐常数项，不声称概率已校准；评估 AUC/AP 排序指标。
M123 控制来自已完成数值重放的原 checkpoint，当前哈希、配对、标签、预测再核对通过。
本轮所有矩阵的预测、目标函数、奇异值惩罚、独立稀疏误差公式、最终近端梯度残差均重算。
两种节点相似图在翻转内部留出标签后完全不变，fit API 只接收训练配对及标签。

PLMF 10 个受保护 checkpoint、图、划分和发布源码保持与此前完整重放相同的哈希。
既有纯 miRNA M123+BPR 原测试均值仍是 0.9575892678/0.9233078960；
要求仍是 0.9733929274/0.9336833737。本轮内部成绩不用于声称测试提升。
'''
    (HERE/'REPORT.md').write_text(report)
    snap = HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):
        shutil.copy2(p,snap/p.name)
    state = dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        current_turn='progress: 24 observed-entry logistic matrix fits, robust/graph ablations and fixed-blend evaluation',
        best_blend=best,promote_to_outer_validation=promote,no_lncrna=True,
        new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Seek a substantively different representation or training hypothesis; matched sparse-error and graph-smoothing variants did not help. Do not repeat these grids or evaluate original test without a frozen promising candidate.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p = HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md'
    progress = p.read_text().split('最新完成')[0]
    progress += f'''最新完成 [24 次观测项低秩矩阵补全](midrug_completion_20261006/REPORT.md)和全部重放。
最佳固定混合 {best['config']}、权重 {best['weight']}：
内部 AUC {best['mean_inner_auc']:.9f}，AP {best['mean_inner_ap']:.9f}；
相对纯 M123 和 unmasked10 的双指标最小增益 {best['selection_score']:+.9f}，推进原验证：{promote}。

此前 [15 次重建+BPR 组合](midrug_reconstruction_ranking_20261006/REPORT.md)均未超过对应重建对照。
[单独重建 unmasked10](midrug_masked_reconstruction_20261006/REPORT.md) 的内部结果为 0.953712482/0.914090581。
[近期文献笔记](midrug_literature_20261006/RESEARCH_NOTES.md)区分摘要证据和实际复现。
没有新候选进入原验证或测试。PLMF 的 10 个 checkpoint、冻结输入和发布源码哈希复核通过。
'''
    p.write_text(progress)
    print(json.dumps(dict(best_blend=best,promote=promote,converged=converged,new_fits=24),indent=2))


if __name__ == '__main__':
    main()
