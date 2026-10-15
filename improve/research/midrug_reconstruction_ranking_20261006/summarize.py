"""Summarize the preregistered paired comparisons after checkpoint replay."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from common import HERE, RELEASE, sha256


def main():
    plan = json.loads((HERE/'results/plan.json').read_text())
    audit = json.loads((HERE/'verification.json').read_text())
    assert audit['status'] == 'passed' and len(audit['records']) == 24
    assert sum(not r['reused_control'] for r in audit['records']) == 15
    for p, h in plan['source_hashes'].items():
        assert sha256(p) == h
    runs = {name: [json.loads(Path(cfg[str(f)]['path']).read_text()) for f in range(3)]
            for name, cfg in plan['controls'].items()}
    for name in plan['configs']:
        runs[name] = [json.loads((HERE/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
    rows = []
    for name, records in runs.items():
        comparisons = ['control']
        if name.startswith('rec'):
            rec, bpr = name.split('_')
            comparisons += [bpr, 'unmasked'+rec[3:]]
        gains = {}
        for comparator in comparisons:
            gains[comparator] = {}
            for metric in ['auc', 'ap']:
                delta = [r['inner_'+metric]-b['inner_'+metric] for r, b in zip(records, runs[comparator])]
                gains[comparator][metric] = {'fold_deltas': delta, 'mean_delta': float(np.mean(delta))}
        row = dict(config=name, mean_inner_auc=float(np.mean([r['inner_auc'] for r in records])),
                   mean_inner_ap=float(np.mean([r['inner_ap'] for r in records])), comparisons=gains)
        if name.startswith('rec'):
            row['selection_score'] = min(gains[c][m]['mean_delta'] for c in comparisons[1:] for m in ['auc', 'ap'])
        rows.append(row)
    best = max((r for r in rows if 'selection_score' in r), key=lambda r: r['selection_score'])
    promote = best['selection_score'] >= plan['promotion_minimum_both_gains_over_each_control']
    selection = dict(status='frozen_inner_selection', rows=rows, best_combination=best,
                     promote_to_outer_validation=promote, no_lncrna=True,
                     outer_validation_evaluated=False, test_evaluated=False,
                     plan_sha256=sha256(HERE/'results/plan.json'), verification_sha256=sha256(HERE/'verification.json'),
                     result_hashes={str(p.relative_to(HERE)): sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))})
    (HERE/'selection.json').write_text(json.dumps(selection, indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['config','fold','inner_auc','inner_ap','best_epoch'])
        w.writeheader()
        for name, records in runs.items():
            for fold, r in enumerate(records):
                w.writerow(dict(config=name, fold=fold, **{k:r[k] for k in ['inner_auc','inner_ap','best_epoch']}))
    original = HERE.parent/'midrug_internal_20261006/plmf_protection.json'
    protected = json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():
        assert sha256(RELEASE/p) == h
    frozen = RELEASE/'data/frozen'
    manifest = json.loads((frozen/'manifest.json').read_text())
    refs = HERE.parent/'baselines_m123_20261004/results'
    protection = []
    for r in protected['records']:
        ds, fold = r['dataset'], r['fold']
        assert sha256(refs/f'incrd_m123_{ds}_f{fold}_s42.pt') == r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{fold}.npz') == r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt') == manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds, fold=fold, checkpoint_sha256=r['checkpoint_sha256'],
                               split_sha256=r['split_sha256'], graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed', records=protection,
        prior_full_replay_sha256=sha256(original), source_unchanged=True), indent=2))
    table = '\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | "
        f"{r['comparisons']['control']['auc']['mean_delta']:+.9f} | {r['comparisons']['control']['ap']['mean_delta']:+.9f} |" for r in rows)
    paired = '\n'.join(f"| {r['config']} | {c} | {g['auc']['mean_delta']:+.9f} | {g['ap']['mean_delta']:+.9f} |"
        for r in rows if 'selection_score' in r for c,g in r['comparisons'].items() if c!='control')
    report = '''# 重建与同药物排序的互补性实验

**双指标绝对 +0.02 尚未达到。本轮不推进新候选到原验证或测试集。**
完成 15 次新增训练和 9 个复用对照的 checkpoint 重放，全部通过。
仅原训练集内部三折、seed42、miRNA/drug 两种节点。

| 配置 | 内部 AUC | 内部 AP | 相对 BCE ΔAUC | 相对 BCE ΔAP |
|---|---:|---:|---:|---:|
'''+table+'''

| 组合 | 对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
'''+paired+f'''

预声明选择分数为相对匹配 BPR 单独和重建单独对照的四项平均增益的最小值。
最佳组合 {best['config']} 的分数为 {best['selection_score']:+.9f}。
所有组合均低于相应重建单独对照，因此没有互补性证据，未达到双指标 +0.002 内部门槛。
内部最佳仍是上一轮 unmasked10（0.953712482/0.914090581）。
BPR 单独在这套较小训练子集上没有稳定提高两项指标；这不推翻之前完整训练集的结果。

核验包含预测/指标、配对 ID/标签/索引、选轮、精确 float32 损失组合、权重爬升、
同药物训练负例采样，每个新模型 6 个正例的 M1/M2/M3/最终 logit 显式删边检查。
组合去除辅助重建解码器后关联预测数值等价；没有 lncRNA 参数、特征或监督。
PLMF 的 10 个受保护 checkpoint、图、划分及发布源码哈希与此前完整重放一致。
未修改发布实现。

既有纯 miRNA M123+BPR 测试均值仍为 0.9575892678/0.9233078960，
最终测试要求仍为 0.9733929274/0.9336833737；内部指标不能当作测试提升。
下一步检查原关联矩阵的稳健低秩建模，与已做过的双线性分支区别，具体设计需先固定。
'''
    (HERE/'REPORT.md').write_text(report)
    snap = HERE/'source_snapshot'
    snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):
        shutil.copy2(p, snap/p.name)
    state = dict(goal_status='active', completion_proven=False, stage='completed_and_audited',
        current_turn='progress: 15 new fits and 24 checkpoint replays; no complementarity over reconstruction-only',
        best_combination=best, promote_to_outer_validation=promote, no_lncrna=True,
        new_candidate_test_evaluations=0, processes_expected_live=[],
        next_action='Assess robust masked matrix completion from original miRNA/drug labels and similarities; avoid repeating neural bilinear fusion.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state, indent=2))
    progress = (HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md').read_text()
    progress = progress[:progress.index('最新完成')]+'''最新完成 [重建与 BPR 的 15 次互补性训练](midrug_reconstruction_ranking_20261006/REPORT.md)，
15 个新模型和 9 个复用对照均通过重放。组合均未超过匹配的重建单独对照，不推进。
内部最佳仍为 [unmasked10](midrug_masked_reconstruction_20261006/REPORT.md)：
AUC 0.953712482、AP 0.914090581，相对纯 M123 +0.001112687/+0.002265472。

没有新候选进入原验证或测试。PLMF 的 10 个 checkpoint、冻结输入和原源码哈希复核通过。
此前 [文献检索](midrug_literature_20261006/RESEARCH_NOTES.md)、
[OCN](midrug_ocn_20261006/REPORT.md)和 [嵌套折外实验](midrug_crossfit_20261006/REPORT.md)保留审计。
下一步查阅近期稳健低秩 miRNA–药物方法，先核对机制与已测方法的差异。
'''
    (HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md').write_text(progress)
    print(json.dumps(dict(best_combination=best, promote=promote, protected=len(protection)), indent=2))


if __name__ == '__main__':
    main()
