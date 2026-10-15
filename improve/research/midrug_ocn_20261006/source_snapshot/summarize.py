"""Freeze inner selection, verify protected assets, and archive this research round."""
import csv,json,shutil
import numpy as np
from common import HERE,CLEAN,sha256,module
from features import CONFIGS

def main():
    verification=json.loads((HERE/'verification.json').read_text())
    assert verification['status']=='passed' and len(verification['records'])==15
    plan=json.loads((HERE/'results/plan.json').read_text());records={};flat=[]
    for name in ('control',*CONFIGS):
        root=CLEAN if name=='control' else HERE;records[name]=[]
        for fold in range(3):
            r=json.loads((root/'results'/f'{name}_f{fold}_s42.json').read_text());records[name].append(r)
            flat.append({k:r[k] for k in ('config','fold','inner_auc','inner_ap','best_epoch')})
    rows=[]
    for name,runs in records.items():
        da=[r['inner_auc']-b['inner_auc'] for r,b in zip(runs,records['control'])]
        dp=[r['inner_ap']-b['inner_ap'] for r,b in zip(runs,records['control'])]
        rows.append(dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in runs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in runs])),
            mean_auc_gain=float(np.mean(da)),mean_ap_gain=float(np.mean(dp)),fold_auc_gains=da,fold_ap_gains=dp,selection_score=min(float(np.mean(da)),float(np.mean(dp)))))
    best=max(rows,key=lambda x:x['selection_score']);promote=best['config']!='control' and best['selection_score']>=plan['promotion_minimum_both_gains']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,plan_sha256=sha256(HERE/'results/plan.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))},no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    # Recheck the full replay's artifacts; this is not a new test evaluation.
    previous=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(previous.read_text());release=module.RELEASE
    for file,h in protected['source_hashes'].items():assert sha256(release/file)==h
    frozen=release/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text())
    refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
    for r in protected['records']:
        ds,fold=r['dataset'],r['fold'];tag=f'incrd_m123_{ds}_f{fold}_s42'
        assert sha256(refs/f'{tag}.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{fold}.npz')==r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=fold,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,prior_verification_sha256=sha256(previous),scope='Previous full replay assets and frozen graph manifests unchanged'),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['mean_auc_gain']:+.9f} | {r['mean_ap_gain']:+.9f} |" for r in rows)
    report='''# OCN 思路的纯 miRNA–药物消融结果

**双指标 +0.02 目标尚未达到。** 本轮查阅近期文献后，实现了 OCN 启发的
3/5 步邻居系数分支，完成 4 配置 × 3 内部折共 12 次训练。
只加载冻结的 miRNA/drug 输入与原训练集，不使用 lncRNA 或外部数据。

| 配置 | 内部 AUC 均值 | 内部 AP 均值 | ΔAUC | ΔAP |
|---|---:|---:|---:|---:|
'''+table+'\n\n'+('达到内部推进门槛，需冻结配置后进入原验证确认。' if promote else '未达到预声明的双指标内部 +0.002 推进门槛，不进入原验证集。')+'''

该结果只评价本地适配，不能宣称推翻官方 OCN 或所有高阶邻居方法。
本地与原文在二部图、每查询正交化、路径质量归一化和残差融合上均有明确区别。
具体公式、原文证据和备选论文见 [文献笔记](../midrug_literature_20261006/RESEARCH_NOTES.md)，
实现与复现说明见 [README](README.md)。

全部 12 份新 checkpoint 和 3 份纯 miRNA M123 对照已重放；
保存的预测、指标、checkpoint 选择均核对通过。每个模型检查 6 条正例的完整
M1/M2/M3/新分支 logit 显式删边，并对 3 正/3 负独立删除监督记录后
用密集邻接矩阵幂重构系数。初始分支与原主干的最大概率差为约 4.47e-7，
任意中间节点池化与显式 PyG 删边计算一致，梯度有限且非零。

PLMF 的 10 个 checkpoint、冻结图/划分和源码与前次完整重推理结果哈希一致；
未修改发布路径。没有对新候选读取原验证或测试标签。
既有纯 miRNA M123+BPR 的测试均值仍为 0.9575892678 / 0.9233078960，
低于最终要求 0.9733929274 / 0.9336833737。

下一步保留训练内嵌套折外残差方案，避免继续只在主干训练内误差上增加分支；
其设计见 ../midrug_subgraph_20261006/NEXT_EXPERIMENT.md。
这是待检验假设，不是本轮已实现的增益。
'''
    (HERE/'REPORT.md').write_text(report)
    snapshot=HERE/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    (snapshot/'imported_clean_common.py').write_bytes((HERE.parent/'midrug_subgraph_20261006/common.py').read_bytes())
    state=dict(goal_status='active',completion_proven=False,stage='completed_12_runs_and_audit',no_lncrna=True,new_candidate_test_evaluations=0,
        literature='../midrug_literature_20261006/RESEARCH_NOTES.md',inner_selection=best,promote_to_outer_validation=promote,processes_expected_live=[],
        next_action='Nested training-only cross-fitted residual learning, using five nested partitions within each outer training subset; base checkpoint duration fixed before any prediction-label access')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    progress='''# MiDrug 专用 miRNA 模型：当前进度

只使用 miRNA 与药物。**AUC/AP 同时绝对提高 0.02 的目标未完成。**

| 项目 | AUC | AP |
|---|---:|---:|
| 原始 iNcRD-HG 测试三种子均值 | 0.9533929274 | 0.9136833737 |
| 必须达到 | 0.9733929274 | 0.9336833737 |
| 既有纯 miRNA M123+BPR 测试均值 | 0.9575892678 | 0.9233078960 |
| 需保住 MDR | 0.9483092901 | 0.9436701562 |
| 需保住 MDS | 0.9508978837 | 0.9490119744 |

最新工作：[近年文献检索](midrug_literature_20261006/RESEARCH_NOTES.md)和
[OCN 适配消融](midrug_ocn_20261006/REPORT.md)，完成 12 次训练及全部重放核验。
'''+f"本轮选中 {best['config']}，内部 ΔAUC={best['mean_auc_gain']:+.9f}、ΔAP={best['mean_ap_gain']:+.9f}；推进原验证={promote}。\n"+'''
这些是原训练集内部开发分数，不是新的测试成绩。原验证/测试未用于本轮选型。
历史含 lncRNA 的候选继续排除；PLMF 保护资产哈希复核通过。
后续候选为嵌套折外残差学习；当前尚未执行。
'''
    (HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md').write_text(progress)
    print(json.dumps(dict(best=best,promote=promote,goal_status='active'),indent=2),flush=True)

if __name__=='__main__':main()
