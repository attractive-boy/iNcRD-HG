"""Archive nested experiments with both matched and optimized comparator metrics."""
import json,csv,shutil
import numpy as np
from common import HERE,CLEAN,PREVIOUS,module,sha256
from residual import CONFIGS

def main():
    ba=json.loads((HERE/'base_verification.json').read_text());ra=json.loads((HERE/'verification.json').read_text())
    assert ba['status']=='passed' and len(ba['records'])==18 and ra['status']=='passed' and len(ra['records'])==9
    plan=json.loads((HERE/'results/plan.json').read_text());rows=[];flat=[]
    optimized=[json.loads((CLEAN/'results'/f'control_f{f}_s42.json').read_text()) for f in range(3)]
    for name in CONFIGS:
        runs=[json.loads((HERE/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
        da=[r['inner_auc']-r['control_auc'] for r in runs];dp=[r['inner_ap']-r['control_ap'] for r in runs]
        oa=[r['inner_auc']-b['inner_auc'] for r,b in zip(runs,optimized)];op=[r['inner_ap']-b['inner_ap'] for r,b in zip(runs,optimized)]
        rows.append(dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in runs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in runs])),
            mean_auc_gain=float(np.mean(da)),mean_ap_gain=float(np.mean(dp)),mean_auc_gain_over_optimized=float(np.mean(oa)),mean_ap_gain_over_optimized=float(np.mean(op)),
            fold_auc_gains=da,fold_ap_gains=dp,fold_auc_gains_over_optimized=oa,fold_ap_gains_over_optimized=op,selection_score=min(float(np.mean(da)),float(np.mean(dp)))))
        flat += [{k:r[k] for k in ['config','fold','inner_auc','inner_ap','control_auc','control_ap','best_epoch']} for r in runs]
    best=max(rows,key=lambda x:x['selection_score']);promote=(best['selection_score']>=plan['promotion_minimum_both_gains'] and
        min(best['mean_auc_gain_over_optimized'],best['mean_ap_gain_over_optimized'])>=plan['also_require_both_gains_over_previous_optimized_control'])
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,
        plan_sha256=sha256(HERE/'results/plan.json'),result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))})
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    prior=HERE.parent/'midrug_ocn_20261006/plmf_hash_recheck.json';protection=json.loads(prior.read_text());assert protection['status']=='passed'
    frozen=module.RELEASE/'data/frozen';refs=HERE.parent/'baselines_m123_20261004/results'
    for r in protection['records']:
        ds,f=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256'] and sha256(frozen/f'{ds}_graph.pt')==r['graph_sha256']
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json'
    for p,h in json.loads(original.read_text())['source_hashes'].items():assert sha256(module.RELEASE/p)==h
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=10,prior_hash_recheck_sha256=sha256(prior),prior_full_replay_sha256=sha256(original),source_unchanged=True),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['mean_auc_gain']:+.9f} | {r['mean_ap_gain']:+.9f} | {r['mean_auc_gain_over_optimized']:+.9f} | {r['mean_ap_gain_over_optimized']:+.9f} |" for r in rows)
    report='''# 训练内部嵌套折外残差：结果

**最终双指标 +0.02 尚未达到。** 完成 18 个固定 300 轮的纯 miRNA M123 基模型，
以及 3 配置 × 3 外层内部折共 9 个子图模型。全部训练/选择都在原训练集内部。

| 配置 | 内部 AUC | 内部 AP | 对固定 300 轮 ΔAUC | 对固定 300 轮 ΔAP | 对既有优化对照 ΔAUC | 对既有优化对照 ΔAP |
|---|---:|---:|---:|---:|---:|---:|
'''+table+'''

固定 300 轮对照均值为 AUC 0.9515018548 / AP 0.9092461719；
既有优化过轮次的纯 miRNA 对照为 0.9525997951 / 0.9118251091。
对照强度不同，不能只选择较弱对照的较大增益作为最终结论。
'''+('达到内部推进规则，需冻结配置进入原验证确认。' if promote else '未同时达到预声明的内部双指标推进门槛，不进入原验证或测试。')+'''

当前最好的 independent25 是独立子图分类器的固定权重组合，
它优于所测的两个折外 offset 残差配置。因此本轮不支持“折外残差优于独立分支”的结论。
训练/推理图密度随嵌套方案变化，不能将结果解释为仅改变误差目标的因果效应。
本轮只排查这几个明确配置，没有证明更复杂的方法均无效或数据已达到上限。

## 关键核验

- 18 个基模型均固定 300 轮，没有使用被预测折标签选轮。
- 每个外层训练样本恰有一次 OOF 预测；其标签与记录不出现在对应基模型的训练中。
- 每个嵌套折独立生成子图缓存，非复用外层图；18 个模型的 72 个子图样本重建通过。
- 18 个基模型和 9 个子图 checkpoint 全部重放；保存的预测、AUC/AP、选轮核对通过。
- 每个子图 checkpoint 另验证 10 个来自五个嵌套折的输入及输出精确一致。
- 三个完整基模型缓存曾修正一个未使用的 prediction_rows 元数据字段；
  原训练代码/plan、修正前后哈希均保留，训练输入、全局索引、logit 与 checkpoint 未变。
- PLMF 的 10 个保护 checkpoint、冻结图/划分及源码哈希均与先前完整重推理一致。

方法、复现与限制见 [README](README.md)。原始日志、计划、验证结果和源码快照随目录保存。
该实验没有使用 lncRNA、外部数据或原验证/测试标签。

本次用户建议的近年文献检索见 [文献笔记](../midrug_literature_20261006/RESEARCH_NOTES.md)。
OCN 思路的 12 次消融已完成且未达到推进门槛。
后续可检验 RGDAGMAE 摘要启发的原始特征掩码重建正则，详见 NEXT_EXPERIMENT.md；
这仍是待验证适配，未声称复现该论文或取得新测试成绩。
'''
    (HERE/'REPORT.md').write_text(report)
    snapshot=HERE/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    state=dict(goal_status='active',completion_proven=False,current_turn='progress: current-paper review, 12 OCN ablations, 18 nested base fits, nine subgraph fits, complete replays',
        inner_selection=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Original-miRNA/drug feature-masking reconstruction regularization; use existing attributes only, preserve M1/M2/M3, fixed training-internal comparison; see NEXT_EXPERIMENT.md')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    progress='''# MiDrug 专用 miRNA 模型：当前进度

**只使用 miRNA 与药物，双指标绝对 +0.02 尚未完成。**

| 项目 | AUC | AP |
|---|---:|---:|
| 原始 iNcRD-HG 测试三种子均值 | 0.9533929274 | 0.9136833737 |
| 必须达到 | 0.9733929274 | 0.9336833737 |
| 既有纯 miRNA M123+BPR 测试均值 | 0.9575892678 | 0.9233078960 |
| 需保住 MDR | 0.9483092901 | 0.9436701562 |
| 需保住 MDS | 0.9508978837 | 0.9490119744 |

用户建议查阅近年论文后，完成了 [文献检索](midrug_literature_20261006/RESEARCH_NOTES.md)、
[12 个 OCN 消融](midrug_ocn_20261006/REPORT.md)和
[18 个嵌套基模型＋9 个子图模型](midrug_crossfit_20261006/REPORT.md)。全部 checkpoint 已重放。

OCN 最佳内部增益约 +0.000031 / +0.000557。
最新 independent25 对同为 300 轮的主干增益约 +0.001728 / +0.003186；
对既有优化过轮次的主干，增益只有 +0.000630 / +0.000607，未达到推进门槛。
没有新候选进入原验证或测试，不能把内部成绩作为最终提升。

PLMF 的 10 个 checkpoint、原始图/划分及源码哈希复核通过。
历史含 lncRNA 的结果继续排除。下一步拟检验原特征掩码重建正则；尚未执行。
'''
    (HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md').write_text(progress)
    print(json.dumps(dict(best=best,promote=promote,new_base_fits=18,new_subgraph_fits=9,goal_status='active'),indent=2),flush=True)

if __name__=='__main__':main()
