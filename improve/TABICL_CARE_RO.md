# CARE-RO：支持比例与排序保持的共享 TabICL 适配器

2026-09-13。**正式比较完成，12 个候选均未通过内部共同门槛，选择 `zero`，没有进入该轮外层。** 三种子 MiDrug 平均 AP 在全部候选上均下降；LncDrug 的平均 AP 虽多为小幅提高，仍未超过原始 TabICLv2 参考。21 个保存上下文状态全部重放，预测最大误差为 0，查询独立性检查通过。见 [完整报告](results/tabicl_care_rank_20260913/REPORT_inner.md) 和 [冻结判定](results/tabicl_care_rank_20260913/SELECTION.json)。

本方案接续 [CARE 两轮实验](TABICL_CARE.md)。此前选定适配器在外层提升 MiDrug，却降低 LncDrug AUC，因此保留失败结果，重新检查训练和推理之间的上下文规模变化。支持比例归一化和教师排序保持已实现，但当前结果不支持将它们作为有效替代模型。

CARE-RO 仍在冻结 TabICLv2 的最后一层 ICL 注意力中工作，共 234 个可学习参数；两个任务使用同一组权重，没有任务 ID、RNA 类型或 LncDrug 恒等分支。此前 RNA 局部证据、药物原子环境、36 维图特征和图上下文组合保持其原有定义。

**支持比例控制。** 原控制器输入查询在支持集中共享 RNA、共享药物的原始计数。训练 episode 的支持集只有 LncDrug 304 / MiDrug 约 777 行，而外层完整上下文为 3,800 / 9,712 行，原始计数混合了关系密度和上下文大小。当前输入改为

\[
\widetilde n_r=512\,n_r/|S|,\qquad
\widetilde n_d=512\,n_d/|S|,
\]

再分别使用 `n/(n+4)`、`n/(n+16)` 的有界变换。其余证据覆盖率、匹配分散度、原分类间隔及不确定度仍作为控制器输入。重复同一支持总体时，新比例统计保持不变；这不意味着基础 TabICL 的预测也对上下文规模不变。

在不使用查询标签的输入诊断中，LncDrug 的 RNA 计数输入，训练/外层均值从原先 0.043/0.379 变为 0.063/0.086；药物侧从 0.195/0.582 变为 0.251/0.258。该结果证明输入分布偏移减小，**不能单独证明原退化由此造成或新指标一定改善**。完整统计见 [输入诊断](results/tabicl_care_rank_20260913/CONTEXT_RATE_AUDIT.json)。

**教师排序保持。** 对训练伪查询的正负案例，令原模型 logit 差为 `Δ₀`，新模型对应差为 `Δ`。只保护原模型正确排列且较接近的案例，即 `0 < Δ₀ < 2`：

\[
L_{\mathrm{order}}=
\operatorname{mean}_{0<\Delta_0<2}
\frac{\max(0,0.9\Delta_0-\Delta)}{\max(\Delta_0,0.1)}.
\]

已经排错的训练案例不受这项保持约束。每个优化步同时累计两个任务的梯度，再更新共享适配器。两任务仍使用交叉熵、权重 0.25 的排序损失和 KL 约束；LncDrug 的 KL 权重为 MiDrug 的四倍，另外使用权重 20 的上述教师排序损失。它是训练惩罚，不是验证分数或未来分布不退化的数学保证。

**固定预算与对照。** 重用原先已核对的 20 个拟合数据 episode，每个优化步每任务至多 128 个伪查询。KL 系数 1/10、优化步数 50/200、注意力强度 1/2/4，共 12 个主候选；置信度指数固定为 0.5，最大注意力 logit 改动为 `2α`。两项同输入、同参数量、同训练预算对照分别去掉支持比例归一化和教师排序保持。它们与主模型共用一次未修改的 TabICL 前层计算。

**判定。** 沿用每任务 seed 666/0/1、每种子 32 个成员。LncDrug 平均 AUC/AP 必须分别不低于原始 TabICLv2 和图上下文参考；MiDrug 平均 AUC/AP 必须超过图上下文，且每项至少两个种子正向。符合条件时优先较小修正、更强 KL 和更少优化步数。按药物宏 AP 单列报告。没有合格候选则停止，不评价该轮外层。

内部选择通过后冻结共享参数，仅扩充原 TabICL 支持上下文，评价选定主模型和两个机制对照；不根据该轮外层改选配置。此前的外层退化已经被看到，本轮属于对**历史开发验证的继续研究**，不能把它称为独立确认。原始测试数组保持未使用。

实现：[实验入口](tabicl_care_rank.py)、[比例控制与排序约束](tabicl_care_rank_modules.py)、[机制检查](test_tabicl_care_rank.py)。结果保存于 [实验目录](results/tabicl_care_rank_20260913/)。

在 `iNcRD-HG` 目录复现，使用新的输出目录：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
CARE_RO_PY=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
CARE_RO_OUT=improve/results/tabicl_care_rank_repeat
"$CARE_RO_PY" -m unittest improve.test_tabicl_care_rank -v
"$CARE_RO_PY" -m improve.tabicl_care_rank prepare --out-dir "$CARE_RO_OUT"
"$CARE_RO_PY" -m improve.tabicl_care_rank train --out-dir "$CARE_RO_OUT" --device cuda:0
for command in run verify audit; do
    for task in LncDrug MiDrug; do
        "$CARE_RO_PY" -m improve.tabicl_care_rank "$command" --out-dir "$CARE_RO_OUT" --task "$task" --device cuda:0
    done
done
"$CARE_RO_PY" -m improve.tabicl_care_rank select --out-dir "$CARE_RO_OUT"
"$CARE_RO_PY" -m improve.tabicl_care_rank report --out-dir "$CARE_RO_OUT"
```

只有内部通过后才执行相同 `run/verify/audit/report --phase outer`。同任务不同 `--seed` 的推理作业也可分配到两张 GPU；共享训练必须在所有源训练 episode 就绪后执行。
