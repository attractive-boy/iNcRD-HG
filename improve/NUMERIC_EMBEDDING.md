# 图特征的数值嵌入对照

> 后续已完成每组 12 配置、114 次训练的调参验证。LncDrug AP 的损失有所收窄，但仍没有双任务共同提升；见 [调参结果与机制切换](NUMERIC_EMBEDDING_TUNING.md)。以下保留第一轮结果。

2026-09-12。在现有 [TabM](GRAPH_TABM.md) 的 36 维图结构特征上，检验逐特征数值嵌入能否提高关联排序。
完整数值与预设判定保存到 [实验报告](results/numeric_embedding_20260912/REPORT.md)。

**结果：MiDrug 出现全局指标的小幅提升，尚未实现双任务共同提升。** 分位数 PLE 在 MiDrug 的 AUC、AP 均为 3/3 种子正向；相对普通数值嵌入对照，这两个指标也都是 3/3 正向。LncDrug 的 AUC 和按药物宏 AP 提高，但全局 AP 明显下降。

| 任务 | 原 TabM AUC → 分位数 PLE | 原 TabM AP → 分位数 PLE | 原 TabM 宏 AP → 分位数 PLE |
|---|---|---|---|
| LncDrug | 0.982052 → 0.983805 | 0.966260 → 0.948366 | 0.825590 → 0.909883 |
| MiDrug | 0.958717 → 0.959045 | 0.921461 → 0.922662 | 0.718469 → 0.709906 |

MiDrug 全局 AP 增加 **0.120 个百分点**，但宏 AP 减少 0.856 个百分点；分位数 PLE 因此保留为面向全局排序的候选。LncDrug 全局 AP 减少 1.789 个百分点，不能据 AUC 或宏 AP 单独宣称全面改善。树分段 PLE 在 MiDrug 的全局指标下降，在 LncDrug 同样是 AUC/宏 AP 上升而全局 AP 下降。

13 项检查、24 个短流程检查点和 56 个正式检查点重放通过。四份特征缓存与原实验完全一致，14 个原 TabM 对照预测的最大误差为 0。以上是同一历史验证划分的三个初始化均值，不是独立测试结果。

## 改变的机制

原模型直接把经过分位数变换的标量送入 TabM。本轮增加逐特征编码，让模型在不同数值区间学习不同的响应，再交给相同的 TabM 组合各特征。

| 组 | 输入编码 |
|---|---|
| `tabm` | 原标量输入，完全复现现有对照 |
| `linear` | 每个标量独立仿射映射到 16 维，再接 ReLU |
| `ple_quantile` | 在拟合数值分位数上划分最多 16 段，编码后独立投影到 16 维，再接 ReLU |
| `ple_tree` | 用拟合标签训练单变量分类树确定最多 16 段，后续与分位数 PLE 相同 |

PLE（piecewise-linear encoding）每个区间的坐标为 `clip((x - left) / width, 0, 1)`。
重复分位点合并，常量特征输出零基函数，不足 16 段时用零填充，拟合范围外截断。每个特征独立学习投影，后续网络宽度 128、深度 3、16 个内部成员。

PLE 来自 Gorishniy 等的 *On Embeddings for Numerical Features in Tabular Deep Learning*（NeurIPS 2022）。这里是兼容现有环境的实现与任务适配，不是新的通用算法。`linear` 控制编码输出维度和后续网络容量，但 PLE 额外包含基函数投影参数。

## 相同条件

四组共享五折生成的图特征、分位数变换、BCE、AdamW、dropout 0.1 和 weight decay 0.0001。
训练样本的整折目标正边在生成图特征前删除；所有分段点仅在拟合行上训练。树分段最小叶样本数为 16；验证标签不参与分段。

每组用 seed 666 搜索学习率 0.001/0.003，按内部最近五轮平均 `(AUC+AP)/2` 选择配置和实际检查点；补充 seed 0、1 后，取三个所选轮数的中位数，在完整拟合集重训三个种子。最大轮数沿用 LncDrug 150、MiDrug 450。正式实验共 56 次训练。

两种 PLE 分别按预设条件判定：两个任务的 AUC/AP 均值都超过原 TabM，且每项至少 2/3 种子为正。各组完整报告，外层结果不用于改选分段方式或参数。

## 复现

在仓库根目录，使用已有 `ncRNAdrug` 环境：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
OUT=improve/results/numeric_embedding_repeat

$PY -m unittest improve.test_numeric_embedding improve.test_graph_tabm -v || exit 1
$PY -m improve.numeric_embedding prepare --out-dir "$OUT" || exit 1
for STAGE in screen confirm outer; do
  $PY -m improve.numeric_embedding run --out-dir "$OUT" --stage "$STAGE" --task LncDrug --device cuda:0 &
  LNC_JOB=$!
  $PY -m improve.numeric_embedding run --out-dir "$OUT" --stage "$STAGE" --task MiDrug --device cuda:1 &
  MI_JOB=$!
  wait "$LNC_JOB" || exit 1
  wait "$MI_JOB" || exit 1
  if [ "$STAGE" != outer ]; then
    $PY -m improve.numeric_embedding select --out-dir "$OUT" --stage "$STAGE" || exit 1
  fi
done
$PY -m improve.numeric_embedding report --out-dir "$OUT" --device cuda:0 || exit 1
$PY -m improve.analyze_numeric_embedding --out-dir "$OUT"
```

短流程在另一个新目录对 `prepare` 加 `--smoke`，只用于核验运行。正式训练源码、输入引用和配置经过哈希冻结；已有完成结果先校验再跳过。报告重放所有检查点，独立分析进一步检查四份特征缓存和十四个标量 TabM 对照与原实验的一致性。

这是反复使用的历史验证划分上的开发实验，三个初始化不能替代新划分。AP 使用 `average_precision_score`；按药物宏 AP 仅纳入同时包含两类标签的药物。本轮不载入测试集。
