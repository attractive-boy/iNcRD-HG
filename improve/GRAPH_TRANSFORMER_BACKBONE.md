# 异构图 Transformer 骨干替换：结构增强 HGT（SA-HGT）

本轮把论文的 GeneralConv 骨干替换为**结构增强的异构图 Transformer**：关系感知的
HGT 注意力（Hu et al. 2020）+ 逐关系入度编码 + LayerNorm 残差，解码器同时接入
交叉拟合的 36 维图结构配对特征（即 TabM 所用的那套无泄漏特征）。训练、选参、
重训与外层评价完全沿用既有冻结协议。

## 结果（历史外层验证，seed 666/0/1 均值 ± 初始化标准差）

| 任务 | 架构 | AUC | AP | 按药物宏 AP |
|---|---|---:|---:|---:|
| LncDrug | base（原 GNN） | 0.980402 | 0.943892 | 0.892087 |
| LncDrug | sa-hgt | 0.980210 | **0.947943** | 0.876563 |
| LncDrug | tabm（当前推荐） | 0.982052 | 0.966260 | 0.825590 |
| MiDrug | base（原 GNN） | 0.955536 | 0.919169 | 0.716735 |
| MiDrug | sa-hgt | **0.956300** | 0.918420 | 0.702278 |
| MiDrug | tabm（当前推荐） | 0.958717 | 0.921461 | 0.718469 |

相对 base 的配对差：

| 任务 | 指标 | sa-hgt − base | 正增益种子 |
|---|---|---:|---:|
| LncDrug | auc | −0.000193 | 1/3 |
| LncDrug | ap | +0.004051 | 2/3 |
| LncDrug | macro_drug_ap | −0.015524 | 1/3 |
| MiDrug | auc | +0.000764 | 3/3 |
| MiDrug | ap | −0.000749 | 0/3 |
| MiDrug | macro_drug_ap | −0.014457 | 0/3 |

## 结论边界

- **不是干净涨点**。全局 AUC/AP 相对 base 是混合的：LncDrug 只有 AP 涨
  （+0.0040，2/3 种子）、AUC 平；MiDrug 只有 AUC 涨（+0.0008，3/3 种子）、AP
  略降。没有任何一个任务在 AUC 和 AP 上同时稳定超过 base。
- **相对当前 TabM 方案仍全面落后**于全局 AUC/AP（LncDrug AP −0.018、MiDrug
  AP −0.003）。唯一例外是 LncDrug 按药物宏 AP +0.051（3/3 种子），把 TabM 破坏的
  单药物排序部分补回（0.877 vs TabM 0.826，仍低于 base 0.892）。
- **机制解释与本项目历史一致**：可学习关系注意力的 HGT 分支贡献被门控压到约
  12%（`graph_gate` sigmoid≈0.12），说明在该协议上关系平滑带来的噪音大于信号；
  有效提升主要来自交叉拟合的结构配对特征。这与 HGT/HAN/LightGCN/超图/PENCIL
  全部负结果的既有记录一致，本轮用同一个冻结协议再次确认：**图关系机器换不来
  全局排序涨点**。
- 因此当前推荐方案仍是图结构特征 + TabM；SA-HGT 不替换 TabM。它作为
  “异构图 Transformer 骨干”的完整实现、训练与核验记录保留为对照，证明该方向在
  本数据上没有胜过表格特征方案。

## 调参能否补回小幅下降（2026-09-11 第二轮）

外层上两个小幅负项（LncDrug AUC −0.0002、MiDrug AP −0.0007）都在种子标准差
（0.0013 / 0.00035）以内。按协议不能直接对着外层调参；合法的做法是扩内层搜索、
在内层选参后重跑 confirm+outer。

结果证明**调参涨不回去**（[第二轮冻结实验](results/ragt_backbone_tuned_20260911/REPORT.md)）：

- 内层确实有提升：MiDrug 内层 AP 从 0.9202 提到 0.9245（`wd 1e-3 + gate -6`），
  LncDrug 内层 AP 到 0.9830。
- 但外层重训后收益全部蒸发：MiDrug 外层 AP 0.9185（与第一轮 0.9184 相同，仍低于
  base 0.9192），LncDrug 外层反而变差（AP 0.9426 vs 第一轮 0.9479，AUC 0.9791 vs
  0.9802）。内层选出的配置在外层分布上不迁移。

即：**两个外层负项是划分迁移/种子噪声，不是超参数窗口**。内层选参选得越好，
越容易在外层掉下去；没有任何在内层合法选择得到的配置能稳定翻正外层符号。
第一轮实验（`ragt_backbone_20260911`）仍是两轮中更好的结果。

## 结构

- 图编码器：1 层 HGTConv，逐关系 key/query/value 注意力，逐节点类型的 Lazy
  Linear 投影 + 类型嵌入 + log1p 逐关系入度编码；LayerNorm + LeakyReLU 残差。
- 解码器（`fusion='add'`）：特征分支 MLP 与图分支 MLP 相加，图分支由可学习标量
  门控（初始 sigmoid≈0.12）；两分支共享同一 BCE 目标。
- 训练：每步按查询半折移除目标与反向正边后再做图编码和入度编码；36 维结构特征
  由五折交叉拟合生成，训练行特征排除其整折目标正边；内部选参用滑窗 5 的
  mean(AUC,AP)，冻结后按三个内部选中轮数的中位数在外层完整拟合集上固定轮数
  重训三个初始化。外层验证此前已用于开发，结果是探索性证据，不是独立测试；
  本轮未读取测试集。

## 复现

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python

$PY -m unittest improve.test_ragt_backbone -v
$PY -m improve.ragt_backbone prepare --out-dir improve/results/ragt_backbone_repeat
$PY -m improve.ragt_backbone run --out-dir ... --stage screen --task LncDrug --device cuda:0
$PY -m improve.ragt_backbone run --out-dir ... --stage screen --task MiDrug --device cuda:1
$PY -m improve.ragt_backbone select --out-dir ... --stage screen
# 完成 confirm 后 select confirm，再 outer，最后 report（重放全部检查点）
```

入口：[模型与解码器](ragt_modules.py)、[训练/选参/重训/报告](ragt_backbone.py)、
[单元与泄漏检查](test_ragt_backbone.py)、[完整结果](results/ragt_backbone_20260911/REPORT.md)。