# LoMETab 网络与 per-drug 分组排序目标

2026-09-12。在 [图结构特征 + TabM](GRAPH_TABM.md) 的完全冻结管线上做骨干替换试验。
**两轮均为负结果：LoMETab（含 (rank, σ) 网格）与分组排序目标在已搜配置内都没有超过
现有 TabM。** 完整数值、逐种子差值和边界见
[第一轮 2x2 报告](results/lometab_20260912/REPORT.md) 与
[第二轮网格报告](results/lometab_grid_20260912/REPORT.md)。

## 两轮结论速览

| 轮次 | 内容 | 结论 |
|---|---|---|
| 1 (lometab_20260912) | LoMETab(r=2,σ=.1) × {BCE, BCE+分组排序} 2x2，64 检查点 | 所有新臂相对 tabm 全局 0/3 种子负向；分组排序一致伤害全局指标 |
| 2 (lometab_grid_20260912) | LoMETab 网格 rank∈{1,2,4,8} × σ∈{.01,.1,.5,1.0} × lr∈{1e-3,3e-3}，88 检查点 | 内层选出高多样性配置，把 AP 差从约 -1.0pp 收窄到约 -0.27pp，仍 0/3 种子低于 tabm |

## 动机

TabM 相对原 GNN 全局 AUC/AP 涨点，但 LncDrug 按药物宏 AP 从 0.892087 掉到 0.825590。
本轮尝试两个对症组件：

1. **LoMETab**（Choi 等，2026，arXiv:2605.14365）：TabM rank-1 BatchEnsemble 的 rank-r
   推广。每个成员权重为 `W_k = W * (1 + A_k B_k^T)`，共享 W、成员专属低秩因子 (A_k, B_k)。
   论文证明 r ≥ 2 严格扩大假设类，且 (r, σ_init) 可控成员多样性。目标是补 TabM 成员
   多样性、改善组内排序。
2. **Per-drug 分组排序损失**：对批内每个同时含正负标签的药物，对全部正负配对计算
   `softplus(logit_neg - logit_pos)` 并按成员与配对求均值；总损失为
   `BCE + rank_lambda * group`。该代理直接对准按药物宏 AP 评价。

## 结构与冻结配置

每个配对仍使用相同的 36 维图结构特征与五折生成、分位数变换。网络 width 128、depth 3、
16 成员。第一轮 LoMETab 固定 rank=2、σ_init=0.1，分组臂搜索 lr × rank_lambda ∈
{1e-3, 3e-3} × {0.5, 2.0}；第二轮把 LoMETab 的 rank/σ_init 放开为配置轴，做
rank ∈ {1,2,4,8} × σ ∈ {0.01,0.1,0.5,1.0} × lr ∈ {1e-3, 3e-3} 网格，并保留 TabM 锚点。

screen 用 seed 666 按最早最优 trailing-5 内部 (AUC+AP)/2 选参，confirm 补 seed 0/1，
outer 用三个选定轮数中位数重训 seed 666、0、1 后只评价一次。历史外层验证数组仅在外层
重训阶段打开，未读取测试集。

## 第一轮结果摘要

| 任务 | 臂 | AUC | AP | 宏 AP |
|---|---:|---:|---:|---:|
| LncDrug | tabm | 0.982052 | 0.966260 | 0.825590 |
| LncDrug | lometab | 0.978009 | 0.955966 | 0.798085 |
| LncDrug | lometab_group | 0.976270 | 0.941368 | 0.830743 |
| MiDrug | tabm | 0.958717 | 0.921461 | 0.718469 |
| MiDrug | lometab | 0.956371 | 0.917314 | 0.706263 |
| MiDrug | lometab_group | 0.949363 | 0.903748 | 0.665960 |

全局 AUC/AP 上所有新臂相对 tabm 均为 0/3 种子负向。`tabm` 臂与官方结果逐位一致，
作为管线正确性锚点。

## 第二轮网格结果摘要

内层选出的配置：LncDrug r=4 σ=1.0，MiDrug r=8 σ=0.5。

| 任务 | 臂 | 配置 | AUC | AP | 宏 AP |
|---|---:|---|---:|---:|---:|
| LncDrug | tabm | — | 0.982052 | 0.966260 | 0.825590 |
| LncDrug | lometab | r=4 σ=1.0 | 0.981087 | 0.963590 | 0.815621 |
| MiDrug | tabm | — | 0.958717 | 0.921461 | 0.718469 |
| MiDrug | lometab | r=8 σ=0.5 | 0.957598 | 0.919442 | 0.713820 |

逐种子相对 tabm 的 AUC/AP/宏 AP 差全部为 0/3 种子负向；LncDrug AP 差从第一轮
-0.010294 收窄到 -0.002670，MiDrug 从 -0.004147 收窄到 -0.002020。

## 复现

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
OUT=improve/results/lometab_repeat

$PY -m unittest improve.test_lometab -v
$PY -m improve.lometab prepare --out-dir "$OUT"
$PY -m improve.lometab run --out-dir "$OUT" --stage screen --task LncDrug --device cuda:0
$PY -m improve.lometab run --out-dir "$OUT" --stage screen --task MiDrug --device cuda:1
$PY -m improve.lometab select --out-dir "$OUT" --stage screen
$PY -m improve.lometab run --out-dir "$OUT" --stage confirm --task LncDrug --device cuda:0
$PY -m improve.lometab run --out-dir "$OUT" --stage confirm --task MiDrug --device cuda:1
$PY -m improve.lometab select --out-dir "$OUT" --stage confirm
$PY -m improve.lometab run --out-dir "$OUT" --stage outer --task LncDrug --device cuda:0
$PY -m improve.lometab run --out-dir "$OUT" --stage outer --task MiDrug --device cuda:1
$PY -m improve.lometab report --out-dir "$OUT" --device cuda:0
```

当前源码即第二轮网格协议（32 组 LoMETab + 2 组 TabM）。两个任务可在两块 GPU 上并行；
prepare 只允许新目录，修改冻结源码后必须新建实验目录。

## 结论与下一步

两轮搜索内 LoMETab 与分组排序都不涨点。剩余可选方向：

- **GNN 与 TabM 的折外堆叠（OOF stacking / 集成选择）**：把已训练的异质 GNN 与 TabM
  的折外预测作为元特征做第二层精排。两模型排序优势互补（GNN 单药排序、TabM 全局排序），
  是前期检索中成功率最高、风险最低的路线。
- **自监督图嵌入作额外特征**：在图全量节点上预训练图嵌入，追加到 36 维特征喂给 TabM。
- **PU 负采样校正**：未观测配对不是真阴性，用 nnPU 或可靠负例加权。
- LoMETab 若继续投入，需超出本网格（更高 rank、逐层 σ、更久训练）；鉴于两轮负结果
  与宏 AP 高方差，优先级低于上述路线。