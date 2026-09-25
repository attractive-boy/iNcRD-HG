# TabICLv2 + TabM 分数融合

2026-09-12。两种模型可以直接做预测分数融合，但本轮内部选出的融合在历史外层验证上未超过单独 TabICLv2。当前继续保留 TabICLv2 作为更强的方案。完整对照、逐种子结果及核验见 [实验报告](results/tabicl_tabm_fusion_20260912/REPORT.md)。

## 结果

以下为 seed 666/0/1 的均值；TabICLv2 使用前一轮内部选出的 LncDrug c3、MiDrug c1，TabM 使用原有 LncDrug c1、MiDrug c0。

| 任务 | 方案 | AUC | AP | 按药物宏 AP |
|---|---|---:|---:|---:|
| LncDrug | TabICLv2 | **0.985403** | **0.968115** | **0.878353** |
| LncDrug | 内部选定融合 | 0.984892 | 0.967424 | 0.858363 |
| MiDrug | TabICLv2 | **0.959524** | **0.924198** | 0.718572 |
| MiDrug | 内部选定融合 | 0.959423 | 0.923195 | **0.722057** |

融合在两个任务上均超过原 TabM，但相对 TabICLv2，LncDrug 的平均 AUC/AP 下降 0.000512/0.000691，MiDrug 下降 0.000101/0.001003。LncDrug 宏 AP 下降 0.019990，MiDrug 宏 AP 提高 0.003485。

预设的概率 50:50 对照同样未达到两任务 AUC/AP 一致提高：LncDrug 为 0.984092/0.966528，MiDrug 为 0.959637/0.923951；MiDrug 只有 AUC 略高于 TabICLv2。判定详见 [OUTCOME.json](results/tabicl_tabm_fusion_20260912/OUTCOME.json)。这些结果不支持把当前预测器替换为本轮融合。

## 融合方式与选参

概率融合：`p = w * p_TabICL + (1-w) * p_TabM`。

Logit 融合：`p = sigmoid(w * logit(p_TabICL) + (1-w) * logit(p_TabM))`，计算 logit 前将概率截断到 `[1e-7, 1-1e-7]`。两个单模型端点直接返回原预测。融合按单个配对计算，改变查询顺序或分批方式不改变结果。

权重步长固定为 0.05。概率方式有 21 个权重，logit 方式有 19 个内部权重，共 40 个不重复配置，包含两个单模型端点。每个任务按三个种子的平均 `(AUC+AP)/2` 选择一套配置；精确同分时优先单模型端点，其次概率融合，再优先较大 TabICLv2 权重。

| 任务 | 内部选定方式 | TabICLv2 / TabM 权重 | 内部 AUC | 内部 AP |
|---|---|---|---:|---:|
| LncDrug | 概率融合 | 0.80 / 0.20 | 0.992786 | 0.985020 |
| MiDrug | Logit 融合 | 0.40 / 0.60 | 0.962803 | 0.928851 |

上述融合在内部验证中均提高了两个单模型的 AUC/AP，但收益未在当前外层验证保留。选择记录在本轮解读外层预测数组前冻结，外层只评估单模型、固定概率 50:50 和内部选定融合。预设成功条件为两个任务都超过两个单模型，平均 AUC/AP 均提高，且每项至少 2/3 个对应种子标签的差值为正。

## 核验与适用范围

复用 24 份已经由前序实验重放的单模型预测：内部与外层各 12 份。本轮没有模型训练或 GPU 推理。逐文件校验缓存、结果、模型选择及源码哈希；每份分数均与同一特征缓存的标签和 RNA–drug 配对逐行对齐，并检查拟合/验证配对无重叠。

10 项检查通过，覆盖同标签下的配对错序、重复配对、外层载入前的选择检查、三种子选参、无效概率、logit 端点、查询分批不变性以及冻结文件变更检测。正式核验重算了全部 240 次内部候选评分，重现相同选择；6 份外层结果文件的 24 列分数（含单模型对照）重算差异均为 0。详见 [VALIDATION.json](results/tabicl_tabm_fusion_20260912/VALIDATION.json)。

历史内部与外层验证已多次参与开发，外层单模型成绩在本轮之前已知。TabM 的内部检查点和两模型配置也已使用内部标签选择，所以这些内部预测不构成新的交叉拟合层。本轮没有读取测试数组，不作独立测试或统计显著性结论。不同模型的同号种子仅用于对齐记账，三个 TabICL 种子共享同一个预训练检查点。

## 复现

实现为 [tabicl_tabm_fusion.py](tabicl_tabm_fusion.py)，使用上一轮的隔离环境与已验证预测缓存。在仓库根目录运行：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
FUSION_PY=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
FUSION_OUT=improve/results/tabicl_tabm_fusion_repeat

"$FUSION_PY" -m unittest improve.test_tabicl_tabm_fusion -v
"$FUSION_PY" -m improve.tabicl_tabm_fusion prepare --out-dir "$FUSION_OUT"
"$FUSION_PY" -m improve.tabicl_tabm_fusion select --out-dir "$FUSION_OUT"
"$FUSION_PY" -m improve.tabicl_tabm_fusion evaluate --out-dir "$FUSION_OUT"
"$FUSION_PY" -m improve.tabicl_tabm_fusion verify --out-dir "$FUSION_OUT"
```

代码、依赖版本、原预测或候选空间变更需要使用新的实验目录。正式结果目录为 `results/tabicl_tabm_fusion_20260912/`，包括完整候选表、冻结权重、逐种子外层分数、汇总指标和核验记录。
