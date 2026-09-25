# TabICLv2 任务内适配

2026-09-12。继概率/logit 融合未超过单独 TabICLv2 后，本轮检验用当前拟合数据更新官方预训练参数。正式实验与核验已经完成，**没有继续提高双任务 AUC/AP**，该轮保留原始 TabICLv2。后续[支持图与上下文匹配](TABICL_GRAPH_CONTEXT.md)已取得小幅双任务全局提升。

| 任务 | 方案 | 外层 AUC | 外层 AP | 按药物宏 AP |
|---|---|---:|---:|---:|
| LncDrug | 原 TabICLv2 | 0.985403 | 0.968115 | 0.878353 |
| LncDrug | 内部选定：仅预测层、8 步 | 0.985399 | 0.967889 | 0.882745 |
| MiDrug | 原 TabICLv2 | 0.959524 | 0.924198 | 0.718572 |
| MiDrug | 内部选定：保留原模型、0 步 | 0.959524 | 0.924198 | 0.718572 |

数字是同一历史外层验证、seed 666/0/1 的均值。LncDrug 内部选择时仅有很小的正增量，外层 AP 反而下降 0.000226；宏 AP 的提高不足以满足全局 AUC/AP 的目标。MiDrug 的各适配候选未超过内部原模型，按协议保留零步。

## 机制与选择边界

比较冻结列/行编码器、仅训练上下文预测层（`icl`），以及更新全部参数（`all`）。输入仍为原 36 维五折图特征，LncDrug 使用已选定的分位数视图，MiDrug 使用原始视图。官方元批次构造器从拟合行生成最多 2,048 行的片段，再按约 80%/20% 分为上下文和伪查询。只有上下文标签进入模型，伪查询标签仅用于交叉熵。

固定 AdamW、学习率 1e-5、权重衰减 0.01、梯度范数裁剪 1、FP32、两个训练预处理成员；48 步预热/余弦日程，在 8/24/48 步保存。所有正式评分均使用 32 个推理成员。每种方式运行三个适配种子，以平均内部 `(AUC+AP)/2` 选择方式及步数，原模型零步也参与选择。两个任务的选择冻结后，才打开本轮外层特征；外层从官方权重重新适配选定步数，保留原 48 步日程前缀，更新期间不评价外层。

本轮直接使用已安装 TabICL 的官方 `_finetune/data.py` 与分类损失形式，固定步数循环、调度和核验由本地实现。高层官方训练器需要不同 sklearn 接口和可选依赖，因此没有直接调用它，也没有改动原环境。

## 核验与限制

11 项单元检查通过。工程短流程的 10 个更新状态全部重载，分数重放误差为零，并检查查询顺序及分批不变性。正式完成 12 条内部训练轨迹、3 条外层轨迹，共 39 个实际适配状态；全部保存权重、重载并重放，分数误差为零。中途快照不是独立训练重复。冻结编码器保持原参数哈希，内部选择重新计算一致，标签与配对逐行对齐。保存时使用 `save_model_weights=True`，适配状态不依赖被覆盖的官方权重。

完整证据：[报告](results/tabicl_adaptation_20260912/REPORT.md)、[核验](results/tabicl_adaptation_20260912/VALIDATION.json)、[预设判定](results/tabicl_adaptation_20260912/OUTCOME.json)、[逐种子差值](results/tabicl_adaptation_20260912/paired_deltas.csv)。来源与适配范围见[研究记录](research/tabicl_adaptation_20260912/RESEARCH.md)。

三个种子共享官方检查点及同一数据划分，只反映适配/推理随机性。历史验证已多次用于开发；没有读取测试数组，不构成独立泛化或显著性证据。当前结果也不能证明所有适配方式均无效。

## 复现

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
TABICL_PY=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
ADAPT_OUT=improve/results/tabicl_adaptation_repeat
"$TABICL_PY" -m unittest improve.test_tabicl_adaptation -v
"$TABICL_PY" -m improve.tabicl_adaptation prepare --out-dir "$ADAPT_OUT"
"$TABICL_PY" -m improve.tabicl_adaptation pipeline --out-dir "$ADAPT_OUT" --devices cuda:0,cuda:1
```

使用新目录；工程短流程在 prepare 时增加 `--smoke`。正式源码及协议已冻结。
