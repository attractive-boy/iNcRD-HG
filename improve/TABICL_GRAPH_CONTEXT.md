# TabICLv2：支持图与上下文匹配

2026-09-12。原有[任务内适配](TABICL_ADAPTATION.md)没有继续提高双任务全局 AUC/AP。本轮保留官方权重，转而检验图支持规模与上下文的一致性。

**正式结果：内部选定方案小幅提高了两个任务的全局 AUC/AP，通过预设双任务判定。** 72 个拟合上下文状态已全部重载重放，分数误差均为 0。完整结果见[报告](results/tabicl_graph_context_20260912/REPORT.md)、[核验记录](results/tabicl_graph_context_20260912/VALIDATION.json)及[逐种子差值](results/tabicl_graph_context_20260912/paired_deltas.csv)。

| 任务 | 方案 | AUC | AP | 按药物宏 AP |
|---|---|---:|---:|---:|
| LncDrug | 原 TabICLv2 | 0.985403 | 0.968115 | 0.878353 |
| LncDrug | 内部选定图上下文方案 | **0.985742** | **0.968844** | 0.862695 |
| MiDrug | 原 TabICLv2 | 0.959524 | 0.924198 | 0.718572 |
| MiDrug | 内部选定图上下文方案 | **0.959934** | **0.925057** | 0.717259 |

以上是历史外层验证、seed 666/0/1 的均值。LncDrug 的 AUC/AP 增量为 +0.000339/+0.000729，MiDrug 为 +0.000410/+0.000859；两任务两指标均为 3/3 个种子标签正向。LncDrug 选定 `query_bag_050`，MiDrug 选定 `fold_match_050`，即各自的新组成项与原模型各占 50%。图视图的完整权重结果也已预先规定报告，但没有用外层成绩改选权重。

**收益集中在全局排序。** LncDrug 的按药物宏 AP 下降约 1.566 个百分点，MiDrug 下降约 0.131 个百分点，因此单药物候选排序仍应保留原模型对照。三个推理种子不构成独立数据重复，历史验证增量不等于未参与开发数据上的收益。

原 36 维拟合特征通过五折遮边生成：一行所属折的全部目标正边从支持图中删除。因而上下文行的支持图约含 80% 拟合正边，而原验证查询特征使用全部拟合正边。度数、路径和邻域支持等数值由不同规模的图得到；这提供一个可检验的分布匹配假设，尚不能从中直接断言它造成误差。

## 对照与预设选择

对每个查询重建相同的五个遮边支持图。比较以下三个组成项，并保留原始 TabICLv2。

| 组成项 | 模型上下文 | 查询图 | 检验目的 |
|---|---|---|---|
| `query_bag` | 全部原五折拟合行 | 分别在五个遮边支持图中编码，平均概率 | 保留全部上下文，匹配支持规模并集成图视图 |
| `fold_full` | 每次只使用对应一折的行 | 全部拟合正边 | 上下文样本数与模型数的对照 |
| `fold_match` | 与 `fold_full` 完全相同的五个拟合状态 | 与各折上下文完全相同的支持图 | 与上述对照相比，检验同图匹配 |

每个组成项分别以 50% 或 100% 概率权重与原模型合成，原模型也独立参与选择，共七个固定候选。所有候选均用 seed 666/0/1 完成内部评价，按平均 `(AUC+AP)/2` 选择；完全同分时优先较小新组成项权重，再按 ID。两个任务选择一起冻结后，才在本轮打开外层数组。

外层预先规定报告原模型、内部选定方案，以及三个未混合组成项。不能根据外层表现选择其中一个。成功条件仍是两个任务平均 AUC/AP 均超过原始 TabICLv2，且每项至少 2/3 个对应种子标签为正差值；按药物宏 AP 单独报告。

## 实现与证据边界

输入名称、36 维编码器、原始/分位数视图、官方 FP32 推理设置与 32 个推理成员保持原设定。LncDrug 的原分位数变换仍只从全部折外拟合行得到；官方预处理在各自模型上下文上拟合。五个图视图使用同一个种子，完整上下文模型的预处理置换不随视图改变。

编码器只接受显式支持正边。各折上下文正例不在其支持图中，所有评价目标边及反向边均排除；查询标签不参与编码。每份新缓存会重新计算全部原折外上下文特征以及完整图查询特征，并逐行与旧缓存比较。其它任务和 circRNA 边遵循原单任务协议，不跨任务拼接缓存。

正式流程共有 72 个拟合上下文状态：每阶段、每任务、每种子各一个完整上下文状态与五个折上下文状态。它们都共享同一官方预训练检查点，没有参数更新；不是 72 次独立模型训练。完整上下文的完整图预测另与既有基线对齐。每个状态均保存并重载，重放其全部查询视图；小规模工程流程额外核对查询顺序与分批不变性。

本设计是图 ICL 输入一致性的本地假设。已阅读 [TFMLinker](https://arxiv.org/abs/2602.08592) 方法中的上下文、链路编码和 TFM 三节，但本轮没有实现它的可学习跨图原型、DRNL+ 或预训练 GNN，不将该实验称为 TFMLinker 复现。参见[来源记录](research/tabicl_adaptation_20260912/RESEARCH.md)。

历史验证已多次用于开发；额外图视图增加了计算量。三个推理种子和五个图视图均不是独立数据划分或独立预训练重复。本轮不读取测试数组。

## 复现

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
TABICL_PY=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
GRAPH_PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
CONTEXT_OUT=improve/results/tabicl_graph_context_repeat
"$GRAPH_PY" -m unittest improve.test_tabicl_graph_context -v
"$TABICL_PY" -m improve.tabicl_graph_context prepare --out-dir "$CONTEXT_OUT"
"$TABICL_PY" -m improve.run_tabicl_graph_context --out-dir "$CONTEXT_OUT" --devices cuda:0,cuda:1
"$TABICL_PY" -m improve.analyze_tabicl_graph_context --out-dir "$CONTEXT_OUT"
```

旧图环境只负责 CPU 特征重建，独立 TabICL 环境负责 GPU 推理；二者都不修改。小规模工程流程使用新目录，在 prepare 时加 `--smoke`，其指标不用于性能结论。

源码：[复现编排入口](run_tabicl_graph_context.py)、[原冻结实验实现](tabicl_graph_context.py)、[图特征重建](tabicl_graph_context_features.py)、[选择与聚合](tabicl_graph_context_modules.py)、[关键检查](test_tabicl_graph_context.py)。

## 数值核验记录

11 项单元检查通过。首个工程短流程在 32 行上下文的分批查询检查中触发 1e-6 容差；失败目录及源码快照保留。第二个工程流程明确沿用原 TabICLv2 对查询换序/分批的 FP32 容差 `atol=rtol=1e-5`，24 个拟合状态全部重载，同形状预测误差为 0，查询换序误差为 0，分批最大差异约 7.45e-6。

正式推理中，完整上下文一次拼接六种查询视图，与原基线单独查询的 GPU 批形状不同。原运行器对这个跨形状比较仍使用同形状重放的 1e-6 容差，导致 LncDrug seed 0 以约 1.67e-6 的差异触发断言。为保留原冻结源码和内部选择，新增[独立数值审核适配器](tabicl_graph_context_audit.py)：只有这项跨形状比较采用原协议已有的 1e-5 分批容差，保存状态的重放仍用 1e-6。适配器在独立工程探针上的六个查询视图分数与原运行器逐位一致。

正式修订以 `NUMERICAL_AUDIT_PROTOCOL.json` 单独冻结；当时已观察到部分外层输出。修订没有改变模型、候选分数计算、输入、任何权重或内部选择，也没有删除失败记录。上述复现入口会在内部选择冻结后、构建本次外层特征前登记该数值审核政策。

选定方案的三种子分数另外导出到 `results/tabicl_graph_context_20260912/selected/<task>/s<seed>/scores.npz`，以 `val` 表示最终分数，并保存原基线、三个组成项、标签及配对。导出清单与数值敏感性诊断见 [SELECTED_PREDICTIONS.json](results/tabicl_graph_context_20260912/SELECTED_PREDICTIONS.json)。诊断只检查浮点扰动下的排序差值，不作为统计显著性或独立泛化证据。

另完成两任务、三个种子、五个图视图的 30 次单视图查询对照，复用原拟合状态。与拼接查询相比，所有种子的 AUC、AP 和按药物宏 AP 完全相同。更保守的每分数 2e-5 最不利扰动下，两任务 AP 与 MiDrug AUC 的平均增量仍为正，但 LncDrug 的微小 AUC 增量可以消失。两个诊断均如实保留，见[详细说明](results/tabicl_graph_context_20260912/POSTHOC_NOTES.md)及[核验记录](results/tabicl_graph_context_20260912/POSTHOC_VALIDATION.json)。

单视图数值诊断可在完成主实验及预测导出后，用 `python -m improve.audit_tabicl_query_batches --out-dir <目录> --task LncDrug --device cuda:0` 运行；MiDrug 对应使用 `--task MiDrug --device cuda:1`。该检查不替换主实验分数。
