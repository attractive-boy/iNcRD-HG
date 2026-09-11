# 当前可用替代：图结构特征编码 + TabM

本轮使用 [TabM（ICLR 2025）](https://arxiv.org/abs/2410.24210) 替换原 GNN 的训练和推理路线。输入仍来自工作区的原图、PCA 表达特征、药物特征与拟合标签；模型直接读取无查询正边泄漏的图结构特征，经共享参数的多成员网络输出一个关联分数。

历史外层验证，seed 666、0、1 的均值如下。完整四架构对照、标准差、按种子差值和限制见 [最终比较](results/backbone_comparison_20260910/REPORT.md)。

| 任务 | 当前原 GNN AUC / AP | 图编码 + TabM AUC / AP | AP 增量 |
|---|---:|---:|---:|
| LncDrug | 0.980402 / 0.943892 | **0.982052 / 0.966260** | **+2.24 个百分点** |
| MiDrug | 0.955536 / 0.919169 | **0.958717 / 0.921461** | **+0.23 个百分点** |

两个任务的全局 AUC、AP 均为 3/3 种子提升。LncDrug 的按药物宏 AP 从 0.892087 降至 0.825590，MiDrug 从 0.716735 升至 0.718469。因此当前方案适合追求此划分上的全局排序指标；LncDrug 单药物候选排序仍应保留原模型对照。两个验证划分此前都用于研究，三个初始化不能替代新划分验证。本轮未用测试集选参数或报告测试分数。

**结构与训练**

图编码器为两个任务各输出 36 维特征：关系度数、两侧协同支持、三步/五步路径、16 阶 SVD 重建值、跨 RNA 桥接、属性支持与端点统计。训练使用五折特征生成，每个训练样本的整折正边都从支持矩阵中删除，再重算所有依赖目标关联的特征。其它任务的关联沿用既有单任务协议。

数值预处理只在这些训练特征上拟合分位数变换。TabM 使用三层、128 维、16 个内部成员。每层共享主权重，但各成员有输入/输出缩放及偏置；第一层输入缩放以随机符号初始化，其余缩放初始为 1，输出层各成员独立。训练对各成员分别计算 BCE 并平均，推理平均各成员概率。这是一个共享权重的内部集成，单个检查点保存全部成员。

同特征的三层普通 MLP 获得相同学习率搜索预算。其外层 AP 为 LncDrug 0.958666、MiDrug 0.916393；TabM 高于该对照。LncDrug 内部验证上普通 MLP 反而更强，表明结果仍有划分依赖。相比 GNN，特征表达、训练支持图构造与优化器也改变了，不能把全部收益单独归于 TabM 算子。

冻结参数：

| 任务 | 学习率 | 权重衰减 | dropout | 完整拟合集重训轮数 |
|---|---:|---:|---:|---:|
| LncDrug | 0.003 | 0.0001 | 0.1 | 141 |
| MiDrug | 0.001 | 0.0001 | 0.1 | 441 |

轮数是各任务三个内部验证选中轮数的中位数，外层标签没有参与确定轮数。默认推理固定使用 seed 666；没有根据外层结果挑最佳种子。

**直接使用已训练模型**

在仓库目录下，使用现有环境（已验证 Python 3.8.8、PyTorch 1.13.1）：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python

# 输入 CSV 的列名为 rna_index,drug_index，索引对应当前冻结图。
$PY -m improve.predict_backbone --task LncDrug \
    --pairs candidate_pairs.csv --out candidate_scores.csv --device cuda:0

# 输出某个药物的未知关联候选，自动排除已知拟合正例。
$PY -m improve.predict_backbone --task MiDrug \
    --drug-index 0 --out mi_drug0_scores.csv --device cuda:0
```

输出 `association_score` 是当前关联预测模型的分数，未校准为实际耐药概率。CSV 不能含已知拟合正例；默认不会覆盖已有输出。用 `--seed 0` 或 `--seed 1` 可调用其余两个固定配置检查点。更换实验目录使用 `--experiment`；原 GNN 可通过 `--experiment improve/results/backbone_replacement_20260910 --arm base` 调用。

两任务各 25 个候选已经通过命令行推理与保存预测的一致性核验，最大差异小于 7e-8；记录见 [PREDICTION_CLI_VALIDATION.json](results/graph_tabm_20260910/PREDICTION_CLI_VALIDATION.json)。

**重训**

```bash
OUT=improve/results/graph_tabm_repeat
$PY -m unittest improve.test_graph_tabm -v
$PY -m improve.graph_tabm prepare --out-dir "$OUT"
$PY -m improve.graph_tabm run --out-dir "$OUT" --stage screen --device cuda:0
$PY -m improve.graph_tabm select --out-dir "$OUT" --stage screen
$PY -m improve.graph_tabm run --out-dir "$OUT" --stage confirm --device cuda:0
$PY -m improve.graph_tabm select --out-dir "$OUT" --stage confirm
$PY -m improve.graph_tabm run --out-dir "$OUT" --stage outer --device cuda:0
$PY -m improve.graph_tabm report --out-dir "$OUT" --device cuda:0
```

两个任务、TabM 与普通 MLP 的完整过程共 28 次正式训练。prepare 冻结输入引用、源码和配置；已有完成结果不会被覆盖。每个训练结果含检查点、逐轮轨迹、预测与哈希，report 会重放全部检查点。

入口：[训练器](graph_tabm.py)、[图编码与网络](graph_tabm_modules.py)、[推理程序](predict_backbone.py)、[原始结果](results/graph_tabm_20260910/results.csv)、[近期方法检索](research/backbone_replacement_20260910/RESEARCH.md)。另一个完整替换候选 [异质 PENCIL](BACKBONE_REPLACEMENT.md) 的代码和否定结果保留为对照。
