# 基于细胞系子集复现性的表达与图学习

本实验延续原 GNN 的 PCA 输入正结果，检验一个具体假设：表达视图的波动、相似边在互补细胞系中的支持程度，可以决定哪些信息值得传播和约束。它是一个待验证的机制候选；目前不宣称首创，也不将随机细胞系子集解释成癌种域。

## 机制

所有 RNA 使用训练节点拟合的原 PCA64，后补三个零以匹配既有 67 维输入。药物、circRNA、监督标签和消息图沿用首轮冻结输入。每个 RNA 类型固定抽取 8 对互补细胞系子集，两种类型独立抽样，不假定其细胞系列已经生物学对齐。

对标准化表达 x、原 PCA 分量 V、子集 C，计算经采样比例校正的部分投影：

`z_C = [d / |C| * sum_{j in C}(x_j * V_j) - PC_mean] / PC_scale`。

互补视图的加权均值等于完整投影，因此均值本身没有新增信息。以视图相对完整投影的均方差 `u_i` 衡量细胞系子集敏感度，定义描述性可靠性 `q_i = 1/(1+u_i)`。这些视图共享一个训练拟合的 PCA 基底，并不是独立训练得到的表示或统计置信区间。

对现有 RNA 相似边，在每个半子集中重新计算 Kendall tau-b，并按原相似度符号对齐；一对互补子集的支持为 `max(0, min(sign(s)*tau_left, sign(s)*tau_right))`。跨分组求均值后除以 `|s|`，截断至 `[0.05,1]`，作为原绝对相似度的收缩因子。保留现有边、自环及每个目标节点的非自环总入边权重。负相关可具有稳定支持，但这里不使用负消息，也不把它解释成抑制调控。

辅助目标是完整表达预测到单个子集视图预测的 Bernoulli KL 一致性，完整表达预测停止梯度。可靠性按训练样本归一化至均值 1，以匹配普通一致性对照的总约束强度。固定 `lambda=0.1`，前 20% 轮次线性升温。不构造节点间对比负例，不新增网络参数。

## 预设实验

| 方案 | 用途 |
|---|---|
| base | 原 GNN、原表达、单位边权 |
| pca | 原 GNN、PCA64＋3 零、单位边权 |
| abs_edges | PCA＋原绝对相似度权重 |
| reliable_edges | PCA＋细胞系子集支持修正的边权 |
| consistency_uniform | PCA＋普通一致性约束 |
| consistency_reliable | PCA＋可靠性加权一致性 |
| joint | 可靠性边权＋可靠性一致性；预设主候选 |
| joint_shuffled | 保留视图，打乱可靠性与节点/边的对应关系 |

边可靠性在原相似度绝对值的十分位、同符号组内按无向边打乱；节点可靠性在拟合节点与剩余节点内分别打乱。它们保留主要分布，供机制检验。

固定 LncDrug 100 轮、MiDrug 300 轮，原 RMSprop 学习率；种子为 666、0、1、2、3。每次以验证 AUC/AUPR 均值的首次最大值选择一个检查点。全程不读取测试数组或报告测试分数。五种初始化衡量优化波动，不能替代独立数据划分。

主候选的筛选条件：相对 PCA，在两个任务上的平均验证 AUC、AUPR 均增加，且各任务验证均值指标在至少 4/5 种子上增加。额外报告相对各组成部分及打乱对照的差值；未超过这些对照就不能将收益归因于可靠性。保留所有结果，不按最优种子汇报。

## 运行

在 `iNcRD-HG` 目录和 `ncRNAdrug` 环境中：

```bash
python -m unittest improve.test_expression_consensus -v
python -m improve.expression_consensus prepare --out-dir improve/results/expression_consensus_20260909
python -m improve.expression_consensus run --task LncDrug --device cuda:0 --out-dir improve/results/expression_consensus_20260909
python -m improve.expression_consensus run --task MiDrug --device cuda:1 --out-dir improve/results/expression_consensus_20260909
python -m improve.expression_consensus report --replay --device cuda:0 --out-dir improve/results/expression_consensus_20260909
```

`prepare --smoke` 在独立目录中冻结同样输入、seed 666、每组两轮，用于检查执行路径；正式训练前需要短跑通过。输入、源码、训练曲线、检查点、预测和实验配置均留存哈希。文献检索与新颖性边界记录在 `research/expression_consensus_20260909/`。
