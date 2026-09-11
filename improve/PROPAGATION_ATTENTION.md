# 传播内部的注意力对照

入口：`propagation_attention.py`。使用上一轮 `attention_mechanisms_20260907/protocol.json`
引用的清理后输入及其哈希，不修改此前的源码、数据或结果。

两个任务 × 5 个种子（666、0、1、2、3）× 9 组，共 90 次训练。

| 实验组 | 改动 | 解释所需的对照 |
|---|---|---|
| base | 原模型 | 共同基线 |
| relation_uniform | 原关系卷积输出做均匀融合 | 关系融合算子变化 |
| relation_global | 同一节点类型共享学习到的关系权重 | 全局关系注意力 |
| relation_node | 每个节点单独学习关系权重 | relation_global、relation_uniform、base |
| edge_uniform_mean | 每个关系内的邻居等权取均值，保留自节点项 | 邻居求和改为均值 |
| edge_gate_mean | sigmoid 边门控，按目标节点归一到权重和为 1 | edge_uniform_mean |
| edge_gate_degree | 相同边门控，归一到权重和等于入度 | base、edge_gate_mean |
| gatv2_uniform | GATv2 的源/目标投影和原自节点投影，边权固定等权 | 消息变换与聚合方式 |
| gatv2_attention | 相同投影下学习 4 头 GATv2 注意力 | gatv2_uniform、base |

关系卷积和原解码器先按同一种子完整初始化，然后再修改编码器。
三个关系组初始均为关系均值；保度门控初始还原原邻居求和；均值门控和两个 GATv2 组初始还原邻居均值。
因此只在上述匹配组之间要求初始函数相同，不能声称所有架构的初始预测相同。

GATv2 头数固定为 4，拼接到原隐藏宽度；保留原自节点投影、不额外增加自环、dropout 为 0。
其源值投影从原消息投影初始化，注意力向量和新增输出偏置初始化为零。
均匀对照冻结不参与其输出的打分参数，其余可训练变换保持一致。

沿用每任务原学习率、RMSprop、无权重 BCE，以及 MiDrug 300 / LncDrug 100 个 epoch。
每次只用验证集 AUC/AUPR 均值选择一个 checkpoint，训练图用于所有阶段的编码。
没有对单个新架构做超参数搜索。

每个主要候选选模后，再将其学习权重替换成对应的等权权重，检查预测依赖；不重训、不重选。
所有训练曲线、checkpoint、预测、注意力和干预指标都写入本轮独立结果目录。

在仓库根目录使用 `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`：

```bash
python -m unittest improve.test_propagation_attention -v
python -m improve.propagation_attention init --out-dir improve/results/propagation_attention_20260907
python -m improve.propagation_attention run --out-dir improve/results/propagation_attention_20260907 --device cuda:0 --shard-index 0 --shard-count 2
python -m improve.propagation_attention run --out-dir improve/results/propagation_attention_20260907 --device cuda:1 --shard-index 1 --shard-count 2
python -m improve.propagation_attention summarize --out-dir improve/results/propagation_attention_20260907
python -m improve.verify_propagation_attention --out-dir improve/results/propagation_attention_20260907
```

两个 worker 写入不重叠的任务目录，可同时运行；相同命令可以恢复未完成任务。
短跑使用独立目录和 `init --smoke`。正式计划冻结后，源码或输入哈希变化会使 worker 拒绝运行。
固定划分、多次探索后的结果仍需独立划分验证；负结果只针对本轮配置。
