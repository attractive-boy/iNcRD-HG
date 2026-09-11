# 边属性与路径实例注意力

本轮接续传播内部注意力实验，检查两个尚未完成正式评估的方向。
运行入口为 `edge_path_attention.py`，模块为 `edge_path_modules.py`，
独立核验入口为 `verify_edge_path_attention.py`。
正式结果写入 `results/edge_path_attention_20260908`。

## 假设与对照

上一轮将邻居求和改为均值后，两个任务的测试 AUPR 都下降约 0.04。
因此边属性方案保留原关系融合、自节点项、自环和非自环入度总权重，
只学习同一相似关系内部的权重分配。路径方案在解码器增加零初始化的残差分支。

| 实验组 | 变化 | 对照目的 |
|---|---|---|
| base | 原模型，同种子完全重复两次 | 测量严格重复运行是否一致 |
| edge_fixed | 按相似度绝对值分配边权，再恢复入度总权重 | 固定属性加权是否已足够 |
| edge_blind | 由源、目标节点表示学习边权 | 学习门控本身的作用 |
| edge_attributes | 在相同打分器中加入有符号相似度及其绝对值 | 边属性的额外作用 |
| edge_shuffled | 在关系内部打乱互反边的属性配对 | 属性与具体边的对应是否重要 |
| path_count | 为每类 RNA 中介路径增加 `log(1 + count)` | 新增路径数量信息 |
| path_uniform | 数量特征加中间节点的均匀池化 | 新增路径表示 |
| path_attention | 用候选 RNA 和药物共同查询中间节点，softmax 池化 | 在相同路径信息上学习选择 |

主要候选预先限定为 `edge_attributes` 和 `path_attention`。
不能仅凭超过 `base` 就将收益归因于注意力；需要同时比较固定、无属性、打乱属性或均匀池化对照。

对非自环边，学习权重为

\[
u_{ij}=\operatorname{sigmoid}\!\left(w^\top\operatorname{LeakyReLU}
(W_s\operatorname{LN}(x_i)+W_t\operatorname{LN}(x_j)+W_e[s_{ij},|s_{ij}|])\right),
\qquad a_{ij}=d_j\frac{u_{ij}}{\sum_{k\to j}u_{kj}}.
\]

`edge_blind` 去掉属性项；`edge_fixed` 用 `|s_ij|` 替代 `u_ij`。
自环的权重始终为 1。这里有符号相似度参与打分，最终消息权重仍为非负数。

路径只枚举训练图内的 `目标 RNA → 同类/另一类 RNA → 药物`，
候选对的所有有效中间节点均保留。剔除中间节点就是目标 RNA 的自环捷径，
不把候选训练正例直接编码成一条路径，不按分数截断或抽样。
基础编码器仍沿用原协议，使用训练正例作消息传递边。

## 固定实验协议

- MiDrug、LncDrug，各使用种子 666、0、1、2、3。
- 8 组 × 2 任务 × 5 种子 = 80 次；每个原模型另重复一次，共 90 次。
- MiDrug 使用 256 维、关系 mean、学习率 0.0001、300 个 epoch。
- LncDrug 使用 64 维、关系 min、学习率 0.0005、100 个 epoch。
- 使用 RMSprop、alpha=0.99、无权重 BCE；本轮不搜索架构超参数。
- 每次仅按验证集 AUC/AUPR 均值选择一个 checkpoint，完全相同时选最早 epoch。
- 所有阶段使用清理后的训练图；沿用上一轮删除 LncDrug 跨划分重复训练正例的实验副本。
- 基础模型、分支初始化、训练随机数分别固定。启用确定性算法，关闭 TF32，同组重复在同一 GPU 上运行。
- 主表只统计 repeat_0。重复运行不作为额外独立种子，历史轮次不能直接用作本轮基线。

## 执行与核验

在仓库根目录，使用 `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`：

```bash
python -m unittest improve.test_edge_path_attention -v
python -m improve.edge_path_attention init --out-dir improve/results/edge_path_attention_20260908
python -m improve.edge_path_attention run --out-dir improve/results/edge_path_attention_20260908 --device cuda:0 --shard-index 0 --shard-count 2
python -m improve.edge_path_attention run --out-dir improve/results/edge_path_attention_20260908 --device cuda:1 --shard-index 1 --shard-count 2
python -m improve.edge_path_attention summarize --out-dir improve/results/edge_path_attention_20260908
python -m improve.verify_edge_path_attention --out-dir improve/results/edge_path_attention_20260908
```

两个 worker 写入不重叠的目录，可以同时运行。已冻结的计划通过相同 `run` 命令恢复，
不重复执行 `init`。源码或输入哈希变化会拒绝续跑。
`results/edge_path_attention_20260908_smoke_v2` 是当前实现的 32 次短跑检查，不能用于评判性能。

核验覆盖全部输入、相似度及资产哈希、划分隔离、独立路径枚举、权重约束、
模型参数哈希、预测对应的指标、选模轨迹、重复运行一致性，
并在 CPU 上重建第一个预设种子的所有 16 个 checkpoint 及其干预预测。

这是同一固定划分上的探索；确定性重复和多初始化种子都不能代替新划分验证。
干预权重衡量预测依赖，不构成生物因果证据。
