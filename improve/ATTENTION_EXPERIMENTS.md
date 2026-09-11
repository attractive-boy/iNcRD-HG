# 三类注意力的独立对照实验

入口是 `attention_mechanisms.py`，原模型与既有实验脚本保持独立。
结果目录是 `results/attention_mechanisms_20260907`。

## 固定设计

在 MiDrug、LncDrug 上分别使用种子 666、0、1、2、3。
每个任务有 9 个实验组，总计 90 次训练；不在本轮叠加不同机制。

| 实验组 | 操作 | 主要对照 |
|---|---|---|
| base | 原编码器和解码器 | 共同基线 |
| channel_mlp | 对最终嵌入增加按节点 RMS 缩放的残差 MLP | 新增参数和非线性 |
| channel_attention | 最终嵌入乘以输入依赖的通道权重 `1 + tanh(MLP(LN(z)))` | channel_mlp、base |
| neighbour_uniform | 对已有 RNA 邻居均匀聚合，增加 RNA 嵌入残差 | 新增邻居分支 |
| neighbour_softmax | 对同一邻居集合进行点积 softmax 加权 | neighbour_uniform |
| neighbour_randomk | 每节点、每关系随机固定最多 8 个邻居，然后 softmax 加权 | 随机筛选；每个种子内掩码固定 |
| neighbour_topk | 按学习到的分数选最多 8 个邻居，然后 softmax 加权 | neighbour_softmax、neighbour_randomk、base |
| layer_uniform | 均匀融合投影后的原始特征、第一层和第二层表示，增加嵌入残差 | 增加多层信息 |
| layer_attention | 每节点学习上述三个层级的 softmax 权重 | layer_uniform、base |

通道分支作用于目标 RNA 和药物嵌入；层间分支也作用于这两类节点。
邻居分支只使用 RNA→RNA 边，不把目标 RNA–药物标签边作为注意力掩码。
所有分支的最终输出层初始化为零，因此新增模块对同一次编码的初始预测没有影响。
同一种子下，各组原模型参数的哈希必须完全一致。

## 数据与选模

- 使用现有节点特征。训练、验证和测试编码均使用本轮训练图。
- 在实验副本中，跨划分重复样本按测试、验证、训练的顺序保留；冲突标签直接报错。
- 实际发现 LncDrug 的正关联 `(449, 10)` 同时在训练和验证集中。本轮移除其训练标签和正反向训练图边，保留验证及测试样本。
- MiDrug 在训练集内部重复的两个正例保留原重数；全部组使用同样输入。
- 原始数据文件不修改。清理后的副本、输入哈希和删除记录保存于结果目录。
- MiDrug 训练 300 个 epoch，LncDrug 训练 100 个 epoch，沿用各自原模型宽度、学习率和关系聚合方式。
- 每次训练以验证集 AUC/AUPR 均值选择一个 checkpoint，测试指标不参与选模。
- Top-k 固定为 8，不按测试集扫描 k。采用原有无权重 BCE 和 RMSprop。

## 执行

在仓库根目录，使用 `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`：

```bash
python -m unittest improve.test_attention_mechanisms improve.test_drug_condition_attention -v
python -m improve.attention_mechanisms init --out-dir improve/results/attention_mechanisms_20260907
python -m improve.attention_mechanisms run --out-dir improve/results/attention_mechanisms_20260907 --device cuda:0 --shard-index 0 --shard-count 2
python -m improve.attention_mechanisms run --out-dir improve/results/attention_mechanisms_20260907 --device cuda:1 --shard-index 1 --shard-count 2
python -m improve.attention_mechanisms summarize --out-dir improve/results/attention_mechanisms_20260907
python -m improve.analyze_attention_mechanisms --out-dir improve/results/attention_mechanisms_20260907
```

两个 `run` 命令可同时执行，写入不重叠的运行目录。相同命令可恢复未完成的任务；完整结果会跳过。
运行计划冻结后，修改源代码或输入会被拒绝。短跑测试使用独立目录及 `init --smoke`。

每次运行保存 checkpoint、学习曲线、验证和测试预测、注意力权重与嵌入变化幅度。
汇总包含逐种子结果、均值与标准差，以及预设对照的配对差值和胜出种子数。
干预分析对所有三个主要方案的已选 checkpoint 关闭新增分支或使用非选择性权重，
并在邻居数大于 k 的节点上计算跨种子支持集 Jaccard。它不会重新训练或选模。
这是固定划分上的探索实验；跨种子稳定不能代替跨划分验证，隐藏通道也不能直接解释为生物特征。
