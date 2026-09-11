# 来源诊断上下文的多实例 GNN

第一轮跨细胞系可靠性实验完成 80 次正式训练，联合方案未超过 PCA。第二轮因此检验不同的标签语义：文献级 RNA–drug 关联可能只需在部分表达背景中成立，跨背景预测无需全部一致。这是由第一轮验证结果启发的自适应探索，不是独立确认。

## 输入与模型

lncRNA 与 miRNA 表达覆盖同一批 938 个 ACH 细胞系 ID。现有 Cellosaurus 的 DepMap 交叉引用可为 899 个细胞系匹配来源诊断。只按覆盖数选择前三个精确诊断，未使用耐药标签：Lung adenocarcinoma（59）、Lung small cell carcinoma（42）、Colon adenocarcinoma（33）。其余 804 个细胞系合为一个剩余组。这些分组很不均衡，也不代表完整癌种范围。

每组表达投影到原训练拟合的 PCA64 坐标，再仅用训练 RNA 拟合该组的分量尺度；后补 3 个零。同一组中两类 RNA 按 ACH ID 对齐。药物、circRNA 与所有消息边沿用原图。完整表达图和四个上下文图组成一个 bag，用同一套原 GNN 参数并行计算。节点和边索引按图复制并偏移，禁止不同上下文之间串边。

令 `l_full` 为完整表达预测、`l_c` 为各上下文预测：

`l_pair = 0.75*l_full + 0.25*2*log(mean_c(exp(l_c/2)))`。

这是平滑的部分上下文支持聚合，不是校准的 noisy-OR 概率或严格存在量词。温度 2、上下文比例 0.25 固定。仅 pair/bag 级标签参与 BCE；不构造癌种级监督。该模型不新增网络参数。

## 预设对照

| 方案 | 定义 |
|---|---|
| pca | 同一原 GNN 的完整 PCA 输入 |
| context_mean | 相同真实诊断分组，改用上下文 logit 均值 |
| context_softor | 真实诊断分组，使用平滑部分支持聚合；主候选 |
| random_softor | 保留各组大小，随机重排细胞系归属，使用相同聚合 |

`context_softor − context_mean` 检验聚合方式，`context_softor − random_softor` 检验诊断分组。若只有相对 PCA 的提升，不足以证明生物学分组或部分支持机制有用。

训练预算与第一轮相同：LncDrug 100 轮、MiDrug 300 轮；seed 666、0、1、2、3。所有组初始化权重、参数量一致，两个指标来自同一验证集选定检查点。不读取测试数组。按药物排名作为额外诊断，不参与模型选择。

## 运行

```bash
python -m unittest improve.test_cell_context_mil -v
python -m improve.cell_context_mil prepare --out-dir improve/results/cell_context_mil_20260909
python -m improve.cell_context_mil run --task LncDrug --device cuda:0 --out-dir improve/results/cell_context_mil_20260909
python -m improve.cell_context_mil run --task MiDrug --device cuda:1 --out-dir improve/results/cell_context_mil_20260909
python -m improve.cell_context_mil report --replay --device cuda:0 --out-dir improve/results/cell_context_mil_20260909
```

正式训练前，在另一目录运行 `prepare --smoke` 和两个任务的 `run`，复核两轮短跑。该方案的多实例学习思想也有先例；新颖性只能围绕具体上下文构造、标签建模和对照证据判断，不能宣称多实例或 log-mean-exp 聚合首次出现。
