# OCN 思路的纯 miRNA–药物适配

目标仍是固定原测试集三种子均值 AUC ≥0.9733929274189016、
AP ≥0.933683373732677，同时保住 PLMF-MDA MDR/MDS。当前尚未达到。

依据 [OCN，NeurIPS 2025，arXiv v2 2026-03-09](https://arxiv.org/abs/2505.19719)，
详细适配边界和其他近期论文见 [文献笔记](../midrug_literature_20261006/RESEARCH_NOTES.md)。
本目录是 OCN 启发的 M123 增量实验，不是官方 OCN 完整复现。

仅用 `../midrug_mirna_only_20261006/graph.pt` 和 `inner_data.npz`；
保留原 M1/M2/M3，新增二部图 3/5 步系数的编码器状态池化分支。
每条查询的正例边先删除，所有根节点系数归零。可选的归一化按该删边图中
每个节点的 k 步游走数进行；可选的 Gram–Schmidt 在查询内部去除高阶系数
在低阶系数方向上的分量。随后 L1 归一化池化，并保留系数幅值的 log1p 特征。
正交化后的负系数只是数值权重，不是新增生物学负关系。

任意节点的删边后池化使用 M1 的一阶状态变化经第二层消息矩阵传播的精确修正。
分支有两个 256→32 投影及 132→128→1 MLP，dropout 0.1，末层零初始化。
训练配置与上一轮相同：RMSprop 1e-4，新增分支 weight decay 1e-4，
最多 800 epochs，至少 300，最优轮后 200 轮无改善停止。
按内部 `(AUC+AP)/2` 选 checkpoint。配置选择按三折相对 M123 对照的
`min(mean ΔAUC, mean ΔAP)`，两项均 ≥0.002 才考虑原验证确认。
这个开发门槛不是最终 +0.02 的替代。

四个配置为 raw/norm/orth/orth_norm，构成归一化×去相关的 2×2 消融。
对照复用上一轮已完整重放的三份纯 miRNA M123 checkpoint；本轮再次核对。
全部选择只使用原训练集内部三折；没有原验证/测试标签访问。

解释器：`/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`。
环境变量：`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`。
顺序运行 `features.py` → `check.py` → `launch.py` → `audit.py` → `summarize.py`。
缓存与训练脚本拒绝覆盖已有同名结果；结果、源码及输入哈希随实验保存。
