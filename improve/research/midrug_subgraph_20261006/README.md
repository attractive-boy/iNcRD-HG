# miRNA–drug 查询子图分支

只使用 `../midrug_mirna_only_20261006/graph.pt` 和原训练集内部三折数据。
不使用 lncRNA、外部数据集或原验证/测试标签。目标仍为原测试 AUC/AP 各增加 0.02，
并保住 PLMF 两个数据集，详见 `TARGET.json`。

在 M1/M2/M3 输出上加入查询删边后的局部子图 GNN logit，比较：

- `sg1_struct`：1-hop，最多各 16 个 miRNA/药物，仅结构特征。
- `sg1_cov`：上述子图加原始 miRNA 表达和药物 ChemBERTa 特征。
- `sg1_signed`：进一步区分原训练正例与抽样负例关系；不表示敏感/耐药方向。
- `sg2_cov`：2-hop，最多各 32 个 miRNA/药物，加原始特征。

所有子图先删除查询的正/负记录，再选择节点、计算距离与度数。截取保留根节点；
1-hop 按查询相关的共同邻居数排序，2-hop 为一阶节点与二阶节点分配名额。
节点 ID 只用于索引原协变量和稳定破同分，不作为可训练 ID embedding。

运行环境：`/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`，
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`。

1. `extract.py --fold 0/1/2` 分别准备三折缓存。
2. `check.py` 检查显式监督记录删除、初始一致性、梯度和成本。
3. `launch.py` 运行固定 12 项训练（每 GPU 两个独立进程）。
4. 全部训练完成后运行 `audit.py` 和 `choose.py`。
5. `context_audit.py` 检查截取上下文覆盖率；`summarize.py` 汇总完成后的证据。

现有缓存/结果不覆盖；launcher 只跳过哈希吻合的完整结果。
在相同研究目录层级复制脚本到新目录后可重跑，不应在 source_snapshot 目录直接执行。
原对照使用上一轮完全相同的纯 miRNA 三折 M123 checkpoint，记录在 plan.json，
本轮重新推理核验；没有把含 lncRNA 的历史结果当作对照。

训练按内部 (AUC+AP)/2 选轮。配置按相对对照的两指标平均增益最小值选择。
只有内部两项平均增益均达到预声明的 0.002，才进入原验证集进一步确认。
0.002 是开发推进门槛，不是最终目标，也不代表最终测试 +0.02。
