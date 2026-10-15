# 近期论文与纯 miRNA–药物实验方向

检索日期：2026-10-06。服务于现有目标：固定 MiDrug 划分、保留 M1/M2/M3，
AUC/AP 均相对原始 iNcRD-HG 绝对提高至少 0.02，同时保住 PLMF-MDA MDR/MDS。
仅允许现有 miRNA 与药物数据，不使用 lncRNA、外部基因网络、序列或注释。
查阅文献不等于引入论文的数据集或预训练权重。

## 本轮结论

优先检验 **OCN 的高阶邻居系数去相关与归一化**。本地已有 M2 简单路径、
固定 3/5/7 步统计、路径注意力和局部子图试验，但它们不等同于先把高阶系数中
与低阶重合的线性分量去掉。OCN 因而提供了一个具体、可消融的新机制。
该判断是实验优先级，不是 +0.02 的效果保证，也不是原创性主张。

对照已在 [OCN 适配目录](../midrug_ocn_20261006/README.md) 中实现。
其余两项备选是掩码重建/路由融合，以及从当前训练关联提取稳健低秩信号。
本轮先完成 OCN 消融，随后也完成了此前提出的嵌套折外残差方案，
结果见 [折外实验报告](../midrug_crossfit_20261006/REPORT.md)。两轮均未达到推进门槛。

## 核对的代表文献

| 文献 | 年份与出处 | 本轮阅读依据 | 对当前任务的意义 |
|---|---|---|---|
| Juntong Wang、Xiyuan Wang、Muhan Zhang. [OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](https://arxiv.org/abs/2505.19719) | NeurIPS 2025；arXiv v2 更新于 2026-03-09 | arXiv 元数据、v2 全文方法 4–6 节/消融/限制、官方代码与 README | Gram–Schmidt 去相关和路径计数归一化；可以只依赖原训练图 |
| Yunyin Li 等. [Router-guided dual attention with graph masked autoencoder for microRNA–drug association prediction](https://doi.org/10.1016/j.compbiolchem.2026.109236) | Computational Biology and Chemistry，2026；Europe PMC 首发日期 07-13，Crossref 卷期日期 10 月 | Europe PMC 摘要、Crossref 元数据 | 路由式全局交互与门控掩码重建互补；备选，目前未核实全部全文实现细节 |
| [IRPCA: An Interpretable Robust Principal Component Analysis Framework for Inferring miRNA-Drug Associations](https://doi.org/10.1021/acs.jcim.4c02385) | Journal of Chemical Information and Modeling，2025-02-20 | Europe PMC 摘要、PubMed 日期核对 | 非凸低秩分解与噪声处理；若适配，GIP 和分解必须在每一训练子集内重算 |
| Harry Shomer 等. [LPFormer: An Adaptive Graph Transformer for Link Prediction](https://doi.org/10.1145/3637528.3672025) | KDD 2024 | Crossref 题名/作者/会议元数据，未返回摘要 | 图 Transformer 的较早参照；本轮未据其全文设计新实现 |
| Ru Nie 等. [Multi-view Graph Learning Framework with Spectral Encoding and Sparse Cross-Attention for miRNA-Drug Association Prediction](https://doi.org/10.1109/TCBBIO.2026.3729865) | IEEE TCBBIO，2026-09-01（PubMed 电子日期） | Europe PMC 摘要、PubMed 日期核对、Crossref 检索记录 | 谱过滤/稀疏交叉注意力可提供思路，但原方法含 gene 介导图及序列相似性，不能原样导入当前实验 |
| [DLP: Duplex Link Prediction via Subspace Segmentation for Predicting Drug-MiRNA Associations](https://doi.org/10.1109/TCBB.2024.3402215) | IEEE TCBBIO；2024 年已有 PubMed 记录，2026 年 3 月编入卷期 | Europe PMC 摘要、PubMed/Crossref 日期交叉核对 | 双向子空间分割是较早的低秩类参照；不能因卷期为 2026 就声称它是新近首发方法 |

检索也出现 mDADGAN 等显式使用 lncRNA 的论文；这些不纳入当前候选。
MPHGNN/MVGSCA 的基因中介结构也不满足本轮仅 miRNA/drug 输入要求。
论文各自的 AUROC/AUPR、Hits@K 或 MRR 不能换算成当前固定划分上的 AUC/AP 增益。

## OCN 原方法与本地适配的区别

原文将不同阶的共同邻居系数视为向量，通过 Gram–Schmidt 去相关，
并通过路径参与次数归一化降低高频邻居支配。官方实现使用批次运行统计，
主要实验包括一般图的 Hits@K/MRR；不是直接在本项目上验证的算法。
原文还指出小图上的收益较弱，加入更高阶数可能带来不稳定，不能只引用其最大增益。

本地适配保留原 M123，在二部关联图上使用 3/5 步游走的中间节点系数：

`C_k(r,d) = B_q^(k-1)[r] * B_q^k[d] + B_q^k[r] * B_q^(k-1)[d] + B_q^k[r] * B_q^k[d]`, `k=2,3`。

`B_q` 是先去掉当前查询边的对称二部邻接矩阵；同侧项在二部图中自然为零。
先将两个端点的系数清零。归一化除以 `B_q^k @ 1` 的逐节点路径质量，
去相关在每条查询内部进行，随后将各阶的 L1 尺度与池化向量分开输入分支。
这些是明确的适配选择，不声称完整复现论文的全图批次正交化或归一化公式。

通过原 M1 的解析修正汇聚删边后的编码器状态，末层零初始化；
设置 raw/norm/orth/orth_norm 四项消融，三折均只来自原训练集。
新增实验成绩单独记录，不把当前内部开发分数与原测试分数相减。

## 检索范围与可复查材料

- 技能：`nature-academic-search`，工作流为 multi-source-search。
- 当前没有学术 MCP；直接调用公开 Crossref、arXiv、Europe PMC、PubMed API。
- 首轮五组检索返回 330 条记录，按已提供 DOI/arXiv ID/规范化标题去重为 328 条；
  其中有大量不相关发现，不能视作 328 篇已阅读文献。定向 Europe PMC 查询另返回 23 条，
  PubMed 定向查询也返回 23 条（两者重叠，不能相加称为独立文献数）。
- 检索以 2025–2026 为主，保留必要的 2024 方法参照。这是有范围的研究检索，不是系统综述。
- `search_results.json`、`focused_fetch.json`、`raw/` 保留查询、原始响应及全文/元数据。
- OpenAlex 备用检索读取超时，没有使用它的未返回结果；GitHub raw 直址一次连接重置后，
  官方源码通过 GitHub API 成功获取。没有绕过任何付费或账号访问限制。
- 未下载或加载文献中的训练数据集；训练脚本仅加载独立冻结的纯 miRNA/drug 输入。


## IRPCA 的进一步核对与独立适配

重建/BPR 互补性试验完成后，进一步核对 IRPCA。Crossref 题名与 DOI 一致，
其 published-online 是 2025-02-21，PubMed electronicPublicationDate 是 2025-02-20，
相差一天；两源都支持它为 2025 年首发。GitHub 的 IRPCA+miRNA 仓库检索未命中，
不能据此断言没有代码。完整记录见 irpca_fetch.json 与 raw/irpca_*.json。
无学术 MCP 可调用，因此复用 Europe PMC 摘要并通过公开 Crossref/GitHub API 定向核对。

摘要具体支持：非凸低秩近似处理噪声、GIP 融合相似性、全连接预测器。
它没有提供足够信息确认原优化公式，因此不将新代码称为复现或作者实现。
[新的矩阵补全适配](../midrug_completion_20261006/README.md) 是凸核范数、
仅训练观测项的 logistic 损失、可消元 L1 误差与原相似图拉普拉斯平滑。
它既不是 IRPCA 的非凸公式，也不同于此前神经网络里加入固定 rank 的双线性交互。
效用必须由原训练内部的匹配消融确认，文献各自的成绩不作为本任务提升证据。


## 注意力之后的追加检索

见 [追加阅读笔记](FOLLOWUP_NOTES.md)：HDIGRL 全局上下文摘要、CMP 与 GegenNet 的公开方法全文，以及与当前抽样关联标签的差异。候选已进入独立的全局特征路由内部对照实验；未使用论文数据或权重。
