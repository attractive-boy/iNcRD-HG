# 近期架构检索与本轮选择

检索日期：2026-09-10。范围是适合现有 ncRNA–drug 关联预测数据的可实现骨干，不是系统综述，也不证明算法首次提出。最终已实现并训练 PENCIL 与 TabM 两条路线；推荐使用图结构特征编码 + TabM，数值与限制见 [完整比较](../../results/backbone_comparison_20260910/REPORT.md)。下表保留检索时的适用性判断。

| 已核验来源 | 核心思路 | 对本项目的判断 |
|---|---|---|
| [PENCIL, 2026 预印本](https://arxiv.org/abs/2602.01553)，*Plain Transformers are Surprisingly Powerful Link Predictors* | 候选链接的局部子图；局部身份与邻接行编码；普通 Transformer 与邻接矩阵乘法残差 | **本轮实现**。直接替换节点编码式 GNN，且保留邻接求和。但原文也报告小数据上的统计效率不足，必须实测 |
| [Refined-GAE, CIKM 2025](https://dl.acm.org/action/doSearch?AllField=Reconsidering+the+performance+of+GAE+in+link+prediction)，*Reconsidering the Performance of GAE in Link Prediction* | PENCIL 文中讨论的正交节点向量与 GAE 改良 | 仅核对到 PENCIL 的参考文献及其描述，未独立核验全文；不作为本轮具体实现依据 |
| [LPFormer, KDD 2024](https://doi.org/10.1145/3637528.3672025)，*An Adaptive Graph Transformer for Link Prediction* | 为每个链接自适应学习配对编码，结合多种链接形成因素 | 已核验元数据和摘要；与已有路径/配对注意力实验相邻，本轮优先更换整体编码方式 |
| [Polynormer, 2024](https://arxiv.org/abs/2403.01232)，*Polynomial-Expressive Graph Transformer in Linear Time* | 线性复杂度的局部与全局图 Transformer | 已核验 arXiv 页面；可作为后续节点骨干候选，本轮不将其与 PENCIL 拼接 |
| [HDBI, 2026](https://doi.org/10.1021/acs.jcim.6c01116)，*Higher-Order Dynamic Disentangled Intent Sensing and Bidirectional Joint Updating Framework for NcRNA-Drug Resistance Association Prediction* | 同任务的多视图超图、解耦表示及双向跨模态更新 | 已核验 PubMed 摘要；说明“高阶、双向、多视图”不是任务空白，不能把名称当新颖性 |
| [TabM, 2024 预印本 / ICLR 2025](https://arxiv.org/abs/2410.24210)，*Advancing Tabular Deep Learning with Parameter-Efficient Ensembling* | 多个共享大部分参数的 MLP 同时训练与预测 | **本轮实现并验证**。已进一步核对官方 README 与 BatchEnsemble 初始化/读出代码，适配到既有图的无查询正边泄漏结构特征 |
| [TabPFN, Nature 2025](https://doi.org/10.1038/s41586-024-08328-6)，*Accurate predictions on small data with a tabular foundation model* | 小样本表格基础模型 | 已核验出版元数据；需要额外模型权重及表格化输入，本轮不引入 |
| [SIGIL, 2026 预印本](https://arxiv.org/abs/2608.08567)，*Neural Message Passing on Structural Interaction Graphs for Fully-Inductive Graph Neural Networks* | 在特征维组成的结构交互图上传播，实现跨图输入空间迁移 | 已核验摘要；主要针对跨图归纳迁移，与当前固定图提升的需求不同 |

PENCIL 的全文 HTML 已读取。具体依据是第 3.1–3.2 节的编码与公式 (1)–(2)，第 5.1 节的小数据限制，以及附录 C.2 的固定正交输入投影。实现不是官方代码复刻：异质关系独占邻接槽、按类型投影现有特征、额外编码当前可见图的关系入度，并缩小到 64 维、两层、34 个上下文节点。

原始记录见 `discovery.json`、`openalex.json`、`targeted.json`、`recent_graph_papers.json`、`selected_sources.json` 和 `pencil_full.html`。MCP 未挂载；公开 arXiv API 出现连接重置，使用 arXiv 官方 HTML 检索及全文兜底。CrossRef 的模糊检索命中并未一律视为目标论文，只有题名和内容相符的条目用于上述判断。检索仅发送公开关键词，没有上传项目数据。
