# 注意力实验期间的追加文献核对

检索日期：2026-10-06。仅为现有 miRNA–drug 输入寻找机制；不导入文献数据或权重。
技能工作流：nature-academic-search / multi-source-search。没有学术 MCP，使用公开
Crossref、Europe PMC 和 arXiv API；PubMed/Crossref/arXiv 连通性检查均通过。

## 阅读结果与适用边界

| 文献 | 阅读依据 | 可借鉴内容与限制 |
|---|---|---|
| Zhou 等，[Heterogeneous dual-channel and interpretable graph representation learning with global virtual nodes for microRNA-mediated drug sensitivity prediction](https://doi.org/10.1007/s11030-026-11648-6)，2026-07-09 | Europe PMC 摘要；Crossref 题名、作者及在线日期交叉核对 | 全局虚拟节点交换上下文、通道门控；尚未获得全文方法，不能称为复现。药物敏感性任务也不等同于本任务的关联标签。 |
| [Learning over Positive and Negative Edges with Contrastive Message Passing](https://arxiv.org/abs/2605.17854v1)，2026-05-18 预印本 | arXiv 元数据及公开 HTML，重点读理论条件、方法第 3 节 | 正负传播分别变换，软半正定约束调节负特征值。论文重点是节点分类，理论讨论高同配、高密度、低标注率；本任务不自动满足这些条件。其补图非边与本任务的抽样零标签也不相同。 |
| [GegenNet: Spectral Convolutional Neural Networks for Link Sign Prediction in Signed Bipartite Graphs](https://arxiv.org/abs/2508.19907v1)，2025-08-27 预印本记录 | arXiv 元数据及公开 HTML，重点读第 4.1.3、4.2 节与实验任务 | 正负两图的多项式过滤交替进入非线性网络；不是仅追加有符号路径计数。实验是评分/投票等符号二部图，不能把采样零标签解释为生物拮抗。v1 式 (6) 与 (7)/(8) 的一阶系数分别为 α+1/2 与 α+1，若复现须进一步核对作者实现，不能自行声称公式一致。 |

RGDAGMAE 与 MVGSCA 本次再次检出；已有阅读依据见 RESEARCH_NOTES.md。
没有把重复检出计为新的独立证据。未根据预印本元数据推断会议接收状态。

## 对下一实验的判断

优先检验**原始特征上的全局路由，再送入现有 M123 编码器**。它与当前边注意力的
差别是：节点可以从没有图边连接的其他原始节点接收特征上下文。与早期 expert router
也不同：这里路由节点特征，不是给已有预测分支加权。

拟采用原表达谱/ChemBERTa 的小型残差变换；设置局部 MLP、均匀全局池和可学习路由
三个匹配结构，并各自比较无重建/原重建。输出从零初始化，初始恢复原模型。
路由输入只含原始节点特征，不含关联标签、查询集合或度数；因此查询删边不改变
路由输出，随后原 M1/M2 的精确删边公式仍可应用。推理属于原冻结节点全集上的
传导设置，不宣称支持新节点的归纳泛化。

这是基于摘要启发的独立适配，不是 HDIGRL 或 RGDAGMAE 的完整复现；效果待检验。
负边传播暂不优先：历史已有能量排斥和有符号路径方向，且新论文的任务/理论条件
不提供当前 MiDrug 上必然收益的依据。

## 检索记录

- followup_search.json：首轮 Europe PMC 返回 HTTP 503；Crossref 链路预测检索返回
  HTTP 429；Crossref miRNA 检索成功。失败没有被解释为“没有相关论文”。
- followup_retry.json：放宽的 Europe PMC 检索成功，返回前 50 条；arXiv 返回前
  15 条，包含不相关条目。这不是系统综述或完整收录。
- signed_methods_fetch.json：两篇 arXiv HTML 读取成功；raw/ 保存全文 HTML/text。
- raw/hdigrl_crossref.json：HDIGRL 的 DOI 元数据。
- 初始 HTML 转换缺少 BeautifulSoup，最终用 Python 标准库读取公开 HTML；不改变
  训练环境或安装依赖。仅记录实际阅读范围，不把下载全文等同于逐段精读全文。
