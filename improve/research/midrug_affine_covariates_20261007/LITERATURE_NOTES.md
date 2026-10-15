# 近期文献与原始信息覆盖复核

检索日期：2026-10-07。使用已读 nature-academic-search 的 multi-source-search 流程；当前无学术 MCP，直接请求 Europe PMC、Crossref 与出版/预印本原站。检索限定2025-01-01至2026-10-07。三组原始响应、查询URL和按DOI去重的73条记录已保存；它们并非73篇全部精读的相关论文。

| 文献 | 实際核对范围 | 对当前工作的含义 |
|---|---|---|
| Li et al., Router-guided dual attention with graph masked autoencoder for microRNA–drug association prediction. 2026. DOI: [10.1016/j.compbiolchem.2026.109236](https://doi.org/10.1016/j.compbiolchem.2026.109236) | Europe PMC摘要、Crossref题名及卷期；首发2026-07-13、卷期10月 | 路由与掩码重建已在先前纯miRNA实验覆盖；未新读全文，不以此作为新机制。 |
| Nie et al., Multi-view Graph Learning Framework with Spectral Encoding and Sparse Cross-Attention for miRNA-Drug Association Prediction. 2026. DOI: [10.1109/TCBBIO.2026.3729865](https://doi.org/10.1109/TCBBIO.2026.3729865) | Europe PMC摘要，2026-09-01首发；对照旧检索记录 | 原方法使用gene介导图和序列相似性，当前禁止增加这些外部输入；谱/扩散思路亦有先前覆盖。 |
| Tang et al., Dual-granularity feature learning reveals static and dynamic mechanisms of ncRNA-drug interactions. 2026. DOI: [10.1109/JBHI.2026.3717803](https://doi.org/10.1109/JBHI.2026.3717803) | Europe PMC摘要、Crossref题名/作者；首发2026-07-28 | 摘要称分离静态结合与动态调节，且分别报告lncRNA和miRNA任务。尚未获得可验证的方法细节，不把统计分支分离直接解释为因果机制，也不使用lncRNA输入。 |
| Souza & Mehta, Parameter-free representations outperform single-cell foundation models on downstream benchmarks. bioRxiv预印本，2026-02-13 v1. DOI: [10.64898/2026.02.11.705358](https://doi.org/10.64898/2026.02.11.705358) | Crossref、bioRxiv元数据API和原站全文HTML；重点读取引言、主要结果IV及方法A1–A4文字 | 可借鉴的是检查表达预处理和训练内特征选择。研究对象为单细胞分类等任务，不是miRNA–药物预测；不能移植其性能结论。 |

对最后一篇预印本，方法文字明确：每细胞归一化计数、log2(x+1)、在同一细胞内跨基因排名、将百分位映射为正态分数；部分分类任务再用训练数据ANOVA选特征、标准化和PCA。原始MiExp表不能假定为原始单细胞计数，因此不能照搬计数归一化、单细胞标签或参考图谱。可单独检查跨miRNA的细胞内秩表示，但必须明示是当前任务的适配，不称论文复现。

原站HTML正文已保存并部分阅读。JATS XML及百分位/逆CDF公式GIF请求返回HTTP429，记录在jats_fetch.json与equation_fetch.json；没有从不可见公式中猜测作者百分位边界或并列值规则。文本提取首次使用bs4失败（环境未安装），随后采用标准库HTMLParser；无需安装依赖。

本轮尺度恢复实验在读到这篇预印本具体方法前已冻结。不能事后称其为该论文的复现或该论文验证了恢复尺度有效。该实验直接来自输入审计：原938/768维均逐行min–max，纯miRNA已有模型未显式得到原最小值与极差。

代码覆盖检索：midrug*中的rankdata此前只出现在AUC审计；未发现表达输入的inverse-normal/QuantileTransformer/ndtri实现。原Drug_Name有216个唯一字符串；此前分组代码读取的是CID、SMILES和miRNA名称，未发现Drug_Name字符表示。药名字符相似性不能自动视作机制/靶点/药理类别。

下一项是否值得实施，要结合本轮完整三折结果及覆盖证据选择。任何新实验须另建目录，保留原训练划分与无lncRNA约束，原测试双+.02和PLMF不下降的要求不变。
