# 配对交换学习：新颖性核对

结论：这是一条可以检验的任务化机制，不能称为“从来没有人考虑过”。本次定向检索覆盖题名、元数据和部分摘要，未完成系统综述或全部全文比对。

## 最接近的已有思想

| 工作 | 已核实来源 | 已有内容 | 本轮不能声称的贡献 |
|---|---|---|---|
| Graham, An Econometric Model of Network Formation With Degree Heterogeneity | 2017, Econometrica, [DOI](https://doi.org/10.3982/ECTA12679)；[NBER 工作论文摘要](https://www.nber.org/papers/w20341) | tetrad logit 对节点异质性的充分统计量做条件化 | 四元组消除节点效应不是新发现 |
| BPR: Bayesian Personalized Ranking from Implicit Feedback | [arXiv:1205.2618](https://arxiv.org/abs/1205.2618)，2012 年上传记录及摘要；原方法更早发表于 UAI 2009 | 用隐式反馈进行成对排序，并结合矩阵分解 | 以未知配对做排序候选、用排序损失训练分解模型均已有先例 |
| Neural Collaborative Filtering | 2017, WWW, [DOI](https://doi.org/10.1145/3038912.3052569) | 神经网络与协同过滤交互评分 | 用神经网络替代简单配对打分不是新颖贡献 |
| Disentangling User Interest and Conformity for Recommendation with Causal Embedding (DICE) | 2021, WWW, [DOI](https://doi.org/10.1145/3442381.3449788)；已读取摘要 | 将兴趣和对热门项目的从众因素分开表示 | 分离偏好与热门程度这一方向也已有明确先例；本模型不继承 DICE 的因果表述 |
| 同任务中的 GSLRDA、AGCLNDA、MIFNDRA、HDBI | 见[上一轮核对](../expression_consensus_20260909/NOVELTY.md) | 自监督、可学习增强、多模态与解耦高阶关系 | 加入对比学习或给模块更换名称不能构成首次方法贡献 |

Graham 工作论文摘要明确写道：“The first, tetrad logit (TL), estimator conditions on a sufficient statistic for the degree heterogeneity.” 本轮的弱排序设定使用未标注交叉配对和 GNN 表示，不满足该统计论文的原设定，不能借用其一致性保证。

## 本轮实际检验的区别

1. 在现有 ncRNA–drug 图上显式限制配对交互的行列均值，将加性端点函数与交互项分开。
2. 用预测正边在正反两个方向均不可见的消息传递图，训练原解码器、分解解码器、普通排序和交换排序。
3. 交换排序对加性端点分数不敏感；同样四个分数用于普通排序对照，检验联合比较是否提供额外收益。
4. 不把未知配对称作实验阴性，不宣称“耐药/敏感方向”或“因果效应”；通过内部等预算调参、固定轮次重训和组内排序评价，区分机制收益与参数收益。

可能形成研究贡献的是上述任务假设及充分实验支持。即使检索没有命中完全相同的组合，也不能推导出它此前不存在。若交换项无法胜过无辅助目标和普通排序，应保留负结果，不能只用某个全局指标的局部上涨支撑机制叙事。

## 检索记录与限制

- 本环境没有挂载 academic-search MCP，使用公开 CrossRef、PubMed、arXiv API；三个接口均通过连通检查。
- 一条 CrossRef 宽泛查询返回 HTTP 429；随后补充 OpenAlex 和精确 DOI 查询。
- 关键词检索包含多条离题的生物医学、物理和逻辑学记录，保留原始记录，但没有用于证明新颖性或“查无先例”。
- 精确 DOI、arXiv 编号与工作论文页面的核对优先于宽泛检索排序。
- 原始材料：search_results.json、targeted_verification.json、prior_art_abstracts.json、graham_abstract.json。工作论文页面提取片段还含网页脚本文字，引用只取与 tetrad logit 相关的摘要句。
