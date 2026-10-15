# 源条件观测组合与互补性文献核对

检索日期2026-10-06；继续使用 nature-academic-search / multi-source-search。
学术MCP未挂载；公开PubMed、Crossref、arXiv连通性检查3/3通过，直接API检索。
只查文献，没有加载其数据集、注释或预训练权重。

## 本轮实际检索与阅读

arXiv以2025–2026年 link prediction 与 negative/conditional/source 搜索，返回25条；
Crossref相应主题返回15条；Europe PMC的miRNA/drug/negative sampling主题返回15条。
各源原响应与完整查询见 source_observed_search.json、raw/source_observed_*。
55是返回记录数，规范化标题合并后为54项，保留全部来源和标识符于
source_observed_triage.json；这不是54篇已阅读文献。Europe PMC结果多为实验生物学、
miRNA疾病或包含lncRNA的其他研究，不作为当前方法候选，也不引入其中数据。

重点核读 Guillaume Méroué、Fabien Gandon、Pierre Monnin：
[Oracle, will I ever learn? A study of prediction convergence and complementarity across link prediction models](https://arxiv.org/abs/2609.02638v1)。
arXiv元数据给出2026-09-02，按预印本引用，不推断会议录用。
已读取公开全文摘要、§3的Definition4、§5有关互补性/饱和的论述、§6讨论与§7结论。

Definition4的oracle是在每个知识图谱补全查询上选择某模型已有的最佳真实实体排名。
这提供了**在给定模型集合中逐查询选择已有排名**的理想参照，不能扩大为任意模型
组合的普遍上界。尤其原文§6明确承认：投票或分数加权可能产生所有单个模型都没有
给出的正确排名，从而超过其选择型oracle。这一点比摘要中笼统的upper bound更精确。
论文没有证明当前MiDrug存在不可学习标签，也不证明当前+.02目标不可能。
MRR/Hit@1的知识图谱实验不可换算为本任务的全局AUC/AP。

其有用启发是检验“新增分支究竟改变了哪些预测、是否提供互补信号”，不把换分支
名称、单次最好成绩或增加随机种子本身视为进展。当前实验在冻结选型后检查移除
额外观测消息/整个源投影的预测变化；这些只是干预诊断，不按干预结果重新选型。
此前56个固定概率向量的混合证书仍仅覆盖证书中列出的集合；不扩大到本轮新模型。

[Structural Alignment in Link Prediction](https://arxiv.org/abs/2505.04939v1)
为2025-05的学位论文记录。已阅读检索摘要；公开HTML虽已下载，但没有逐节精读，
不根据它提出新的具体实现或效果主张。
CGLE、Norm Augmented GAE等也检出；仅摘要层面筛选，其中CGLE类先验已有本地试验。

## 对本轮组合检验的影响

新实验不是上述Oracle论文的复现。它是在已有三层完整药物源传播中增加原训练零
观测关系，检验“源标记×观测关系”的交互。此前普通节点编码器中的观测上下文、
符号路径、cavity和profile结果仍作为负证据保留，不能把零标签传播声称为新家族。
此前GegenNet/CMP/BioPathNet阅读的任务差异与数据边界不变；本轮不再把重复引用
计为新文献支持，也不将抽样零标签解释为生物学上的负作用。

四个配置及三折规则在结果出现前冻结。仅当零观测版本对纯M123、unmasked10、
对应旧source和新增重复正例对照均有双指标≥.002增益，才推进原验证；最终仍须
原测试三种子AUC/AP同时绝对+.02，且PLMF不下降。
