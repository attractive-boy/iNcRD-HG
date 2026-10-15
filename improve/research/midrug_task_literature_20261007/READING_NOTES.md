# 近期miRNA–药物方法与训练案例检索的适用性

北京时间2026-10-07，nature-academic-search / multi-source-search。
学术MCP未挂载，使用公开Europe PMC、Crossref、arXiv和作者GitHub。
公共API预检3/3通过；四组查询65条发现记录，按DOI及题名/首作者归并为62条。
Europe PMC/arXiv检索2025年至2026-10-06；复用Crossref获取器保留2024参照。
这些是检索条目，不是62篇相关或已读全文。没有下载模型数据、注释或权重。

## 近期同领域文献

| 文献 | 核对范围 | 当前输入约束下的判断 |
|---|---|---|
| Nie等，[MVGSCA](https://doi.org/10.1109/TCBBIO.2026.3729865)，2026-09 | Europe PMC摘要，Crossref重复命中 | 基因中介图、miRNA序列和药物SMILES相似度、谱编码和稀疏跨注意力。含额外基因与序列输入，不能原样迁移；项目先前已经查过。 |
| Zhu、Zhang，[Prior-RDRGSE](https://doi.org/10.3390/computers15090615)，2026-09-14 | Crossref元数据，出版社公开JATS全文§3–5及数据可用性 | Bernoulli图后验、Gumbel采样、重建、语义注意力和先验一致性。人工描述、BioMedBERT和外部机制/文献先验不符合当前约束；原基准含多类ncRNA。不能使用其数据/权重，也不能把去掉主要先验的改写称为复现。 |
| Zhang，[DMAGT](https://arxiv.org/abs/2512.05287v1)，2025-12预印本 | arXiv元数据、公开全文§2–4.1 | miRBase序列、PubChem SMILES、CBOW、Laplacian位置编码及邻域Transformer。miRNA序列输入不允许新增；邻域注意力已有严格本地实验。不能把论文不同数据及平衡采样AUC用于本任务验收。 |
| Cao等，[MIRCID](https://arxiv.org/abs/2609.21280v2)，2026-09预印本 | arXiv摘要 | 从L1000基因表达推断Hub-miRNA并做药物机制任务；不是当前配对关联标签任务，所需基因输入和预训练推断器不在允许输入内。未据摘要宣称已读方法。 |
| Fu等，[GCFH-MDA](https://doi.org/10.1007/978-981-92-3485-1_9)，2026-07在线 | 此次Crossref；既有EXPRESSION_FOLLOWUP_NOTES中的出版社摘要 | 先前已确认额外gene网络和序列/结构视图，不作新候选。 |

mDADGAN明确使用lncRNA，排除。MSMP-GAE、DLP、PLMF-MDA及DLST-MDA也再次检出，
保留原摘要条目并与既有笔记查重，没有把它们计作新发现。DLP的2026卷期不能覆盖其
2024首次记录。没有从题名或单篇高指标推断可以在本数据上获得双+.02。

Prior-RDRGSE主站HTML/XML返回403，但出版社公开附件JATS成功，文件及哈希保存。
DMAGT全文成功。TabR首次请求了不存在的v3，404保留；元数据确认v2后取得正文。
没有请求机构凭证、联系作者或访问受限内容。

## 公式的适用边界

`formula_check.py/json`只检查原文公式的字面含义，不代表作者未公开代码的行为。

1. DMAGT式(4)先将attention logits逐元素乘邻接，再在式(5)softmax。
   当非邻接位置变成0时，其softmax质量仍大于0。例如logits=[2,1,4]、
   邻接=[1,1,0]时，第三项仍有约0.0900权重；若要严格只聚合邻居，应使用
   排除位置为负无穷的掩码。没有作者代码核验，不能据此断言论文程序实际泄漏。
   原文§2.3另有miRNA/药物的SMILES和核苷酸名称互换，不能据这句误述构造输入。
2. Prior-RDRGSE式(14)写成sum P_prior log(P_prior/Yhat)。若Yhat是逐边独立
   sigmoid概率，未在相同集合归一化，该表达式不能直接当作离散分布KL；
   P=[.5,.5]、Yhat=[.8,.8]给出负值。逐边Bernoulli KL需要补上(1−P)项。
   原文没有在该式明确归一化集合；因此不照抄该式为已验证的先验约束。
   §3.4的Bernoulli图变量、§3.6的normal/Dirichlet潜变量先验也需在实现前逐一
   明确对应关系，不能默认它们可以直接计算同一个KL。

这些检查没有使用MiDrug标签，也不构成对整篇论文实验真实性的评价。

## 可实施的较早方法参照：TabR

[TabR](https://arxiv.org/abs/2307.14338v2)是2023年首发的较早参照，不能称为2026新方法。
v2题名为“TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023”；作者仓库
README使用“TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning”。
本轮读§3.2步骤0–4、式(5)，及固定commit
`17baa9082506f8e7a0f8d11bb1e08212926a1507`的`bin/tabr.py`。

原方法在训练案例的学习key空间做top-m近邻检索，softmax分数为负平方L2距离，
没有额外维数缩放。值为`label_embedding(y_j)+T(key_q−key_j)`；T为两层MLP，
第二层无偏置。聚合结果加回查询的隐藏表示，再做预测。代码先离散选近邻，
随后重算距离和values以传递梯度；训练排除查询自身，其他训练标签可作为案例。
分类label_embedding初始uniform[-1,1]。这不需要外部预训练权重。

## 本地查重与下一步边界

先前`support_operator_modules.py`是药物条件化的可微岭回归；
`local_support_modules.py`已做配对相关top-k原属性软检索和有界Newton残差，
32次旧协议拟合未通过MiDrug双指标门槛。`tabicl_midrug_memory_modules.py`
是原属性无监督分组路由配合冻结TabICL，涉及旧混合图/权重，不可执行其加载器。
因此，不能声称“检索训练标签”是项目首次尝试。

尚需检验的具体差异是：在当前纯miRNA M123+unmasked10旁加入**直接学习标签
嵌入与key差值的非线性值聚合**，与旧闭式求解和有界预测残差不同。只用原始
miRNA/药物特征产生检索key；不能用含查询标签的图状态为其他案例生成key。
案例库只有当折训练对，正负查询自身都排除；固定top-k和编码维数，比较完整标签值、
移除标签值、无检索直接分支，才可判断是否是训练案例的实际增益。
置换全部案例标签可能把查询自己的标签移到其他案例，故不能直接用这种置换作
“无泄漏”对照。负标签只代表本任务的观测类别，不解释为药物敏感性。

本笔记提出可核验的适配方向，尚未创建检索模型或训练协议，不能报为训练结果。
下一步先做实现/信息边界检查再冻结正式训练。最终原测试双+.02目标仍未达到。
