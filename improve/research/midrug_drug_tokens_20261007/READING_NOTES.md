# 近期文献与可用迁移边界

检索时间2026-10-07。使用nature-academic-search多源流程；无学术MCP，直接公开API。
CrossRef三条检索（2024-01-01至2026-10-07），其中drug-target查询HTTP429；另外两条
共24条结果。Europe PMC另两条查询返回40条结果；只代表有限候选集，不是系统综述。
原始响应、URL和错误在literature目录。包含lncRNA/circRNA的检索命中不用于模型数据。

1. Qin C et al. **CMA-DTI: a cross-modal fusion and attentive interaction network for
   interpretable drug-target interaction prediction.** Frontiers in Bioinformatics (2026),
   2026-06-11，DOI https://doi.org/10.3389/fbinf.2026.1861685 ，PMCID PMC13294260。
   获取合法OA全文XML，部分精读引言、Drug Representation Module、Intra-drug Cross-modal
   Fusion及Drug-Protein Interaction（式5–8及周围定义）。论文以GCN药物节点为query，
   ChemBERTa token为key/value；再对蛋白ESM-2残基做注意力。原文明确不假定原子与SMILES
   token一一对应。启发：延后药物token池化，让交互对象影响局部汇总。此处没有GCN原子
   query和蛋白残基，改用原miRNA表达向量；不能称为该论文复现或物理结合机制预测。

2. Wang B et al. **MAGC-DTI: modality-shared space and adaptive gated interactive
   cross-attention for drug–target interaction prediction.** Scientific Reports (2026),
   2026-06-04，DOI https://doi.org/10.1038/s41598-026-50141-w ，PMCID PMC13473133。
   OA XML部分精读AGICA、MPRC、划分和discussion。该方法使用共享空间、级联双向注意力、
   自适应门控和残差分类；本轮没有照搬这些模块。可借鉴的控制思路：保留共同全局配对
   交互，再检验局部注意力贡献。论文的Human偏倚讨论和自身解释性限制也表明，高注意力
   不能直接当作物理结合位点。本轮不采用其数据、蛋白序列、模型参数或划分。

另外发现2026年miRNA-drug专门论文（路由注意力+图掩码自编码器、谱编码+稀疏交叉注意力），
仅获取元数据/摘要，未声称全文读过。此前相关路由、重建和表达列注意力已经实验覆盖；
不因新名称重复同一机制。检索命中标题里的“冷配对”也不能代替核对实际实体划分。

## 与已做工作区别

midrug_chemberta_adaptation_20261006：最后层Q/V LoRA及等参数池化适配均先masked mean。
midrug_cell_tokens_20261007：用药物整行表示选择miRNA表达列，本轮用miRNA整行表示选择药物token。
midrug_cell_qknorm_20261007：上述表达列注意力归一化变体，非本轮方向。
旧mirna_general_architecture_20260912提案已有局部表示讨论，故不声称全局新颖。
不执行旧提案外部RNA编码器或混合RNA加载器。

冻结前的主假设仅为“全局池化可能丢掉有用的条件信息”，并不预期或承诺+.02。
学习目标仍是miRNA–药物关联，不是药物–蛋白结合。新分支只是可检验的迁移假设。
