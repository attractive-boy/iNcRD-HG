# RNA → 下游基因 → 通路：补齐方案

日期：2026-09-13。范围：复用本地身份注释、核对公开接口、制定数据契约。本次未完成全体 RNA 的靶基因下载，未进行新模型训练。

## 要补的关系

当前每条 RNA 的 938 维输入是跨细胞系表达。接入通路知识需要另建“RNA 调控或结合哪些下游基因”的关系表，再连接基因的通路注释。把 lncRNA 名称转换成它自身的 Ensembl gene ID 只完成身份对齐，不能替代下游基因关系。

由靶基因集合得到的是 RNA 的通路关联表征。若要估计某一细胞系中的通路活性，还需要该细胞系的 mRNA 表达或合适的扰动数据；938 维 RNA 表达不能直接冒充基因表达输入。

## 可直接复用的身份数据

已生成 [rna_identity.tsv](rna_identity.tsv)，与原实体 CSV 的行序逐项核对一致，共 1,927 行。`rna_node_index_0based` 明确使用从 0 开始的行索引。

| 实体 | 本地总数 | 已有身份 | 尚需处理 |
|---|---:|---|---|
| miRNA | 605 | 605 个成熟链 MIMAT 编号 | 其中 10 条仅有历史版记录，查询靶库时保留历史状态 |
| lncRNA | 1,322 | 1,292 个 GENCODE Ensembl gene ID | 30 条待做 HGNC/Ensembl 别名与基因类型核对 |

miRNA 保留 5p/3p 区分，不能把 precursor 与 mature 混用。lncRNA 保留原始带版本 gene ID，同时提供去掉版本后缀的连接键；需要记录 GENCODE 与下游注释版本。歧义别名不能随意选一个基因。

身份来源为 `improve/results/tabicl_reli_inputs_20260912/{MiDrug,LncDrug}_sequences.json`。这些数字均为身份覆盖，尚不是靶基因或通路覆盖。

## 数据源与证据含义

| 要补的边 | 第一批候选来源 | 使用规则 |
|---|---|---|
| miRNA → mRNA/基因 | miRTarBase 2025；ENCORI 作结合证据补充 | 限定人类、对应成熟链；功能实验、CLIP、降解证据、计算预测保留各自类型 |
| lncRNA → 下游基因 | LncTarD 2.0 | 区分功能调控与表达关联，区分直接与间接机制；疾病中上/下调不能直接作为对靶基因的作用方向 |
| lncRNA ↔ mRNA | ENCORI RNA–RNA | 属于相互作用证据，不能自动解释为促进或抑制 |
| lncRNA → miRNA → 基因 | 本地 LncMiRNA_interaction.csv 与 miRNA 靶基因表 | 仅作为间接候选关系，保留路径；不能仅凭两条边声称已验证 ceRNA 机制 |
| 基因 → 通路 | Reactome 人类注释 | 保留基因 ID、通路稳定 ID、物种、注释版本；先检查实际下载字段及 ID 类型 |

所有下游基因统一到人类 Ensembl gene ID；保留来源 ID、symbol、映射状态。若采用 RNA–protein 或药物–protein 数据，另做 UniProt → gene 映射，保留蛋白复合物成员。

缺少证据的 RNA 保留缺失标记。计算预测可作为独立候选通道，不填成“已验证靶基因”。证据权重属于建模选择，未经校准不能称为真实概率。

## 实际访问核查

- ENCORI miRNA API 可访问。本次 `hsa-miR-21-5p` 查询返回 1,820 条位点记录；在客户端重新限定 `MIMAT0000076`、`protein_coding`、`clipExpNum >= 1` 后，保留 1,289 条位点记录，对应 1,098 个唯一基因。这是一个 miRNA 的结合候选示例，不能外推到 605 条 miRNA，也不是 1,098 条已证实功能调控。
- 可见真实示例：本地 `HSA-MIR-21-5P` → `MIMAT0000076` → `PDCD4 / ENSG00000150593`。来源记录包含 CLIP 支持；功能方向需其他证据确认。
- 查询参数请求 `clipExpNum=1`，原响应仍有 531 条 `clipExpNum=0` 记录，因此生产流程必须在本地复核阈值，不能仅信任 URL 参数。
- ENCORI 的 NEAT1 RNA–RNA 接口返回了配对 Ensembl ID，但本次约 2 MB 后停止读取，只保存预览，不参与完整数量统计。
- miRTarBase 官网本次 TLS 握手失败；LncTarD 主站返回 403，另一公开入口有证书域名不匹配。尚未获得它们的完整靶基因表。未绕过站点验证或关闭证书检查。
- 原工程 `plot_case_drug.py:95` 已有读取 ENCORI 的后处理代码，但其引用的两个原始数据文件当前不在本地。`TDMDScore > 0.8` 筛的是 TDMD 候选，不能沿用为经典 miRNA 抑制靶点的通用置信阈值。

原响应、URL、状态及哈希见 [access_probes.json](access_probes.json)；整理后的身份与探针统计见 [PREPARATION_AUDIT.json](PREPARATION_AUDIT.json)。

## 产物契约与接入

目标边表 `rna_to_gene.tsv` 至少包括：

```text
task, rna_node_index_0based, rna_name, rna_stable_id,
target_gene_id, target_gene_symbol, relation_type, evidence_type,
source_evidence_label, effect_direction, context, PMID,
source, source_version, source_record_id, inference_path
```

`effect_direction` 缺证据时为 unknown。同一论文、位点或数据库间重复收录不能累计成多份独立支持。原始证据保留在明细表，特征生成时再按 RNA–gene–关系类型去重。

目标通路表 `gene_to_pathway.tsv` 至少包括 `gene_id, pathway_id, pathway_name, species, source_version`。分别构建 miRNA 的 605×G、lncRNA 的 1322×G 稀疏关系矩阵，以及同一份 G×P 基因–通路矩阵。

对每种证据类型分别生成通路支持向量，并处理靶基因数、通路大小与高连接基因造成的偏置；不能简单把命中数称为显著性或通路活性。先使用简单的二值去重与预先规定的归一化规则做信息增量对照。

938 维表达继续作为原表达模态；通路支持向量和缺失标记作为新增模态，通过两任务共享的表征/融合模块接入。已有 ChEMBL 药物机制靶点可连接到相同的基因–通路字典，用于后续 RNA 与药物程序匹配。

先统计身份、靶基因、通路三层覆盖率；分证据类型报告每条 RNA 的靶基因数、缺失及歧义。随后与强图上下文 TabICLv2 做同一评估协议下的增量对照，LncDrug AUC 和 AP 均不下降为接受条件，按药物宏 AP 另报。任何模态或训练代理都不能预先保证真实指标不降。

## 来源边界

miRTarBase 2025 的药物耐药模块整合了 ncRNADrug。此次只获取 RNA–gene 证据，不把 RNA–drug 关联重新输入特征；保留文献来源，以便按实际时间/分组评估检查证据边界。

- [miRTarBase 2025 原文](https://doi.org/10.1093/nar/gkae1072)：功能实验与高通量证据，以及 ncRNADrug 来源说明。
- [LncTarD 2.0 原文](https://doi.org/10.1093/nar/gkac984)：5,531 条功能调控与 2,829 条表达关联，不能合并声称为直接功能调控。
- [ENCORI 官方 API 教程](https://rnasysu.com/encori/tutorialAPI.php)。
- [Reactome 官方下载入口](https://reactome.org/download-data)：本次未下载全量通路注释，作为后续接入来源。
