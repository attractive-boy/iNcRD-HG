# iNcRD-HG 精读与 ncRNA–药物耐药领域改进方案

> 目标论文：*Semantic-enhanced heterogeneous graph learning for identifying ncRNAs associated with drug resistance*（Bioinformatics, 2026）

## 1. 论文核心内容

### 1.1 任务定义

iNcRD-HG 预测的是 **ncRNA–药物耐药/敏感性关联**，不是直接预测药物响应值，也不直接推断因果关系。输入为一个包含 lncRNA、miRNA、drug 的异质图，输出某 ncRNA–drug pair 的关联概率。

论文使用的数据主要包括：

- ncRNADrug 中实验验证的耐药关联；
- CCLE 中 938 个癌症细胞系的 miRNA/lncRNA 表达谱；
- PubChem/SMILES 与 ChemBERTa 药物结构表示；
- lncRNA–miRNA 交互网络；
- lncRNA/lncRNA、miRNA/miRNA、drug/drug 相似性网络。

预处理后包含 1322 个 lncRNA、605 个 miRNA、216 个药物、2109 条 lncRNA–drug 耐药关联和 5394 条 miRNA–drug 耐药关联。虽然论文称 pan-ncRNA，但当前版本 **没有 circRNA**；作者也在结论中把 circRNA 数据列为未来扩展。

### 1.2 模型结构

模型由两部分组成：

1. **异质图构建**
   - 六类边：lncRNA–lncRNA、miRNA–miRNA、drug–drug、lncRNA–miRNA、lncRNA–drug、miRNA–drug；
   - ncRNA 初始特征来自表达谱；
   - drug 初始特征来自 ChemBERTa。

2. **图学习与解码**
   - 使用 PyG `to_hetero` 包装的 `GeneralConv`，对不同关系分别学习参数；
   - 两层图编码器得到 ncRNA 和 drug embedding；
   - 解码器拼接 ncRNA 与 drug embedding，经三层 MLP 输出关联分数；
   - 训练目标为 binary cross-entropy。

论文报告的主要结果是 LncDrug AUC 0.9857 / AUPR 0.9585，MiDrug AUC 0.9564 / AUPR 0.9165，均优于 GAT、GraphSAGE、GCMDR、DeepLDA、AGCLNDA、NDSGCL 等基线。

## 2. 关键问题

### 2.1 评估协议存在药物流行度捷径

代码使用 `T.RandomLinkSplit(..., neg_sampling_ratio=2.0)` 生成负样本，见 `heter_data_prepare.py:227`。正样本高度集中在少数药物上，而负样本近似均匀采样，导致模型可以主要学习“这个药物是否热门”，而不是学习特定 ncRNA 的生物学关联。

已有复现实验显示：

- 只使用药物训练度、完全忽略 ncRNA 的控制模型，在 LncDrug 上达到 AUC 0.9775 / AUPR 0.9513，接近论文模型；
- 模型分数与药物度数的 Spearman 相关在 LncDrug/MiDrug 上分别为 0.837 和 0.877；
- 采用同药物匹配负样本后，度数控制模型 AUC 变为 0.500；
- 在去偏评估下，原论文的方法排序不再稳定，多个冷启动单元被 GCN 等简单模型超过。

详细结果见 `analysis/FINDINGS.md:7` 和 `analysis/FINDINGS.md:37`。

### 2.2 冷启动实验可能有跨任务边泄漏

`RandomLinkSplit` 只切分当前任务边。训练 LncDrug 时，MiDrug 边保持完整；训练 MiDrug 时，LncDrug 边保持完整。对单任务评估而言，目标药物在另一种 ncRNA 任务中的已知关联仍会通过图传播到 drug 节点，再经 lncRNA–miRNA 边影响目标预测。

在 leave-one-drug-out 实验中，代码只移除当前 edge type 中目标药物的边，见 `predict_simulated_drug.py:189`。如果评估的是“全新药物”，更严格的设置应至少提供两档：

- **单任务冷启动**：移除所有 ncRNA 类型中指向该药物的耐药关联；
- **pan-ncRNA 冷启动**：明确允许其他 ncRNA 类型边存在，并使用多任务训练目标。

### 2.3 模型选择存在测试集使用风险

主训练脚本每个 epoch 都评估 test set，但未真正使用 validation set 选择模型，见 `predict_association.py:180`。冷启动脚本进一步直接用 test 的 Precision@30 与 NDCG@30 更新最优结果，见 `predict_simulated_drug.py:288`。这会造成测试集选择偏差。

正确做法是：训练中只看 train/val，早停和超参选择只依赖 val，最终在 test 上一次性评估。

### 2.4 论文与代码的相似度阈值不一致

论文正文写相似性边阈值大于 0.5，但代码中 lncRNA、miRNA、drug 相似网络均使用 0.3，见 `heter_data_prepare.py:45`。这会影响网络密度、消息传递路径和结果可比性，应统一论文、代码与配置文件。

### 2.5 “语义增强”的实际表达能力有限

`GeneralConv` 通过异质图包装实现了关系特定参数，但模型没有显式的 edge-type embedding、关系注意力、元路径实例、路径约束或跨模态门控。相较 HGT、R-GCN、HAN、meta-path GNN、超图和图 Transformer，其关系语义建模仍偏基础。

### 2.6 解释性主要是后验归因

Captum/IntegratedGradients 能说明模型使用了哪些节点和边，但不能证明生物学因果。当前案例研究中的 PubMed 证据也可能与训练数据存在同源风险。解释性需要补充稳定性、忠实性、反事实删除实验和时间分割外部验证。

## 3. 领域文献脉络

### 3.1 数据库与基准

| 工作 | 年份 | 核心贡献 | 对改进的启示 |
|---|---:|---|---|
| NoncoRNA | 2020 | 实验支持的 ncRNA–drug–cancer 关联数据库 | 引入 cancer/disease 节点，区分肿瘤类型 |
| ncRNADrug | 2024 | 29,551 条验证耐药关联，覆盖 miRNA、lncRNA、circRNA | 当前论文只用 lncRNA/miRNA，应扩展 circRNA 与新版数据 |
| SGTCDA | 2024 | 强调 5/10 折 CV 过乐观，使用独立测试集 | 应建立 time-split/independent test benchmark |

### 3.2 泛 ncRNA 耐药预测

| 工作 | 年份 | 方法要点 | 对改进的启示 |
|---|---:|---|---|
| GSLRDA | 2022 | LightGCN + 自监督对比学习 | 利用自监督缓解稀疏性 |
| NDSGCL | 2024 | 局部结构邻居 + 全局语义邻居对比学习 | 增加语义邻居与多视图增强 |
| RDRGSE | 2023 | 图骨架提取 + 注意力融合 | 显式去噪边，降低噪声相似性影响 |
| DMGAT | 2025 | diffusion map + 异质 GAT + 随机森林可靠负样本 | 负样本不应只是均匀随机 |
| AGCLNDA | 2025 | 可学习增强视图 + 去噪视图 | 增强视图应与度数去偏结合 |
| MIFNDRA | 2026 | GIN + SpliceBERT + ncRNA 交互网络 + GraphSAGE/对比学习 | 序列基础模型与多类型 ncRNA 交互结合 |
| NCRDLLM | 2026 | RNA-FM/ChemBERTa/结构特征映射到 LLaMA，LoRA 微调 | 可测试基础模型，但需强基线和算力收益证明 |
| HDBI | 2026 | 高阶动态解耦 + 双向跨模态更新 | 分离稳定语义与流行度/噪声因子 |
| DC-MetaMG | 2026 | 因果解耦，静态结合与动态调控两种机制 | 用因果表示抑制虚假关联 |

### 3.3 miRNA–药物方向

| 工作 | 年份 | 方法要点 | 对改进的启示 |
|---|---:|---|---|
| GCMDR | 2020 | 图卷积 + miRNA 表达 + 药物指纹 | 作为经典强基线 |
| AMMGC | 2021 | 多模态图卷积 + 注意力融合 | 门控融合优于简单拼接 |
| NASMDR | 2022 | NAS + GIN，用属性替代交互图 | 冷启动时应更依赖属性 |
| GAM-MDR | 2024 | 图自编码 + 随机路径掩码 | 用路径扰动提升鲁棒性 |
| MPHGNN | 2026 | miRNA–gene–drug 元路径 GNN | 加入 gene/pathway 中介节点 |
| PLMF-MDA | 2026 | 预训练语言模型 + 跨模态注意力 | 预训练特征可提升冷启动 |
| mDADGAN | 2026 | miRNA–lncRNA–drug 异质图 + diffusion GAN + cost-sensitive loss | 多任务/跨类型 ncRNA 图与代价敏感训练 |

### 3.4 lncRNA–药物方向

| 工作 | 年份 | 方法要点 | 对改进的启示 |
|---|---:|---|---|
| DeepLDA | 2023 | lncRNA–drug 深度学习与注意力机制 | 作为 lncRNA 基线 |
| HGNNLDA | 2023 | 双通道超图神经网络 | 高阶关联适合超图建模 |

### 3.5 circRNA–药物方向

| 工作 | 年份 | 方法要点 | 对改进的启示 |
|---|---:|---|---|
| AutoEdge-CCP | 2024 | circRNA–cancer–drug 多关联异质网络 + autoGNN | cancer-centered 三元组建模 |
| MAGSDMF | 2024 | 多视图多注意力 + 深度矩阵分解 | 处理假阴性/稀疏问题 |
| SGTCDA | 2024 | SDNE + graph transformer + EdgeSHAPer | 长程依赖、独立测试和边解释 |
| GraphCDD | 2025 | circRNA–disease–drug 多模态图 | 疾病/肿瘤上下文直接进入图 |
| DeepHeteroCDA | 2025 | 多尺度异质网络 + GAT/GCN | 分子结构子图与关联图联合训练 |
| MiGNN2CDS | 2025 | 多实例学习 + 伪元路径 + BiTrans | 路径级解释更贴近机制 |
| AGCECDA | 2026 | 注意力异质图协同嵌入 + Top-K 稀疏邻接 | 稀疏图可降低噪声传播 |
| CDBGT | 2026 | 二部图 Transformer + RNA-FM/ChemBERTa + 度/谱位置编码 | 预训练模态与拓扑编码结合 |
| HECLCDA | 2026 | 异质跨尺度对比学习 | 跨尺度自监督缓解标注稀疏 |

## 4. 推荐改进方案

### 4.1 第一优先级：先修正 benchmark，再改模型

这是最重要的改进，因为当前高分很可能被药物流行度解释。建议建立 **NcRD-Bench v2**：

> 当前仓库已实现基础版：`analysis/ncrd_bench.py` 生成
> `matched_drug`、`matched_rna`、`matched_both`、`cold_rna`、`cold_drug`
> 和 `cold_pair`；`analysis/audit_bench.py` 自动检查 held-out 边泄漏、
> 负样本撞正例、匹配边际和冷药物跨任务泄漏；`analysis/run_bench.py`
> 提供 drug degree、ncRNA degree 和双端度数控制基线。

1. **度数匹配负样本**
   - 每个正样本 `(ncRNA, drug, cancer)` 选择同 drug、同/近似 cancer、同 ncRNA 度数档位的未观察 pair；
   - 同时报告 drug-matched、ncRNA-matched、drug+ncRNA 双匹配三档；
   - 保留随机负样本作为参考，但不能作为主结果。

2. **分层 split**
   - random；
   - cold ncRNA；
   - cold drug；
   - cold pair；
   - cancer/cell-line holdout；
   - literature time split：用某日期前证据训练，日期后证据测试。

3. **度数控制基线**
   - drug degree only；
   - ncRNA degree only；
   - degree + attribute logistic regression；
   - matrix completion；
   - GCN/GAT/GraphSAGE/GeneralConv；
   - 最新 MIFNDRA/DMGAT/AGCLNDA 等方法。

4. **指标**
   - per-drug macro AUC/AUPR；
   - within-drug Recall@k、NDCG@k、Precision@k；
   - global AUC/AUPR 单独报告，不作为唯一 headline；
   - coverage、calibration、uncertainty；
   - 10 个以上 seed，报告均值、标准差、置信区间和 paired significance test。

5. **泄漏控制**
   - train/val/test 目标边从 message passing 图中删除；
   - 对单向图和 reverse edge 均删除；
   - 单任务冷启动时移除所有任务中目标药物相关耐药边；
   - 相似性网络只能使用训练期可见信息，不能由全量关联矩阵计算；
   - 固定 split ID、数据版本、负样本 ID 和随机种子。

### 4.2 第二优先级：扩展数据与标签语义

当前数据只有二值关联，缺少上下文和方向。建议扩展为：

1. **ncRNA 类型**
   - miRNA、lncRNA、circRNA、必要时加入 snoRNA/piRNA；
   - 每个节点加入 `ncRNA_type` embedding；
   - 不同类型使用独立 encoder 和共享 projection。

2. **肿瘤/细胞上下文**
   - 增加 cancer type、cell line、tissue、disease 节点；
   - 将 `(ncRNA, drug, cancer)` 作为三元组而非全局 pair；
   - 对同一 ncRNA–drug 在不同肿瘤中的相反效应分开建模。

3. **标签粒度**
   - resistance / sensitivity / target / regulation；
   - 上调/下调、过表达/敲低、恢复敏感性的实验方向；
   - 证据等级、实验类型、文献年份、效应量；
   - 将未知样本视为 unlabeled，而不是确定负类。

4. **模态**
   - ncRNA：序列、二级结构、RNA family、表达、subcellular localization、exosome 标记、m6A/m5C 等修饰；
   - circRNA：back-splice junction、host gene、circBase/circAtlas ID、环化长度、miRNA sponge；
   - drug：SMILES、分子图、target、ATC、mechanism of action、pathway、副作用；
   - context：CCLE/GDSC/CTRP/PRISM/DepMap 的表达与 drug response，TCGA/单细胞数据中的肿瘤上下文。

### 4.3 第三优先级：模型 v2

建议采用 **上下文感知、度数去偏、多类型 ncRNA 的多任务异质超图模型**：

1. **图结构**
   - 节点：ncRNA(type-aware)、drug、gene、pathway、cancer/cell line；
   - 边：ncRNA–drug、ncRNA–ncRNA、ncRNA–gene、gene–pathway、drug–target、drug–drug、drug–cancer、ncRNA–cancer；
   - 对多对多调控模块使用超边，而不是全部拆成二部边。

2. **编码器**
   - RNA 侧：RNA-FM/SpliceBERT/RNA sequence encoder + expression/structure encoder；
   - drug 侧：ChemBERTa/MolFormer + molecular graph GIN；
   - 关系侧：R-GCN/HGT/HAN 或 relation-aware Transformer；
   - 使用 meta-path 实例或 path-level attention 建模 `ncRNA–ncRNA–drug`、`ncRNA–gene–pathway–drug`。

3. **训练目标**
   - within-drug pairwise BPR/ranking loss：优先学习同一药物内部哪个 ncRNA 更可能耐药；
   - PU learning 或 reliable negative sampling；
   - 多任务 uncertainty weighting / GradNorm / PCGrad，避免 lncRNA、miRNA、circRNA 任务干扰；
   - contrastive loss 只对同类型、同上下文、度数相近的样本做强约束；
   - 对不同证据等级加权。

4. **去偏**
   - batch 内按 drug degree 分层采样；
   - 对 drug degree 做 adversarial removal 或 gradient reversal；
   - inverse propensity weighting；
   - 加入 counterfactual loss：改变 ncRNA 类型或替换相似药物时，预测应发生合理变化，而 drug degree 不应主导输出。

5. **不确定性**
   - deep ensemble 或 MC dropout；
   - conformal prediction 给出候选集与覆盖率；
   - 对 cold-start 样本输出 abstain，而不是强行高分。

### 4.4 第四优先级：解释性升级

1. **解释单位**
   - 从 node/edge attribution 升级为 meta-path/pathway explanation；
   - 对 top prediction 输出完整路径：`circRNA → miRNA → target gene → pathway → drug`。

2. **反事实验证**
   - 删除候选 ncRNA–miRNA 边、替换相似药物、替换 cancer context；
   - 记录 score 变化是否与预期机制一致；
   - 与 Captum attribution 比较，避免只报告事后归因。

3. **解释质量指标**
   - faithfulness：删除重要子图后预测下降幅度；
   - stability：同一样本扰动后解释是否稳定；
   - sparsity：解释子图大小；
   - external recall：解释路径中的基因/通路是否能被时间分割后的文献或 KEGG/Reactome 支持。

4. **案例研究规范**
   - 只使用训练截止日期后的证据评估 top-k；
   - 与 degree-matched random candidates 比较；
   - 报告 top-k 支持率和置信区间；
   - 明确区分“训练集中已知”“外部文献支持”“完全未验证”。

## 5. 一个可行的最小改造成果

如果要快速形成新工作，可以聚焦以下贡献：

1. **NcRD-Bench v2**：度数匹配负样本 + 冷启动 + 时间分割 + 泄漏控制的统一 benchmark；
2. **CircRNA inclusion**：加入 circRNA，使模型真正覆盖 miRNA/lncRNA/circRNA；
3. **Context-aware multi-task graph**：`(ncRNA, drug, cancer)` 三元组建模；
4. **Popularity-debiased ranking objective**：within-drug BPR + degree adversarial/重加权；
5. **Path-level interpretability**：meta-path explanation + 反事实删除；
6. **External validation**：使用 ncRNADrug/NoncoRNA 更新日期后的记录和独立细胞系 drug response 数据。

这个组合的 novelty 不只是“换一个 GNN”，而是解决当前领域普遍存在的评估偏差、上下文缺失和 circRNA 覆盖不足。

## 6. 结论

iNcRD-HG 的异质图和 ncRNA 交互建模方向是合理的，但当前评估协议容易被药物度数捷径主导；同时模型没有 circRNA，缺少肿瘤上下文，解释性偏后验。最有价值的改进不是盲目增加复杂模块，而是先建立去偏和防泄漏 benchmark，再在该 benchmark 上引入 circRNA、多任务上下文建模、可靠负样本和路径级解释。
