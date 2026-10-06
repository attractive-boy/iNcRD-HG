**这 14 个方向仍有提升空间。当前最值得投入的是功能证据、局部 RNA 功能、条件化效应与标注偏差；常规网络替换的研究优先级较低。**

检索截止：2026-09-13。对象是 iNcRD-HG 的 ncRNA–drug resistance association prediction。以下区分研究价值与当前基准提分，优先级不代表预期 AUC/AP 增益，也不构成首创性证明。

本轮先复核项目代码与历史实验，再进行 OpenAlex 发现检索、8 组 Europe PMC 检索和 Crossref DOI 核验。Europe PMC 返回 155 条记录，对应 153 个不同 DOI；这不是穷尽检索。定向核验共 26 篇，包含 22 篇期刊论文和 4 篇预印本。正式版与预印本同时存在时优先引用正式版，例如 RIME。日期优先采用出版社在线出版日期，避免把卷期日期误写为首次发表日期。

原始记录保存在 [search_manifest.json](search_manifest.json)、[supplemental_verification.json](supplemental_verification.json)、[functional_context_verification.json](functional_context_verification.json) 和 [references.json](references.json)。arXiv 批量查询返回 HTTP 429，未采用其结果。CRISP 出版社页面只有摘要与预览；CoCoBind 的 PubMed 网页返回 cookies 检查，相关结论采用已取得的 Europe PMC/Crossref 摘要。PARNET 的 v1 公开全文成功取得，并阅读了功能迁移、Discussion、限制与未来方向部分。其余具体阅读范围列在文末。

**当前项目的起点。** 原始实现为两层 GeneralConv、拼接配对解码，LncDrug 与 MiDrug 分别训练。项目已尝试边权/top-k、PCA/AE、RNA-FM、HGT、多种配对交互及自监督方法；最新开发版本为冻结 TabICLv2 加图上下文与分层 RNA–gene–pathway 证据。因此这些方向的常规版本不能当成尚未探索的空白。

最新[通路实验记录](../../TABICL_PATHWAY_CONTEXT_FINDINGS.md)中，MiDrug AP 从 0.925057 变为 0.926064，LncDrug AP 从 0.968844 变为 0.971320，分别增加约 0.101 和 0.248 个百分点。这是重复使用的历史验证划分上的开发结果，三个种子反映推理随机性，不能解释为独立测试或显著泛化优势。药物基因注释覆盖为 93/216；RNA 功能证据覆盖为 miRNA 501/605、lncRNA 317/1322。缺失不等于没有功能。

|原方向|建议优先级|更值得研究的具体增量|主要依据|
|---|---|---|---|
|④ 边加权／正负／top-k|中；固定 top-k 调参低|边保留来源、证据等级、方向和条件；分别编码结合、调控和表达相关，研究置信度与稀疏化|SIGMA-KG/FLASH 预印本；已有本地加权图实验|
|⑤ mRNA／protein／pathway 节点|**高，近期主线**|保留具体 RNA 调控基因、药物 target/action，增加基因依赖性或药物扰动证据，检验功能路径的配对作用|MPHGNN 已做 gene 中介；FORGE、DiSPA 提供下一步依据|
|⑥ 癌种／细胞系上下文|**高；数据成本较高**|输入每条记录对应的细胞状态、癌种、剂量、时间；评估未见背景迁移|CRISP、PertDiff、scRADAR；2026 扰动模型评测|
|⑦ 表达降噪压缩|中，基础环节|在训练折内拟合压缩，审计批次与癌种组成影响；保留有用表达尺度并比较简单基线|OmicsTransformer；本地 PCA/AE 已有记录|
|⑧ 序列／结构特征|**高，优先功能与局部区域**|miRNA–靶位点抑制先验；lncRNA–RNA/RBP 功能区域与定位；预测结构保留置信度|miRBind2、RIME、PARNET、NucleicBERT|
|⑨ drug 多视图与对齐|中高|结构、靶点作用、处理后表达变化互补；条件与模态缺失显式进入表示|DiSPA、scRADAR、PertDiff|
|⑩ HAN/HGT 融合|基础版本低；具体交互中高|先确定需要交换的功能证据，再选择图算子；与同信息的简单模型公平比较|MVGSCA、HDBI 已覆盖复杂融合；本地 HGT 已试|
|⑪ 自监督／对比|中；目标设计关键|从可靠功能证据做预训练，处理缺失模态；核验哪些对象应对齐、哪些可作负例|MIFNDRA、OmicsTransformer；PARNET 展示功能监督预训练的另一条路线|
|⑫ 过平滑缓解|低，诊断后再做|出现深层表示退化后再比较残差、归一化和多尺度传播|原 GNN 仅两层，当前主干也不是深层 GNN|
|⑬ 配对交互|中；局部功能交互中高|RNA 调控区域/基因程序与药物靶点/子结构交互，比较同一信息的汇总与局部匹配|PLMF-MDA、DiSPA、HDBI|
|⑭ 负样本策略|**高，先做偏差审计**|区分未观测与实验阴性；检测药物/RNA 频率捷径，研究 PU 与来源选择偏差|miRBench、PULSNAR；提高生物可信度不保证旧 AP 上升|
|② 精细多任务|中高；标签就绪后可作主线|关联、方向、RNA–gene 功能及药敏等有真实标签的任务联合学习；允许任务差异与缺失标签|FORGE 的依赖性–IC50 联合学习；CoCoBind 的位点–结合联合学习只适用于对应任务|
|③ 细粒度关联|**高，长期主线**|记录 RNA 干预、效应方向/大小、细胞背景、剂量、时间、实验类型和 PMID|MPHGNN 明确承认聚合二分类的上下文与连续效应局限|
|① ncRNA 多模态融合|**高；需补互补信息**|序列/结构、表达、功能结合与定位融合；按证据质量和模态缺失报告表现|MIFNDRA、NCRDLLM 已覆盖普通多模态；PARNET/RIME 提供更具体功能线索|

**直接竞争论文已经覆盖了什么。**

- [MPHGNN，Briefings in Bioinformatics，2026-01-29](https://doi.org/10.1093/bib/bbag013)：miRNA–gene–drug 异构图，元路径与全局表示。单独新增 gene 节点已有直接先例。
- [PLMF-MDA，PLOS Computational Biology，2026-02-10](https://doi.org/10.1371/journal.pcbi.1013968)：RNA-FM/ChemBERTa-2、局部 CNN/GCN、跨模态注意力，用于耐药与敏感关联预测。普通局部＋全局融合也已有先例。
- [NCRDLLM，JCIM，2026-03-12](https://doi.org/10.1021/acs.jcim.5c03011)：三类 ncRNA、序列/结构/关联特征、LLaMA-3.2-3B 与 LoRA。换大模型或统一多个 RNA 类型本身不足以形成清晰贡献。本轮依据摘要。
- [MIFNDRA，Briefings in Bioinformatics，2026-06-17](https://doi.org/10.1093/bib/bbag322)：SpliceBERT、预训练 GIN、跨 ncRNA 网络、残差 GraphSAGE 与对比学习。
- [HDBI，JCIM，2026-07-10](https://doi.org/10.1021/acs.jcim.6c01116)：多视图超图、解耦表示、双向跨模态更新。本轮依据摘要；模型使用的“动态”“解耦”名称不能直接视为已测动态机制。
- [MVGSCA，IEEE TCBBIO，2026](https://doi.org/10.1109/TCBBIO.2026.3729865)：基因介导图、谱编码与稀疏跨注意力。本轮依据摘要，Crossref 尚无精确在线日期。

**最值得形成后续研究问题的四条路线。**

1. **RNA 调控程序与药物作用程序之间的功能匹配：⑤＋④＋⑨＋⑬。** [FORGE，Nature Communications，2026-06-08](https://doi.org/10.1038/s41467-026-73977-2)用联合矩阵分解共同解释 CRISPR 基因依赖性和药物 IC50；[DiSPA，Bioinformatics，2026-07-07](https://doi.org/10.1093/bioinformatics/btag272)让药物子结构与通路表达双向交互；[scRADAR，2026-06-26](https://doi.org/10.1371/journal.pcbi.1014392)结合静态靶点/通路和 LINCS L1000 扰动指纹。可检验假设是：具体功能证据是否比靶点重叠或通路汇总更有增量。FORGE 目前按 drug–target pair 建模；基因敲除依赖性不等于 ncRNA 干预效应。DiSPA 的单细胞/空间迁移缺少实测药敏真值；scRADAR 部分标签为回顾性操作定义，因此不能把它们的解释直接当因果验证。

2. **用 RNA 功能监督提升序列表示：⑧＋①＋⑪。** [miRBind2，Bioinformatics，2026-08-21](https://doi.org/10.1093/bioinformatics/btag410)将局部靶位点学习迁移到 3′UTR 的抑制预测；[RIME，Nature Communications，2026-01-23](https://doi.org/10.1038/s41467-026-68356-w)研究低复杂度重复区域与长 RNA–RNA 互作，包含 lncRNA。对 lncRNA 特别值得补看 [PARNET，2026-08-13，预印本 v1](https://doi.org/10.64898/2026.08.08.743506)：用 223 个 eCLIP 实验、150 个 RBP 的结合信号训练功能表示，并研究 lncRNA 染色质定位动力学。其全文明确限制为 K562/HepG2 两个细胞系，RBP 覆盖有限；较大容量与多任务效果也没有完全分离。它目前未显式输入二级结构或修饰。可比较 RNA-FM、[NucleicBERT，Nature Machine Intelligence，2026-09-03](https://doi.org/10.1038/s42256-026-01295-9) 等通用表示与功能监督表示，检验长度、GC、家族相似度以外的增量。miRBind2 的功能预测仍约为 Pearson 0.30、ROC-AUC 0.60，预测靶标只能作为带不确定性的先验，不能当实验真值。

3. **预测特定背景下的 ncRNA 干预效应：③＋⑥＋②。** [CRISP，Nature Computational Science，2025-10-03](https://doi.org/10.1038/s43588-025-00887-6)针对未见细胞类型迁移；[PertDiff，Bioinformatics，2026-06-30](https://doi.org/10.1093/bioinformatics/btag482)结合对照表达、药物表示和细胞描述预测扰动。它们解决的是相关任务，尚不能替代 ncRNA 干预标签。将数据单位明确为 `(RNA, 干预方向/强度, drug, cell/cancer, dose, time, assay, PMID)`，在有真实测量时学习耐药/增敏和效应大小。2026 年 9 月的 [Science Advances 系统评测](https://doi.org/10.1126/sciadv.aed3414)摘要报告：13 类方法/基线、25 个数据集，跨细胞类型任务没有持续占优的方法。可迁移的条件机制仍是值得研究的问题。聚合正例不能复制成多个背景的真标签；输入当前 RNA×细胞系矩阵也不等于获得记录级上下文。

4. **显式处理发现过程与未标注样本：⑭，并反过来约束⑪。** [miRBench，Bioinformatics，2025](https://doi.org/10.1093/bioinformatics/btaf233)显示，单看 miRNA、不输入靶位点的模型也能利用正负类频率偏差取得高 AP。先在本项目量化 RNA-only、drug-only、训练度数等简单预测器，再比较随机采样与频率匹配采样。[PULSNAR，2024](https://doi.org/10.7717/peerj-cs.2451)研究非完全随机标注下的类别比例估计，仍要求未标注正例子类型有相应已标注支持等条件，不能消除任意来源偏差。PU 可作为后续候选；仅靠正例和未知配对不能同时无假设地识别真实关联概率与被研究/记录概率。高分未知样本不宜直接施加强负监督。

**两类容易混淆、但必须保持区别的任务。** [CoCoBind，J. Med. Chem.，2026-08-17](https://doi.org/10.1021/acs.jmedchem.6c00902)联合 RNA–小分子结合与位点定位，其跨任务一致性有位点–结合语义依据；[GRASSP，Bioinformatics，2026-08-22](https://doi.org/10.1093/bioinformatics/btag638)研究 RNA–小分子结合位点。这些可启发局部表示，却不能把“药物直接结合 RNA”当成所有 ncRNA 耐药关联的机制。类似地，相关系数正负、RNA 调控方向和耐药/增敏方向是不同的量。若扩展 circRNA，[circMAC，2026-09-09](https://doi.org/10.1093/bib/bbag492)提示需要成熟 miRNA 条件、完整环状转录本与反向剪接接头连续性，普通线性序列截断会遗漏特定局部语境。

**建议的最小实验与验收。**

- 近期投入优先放在⑤和⑧；启动前先做低成本⑭偏差审计。完整多任务与背景效应研究③＋⑥＋②，需要记录级标签整理。
- 新证据先比较“当前模型／新增同一批信息的简单汇总／同一批信息的局部交互”。实体归属打乱应保持覆盖、缺失和重要度数分布，检验增益是否来自正确生物对应关系。
- 功能预训练与普通预训练比较时，匹配输出维度、下游训练预算与数据划分，控制长度/GC/家族捷径；分别报告两个 RNA 任务及模态缺失组。不要把网络容量差写成监督目标的单独贡献。
- 关联、方向、功能调控和连续药敏只在有标签时计算对应损失；用实测任务关系决定共享部分，记录负迁移。全局 AUC/AP、按药物宏 AP、长尾与覆盖组性能同时看。
- 下游分组应覆盖药物骨架、RNA 家族、来源/时间，以及条件任务中的细胞类型；同一来源的重复记录或同源辅助数据库要防止回流。原论文已有部分留一实体实验，严格双端未见或跨任务共同药物屏蔽才是更进一步的设置。
- 预训练模型预测、注意力、富集或数据库中与输入重复的证据，只能支持候选机制；机制验证需要来源独立的功能读数或针对性扰动。新数据和新算子通过开发筛选后，再用未参与选模的外部/时间数据检验。

本轮完成文献与代码核查，没有运行新的模型训练。以下文献表中“全文相关部分”仅指本轮阅读了与上述判断有关的方法、结果或限制段落，未逐句审阅全部正文及补充材料。

|编号|论文与 DOI|作者、出处与日期|本轮阅读范围|作用|
|---|---|---|---|---|
|1|[MPHGNN: metapath-guided heterogeneous graph neural network for miRNA-drug resistance association prediction](https://doi.org/10.1093/bib/bbag013)|Guangsheng Huang 等；Briefings in Bioinformatics；2026-01-29|公开全文缓存：方法与限制相关段落|直接任务：gene 中介图已有先例|
|2|[A pre-trained language model-based cross-modal fusion framework for predicting miRNA-drug resistance and sensitivity associations](https://doi.org/10.1371/journal.pcbi.1013968)|Nan Sheng 等；PLOS Computational Biology；2026-02-10|公开全文缓存：Discussion 与结论相关部分|直接任务：局部与全局多模态已有先例|
|3|[NCRDLLM: Predicting ncRNA-Drug Response Associations via Multimodal Feature Fusion and Large Language Models](https://doi.org/10.1021/acs.jcim.5c03011)|Zihan Zhang、Yuchen Zhang；Journal of Chemical Information and Modeling；2026-03-12|Europe PMC/Crossref 摘要|直接任务：三类 ncRNA、多模态与 LLM 已有先例|
|4|[MIFNDRA: an innovative knowledge-enhanced multimodal fusion and graph learning framework for predicting drug resistance-related ncRNAs](https://doi.org/10.1093/bib/bbag322)|Jianan Sui 等；Briefings in Bioinformatics；2026-06-17|公开全文缓存：数据集相关部分；摘要|直接任务：序列＋图＋对比已有先例|
|5|[Higher-Order Dynamic Disentangled Intent Sensing and Bidirectional Joint Updating Framework for NcRNA-Drug Resistance Association Prediction](https://doi.org/10.1021/acs.jcim.6c01116)|Tiyao Liu 等；Journal of Chemical Information and Modeling；2026-07-10|Europe PMC/Crossref 摘要|直接任务：超图、解耦、跨模态更新已有先例|
|6|[Multi-view Graph Learning Framework with Spectral Encoding and Sparse Cross-Attention for miRNA-Drug Association Prediction](https://doi.org/10.1109/tcbbio.2026.3729865)|Ru Nie 等；IEEE Transactions on Computational Biology and Bioinformatics；2026|Europe PMC 摘要；Crossref 元数据|直接任务：谱与稀疏跨注意力已有先例|
|7|[Gene dependency-informed inference of response to targeted cancer therapies](https://doi.org/10.1038/s41467-026-73977-2)|Nilabja Bhattacharjee 等；Nature Communications；2026-06-08|新取公开全文：方法、Discussion 与限制|可迁移方法：功能依赖性与药敏联合学习|
|8|[DiSPA: differential substructure-pathway attention for drug response prediction](https://doi.org/10.1093/bioinformatics/btag272)|Yewon Han 等；Bioinformatics；2026-07-07|公开全文缓存：迁移结果、Discussion 与限制；摘要|可迁移方法：子结构–通路交互|
|9|[scRADAR: Dissecting intratumoral drug response heterogeneity at single-cell resolution via mechanism-guided prototype routing](https://doi.org/10.1371/journal.pcbi.1014392)|Ren Qi 等；PLOS Computational Biology；2026-06-26|公开全文缓存：指纹构建、标签与限制|可迁移方法：静态机制与药物扰动|
|10|[miRBind2 enables sequence-only prediction of miRNA binding and transcript repression](https://doi.org/10.1093/bioinformatics/btag410)|David Čechák 等；Bioinformatics；2026-08-21|公开全文缓存：功能迁移方法、结果与讨论|上游功能：miRNA 靶向到抑制迁移|
|11|[The role of low-complexity repeats in RNA–RNA interactions and a deep learning framework for duplex prediction](https://doi.org/10.1038/s41467-026-68356-w)|Adriano Setti 等；Nature Communications；2026-01-23|新取公开全文：Discussion 与限制|上游功能：长 RNA–RNA 互作与局部区域|
|12|[PARNET: A CLIP-SEQ-BASED FOUNDATION MODEL FOR RNA SEQUENCE REPRESENTATION LEARNING](https://doi.org/10.64898/2026.08.08.743506)|Lambert Moyon 等；bioRxiv（预印本）；2026-08-13|新取 v1 公开全文：功能迁移、Discussion 与限制|上游功能：实验 RBP 监督与 lncRNA 功能|
|13|[NucleicBERT interprets RNA sequence space through self-supervised language modelling](https://doi.org/10.1038/s42256-026-01295-9)|Utkarsh Upadhyay 等；Nature Machine Intelligence；2026-09-03|公开全文缓存：预训练/随机初始化对照及 Discussion|上游表示：序列自监督与结构/功能|
|14|[OmicsTransformer: self-supervised masked consistency and uncertainty-aware fusion for robust multi-omics prediction](https://doi.org/10.1093/bioinformatics/btag468)|Junxuan Feng 等；Bioinformatics；2026-06-29|公开全文缓存：Discussion 相关部分；摘要|可迁移方法：缺失模态与置信融合|
|15|[Predicting drug responses of unseen cell types through transfer learning with foundation models](https://doi.org/10.1038/s43588-025-00887-6)|Yixuan Wang 等；Nature Computational Science；2025-10-03|出版社摘要与预览；无全文访问|可迁移任务：未见细胞类型|
|16|[Enhancing cross-context generalization in drug perturbation prediction with a multimodal conditional diffusion framework](https://doi.org/10.1093/bioinformatics/btag482)|Yanjie Ma 等；Bioinformatics；2026-06-30|新取公开全文：跨细胞评估与 Discussion|可迁移任务：跨细胞/药物扰动|
|17|[A systematic comparison of single-cell perturbation response prediction models](https://doi.org/10.1126/sciadv.aed3414)|Lanxiang Li 等；Science Advances；2026-09-11|Crossref/Europe PMC 摘要|评测：条件泛化仍有瓶颈|
|18|[miRBench: novel benchmark datasets for microRNA binding site prediction that mitigate against prevalent microRNA frequency class bias](https://doi.org/10.1093/bioinformatics/btaf233)|Stephanie Sammut 等；Bioinformatics；2025-07-15|公开全文缓存：频率偏差、采样方法、结论|评测：负采样与实体频率偏差|
|19|[Positive Unlabeled Learning Selected Not At Random (PULSNAR): class proportion estimation without the selected completely at random assumption](https://doi.org/10.7717/peerj-cs.2451)|Praveen Kumar、Christophe G. Lambert；PeerJ Computer Science；2024-11-05|公开全文缓存：假设、子类估计与限制|观察模型：非随机标注与类别比例|
|20|[A Scalable Sign-Aware Multi-Omics Knowledge Graph Foundation Model for Mechanistic Drug Action and Clinical Response Predictions](https://doi.org/10.64898/2026.04.29.721775)|Mohammadsadeq Mottaqi 等；bioRxiv（预印本）；2026-05-04|Crossref 摘要|可迁移表示：有符号、有方向机制图|
|21|[Dual-granularity feature learning reveals static and dynamic mechanisms of ncRNA-drug interactions](https://doi.org/10.1109/jbhi.2026.3717803)|Runzhou Tang 等；IEEE Journal of Biomedical and Health Informatics；2026|Europe PMC 摘要；Crossref 元数据|直接任务：解耦语义已有先例；非因果实验证明|
|22|[CoCoBind: Consistency-Contrastive Multitask Learning for RNA–Ligand Interaction and Binding Site Prediction](https://doi.org/10.1021/acs.jmedchem.6c00902)|Shihang Wang 等；Journal of Medicinal Chemistry；2026-08-17|Europe PMC/Crossref 摘要；PubMed 网页不可用|邻近任务：结合–位点多任务，非耐药关联|
|23|[GRASSP: RNA language model–enhanced graph attention with adaptive gating for RNA–small molecule binding site prediction](https://doi.org/10.1093/bioinformatics/btag638)|Thi Lan Nguyen、Nguyen Quoc Khanh Le；Bioinformatics；2026-08-22|新取公开全文：结论与结构噪声限制|邻近任务：结合位点与结构置信度|
|24|[circMAC: microRNA-conditioned binding-site localization on circular RNA isoforms](https://doi.org/10.1093/bib/bbag492)|Juseong Kim 等；Briefings in Bioinformatics；2026-09-09|Europe PMC/Crossref 摘要|上游功能：环状转录本的局部条件定位|
|25|[Tabular Foundation Models Are Competitive Cellular Perturbation Predictors Across Biological Scales](https://doi.org/10.64898/2026.06.28.735106)|Giovanni Palla 等；bioRxiv（预印本）；2026-07-01|Europe PMC/Crossref 摘要|背景评测：表格基础模型用于扰动预测|
|26|[Signature Recontextualization: Mapping perturbational signatures across biological contexts](https://doi.org/10.64898/2026.08.14.744937)|Andrew Chen 等；bioRxiv（预印本）；2026-08-19|Europe PMC/Crossref 摘要|背景评测：目标上下文数据可用性与迁移|
