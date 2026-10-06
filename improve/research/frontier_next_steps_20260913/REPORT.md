**iNcRD-HG 还有改进空间。当前最值得推进的是功能型 RNA 表示、带来源的上下文监督，以及细胞状态下的基因依赖性；常规图算子替换的优先级应下调。**

核查日期：2026-09-13。本文结合原始 iNcRD-HG、当前已接受的 TabICLv2 分层证据方案，以及其后的功能证据扩展、GNN/表格模型对照。优先级表示研究价值和当前可执行性，不表示保证提高 AUC/AP，也不构成首创性证明。本轮开展公开文献检索、相关原文阅读和只读数据核查，没有新增模型性能实验。

先明确当前起点。原始 [predict_association.py](../../../predict_association.py) 是两层 GeneralConv，LncDrug、MiDrug 分别初始化并训练。图的相似度在 [heter_data_prepare.py](../../../heter_data_prepare.py) 中取绝对值后阈值化。后续项目已经有加权边、PCA、RNA-FM、HGT、配对交互、共享适配和对比/一致性实验。因此，原表中很多方向现在适合作为对照或组件。

当前已接受的模型是图上下文 TabICLv2 加 26 列分层基因/通路证据，两个任务采用相同的 0.5 混合规则。其历史外层开发结果为 MiDrug AUC/AP 0.960638/0.926064，LncDrug 0.986929/0.971320；这些来自已反复使用的数据，不是独立测试结论。详情见 [已接受方案](../../TABICL_PATHWAY_CONTEXT_FINDINGS.md)。

更晚的结果改变了此前“优先扩靶点覆盖”的建议：

| 最新项目证据 | 对下一步的含义 |
|---|---|
| [功能证据扩展](../../FUNCTIONAL_PRIORITY_FINDINGS.md)：药物覆盖 93/216 → 156/216，drug–gene 边 645 → 2,227；8 个候选均未通过既定共同验收 | 覆盖增加已经实际测试。进一步工作要检验信息语义、功能强度和选择规则，不能只继续加边 |
| 当前 `target_local` 保持 62 维，将通路相似度换为逐靶点匹配的均值/top-3，正确对应相对保度错配没有稳定优势 | 这个版本仍是静态通路摘要；它没有读入 RNA 实测调控方向、药物 action 或细胞依赖性。其负结果不代表所有机制模型都无效 |
| [GNN 对照](../../GNN_TABICL_COMPARISON_FINDINGS.md)：SA-HGT 在三档输入中都未超过 TabICLv2 的全局 AUC/AP；MiDrug 的逐药物 AP 可以更高 | 应先明确全局排序和单药物筛选的目标；不能仅凭模型类别判断优劣 |
| [无预训练预测器对照](../../NONPRETRAINED_TABULAR_FINDINGS.md)：现有预算下 TabM/CatBoost 也有竞争力；MiDrug 扩证据后的 AP 同样回落 | 问题不只发生在 GNN，也不能完全归因于 TabICL 的预训练优势。各模型训练预算尚未严格等算力 |
| miRBind2 已试运行 35 个位点窗口、27 个 miRNA、15 个基因；PARNET 已试运行 32 个完整 lncRNA 转录本 | 功能表示管线可用，但尚未证明全量下游增益。试运行覆盖不能当成整库覆盖 |

你的 14 项可以这样重新排序。对应文献的正式题名、日期、阅读层级在文末列出。

| 原编号 | 当前优先级 | 建议推进的具体版本 |
|---|---|---|
| ④ 加权 / 正负 / top-k | **中**；固定 top-k 调参低 | 从已有固定权重推进到证据类型、来源可靠性和条件化边选择。表达相关正负、RNA 调控方向、药物激活/抑制、耐药/增敏标签分别编码；未知方向单独保留 |
| ⑤ mRNA / protein / pathway 节点 | **中高**；单纯扩节点低 | 保留具体基因身份和作用机制，加入可对齐的基因依赖性。先检验新信息的增量，再比较是否需要显式 gene 节点；[MPHGNN][mphgnn] 已有 miRNA–gene–drug 图 |
| ⑥ 癌种 / 细胞系上下文 | **高** | 从真实记录的 Condition 和来源出发；癌种 ID、干预前细胞状态、耐药衍生株区别处理。现有癌种模型也是对照，不能重新包装为新贡献 |
| ⑦ 表达降噪压缩 | **中低** | 保留训练折拟合的 PCA/尺度统计作为强基础对照；只有发现新的批次/条件混杂再扩展。已有原 GNN 上的局部正结果不自动迁移到当前强参考 |
| ⑧ 序列 / 结构 | **高，适合下一轮小试** | miRNA 做成熟链–靶位点/3′UTR 的功能表示；lncRNA 做 RBP 结合、定位或功能区域表示。结构先用可及性/配对概率及置信度，避免把单一预测构象当确定真值 |
| ⑨ drug 多视图 / 对齐 | **中高** | 结构、靶点 action、细胞/剂量/时间对应的扰动程序互补；比较同药物不同状态，不直接把跨条件 L1000 均值叫上下文建模 |
| ⑩ HAN / HGT | **低，作为基线** | [HDBI][hdbi] 已有高阶多视图超图与双向更新，[MVGSCA][mvgsca] 已有谱编码与稀疏跨注意力。新贡献需要落到新增信息和可验证的交互对象 |
| ⑪ 自监督 / 对比 | **中，作为辅助目标** | RNA–gene 证据重建、功能表征迁移、质量感知的模态缺失训练。[MIFNDRA][mifndra] 已结合序列预训练和图对比；不能默认互作 RNA 必须同方向或同表示 |
| ⑫ 过平滑缓解 | **低** | 原模型只有两层。最新 HGT 长训练的训练损失下降、开发指标恶化更支持先排查过拟合和训练/推理图差异；尚不能定位为过平滑 |
| ⑬ 配对交互 | **中**；有功能对象时中高 | 比较 RNA 功能区域/调控基因与药物靶点/子结构的交互。普通 concat/product/bilinear、池化和同维度随机控制都应保留 |
| ② 精细多任务 | **中高，有标签才推进** | 关联存在、已观测的耐药/增敏描述、RNA–gene 功能、细胞药敏分别设任务；屏蔽缺失标签。共享机制层、允许 RNA 类型差异，实际监测负迁移 |
| ③ 细粒度关联 | **高** | 优先修复实验语义与记录来源；从条件关联逐步扩至真实干预效果。剂量、时间和四组对照不全时，不能输出未经识别的因果增敏标签 |
| ① ncRNA 多模态 | **高，重点在功能互补和缺失处理** | RNA-FM/新 RNA 模型、表达、局部功能和实验证据有各自适配器与缺失标记。直接拼接已有很多先例，[NCRDLLM][ncrdllm] 也是近邻方法 |
| ⑭ 负样本 | **高，优先保证结论可靠** | 把真实阴性与未标注配对区分；比较 PU、同药物候选排序和实体频率控制。高分未标注不能自动变成 hard negative，也不能自动变成真阳性 |

**1. 最适合立即推进：把已有 RNA 功能表示试运行变成有对照的下游实验。**

[miRBind2，Bioinformatics 2026][mirbind2] 用 miRNA–靶位点核苷酸交互学习结合，再迁移到完整 3′UTR 上的 RNA 抑制预测。这比只把整条 RNA 向量放进图，更接近“这个 RNA 会影响哪些 drug-relevant genes”。但要区分两个模型：本地跑通的是 binding-site 模型，论文的功能抑制结果来自另一个经过迁移训练的 3′UTR 模型，不能把本地 binding score 直接解释为抑制强度。论文功能任务 Pearson 约 0.30、ROC-AUC 约 0.60，预测靶向关系需要保留不确定性。

[PARNET，2026-08 预印本][parnet] 用 223 个 eCLIP 实验、150 个 RBP 的结合信号预训练，并研究 lncRNA 染色质定位等功能。它对 lncRNA 比强行复用 miRNA seed-matching 更合适。论文明确承认预训练主要来自 K562/HepG2，未显式输入二级结构/修饰，不能承诺跨癌种普遍有效。本地采用的兼容旧权重也不能自动视为论文全部最终实验配置。

最小实验应冻结同一预测器、样本、压缩维数和调参预算，比较：现有证据参考；加入长度/GC/缺失等简单特征；加入通用 RNA 表示；加入功能监督表示；以及保留覆盖、长度/GC 分层的实体归属打乱。通用表示可先复用现有 RNA-FM，再考虑 [NucleicBERT，Nature Machine Intelligence，2026-09-03][nucleicbert]。局部特征还要和同一批局部输入的池化对照比较。

先报告覆盖的 RNA/药物/配对和各覆盖组表现。如果功能表示不优于通用表示或打乱对照，就不能把收益归为功能语义；如果只在少量有注释实体上有效，就缩小主张。应复用 [rna_functional_pilot.py](../../rna_functional_pilot.py) 的来源与版本记录，另建下游实验，不把 27 个 miRNA 或 32 个 lncRNA 的试运行当成全量模型。

**2. 最有论文问题价值：区分上下文变化、文献来源和实验语义。**

现有数据并非完全缺少上下文。[direction_labels.py](../../../../incrd-run/analysis/direction_labels.py) 保留 `Effect、Condition、PMID、Throughput、Phenotype、year`；[context_controls.py](../../../../incrd-run/analysis/context_controls.py) 已实现条件三元组。这次对现有 CSV 的只读复核得到：

| 统计 | MiDir | LncDir |
|---|---:|---:|
| 原始方向记录 | 10,506 | 2,451 |
| 不同 RNA–drug 配对 | 5,393 | 2,108 |
| 同时存在两类 Effect 的配对 | 1,107 | 56 |
| 按原始 Condition 区分后不再有条件内冲突的配对 | 847 | 53 |
| 现有 disease 映射下的无冲突三元组 | 8,667 | 2,128 |

847/1,107 是按原始 Condition 字符串分组的描述性结果，不能称为“已确认的癌种导致的方向反转”。使用现有规范化 disease 字段时，有映射的 1,085 个原始冲突配对中，549 个不再有 disease 内冲突；仍有 678 个冲突三元组。不同分组口径不能混报。细胞系列表的展开也不增加独立实验数。

更重要的是来源独立性：复现旧 MiDir 划分算法的 seed=1，`cold_rna` 测试有 1,703 个三元组，其中 1,210 个（71.1%）的证据 PMID 至少有一个也出现在训练侧；`cold_drug` 为 444/2,363。这里只统计已有记录，没有重新计算模型分数。共享文献不是直接标签泄漏的证明，但这种拆分不能支持“对新研究泛化”的结论。数据、定义和哈希见 [context_readonly_audit.json](context_readonly_audit.json) 与 [audit_context.py](audit_context.py)。

因此可研究的目标是：在新的研究来源或细胞状态中，预测特定 RNA–drug 的方向，并对冲突或支持不足的证据保留无法判断的情况。PMID 用于分组、来源审计和训练中的观测效应；不能靠记忆 PMID 预测新文献标签。预先保留完整原始 record，同一个 record 的多个细胞系作为同一观测单元处理。不同来源/状态缺少交叉覆盖时，来源效应和生物状态效应可能不可区分。

实验语义要先修复。仓库 [已有原文核查](../intervention_design_20260909/ASSESSMENT.md) 发现，32 个优先配对中有 16 个涉及一篇亲本–耐药株 miRNA 表达芯片研究；这些特定记录没有测量各 miRNA 干预后的药敏改变。整篇文章中存在某个 RNA 的功能实验，不能让所有差异表达记录都变成功能标签。后续应从已有严格细胞系 accession/来源审计继续，而不是仅凭上表重新给所有记录赋硬标签。

若发展成真正的干预任务，记录至少包含 `(RNA, 抑制/补充及强度, drug, cell/state, dose, time, assay, source)`。普通药物 IC50、RNA 结合分数、耐药株差异表达都可提供辅助信息，但不等于 RNA 干预导致的增敏效果。

**3. 值得增加的新信息：在具体细胞状态中，RNA 所影响的基因是否决定药物反应。**

[FORGE，Nature Communications，2026-06-08][forge] 联合建模 CRISPR 基因依赖性与药物 IC50，以共享的稀疏低维表达空间连接两个任务。其启发是：一个药物靶点是否存在之外，还要看该背景是否依赖该靶点。FORGE 当前按 drug–target pair 建模，有线性模型、外部样本量和批次效应限制；基因完全敲除的依赖性也不能当作 RNA 轻度调节的效果。

[DiSPA，Bioinformatics 2026][dispa] 用药物子结构和通路表达做双向交互；[scRADAR，PLoS Computational Biology 2026][scradar] 用靶点/通路先验、LINCS L1000 特征和药物条件化的细胞表示。它们提供可迁移的接口，不证明本项目接上相同模块就会提高 AP。DiSPA 的单细胞/空间扩展缺少实测药敏真值；scRADAR 的部分标签是回顾性操作定义，且其原实现会对同药物多条 LINCS signature 取均值。

可检验的候选结构是：

`RNA 局部功能/实验调控证据 → 具体 gene/program ← drug 靶点/action/子结构`，再由 `cell/state 的表达与依赖性` 调节各基因贡献。

当前 [FunctionalPrograms.table](../../tabicl_functional_modules.py) 用静态二值 drug–gene 矩阵、固定证据层权重和通路相似摘要；代码明确没有推断方向、因果效应或实测通路活性。下一步应先加入一小组可解释的条件功能特征，检验是否有独立增量，再比较带 gene 节点的 GNN 或局部交互网络。每种架构都要访问相同新证据，并有同预算简单预测器和打乱对照。

RNA 抑制 gene、drug 抑制 protein、gene dependency 三者的符号不能机械相乘得出耐药方向：间接作用、反馈、剂量以及组织背景都会影响净效应。没有依据的方向保留 unknown；mRNA 水平变化和蛋白活性变化分别描述。若输入预测时不可得的药物后表达，会改变预测任务，应只把它用于训练目标或明确的已测背景设置。

**4. 把⑭升级为观测过程建模，并增加按药物评价。**

本地 [采样审计](../../results/sampling_bias_audit_20260913/REPORT.md) 中，仅药物训练正例度数的 AUC 就约为 MiDrug 0.936、LncDrug 0.977。这表明全局随机配对评价有强实体频率信号；它不能单独证明频率全部来自发表偏差。[miRBench，Bioinformatics 2025][mirbench] 在 miRNA 靶位点任务中展示了类似的类别频率伪信号，并设计了相应控制。

令 Y 表示真实关联、S 表示被记录为正例。在正标签正确的假设下，`P(S=1|x) = P(Y=1|x) × P(S=1|Y=1,x)`。只凭当前正例和未观测配对，无法无假设地分解这两个概率。[PULSNAR，2024][pulsnar] 可以提供正例子类型和未标注正例比例估计的思路，但要求已标注正例正确、未标注正例的各子类型有已标注支持；本项目的混合证据可能违反这些条件。

建议先固定评价集，比较现有 BCE、PU 风险、可靠阴性子集/同药物候选排序，保持其他输入和预算一致；对未知正例比例做敏感性分析。训练中同药物采样可帮助关注药物内区分，但不能把所有抽中的未知对象解释为真正阴性。不要通过改变负例比例后直接比较 AP。

同时报告全局 AUC/AP、逐药物宏 AP、可用候选全集上的 Recall@K，以及长尾/缺失模态组表现。若只有数据库正例，Recall@K 应叫已知关联恢复率，不能当成真实生物命中率。按药物宏 AP 需要说明药物数、有效正/负例数和排除规则。对实体或研究做分组重采样估计不确定性；同一模型的三个推理 seed 不代表三个独立生物样本。

**原表之外，我会再加两项。**

- **证据来源和时间的独立验证。** 优先做 PMID 分组及标签同源数据库审计，再推进 RNA 家族、药物骨架、跨来源/时间和必要的双端未见。多个 PMID 支持同一关系时要明确证据分配规则；若需要严格实体不重叠，要按共享关系/来源构造连通组，检查巨型分组。原文已有部分留一实体，新增贡献应说清更严格的设置。
- **带不确定性的实验选择。** [2026 年 Molecular Systems Biology 的主动学习研究][active] 把机制模型、不同候选选择规则和实际扰动实验连接；[SHIFT-DRP，2025][shift] 研究多样性到不确定性的动态选择。对本项目可比较随机选、最高分选、来源/实体多样性选和不确定性选，在同等实验预算下看确认的 RNA–drug–context 命中。只有点预测 entropy 或三个 TabICL 推理种子的波动，不足以证明不确定性已校准。基于未标注配对的校准也不能自动覆盖真实生物阴性；若使用 conformal 方法，仍需满足对应校准分布假设。没有真实反馈标签时只能提出或做回顾性模拟，不能声称已经提高实验成功率。

来源审计有直接文献依据：[miRTarBase 2025][mirtarbase] 的药物部分整合了 ncRNADrug；[LncTarD 2.0][lnctard] 同时收录功能调控和表达关联。当前工程已经分层保存部分证据与 PMID，这是可复用基础；仍应检查共享研究是否从辅助模态回流至评价标签。此处不声称现有扩展已经发生标签回流。

多模态缺失训练可参考 [OmicsTransformer][omicstransformer] 的 modality dropout 和置信融合，但其推理时使用的是训练出来的置信度预测器，真实标签对应的概率只在训练时可用。模态缺失本身可能标识研究热度，因此需要“只输入缺失标记/证据计数”的基线，以及有无该模态的分层评估。

建议按下面三个阶段排实验，先固定候选和主要终点，再运行。

| 阶段 | 最小工作 | 什么结果才支持继续 |
|---|---|---|
| A：现有数据即可开始 | 冻结当前已接受参考；复核 Effect 语义；建立 record/PMID 分组；功能表示做同信息、同维度小试 | 增益依赖正确实体/位点对应，且在来源分组和覆盖组中有一致证据；仅旧开发集微小变化不够 |
| B：补功能数据 | 在可可靠对齐的子集补基因依赖性、drug action、状态/剂量/时间；简单汇总与条件交互采用相同输入 | 新信息有增量，条件交互又超过同信息简单模型；正确基因/状态匹配胜过保覆盖的打乱对照 |
| C：完整研究主线 | 条件关联/干预任务加缺失标签多任务与 PU；新增来源或时间数据上冻结验证；具备条件时做主动实验选择 | 能回答新研究/新状态下的候选选择，报告不确定性；如主张干预增敏，要有对应实际功能读数 |

若继续沿用仓库当前“两任务共同通过”的验收要求，应在开发阶段预先冻结逐任务标准和适配规则，不能按外层结果临时给 LncDrug 绕过新分支。是否提高当前全局 AUC/AP，与能否改善药物内筛选、条件泛化，是不同需要分别报告的结果。

最新的 [Science Advances 扰动模型评测，2026-09-11][benchmark] 比较 13 类方法/基线、25 个数据集与多种终点，其摘要指出跨细胞类型没有持续占优的方法。它支持认真研究泛化与任务定义，不能直接给出本项目的性能上限或预期提升。

本轮使用 PubMed E-utilities、Europe PMC、Crossref 和公开全文，重点关注 2025–2026 年，保留关键较早方法。完成 5 组定向 Europe PMC 查询、PubMed 定向查询和 20 个 DOI 元数据核对；相关全文复用工作区已有合法公开缓存并补取新全文。arXiv 补充查询和 OpenAlex 检索超时；主动学习论文的 nature.com 直址返回 404，但 Europe PMC 公开全文成功取得。未将失败来源当作阅读依据。本轮为有范围的研究评估，不是系统综述；摘要级论文在下面单独标明。

可复核材料：[检索与 DOI 原始记录](search_manifest.json)、[补充主动学习来源](supplemental_sources.json)、[只读上下文审计](context_readonly_audit.json)。下面的“全文关键部分”表示阅读相关方法、结果或局限，不表示逐句审阅全部正文及补充材料。

<!-- REFERENCE_TABLE -->

| 文献 | 第一作者等 | 期刊 / 状态 | 在线/登记日期 | 阅读依据 |
|---|---|---|---|---|
| [MPHGNN: metapath-guided heterogeneous graph neural network for miRNA-drug resistance association prediction](https://doi.org/10.1093/bib/bbag013) | Guangsheng Huang 等 | Briefings in Bioinformatics | 2026-01-29 | 全文关键部分 |
| [MIFNDRA: an innovative knowledge-enhanced multimodal fusion and graph learning framework for predicting drug resistance-related ncRNAs](https://doi.org/10.1093/bib/bbag322) | Jianan Sui 等 | Briefings in Bioinformatics | 2026-06-17 | 全文关键部分 |
| [Higher-Order Dynamic Disentangled Intent Sensing and Bidirectional Joint Updating Framework for NcRNA-Drug Resistance Association Prediction](https://doi.org/10.1021/acs.jcim.6c01116) | Tiyao Liu 等 | Journal of Chemical Information and Modeling | 2026-07-10 | Crossref 摘要 |
| [Multi-view Graph Learning Framework with Spectral Encoding and Sparse Cross-Attention for miRNA-Drug Association Prediction](https://doi.org/10.1109/tcbbio.2026.3729865) | Ru Nie 等 | IEEE Transactions on Computational Biology and Bioinformatics | 2026 | Europe PMC 摘要；Crossref 元数据 |
| [NCRDLLM: Predicting ncRNA-Drug Response Associations via Multimodal Feature Fusion and Large Language Models](https://doi.org/10.1021/acs.jcim.5c03011) | Zihan Zhang 等 | Journal of Chemical Information and Modeling | 2026-03-12 | 题名/元数据；未据此展开方法细节 |
| [miRBind2 enables sequence-only prediction of miRNA binding and transcript repression](https://doi.org/10.1093/bioinformatics/btag410) | David Čechák 等 | Bioinformatics | 2026-08-21 | 全文关键部分 |
| [PARNET: A CLIP-SEQ-BASED FOUNDATION MODEL FOR RNA SEQUENCE REPRESENTATION LEARNING](https://doi.org/10.64898/2026.08.08.743506) | Lambert Moyon 等 | bioRxiv（预印本） | 2026-08-13 | 预印本 v1 全文关键部分 |
| [NucleicBERT interprets RNA sequence space through self-supervised language modelling](https://doi.org/10.1038/s42256-026-01295-9) | Utkarsh Upadhyay 等 | Nature Machine Intelligence | 2026-09-03 | Crossref 摘要 |
| [Gene dependency-informed inference of response to targeted cancer therapies](https://doi.org/10.1038/s41467-026-73977-2) | Nilabja Bhattacharjee 等 | Nature Communications | 2026-06-08 | 全文方法、结果和 Discussion |
| [DiSPA: differential substructure-pathway attention for drug response prediction](https://doi.org/10.1093/bioinformatics/btag272) | Yewon Han 等 | Bioinformatics | 2026-07-07 | 全文关键部分 |
| [scRADAR: Dissecting intratumoral drug response heterogeneity at single-cell resolution via mechanism-guided prototype routing](https://doi.org/10.1371/journal.pcbi.1014392) | Ren Qi 等 | PLOS Computational Biology | 2026-06-26 | 全文关键部分 |
| [miRBench: novel benchmark datasets for microRNA binding site prediction that mitigate against prevalent microRNA frequency class bias](https://doi.org/10.1093/bioinformatics/btaf233) | Stephanie Sammut 等 | Bioinformatics | 2025-07-15 | 全文关键部分 |
| [Positive Unlabeled Learning Selected Not At Random (PULSNAR): class proportion estimation without the selected completely at random assumption](https://doi.org/10.7717/peerj-cs.2451) | Praveen Kumar 等 | PeerJ Computer Science | 2024-11-05 | 全文方法与假设 |
| [OmicsTransformer: self-supervised masked consistency and uncertainty-aware fusion for robust multi-omics prediction](https://doi.org/10.1093/bioinformatics/btag468) | Junxuan Feng 等 | Bioinformatics | 2026-06-29 | 全文关键部分 |
| [A systematic comparison of single-cell perturbation response prediction models](https://doi.org/10.1126/sciadv.aed3414) | Lanxiang Li 等 | Science Advances | 2026-09-11 | Crossref 摘要 |
| [Active learning-guided mechanistic modeling reveals context-specific regulators of CXCL9 expression in pancreatic cancer cells](https://doi.org/10.1038/s44320-026-00221-w) | Bi-rong Wang 等 | Molecular Systems Biology | 2026-06-04 | 新取公开全文：方法、结果和 Discussion |
| [SHIFT-DRP: Dynamic Multi-Scale Active Learning for Drug Response Prediction](https://doi.org/10.1021/acs.jcim.5c02453) | Xintao Wang 等 | Journal of Chemical Information and Modeling | 2025-12-15 | Europe PMC 摘要；Crossref 元数据 |
| [miRTarBase 2025: updates to the collection of experimentally validated microRNA–target interactions](https://doi.org/10.1093/nar/gkae1072) | Shidong Cui 等 | Nucleic Acids Research | 2024-11-23 | 全文数据库来源段落 |
| [LncTarD 2.0: an updated comprehensive database for experimentally-supported functional lncRNA–target regulations in human diseases](https://doi.org/10.1093/nar/gkac984) | Hongying Zhao 等 | Nucleic Acids Research | 2022-11-02 | 全文数据/证据类型段落 |
| [Semantic-enhanced heterogeneous graph learning for identifying ncRNAs associated with drug resistance](https://doi.org/10.1093/bioinformatics/btag029) | Hang Wei 等 | Bioinformatics | 2026-01-14 | 元数据；另核对原始模型和数据构建代码 |

日期优先采用 Crossref 的 published-online，缺失时采用 published；MVGSCA 的 Crossref 仅提供 2026 年，Europe PMC 返回的首发日期为 2026-09-01。miRTarBase 2025 与 LncTarD 2.0 的在线日期早于其数据库号卷期，表中保留在线日期。未把预印本当作已同行评议期刊论文。

[mphgnn]: https://doi.org/10.1093/bib/bbag013
[mifndra]: https://doi.org/10.1093/bib/bbag322
[hdbi]: https://doi.org/10.1021/acs.jcim.6c01116
[mvgsca]: https://doi.org/10.1109/tcbbio.2026.3729865
[ncrdllm]: https://doi.org/10.1021/acs.jcim.5c03011
[mirbind2]: https://doi.org/10.1093/bioinformatics/btag410
[parnet]: https://doi.org/10.64898/2026.08.08.743506
[nucleicbert]: https://doi.org/10.1038/s42256-026-01295-9
[forge]: https://doi.org/10.1038/s41467-026-73977-2
[dispa]: https://doi.org/10.1093/bioinformatics/btag272
[scradar]: https://doi.org/10.1371/journal.pcbi.1014392
[mirbench]: https://doi.org/10.1093/bioinformatics/btaf233
[pulsnar]: https://doi.org/10.7717/peerj-cs.2451
[omicstransformer]: https://doi.org/10.1093/bioinformatics/btag468
[benchmark]: https://doi.org/10.1126/sciadv.aed3414
[active]: https://doi.org/10.1038/s44320-026-00221-w
[shift]: https://doi.org/10.1021/acs.jcim.5c02453
[mirtarbase]: https://doi.org/10.1093/nar/gkae1072
[lnctard]: https://doi.org/10.1093/nar/gkac984
[incrd]: https://doi.org/10.1093/bioinformatics/btag029
