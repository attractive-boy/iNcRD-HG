# 阅读与实现核查记录

日期：2026-09-13。本文说明证据读到哪一层，供核查 [研究建议](RECOMMENDATIONS.md)。8 篇跨领域选文阅读了全文关键部分，另有 MotherNet 官方摘要/版本核对。没有将检索命中数、下载数当作精读数。

## 1. DiSPA

- 来源：[正式论文](https://doi.org/10.1093/bioinformatics/btag272)，本地 `broader_search/dispa.xml`。
- 阅读：2.3.1/2.3.2 的通路与药物输入、2.5 差分交互、4 Discussion。
- 核心操作为 `softmax(Q1 K1ᵀ) − λ softmax(Q2 K2ᵀ)`，分别用于 Path2Sub 和 Drug2Path；这是两个学习到的注意力图之差，不是相关性正负号。
- 作者将单细胞/空间迁移限定为缺少真实药敏测量的探索，并指出注意力是模型归因。
- 核对 [官方代码](https://github.com/sslim-aidrug/DiSPA/tree/2df32f6bc53058dab3f0d13f9d6f1fbed16fc9e6)：`src/modules/diff_cross_attn.py` 和 `src/model.py`，可见双 attention slot、学习到的 λ、RMSNorm 和双向调用。没有运行其训练、所有数据预处理或补充消融。
- 对本项目的具体限制：需要蛋白编码基因/通路信息；原文的 BRICS 失败药物排除规则不适合直接用于保持原评测集合的含铂药耐药任务。

## 2. scRADAR

- 来源：[PLoS 正式全文](https://doi.org/10.1371/journal.pcbi.1014392)，本地 `broader_search/scradar.xml`。
- 阅读：4.5 静态/动态药物指纹、4.6 FiLM、4.7 原型路由、4.9.4 额外未见药物分析、3 Discussion。
- 动态指纹来自 LINCS 小分子扰动后表达，原文将同药物的多条签名先取未加权均值，再投影到 PROGENy。不能描述为已经解决剂量/时间条件化。
- 作者明确说部分响应标签是操作性二分类标签，而非逐细胞功能测量；额外药物留出仍在单个研究和共享细胞系背景中。14 条 PROGENy 通路也限制了开放机制发现。
- 核对 [代码版本](https://github.com/bliulab/scRADAR/tree/31cc93fe76e7c21e24089f0de3ffebc7199faf94) 的 README、`code/models/proto_mech.py`、`code/features/drug_fingerprints.py`：原型是直接训练的参数，FiLM 按药物特征生成通道尺度/偏移，预测使用余弦 top-k 原型加权。它不是由当前支持案例生成的新原型。
- 实现差异：论文 4.7.1 写有 `q=Wq z+bq`；本次读取的前向路径直接归一化 FiLM 后输入计算原型余弦，没有对应的独立 query 投影。FiLM 还受配置开关控制。这里仅说明所读版本的行为，未做完整复现或代码正确性认证。

## 3. Madrigal

- 来源：[arXiv:2503.02781v2](https://arxiv.org/abs/2503.02781v2)，2025-09-24 版本；首发 2025-03-04。按预印本引用，未将其当作已核实的期刊正式版。
- 官方 [HTML 全文](https://arxiv.org/html/2503.02781v2) 保存为 `broader_search/madrigal_full.html`；提取的分块在 `madrigal_blocks.json`。
- 阅读：Discussion、Online Methods 2.1 与 2.2.2–2.2.4。
- 结构、知识图、细胞活性加 16 个细胞系扰动表达，共最多 19 个模态；模态编码对齐后通过 attention bottleneck 融合，训练时随机保留可用模态的非空子集。
- 结构是对比对齐的锚。原文对互信息目标的解释带有条件独立假设，不意味着本项目应强制不同癌种、剂量和 RNA 调控状态的表示相同。
- Discussion 明确指出适应症、人口、剂量、时间结局与少见药物类别限制；丰富的单药模态不能消除整个药物类别缺少训练案例的问题。
- PMC 页面只提供摘要，并说明所选许可不允许 PMC 存档全文；全文从论文指定的 arXiv 公开入口取得。没有通过 PMC 空摘要记录推断方法，也没有声称复现其临床实验。

## 4. HyperPCM

- 来源：[JCIM 正式论文](https://doi.org/10.1021/acs.jcim.3c01417)，`broader_search/hyperpcm.xml`。
- 阅读：Context Module、Ablation Study、Discussion and Conclusions。
- Modern Hopfield Network 用训练中独立目标蛋白的表示作为记忆，增强查询蛋白表示；hypernetwork 再生成该目标的 QSAR 参数。不是普通的无条件单预测头。
- 作者在 Discussion 提出未来研究 microRNA 的药物相互作用；也承认零样本指标尚不足以支持实际应用，以及训练成本/小数据情况下树模型的竞争力。
- 本项目可以借鉴条件化计算，但“从当前支持集生成原型”是项目延伸，不是 HyperPCM 原论文已经完成的功能；它的结合亲和力与我们的耐药关联监督也不同。

## 5. TDMD 实验研究

- 来源：[The EMBO Journal，2026-08-17](https://doi.org/10.1038/s44318-026-00889-8)，出版商全文 `broader_search/tdmd.html`。
- 阅读：摘要、NREP–miR-29b 的相关结果、ZSWIM8 非依赖 miR-30c 降解机制及 Discussion 的对应段落。
- 原文通过实验区分经典靶向抑制和转录本诱导 miRNA 降解；部分靶标由直接结合与过表达支持，内源必要性仍需其他实验。
- 对 NREP 与肿瘤表型，原文保留关联与因果之间的界限；对 ZSWIM8 非依赖机制，也明确要求专门的后续机制筛查。
- 用途是指导关系语义与证据等级设计，不是为所有相关图边赋予“激活/抑制”的真标签。本轮没有下载其全部实验数据或补充材料。

## 6. OmicsTransformer

- 来源：[Bioinformatics 正式论文](https://doi.org/10.1093/bioinformatics/btag468)，`broader_search/omicstransformer.xml`。
- 阅读：2.4.2 遮蔽一致性、2.4.5 置信度融合、2.5 模态 dropout、4.6 Limitations and future directions。
- 真实类别概率 TCP 只在训练时可知，另训练网络预测 TCP 后用于推理。不能在验证或部署中直接读取查询标签作为“置信度”。
- 作者指出简单加权的模态交互不足，需更丰富的机制交互；还需跨中心验证与分布变化下的校准。
- 这与当前项目已试的表达视图一致性有重叠，因此不把“加一致性损失”重新列为独立新主线。新用途限于有不同信息的模态，以及对缺失/不可靠模态的处理。

## 7. PULSNAR

- 来源：[PeerJ Computer Science 正式论文](https://doi.org/10.7717/peerj-cs.2451)，`broader_search/pulsnar.xml`。
- 阅读：PU data assumptions、PULSNAR algorithm、Clustering rationale，以及 Discussion 的选择机制与聚类局限。
- 估计的 α 是未标注样本中阳性的比例，不直接等同于全部数据的类别先验。流程先交叉验证预测，再对正例按关键特征分群，分群调用 SCAR 子程序。
- 假设正标签正确，而且未标注正例的每个子类型均在标注正例中有支持。簇内选择机制假设仍然存在。
- 作者承认不知道 SCAR/SNAR 时的选择困难；SCAR 下误分多个近重复簇会过计数，并且 SNAR 聚类版本没有通用上界保证。
- 本项目未知配对多、来源偏差明显，构成研究动机；这些事实并不能验证该方法所需的全部假设。它也不能处理任意错误正标签。

## 8. miR-CellMap

- 来源：[Genes 正式论文](https://doi.org/10.3390/genes17080846)，`broader_search/mircellmap.xml`。
- 阅读：摘要与 Discussion，包括结果解释、数据范围和 Future Work。
- 作者将恢复已知来源支持的调控边限定为 evidence recovery；19 个 matched single-cell profiles 不足以支撑广泛单细胞泛化。
- 将其作为“上下文相关调控程序”已有研究的依据，不作为本项目新增 HGT 的性能依据；本轮没有复现其 Bayesian topology 或竞争模块。

## 摘要级补充：MotherNet

[arXiv:2312.08598v2](https://arxiv.org/abs/2312.08598v2) 官方摘要与提交历史已核对。其核心概念是一次读入训练支持集后生成子网络参数。HTML 与 PDF 全文请求失败，保留错误；本轮只引用这一摘要明确陈述的结构思想，不声称阅读了全文、Discussion 或精确复现结果。

## 与本项目最直接的前序工作与数据资源

- [MPHGNN](https://doi.org/10.1093/bib/bbag013)：阅读数据、gene similarity、元路径/全局双模块与 Conclusion。已直接构建 miRNA–gene–drug 图；作者仍指出上下文和连续剂量反应不足。
- [MIFNDRA](https://doi.org/10.1093/bib/bbag322)：阅读预训练 GIN/SpliceBERT、多类 ncRNA 互作、多模态融合和 triplet loss。限制了“几个已有模块拼起来就是新方法”的叙事。
- [miRTarBase 2025](https://doi.org/10.1093/nar/gkae1072)：阅读 drug resistance、CLIP/PAR-CLIP 证据段。药物部分来自 ncRNADrug；高通量结合证据与报告基因/蛋白实验的证据强度有区分。
- [LncTarD 2.0](https://doi.org/10.1093/nar/gkac984)：阅读数据整理与功能调控类别。该版有 5,531 条功能调控、2,829 条表达关联，不能全部解释为直接功能调控边。

上述 4 篇的 XML 与检索元数据保存在本目录根部。用户提供的 iNcRD-HG 原文引述与代码对应关系另见 [16 项核查](AUDIT_16_DIRECTIONS.md)。

## 方法与可复核性

采用此前已使用的 `nature-academic-search` 与 `nature-downloader` 工作方式：多源发现后回到官方元数据、公开原文和作者代码。当前没有挂载学术 MCP，本次通过 Europe PMC、Crossref、OpenAlex、arXiv、出版商和 GitHub 公共接口完成。

检索截至 2026-09-13，主要关注近两年，也保留与任务直接相关的较早方法。原始查询、无关命中、失败请求、全文和代码哈希均保留。arXiv API 限流后使用其他发现来源；公开论文页面的后续请求按具体论文进行。未请求机构权限、未接入外部聊天服务、未运行第三方模型训练。

论文与代码阅读支持的是方法选择和实现可行性判断。跨论文不同数据集上的数值没有与本项目 AUC/AP 直接比较；本轮没有新增任何性能候选。
