# PLMF-MDA 上"更有创新点、指标能拉开"的方向：前沿文献扫描 + 冷启动试点

目录：`improve/research/plmf_frontier_20260922/`　日期：2026-09-22

> 问题（用户）：目前 iNcRD-HG 在 PLMF-MDA 上最强的是"图上下文 + 扩散谱特征 + 线性读头"
> （MDR 0.951 / MDS 0.954 AUC），不够新颖、指标也不够亮眼；有没有更具创新点、好发论文的技术？

---

## 0. 结论

1. **随机划分（论文 5 折）上不要再找"新架构"。** 仓库已用 100+ 机制、注意力、跨模态 co-attention、
   MPHGNN、TabICLv2 等证明：该协议下单模型天花板 ≈ 0.951/0.954，`drug_deg` 一列就 0.90，
   把测试正边放进图的泄漏探针也只有 0.96。2024–2026 的链路预测文献（§2.1、§2.2）给出了同样的结论：
   随机负采样的链路预测被节点度主导，启发式 ≈ GNN，"null 模型（只看度）接近最优"。
2. **空间在冷启动（inductive）。** PLMF-MDA 论文自己的 Table 2：**drug-cold AUC 只有 0.664 (MDR) / 0.658 (MDS)**。
   今天在完全相同的冷启动划分上：PLMF-MDA 官方模型复现 0.647 / 0.667（test-best，1 seed 已完成，其余在跑），
   而我们现有的图上下文 + 内容 kNN 迁移特征 × LR 达到 **MDR 0.828 ± 0.010 / MDS 0.777 ± 0.025**
   （MDS 用 warm 模型补全后核迁移 0.792 ± 0.026），**领先 +0.12 ~ +0.18 AUC**，远大于 seed 方差。
   这才是能在论文里"拉开"的指标。
3. **最有发表价值的论文形态**（§4 方案 A）：
   "**面向新药物/新 miRNA 的归纳式 MDA 预测 + 度偏差校正评估**"——贡献 = ① 系统揭示现有 MDA 基准的
   度饱和与泄漏（已有全部复现材料）；② 提出 cold-drug / cold-miRNA / leave-drugs-out + 逐药物指标 + 度匹配负例的评估；
   ③ 一个 **统一 warm/cold、支持 k-shot 免重训** 的内容条件化图滤波/启发式学习模型；④ warm 持平 SOTA、cold +0.15 AUC。
   对应的领域趟路：Bioinformatics 2026 已连发 "数据泄漏基准"（btag608）、"cold-start 泄漏控制的 miRNA–disease"（RGCNMDA, btag620）、
   "线性可解释模型在冷启动 DTI 上 SOTA"（LI-DTI, btag428）、"冷启动 DTA"（PCIM-DTA, btag646）；BiB 2026 的 CAREPath 主打 disease cold-start。
4. **两条必须知道的先验工作**：
   - **TFMLinker**（arXiv 2602.08592，2026-02）已发表"表格基础模型 + 图上下文 in-context 链路预测"，
     与仓库"frozen TabICLv2 图上下文参考"同构 → 这条线不能再当方法创新点写；而且今天的试点显示 TabICLv2 在冷启动下反而不如 LR。
   - **HL-GNN**（KDD 2024, 2406.07979）给了"所有局部/全局启发式 = 邻接矩阵多项式"的统一可学习形式，
     是把 `ctx+diff` 变成端到端"可学习启发式"的现成理论骨架（§4 方法 A-a）。

---

## 1. 现状回顾：为什么随机划分上没有空间

| 证据 | 数值 | 来源 |
|---|---|---|
| 只用训练正例的药物度 | AUC 0.9031 (MDR) / 0.9135 (MDS) | `plmf_innov_20260921/REPORT_PLMF_INNOV.md` §4 |
| 度之和 `deg_r+deg_d` | 0.9149 / 0.9272 | 同上 |
| 最强单模型 ctx+diff × LR | 0.9510 / 0.9540 | 同上 §3 |
| TabICLv2 图上下文参考 | 0.9486 / 0.9513 | `plmf_strongest_20260920` |
| PLMF-MDA / ML-DTI / MPHGNN 复现 | 0.924–0.930 | `plmf_baselines_20260920`, `mphgnn_plmf_20260921` |
| 泄漏探针（测试正边入图） | 0.9616 (MDR) | `incrd_baseline_plmf_20260921/run_v4_leakprobe.py` |
| 100+ 机制 / 注意力 / co-attention / 元路径 | 增量 < 0.005，多数 ≈ 0 或负 | `NEW_MECHANISM_FINDINGS.md`, `REPORT_PLMF_INNOV.md` §7–8 |
| 线性 ≥ 树 ≥ MLP | 特征空间近似线性可分 | `REPORT_PLMF_INNOV.md` §3.2 |

MPHGNN 的"元路径"在 PLMF-MDA 上 none/shuffled = real，其全局分支还读了测试 batch 的度结构（+0.01–0.02 泄漏）。
换句话说：**warm 协议上的差异要么是 <0.005 的噪声，要么是泄漏。**

---

## 2. 前沿文献扫描（2024-09 → 2026-09）

检索：arXiv export API（18 组查询、~400 条）、Europe PMC（9 组）、bioRxiv API（2026-06 → 09）；
Semantic Scholar / OpenAlex 429、OpenReview 403、GitHub raw 超时（本机 egress 限制）。
原始结果：`lit/scan_results.json`；本节只列改变判断的论文，方括号为 arXiv id 或 DOI。

### 2.1 链路预测方法论：启发式与 GNN 的关系（决定"新架构"有没有空间）

| 论文 | 要点 | 对我们的含义 |
|---|---|---|
| HL-GNN: Heuristic Learning with GNNs, KDD 2024 [2406.07979] | 局部/全局启发式（CN、Katz、RWR…）都可写成邻接矩阵多项式，用 ~20 层"层内传播 + 层间连接"学系数 | `ctx+diff` 的端到端版本；我们的 PPR/Katz 两组 (α,β) 手选 → 可学 |
| Can GNNs Learn Link Heuristics? [2411.14711] | 消息传递 GNN 学不出共邻数（集合池化）；可训练 ID embedding 才补上，越稠密越明显 | 解释了为何 GeneralConv/GAT 在 PLMF 上 0.925 < 特征 LR 0.951 |
| Heuristic Methods are Good Teachers to Distill MLPs [2504.06193] | 用启发式做教师蒸馏 MLP，接近 GNN4LP | "启发式即上界"的又一证据 |
| NCNC (ICLR 2024) [2302.00890], MPLP [2309.00976], OCN [2505.19719] | 共邻条件化 MPNN / 高阶共邻 | 二部图上直接共邻为 0，需要 path-3 版；仓库 NCN 首版已失败 |
| D-PPR: Diffusion Signals Reveal Hidden Connections [2511.17569] | 用 PPR 信号的扩散距离做链路预测，跨拓扑稳定 | 与 `diff` 族同源，可引为"扩散特征"的理论依据 |
| Mixture of Link Predictors [2402.08583]; LPFormer [2310.11009] | 按 pair 自适应混合启发式 / 自适应图 Transformer | 仓库 pair-adaptive shrinkage 已试 ≈ 0 |

### 2.2 评估偏差：度、泄漏、冷启动（决定论文"贡献"怎么写）

| 论文 | 要点 |
|---|---|
| **Implicit degree bias in the link prediction task** [2405.14985] | 常规负采样对高度节点有隐式偏置，"只看度"的 null 模型接近最优；提出 **度校正的链路预测任务**，与推荐任务表现更一致 |
| Networked Inequality: PA bias in GCN link prediction [2309.17417] | 对称归一化图滤波带来组内"富者更富" |
| Origins of Degree Bias in GNNs [2404.03139] | 高度节点误分类概率天然更低，与训练方式无关 |
| Disentangling Node Attributes from Graph Topology (UPNA) [2307.08877] | 幂律图上 GNN 走拓扑捷径；用预训练属性做 inductive 链路函数更能泛化 |
| Domain matters: domain-informed LP evaluation [2512.23371] | 740 网络：算法排名跨域不一致，不存在"通用最优" |
| **Benchmarking data leakage in biomedical KGE**, Bioinformatics 2026 [10.1093/bioinformatics/btag608] | 训练/测试冗余、度作为非法特征（置换实验）、随机 vs cold-start vs 独立 Orphanet 测试集——**随机划分显著高估** |
| **RGCNMDA**, Bioinformatics 2026 [btag620] | miRNA–disease：折内重建全部关联派生特征；报告 random / cold-disease / cold-miRNA 三口径（0.9589 / 0.9109 / 0.8836） |
| **LI-DTI: linear interpretable DTI**, Bioinformatics 2026 [btag428] | 相似度线性组合模型，在药物/靶点无已知交互时仍能恢复 DTI，含前瞻性时间划分 —— "线性 + 相似度"在冷启动上 SOTA |
| FlexBIP, BiB 2026 [bbag471] | 解耦分子内在特征与图拓扑，强调 cold-start 数据稀缺场景 |
| miRBench (Bioinformatics 2025, 先前报告) | 负采样频率偏差比网络更重要 |

### 2.3 推荐系统：图滤波（训练无关/谱）与生成式协同过滤——与 ctx+diff×LR 同源的一整条线

| 论文 | 要点 |
|---|---|
| GF-CF [2108.07567], Turbo-CF [2404.14243], PolyCF [2401.12590], ChebyCF SIGIR 2025 [2505.00552], GSPRec [2505.11552] | 训练无关/多项式图滤波器在 CF 上打平或超过 LightGCN 类 GNN |
| How Powerful is Graph Filtering for Recommendation, KDD 2024 [2406.08827] | 图滤波去噪在稠密数据无监督即可，稀疏数据需监督；线性 GCN 表达力有限 |
| **ASPIRE** [2604.22549, 2026-04] | 传统推荐目标导致"低频爆炸"，滤波器学不动；双层优化让滤波器可学 |
| Mem-GF [2606.21540, 2026-06] | 训练无关图滤波的内存瓶颈 |
| **Training-free Adjustable Polynomial Graph Filtering for Multimodal Rec** [2503.04406] | 内容（多模态）相似图 + 交互图 → 多项式滤波融合，**训练无关**，天然处理冷物品 |
| GiffCF SIGIR 2024 [2311.08744], S-Diff [2501.00384], FlowCF [2502.07303], Graph-based Diffusion CF [2504.05029] | 把 CF 当条件生成：图热方程扩散 / 谱域各向异性扩散 / 离散流匹配 |
| **DiffCold** [2606.12245, 2026-06] | 冷启动"跷跷板困境"：warm 行为流形 vs cold 语义流形不一致，用扩散桥接 |
| S2CDR [2603.02725] | 平滑-锐化过程模型跨域推荐 |

→ ncRNA–drug 领域**没有人**用这套"图滤波/训练无关 CF"框架；PLMF-MDA 的 MDR/MDS 本质就是 user–item 隐式反馈矩阵 + 双侧内容。

### 2.4 冷启动（推荐 / DTI / 生物网络）

| 论文 | 要点 |
|---|---|
| **On Inherited Popularity Bias in Cold-Start Item Rec** [2510.11402] | 冷启动模型从 warm 模型继承并**放大**流行度偏差（冷物品无法从交互推断流行度） |
| Sparse Contrastive Learning for Content-Based Cold Item Rec [2604.12990] | 放弃 CF 对齐，纯内容 item–item 相似度 + α-entmax 稀疏 softmax |
| Bridging the Semantic-Collaborative Gap (Shallow-RHS) [2606.06225] | 非对称双塔：warm 侧消息传递、cold 侧纯内容，归纳式二部图补全 |
| SPARC: spectral cold-start [2411.01532] | 可泛化的谱嵌入，推理无需邻接 |
| AlphaFree [2603.02653] | 无 user embedding、无 ID、无 GNN 的推荐 |
| PCIM-DTA, Bioinformatics 2026 [btag646]; Co-Diffusion [2603.11125]; ColdDTI [2510.04126]; LaPro-DTA [2603.14792] | DTI/DTA 冷启动（cold-drug / cold-target / scaffold） |
| CAREPath, BiB 2026 [bbag472] | KG+LLM，disease cold-start 上 18 基线最优 |
| WBT-BGRL [2510.24927] | 归纳式加权二部图链路预测（非对比自举） |

### 2.5 基础模型 / in-context 链路预测（注意先验工作）

| 论文 | 要点 |
|---|---|
| **TFMLinker** [2602.08592, 2026-02] | 表格基础模型 + 图 in-context 学习 = 通用链路预测器 —— **与仓库 TabICLv2 图上下文参考同构** |
| Reification as a Transferable Vocabulary [2609.11347]；SIGIL [2608.08567]；HYPER [2506.12362]；GraphPFN / UniLP（先前报告） | 零样本/全归纳 GNN 与图基础模型 |

### 2.6 PU 学习与缺失机制

PULL [2405.11911]（把观测边当正、未观测当无标签的链路预测）、Conformal link prediction with FDR control [2507.07025]、
Link prediction under imperfect detection [1910.03659]。与"负例 = 未观测"的 MDA 数据完全对应，但作为主创新点风险高（需先验/假设）。

### 2.7 ncRNA–drug 2026 同行

MIFNDRA (BiB)、MPHGNN (BiB)、PLMF-MDA (PLoS CB)、MVGSCA (TCBBIO)、HDBI (JCIM)、mDADGAN (JCIM)、DC-MetaMG (JBHI)、NCRDLLM (JCIM)
—— 全部是 warm 随机划分 + 新架构；只有 PLMF-MDA 报了冷启动（且只有 0.66）。**没有一篇做度校正评估或系统的归纳式评测。**

---

## 3. 试点实验：PLMF-MDA 冷启动药物（今日完成，代码 `cold_probe.py`）

### 3.1 协议

- **plmf**（= PLMF-MDA Table 2）：测试药物 = 关联数 11–20 的药物（MDR 10 个/151 正例；MDS 14 个/225 正例）。
  测试对 = 这些药物的全部正例 + 等量随机负例（只在测试药物列采样）；训练对 = 其余药物的正例 + 等量负例。
  测试药物在训练中**完全不出现**。5 个负采样 seed。
- **ldo**：全部药物按度分层 5 折 leave-drugs-out（3 seeds），报折均值、pooled、逐药物 macroAUC。
- **plmf_mirna**：PLMF Table 2 的 miRNA-cold（5–6 条关联的 miRNA）。
- **k-shot**：每个测试药物先揭示 k 条正例（+k 负例）进训练集。
- 特征全部**只用训练正例**做 support；训练特征交叉拟合（`warm` = 按 pair 5 折；`drugcv` = 按药物分组；`lodo` = 留一药物）。
- 对照：PLMF-MDA **官方模型**（缓存冻结编码器、与 `plmf_baselines_20260920` 同实现）跑**同一划分**（`plmf_cold_ref.py`）。

### 3.2 关键结果（AUC / AUPR，均值 ± seed 标准差；完整表见 `RESULT_TABLES.md`）

| 方法 | MDR drug-cold | MDS drug-cold | 备注 |
|---|---|---|---|
| PLMF-MDA 论文 Table 2 | 0.6642 / 0.6534 | 0.6579 / 0.6352 | 论文值 |
| PLMF-MDA 官方模型，同划分（test-best） | 0.647 / 0.628 (seed0) | 0.667 / 0.658 (seed0) | 其余 seed 在跑，见 `results/plmf_cold_*.csv` |
| 只用 miRNA 度 `degree_r`（无训练） | 0.781 / 0.784 | 0.769 / 0.764 | 冷启动下的"流行度 null 模型" |
| 内容 kNN 标签迁移 `knn_tan0_mean`（无训练） | 0.801 / 0.812 | 0.769 / 0.780 | Tanimoto 加权邻居药物 profile |
| ctx (33) × LR，warm 交叉拟合 | 0.823 / 0.808 | 0.772 / 0.773 | 原 iNcRD-HG 图上下文档 |
| **ctx + knn × LR，warm** | **0.828 ± 0.010 / 0.818 ± 0.022** | 0.777 ± 0.025 / 0.781 ± 0.024 | 最优（MDR） |
| warm 模型补全矩阵 → 核迁移 (`transfer`) | 0.815 / 0.814 | **0.792 ± 0.026 / 0.779 ± 0.028** | 最优（MDS） |
| frozen TabICLv2 in-context（同特征） | 0.787 / 0.799 | 0.782 / 0.780 | 冷启动下不如 LR |
| 机制桥 ENCORI×ChEMBL×HumanNet (`mech`) 单独 | 0.700 / 0.684 | 0.670 / 0.653 | 覆盖 79/105、96/140 药 |
| 两侧核岭回归 KronRLS（k-mer 余弦 × Tanimoto） | 0.632 / 0.607 | – | k-mer 核过平滑 |

读数：
1. **+0.15–0.18 AUC 的差距是真实且稳定的**（sd 0.01–0.03），来自两个简单信号：miRNA 流行度先验 + 内容邻居药物的关联 profile 迁移。
   PLMF-MDA 用 RNA-FM/ChemBERTa + 跨模态注意力从 100 个药物学"内容→关联"映射，样本远远不够，所以只有 0.65。
2. **训练/测试一致性（drugcv/lodo）没有帮助全局 AUC**：warm 交叉拟合最好。原因：warm 下模型学到的 `degree_d` 在测试时对所有冷药物同为 0，
   只是常数平移；而 drugcv 改变 support 规模导致特征尺度漂移。lodo 略提升逐药物 macroAUC（MDR 0.860 vs 0.839）。
3. **加更多特征族反而变差**（cdiff/align/mech/diff 在 drugcv 下 0.75–0.79）：冷启动是小样本问题，LR 也会过拟合到 warm 特有的列。
4. **全药物 LDO 揭示评估陷阱**：pooled AUC 0.69 但逐药物 macroAUC 0.85（MDR）。冷药物的流行度不可知 → 跨药物分数不可比 →
   全局 AUC 崩塌。**冷启动论文必须报逐药物指标**（PLMF-MDA 用度相近的 10 个药物回避了这一点）。
5. TabICLv2（仓库 warm 最强）在冷启动下**低于 LR**：基础模型 in-context 不是冷启动的杠杆，也印证 §2.5 该路线不宜作为创新点。

### 3.3 miRNA-cold（PLMF Table 2 的另一半）与 k-shot 曲线

| 协议 | 方法 | MDR AUC / AUPR | MDS AUC / AUPR |
|---|---|---|---|
| miRNA-cold（5–6 条关联的 miRNA；MDR 150 个、MDS 116 个） | PLMF-MDA 论文 Table 2 | 0.9115 / 0.8977 | 0.9190 / 0.9037 |
| miRNA-cold | 只用药物度 `degree_d`（无训练） | 0.9242 / 0.9049 | 0.9331 / 0.9197 |
| miRNA-cold | **ctx × LR，warm** | **0.9327 ± 0.005 / 0.9200 ± 0.009** | **0.9360 ± 0.006 / 0.9235 ± 0.011** |

miRNA-cold 对所有方法都"容易"，因为它仍由**药物流行度**主导（`degree_d` 一列就超过 PLMF-MDA）。真正的难点只在 drug-cold。

k-shot（每个冷药物先揭示 k 条正例 + k 条负例进训练集；ctx+diff × LR，warm 交叉拟合，**不重训特征提取、只重算 support**）：

| k | MDR AUC / AUPR | MDR macroAUC | MDS AUC / AUPR | MDS macroAUC |
|---|---|---|---|---|
| 0 | 0.8094 / 0.7995 | 0.822 | 0.7682 / 0.7678 | 0.780 |
| 1 | 0.8198 / 0.7935 | 0.842 | 0.7961 / 0.7773 | 0.801 |
| 3 | 0.8514 / 0.8429 | 0.856 | 0.8130 / 0.7932 | 0.812 |
| 5 | 0.8686 / 0.8657 | 0.874 | 0.8266 / 0.8125 | 0.820 |
| 10 | 0.8764 / 0.8732 | 0.890 | 0.8504 / 0.8409 | 0.826 |

frozen TabICLv2 在 MDS 上同样单调上升（k=1/3/5：0.786 / 0.811 / 0.826），但每个 k 都低于 LR。
这条曲线是端到端模型（PLMF-MDA/MPHGNN）给不出来的：它们要重训才能吸收新药物的少量关联；图滤波/启发式模型只需重算 support。


---

## 4. 推荐的论文方案

### 方案 A（首选）：归纳式 MDA 预测 + 度偏差校正评估

**标题方向**："Popularity-aware inductive prediction of miRNA–drug resistance associations: why warm benchmarks are saturated and how to generalise to new drugs"

**贡献**
1. **审计**：在 PLMF-MDA、DLST/DLMVF、MPHGNN 三个 2026 基准上证明：度-only null 模型 0.90+，随机划分 SOTA 之间 <0.005，
   MPHGNN 的 batch 图泄漏，PLMF-MDA 基线被低估 0.1–0.15（材料全部已有：`plmf_baselines`, `mphgnn_plmf`, `DLMVF_paper_baselines`）。
   引用 [2405.14985][btag608][miRBench] 说明这是领域共识而非本文孤例。
2. **评估协议**：cold-drug / cold-miRNA / leave-drugs-out 分层 + **逐药物 macro AUC/AP** + **度匹配负例**（warm 协议的度校正版，来自 [2405.14985]）。
3. **方法**（真正的"技术"）：一个 **统一 warm/cold 的内容条件化图滤波 / 可学习启发式模型**，两种可选实现（先做 a，b 作对照）：
   - **(a) HL-GNN 式可学习启发式，跑在内容增强的异质邻接上** `Â = [[γ_r S_r, A],[Aᵀ, γ_d S_d]]`：
     把 ctx/diff/cdiff/knn 手选的 PPR/Katz/路径全部收进一个多项式滤波器 `Σ_k β_k Â^k`，系数、γ、度归一化指数一起学
     （度保持归一化是关键，仓库已证明 softmax 归一化毁掉度信号）。冷药物在 A 中没有边，但通过 `S_d` 有路径 → 天然归纳。
     文献中 HL-GNN 只做同质图、无内容边；[2503.04406] 做了内容图滤波但训练无关且非生物 → 组合是新的，
     并且读头仍是线性的（可解释：每个 β_k 对应一种"路径语义"）。
   - **(b) 流行度解耦双分支** `score = pop(m) + pop(d)·1[warm] + compat(m,d)`：compat 只在度匹配负例上训练
     （借 DDC KDD 2026 与 [2510.11402] 的分析），cold 时 pop(d) 边缘化；对比"继承流行度偏差"的基线。
4. **结果预期**：warm 持平 0.951/0.954（不可能更高，要在文中说清），cold-drug **0.83 vs 0.65**，cold-miRNA 待测，
   k-shot 曲线（免重训），逐药物 macro 指标；再加 iNcRD-HG 自有数据的 cold_drug surface 作为第二个数据集。

**风险**：审稿人问"为什么不是更强的 GNN"→ 用 §1 的饱和证据 + [2411.14711][2504.06193] 回答；
"冷启动测试集太小"→ 用 LDO 全药物 + 多 seed + 逐药物指标；"kNN 迁移太简单"→ 这正是卖点（LI-DTI 2026 在 Bioinformatics 用同样的论证）。

### 方案 B：评估/基准论文（BiB "Problem Solving Protocol" 或 Bioinformatics）

不提新模型，只做 ① 度饱和审计、② 泄漏审计（MPHGNN batch 图、DLMVF/DLST 基线复制）、③ 冷启动重评 2026 的 8 个方法。
工作量最小（复现材料全在），新颖性靠"第一个系统审计 ncRNA–drug 基准的人"。缺点：不满足"新技术"。

### 方案 C：换到 MPHGNN 自有 MiDrug 基准追 0.97+

只有"数字好看"一个好处，且该基准的官方协议 transductive + 测试边入图，本身有泄漏争议。不推荐作为主线。

### 不建议再投入
- 图注意力 / 跨模态 co-attention / HGT / 对比学习 / 元路径的变体（仓库均 ≈ 0 或负）。
- 把 TabICLv2 in-context 当核心创新（TFMLinker 已发表；冷启动下不如 LR）。
- 预训练 RNA/化学大模型做主特征（PLMF-MDA 已做，warm 0.926、cold 0.66）。

---

## 5. 可直接执行的实验计划（按顺序）

1. **补全冷启动评测面**（1 天）：`cold_probe.py` 已支持 plmf / ldo / plmf_mirna / kshot。加 5 seeds × LDO；
   warm 协议加"度匹配负例"（对每个正例按 drug 度分箱采负例）；每张表报 AUC/AUPR + macro-drug AUC/AP。
2. **基线上冷启动**（1–2 天，GPU）：PLMF-MDA（`plmf_cold_ref.py`，在跑）、ML-DTI / GraphDTA / SubMDTA / GCNNMMA
   （改 `plmf_baselines_20260920/run_bench_plmf.py` 读 `cold_split`）、MPHGNN（`mphgnn_plmf_20260921`，需逐对打分）。
3. **方法原型 A-a**（3–5 天）：在 `[[γS_r, A],[Aᵀ, γS_d]]` 上实现可学习多项式滤波（K≤8，度保持归一化 `D^{-p} Â D^{-(1-p)}`），
   线性读头，BCE + 度匹配负例；对照 ctx+knn×LR、HL-GNN 原版（无内容边）、训练无关多项式滤波 [2503.04406]。
   通过标准：warm 不低于 0.950/0.953，cold-drug ≥ 0.83/0.79，且 permutation 对照（打乱 S_d）显著下降。
4. **方法原型 A-b**（2 天）：流行度解耦头 + 度匹配训练；看 macro-drug AP 是否补回（仓库在 iNcRD-HG 上 TabM 的 macro AP 掉 6.6 个点是同一问题）。
5. **外部药物先验消融**（可选，1 天）：ATC/MoA 类别、ChEMBL 靶点 Jaccard、（若可得）LINCS 签名相似度 → 只在 cold-drug 上验证是否超过 Tanimoto。
6. **第二数据集**：iNcRD-HG 自有 MiDrug/LncDrug 的 cold_drug surface（仓库已有划分），复用同一 pipeline。

---

## 6. 产物与复现

```
plmf_frontier_20260922/
├─ lit/search.py, lit/scan_results.json     # 文献扫描（arXiv/EuropePMC/bioRxiv）
├─ cold_probe.py                            # 冷启动协议 + 特征族 (ctx/diff/cdiff/scf/align/knn/mech) + LR/HGB + 无训练 scorer
├─ plmf_cold_ref.py                         # PLMF-MDA 官方模型跑同一冷启动划分
├─ tabicl_cold.py                           # frozen TabICLv2 in-context 跑同一划分（.venv-tabicl）
├─ run_cold_batch{,2,3}.sh, run_tabicl_cold.sh, logs/
├─ summarize.py → RESULT_TABLES.md          # 全部结果表
└─ results/cold_probe.csv, tabicl_cold.csv, plmf_cold_{MDR,MDS}_plmf.csv
```

```bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
$PY cold_probe.py --dataset MDR --protocol plmf --families ctx,knn --fit warm            # 0.828 / 0.818
$PY cold_probe.py --dataset MDS --protocol plmf --families ctx,diff --scorer transfer    # 0.792 / 0.779
$PY cold_probe.py --dataset MDR --protocol ldo  --families ctx --fit warm --seeds 0,1,2  # 逐药物 macroAUC 0.85
CUDA_VISIBLE_DEVICES=0 $PY plmf_cold_ref.py --dataset MDR --protocol plmf               # PLMF-MDA 官方模型同划分
$PY summarize.py
```
