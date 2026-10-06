# 最新论文里可用的技术（2026-08 ~ 2026-09）— 面向 incrd-hg / PLMF-MDA

扫描范围：arXiv / Europe PMC / bioRxiv（`plmf_frontier_20260922/lit/scan_results.json`，68 篇 ≥2026-08）
+ 定向 Web 检索（2026 年 8–9 月最新）。只收录**能落到本项目**的技术，并给可用性判断。

项目现状（决定"什么才叫可用"）：
- warm 5cv 已饱和（度/流行度主导，87 机制增量 ≈1e-4）；当前最强 `ctx+diff × LR` = 0.9510/0.9540。
- 实测仍有余量的杠杆只有三条：**跨数据集迁移**、**负样本/偏差**、**新信息/新监督**；冷启动次之（最冷档 +0.008）。
- 所以下面按"是否落在这些杠杆上"排序，而不是按论文热度。

---

## 0. 一句话结论

有 **3 条可直接用**的技术，且都落在已验证的杠杆上：
**Sigil**（跨图零样本链接预测）、**PCIM-DTA**（pair-conditioned 冷药）、**Better Negatives**（负样本选择）。
**HAWKEYE** 可用但只作**诊断**（实测其 2-hop 桥救不了冷药，见 §7）。
另有 3 条低成本可试：**Reification**（零样本 LP 词表）、**LNMGAT**（可靠伪负样本）、**ADHGMDA**（虚拟节点桥）。

---

## 1. 直接可用（Tier 1）

### 1.1 Sigil — 全归纳图基础模型（跨 MDR↔MDS 迁移首选）
- **论文**：Neural Message Passing on Structural Interaction Graphs for Fully-Inductive GNNs，arXiv **2608.08567**（2026-08-13）。
- **技术**：把任意带属性图"提升"成**结构交互图 SIG**（节点=输入特征维度，边=特征在多个邻域阶上的对齐），
  再用关系消息传递把任意维特征映射到**固定维统一空间**。一个模型在**单张图**上预训练即可**零样本**迁移到
  特征空间完全不同的新图；单 hop 离散关系时**严格泛化 KG 基础模型（复现 ULTRA）**。
- **为何对本项目可用**：MDR(1317/105) 与 MDS(1252/140) 共享 946 miRNA、81 drug，但**特征空间不同**——
  Sigil 天生就是解决"跨图、特征维度不一致"的；论文里它同时击败 TFMLinker、NBFNet、ELPH、NCNC、MPLP+。
- **对应路线**：非冷启动的**跨数据集迁移**（报告 §9.1）。
- **落地**：MDR 预训练 → MDS 零样本/少样本；对照 = 逐数据集训练的 LR/TabICLv2。目标域 warm −0.02~−0.05 内即成立。
- **可用性**：高（论文方法清晰、基于 PyG）；代码需在 GitHub 上找（arXiv 未直给链接）。

### 1.2 PCIM-DTA — 成对条件化，正好补上"静态 content→latent 失败"的坑
- **论文**：PCIM-DTA，*Bioinformatics* 2026, btag646；**代码公开** `github.com/1322469934/PCIM-DTA`。
- **技术**：从全局 drug/target 特征导出**pair-specific condition vector**，用它去**调制注意力、pair-token、
  分布感知重标定、回归参数**；图消息传递抓高阶依赖。在 **Warm / Cold-drug / Cold-target / Cold-both / Scaffold-drug**
  五种设置评测。
- **为何对本项目可用**：本仓库实测"线性 content→latent 映射"失败（ridge 0.61–0.65），根因是**静态映射、不随 pair 变**。
  PCIM-DTA 的核心恰是**成对条件化**——把条件向量用起来，是让冷药内容迁移真正 work 的正解。
- **对应路线**：R1 冷药方法（把 `latent profile` 升级为 conditioned 版）。
- **落地**：在 `asym_probe.py` 的 `lat_profile` 上，用 miRNA 侧全局特征生成 condition，调制 drug 内容→协同映射。
- **可用性**：很高（代码可直接改，任务结构相近）。

### 1.3 HAWKEYE — 2-hop 内聚桥 + "结构何时有用"的免费诊断
- **论文**：HAWKEYE: Seeing One Layer Deeper，CIKM **2026**，arXiv **2608.22727**（2026-08-24）；代码/数据已放出。
- **技术**：实测发现稀疏图上 **1-hop 共邻近乎随机（discAUC≈0.50）**，真正有判别力的是 **2-hop 内聚桥**
  （degree → k-core → k-truss，discAUC 0.73–0.98），作为结构通道的**即插即用替换**；
  并给出**训练无关的 2-hop discAUC 预测器**来判断"加了结构会不会有用"。
- **在本项目的实测（见 §7，`disc_auc_probe.py`）**：HAWKEYE 的"看深一层"框架成立，但结论要修正——
  warm 面**度数仍是主信号**（`deg_d` 0.905/0.915 > `L3` 0.841/0.850），且**归一化后的结构特征反而低于随机**
  （`L3_norm` 0.43/0.49）；**冷药面所有图结构特征全部塌成 0.5000**（冷药在训练图里没有边，L3/L5 恒为 0），
  只有 `deg_r`(0.78) 存活。→ HAWKEYE 的**诊断价值**（先算 discAUC 再决定加不加结构）可用，
  但其 2-hop 桥**救不了冷药**（根因是冷节点无边，不是"看得不够深"）。
- **对应路线**：R2 结构表达升级（warm）+ R1 分层诊断；**别指望它解决冷启动**。
- **可用性**：诊断框架高；直接当冷药解法：低（实测证否）。

### 1.4 Better Negatives — 负样本选择策略（非冷启动，低成本高价值）
- **论文**：Better Negatives, Better Predictions，CIKM **2026**，DOI 10.1145/3774905.3794655；
  代码 `github.com/SLIMlaboratory/glow26` + Zenodo。
- **技术**：用**潜空间图嵌入 + 节点类型 + 边语义**生成负边，显著优于随机负采样（balanced accuracy 统计显著提升），
  且**与下游模型无关**。
- **为何对本项目可用**：PLMF-MDA 负例是**随机零**（`neg_MDA_*.edgelist`）；本仓库已实测"单列度数/属性支撑就能刷分"。
  换负样本协议是**不改数据、不改任务**就能换一个更公平的评测面，天然是 benchmark/分析论文。
- **对应路线**：报告 §9.2 负样本/偏差基准。
- **可用性**：高（策略通用，直接套到 miRNA–drug 二部图）。

---

## 2. 低成本可试（Tier 2）

| 论文 | 日期/来源 | 可用技术 | 对项目的用法 |
|---|---|---|---|
| **Reification as a Transferable Vocabulary** | arXiv **2609.11347**（09-10） | 把图"具体化"（事实→节点 + 6 条元关系），**普通 GAT** 零样本迁移 40 个归纳基准，追平 ULTRA | 最便宜的跨数据集迁移基线；实现量极小，先跑通再上 Sigil |
| **LNMGAT** | Front. Bioinf. 2026-07 | LapRLS 引导的**可靠伪负样本挖掘**，明确"未观测≠负"，含 12 个冷启动设置 | 替换随机负采样；与 Better Negatives 二选一或互补 |
| **ADHGMDA** | Eng. Appl. AI 2026-04 | **K-means 虚拟节点**作语义桥，把孤立节点接入网络；训练时模拟稀疏做冷启动；梯度冲突感知多任务 | 冷药/孤立 miRNA 的廉价桥接技巧，加到现有图即可 |
| **MAGMDA** | BMC Bioinf. 2026-03 | **自适应多阶矩卷积**（Gumbel-Softmax 选阶 + 动态阈值 + 跨阶注意力），替代固定阶数 | 替换 `diff` 里写死的 path-3/5/7 阶数 |
| **RGCNMDA** | 2026-08-01 | pathway-bridged 关系图 + 自适应多视图，用于 miRNA–disease | 若走"新信息/通路"路线（§9.3） |
| **mDADGAN** | JCIM 2026-04 | 引入 lncRNA 扩图 + **扩散扰动去噪** + 代价敏感对抗 | 走"新信息 + 类别不平衡"路线 |

---

## 3. 任务重定义类（Tier 3，需新标签/改任务）

- **DHGCMDA**（BMC Bioinf. 2026-04）：预测**关联类型**（up/down-regulation）而非存在性——对 MDR/MDS 可直接做
  "耐药 vs 敏感"二分类；双视图超图 + 对比 + HGT。
- **MRGBMDAT**（*Bioinformatics* 2026-07）：多关系图卷积 + **双线性融合** + 迭代相似度负采样，做关联类型预测。
- **CS-DTA**（Front. Chem. 2026）：LLM 双编码器（ESM2/ChemBERTa）+ 跨模态注意力；**关键负结论**——
  显式交互建模在 warm 有用、在 cold 反而无益，鲁棒性来自**预训练单模态编码器**。→ 冷启动别押跨模态注意力。
- **HierHGT-DTI**（BMC Bioinf. 2026-09-14，代码可用）：多尺度关系感知异质图 Transformer 做冷启动 DTI。

---

## 4. 建议立刻做的 3 个最小实验

1. **HAWKEYE 诊断**（**已完成**，见 §7）：在 PLMF-MDA 上算 1-hop/2-hop/度数的 discAUC。
   结论：warm 由药物度数主导（0.905/0.915），2-hop 桥并不更强；冷药面所有结构特征塌成 0.5000。
   → "结构负迁移"有了硬证据，也说明别指望深阶结构救冷药。
2. **Reification 跨数据集基线**（1–2 天）：MDR 训练 → MDS 零样本，与逐数据集 LR 比；通了再上 Sigil。
3. **Better Negatives 负样本替换**（1 天）：用潜空间嵌入挑难负例替换随机零，重跑 warm 与冷药，
   看指标是否"更难但更真"，并量化随机负采样带来的度数捷径。
4. **PCIM-DTA 式成对条件化**（2–3 天）：给 `asym_probe.py` 的 `lat_profile` 加 miRNA 侧 condition 向量，
   检验冷药最冷档能否超过当前 0.8698。

---

## 5. 与已有结论的关系

- 这些技术**没有推翻**"warm 二分类已饱和"的结论：它们作用在**跨数据集 / 负样本协议 / 冷启动**这些还没被榨干的面上。
- 最匹配的落点是上一轮报告 §9.1（跨 MDR↔MDS 迁移）与 §9.2（负样本/偏差基准）：
  **Sigil/Reification 服务前者，Better Negatives/LNMGAT 服务后者，PCIM-DTA/HAWKEYE 服务冷药分层方法。**
- 明确不建议：再在 warm 上换 GNN/注意力/融合；CS-DTA 已说明冷启动押跨模态注意力收益有限。

## 6. 参考（可点开核对）

Sigil arXiv:2608.08567 · HAWKEYE arXiv:2608.22727 (CIKM'26) · Reification arXiv:2609.11347 ·
PCIM-DTA Bioinformatics 2026 btag646 (github.com/1322469934/PCIM-DTA) ·
Better Negatives CIKM'26 10.1145/3774905.3794655 (github.com/SLIMlaboratory/glow26) ·
LNMGAT Front.Bioinf. 2026 10.3389/fbinf.2026.1882476 · ADHGMDA Eng.Appl.AI 2026 10.1016/j.engappai.2026.114812 ·
MAGMDA BMC Bioinf. 2026 10.1186/s12859-026-06433-z · RGCNMDA 2026-08-01 · mDADGAN JCIM 2026 10.1021/acs.jcim.6c00254 ·
DHGCMDA BMC Bioinf. 2026 10.1186/s12859-026-06436-w · MRGBMDAT Bioinformatics 2026 btag534 ·
CS-DTA Front.Chem. 2026 10.3389/fchem.2026.1834317 · HierHGT-DTI BMC Bioinf. 2026 10.1186/s12859-026-06637-3

---

## 7. 实测：结构信号在本数据上的判别力（HAWKEYE 诊断，2026-09-22）

脚本 `disc_auc_probe.py`，结果 `results/disc_auc_probe.txt`。对每个 pair 直接用训练正例图算训练无关的
discAUC（越高越能区分正负）：`deg_d`=药物度、`deg_r`=miRNA 度、`L3`=长度 3 路径（2-hop 内聚桥）、
`L5`=长度 5 路径、`*_norm`=按 deg_r·deg_d 归一化。

| 面 | deg_prod | deg_r | deg_d | L3 | L3_norm | L5 | L5_norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| MDR warm 5cv fold0 | 0.8390 | 0.6624 | **0.9050** | 0.8410 | 0.4307 | 0.8316 | 0.4081 |
| MDS warm 5cv fold0 | 0.8436 | 0.6551 | **0.9152** | 0.8501 | 0.4865 | 0.8361 | 0.4520 |
| MDR 冷药 deg11-20 | 0.5000 | 0.7781 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| MDS 冷药 deg11-20 | 0.5000 | 0.7670 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

**读数（三条硬结论）**：
1. **warm 面主信号是药物度数**（0.905/0.915），`L3`(2-hop) 只有 0.84–0.85，且与度数高度相关 →
   HAWKEYE 的"2-hop 更判别"在本二部图上不成立，度数才是捷径。这也再次解释了 warm 为何饱和。
2. **归一化结构特征低于随机**（0.41–0.49）：按 deg 归一化会抹掉流行度信号、留下反向噪声——
   与仓库"softmax 注意力崩溃、需度数保持"的旧结论一致。
3. **冷药面所有结构特征塌成 0.5000**：冷药在训练图里没有边，`deg_d=L3=L5≡0`，只有 miRNA 度(0.77–0.78)存活。
   → 冷启动的根因是**冷节点无边**，不是"结构看得不够深"；HAWKEYE 类深阶结构**救不了冷药**，
   内容 kNN/成对条件化（PCIM-DTA 式）才是对的方向。

**对选型的直接影响**：HAWKEYE 当**诊断工具**（先算 discAUC 再决定加不加结构）可用且便宜；
不要把它当冷药解法。Sigil/Reification 的跨数据集迁移不受此影响（目标域图里有边）。
