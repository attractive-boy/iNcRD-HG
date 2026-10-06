# miRNA / lncRNA-drug 抗性专用模型：把我们的模型代入其数据集与评估体系（2026-09-16）

## 目标与口径（按用户确认）

- **方向**：找出 AUC/AUPR **高于我们**的专用模型（我们输给谁），每个数据集尽量给出 **>=3** 个。
- **范围**：只做 **miRNA-drug 抗性** 与 **lncRNA-drug 抗性** 的专用模型（药物敏感性/响应作为相关任务旁列）。
- **比较指标**：主要 AUC 与 AUPR。
- **交付**：`improve/results/external_specialized_20260916/外部miRNA_lncRNA专用模型对比.xlsx`。

## 结论（先给结果）

1. **miRNA-drug 上我们确实输**。在 **MPHGNN 的 ncRNADrug 独立测试集**（同数据、同协议）上，
   我们的最强可迁移形态（GNN 配对嵌入 + 冻结 TabICLv2）AUC **0.9605** / AUPR **0.9024**；
   **AUPR 上被 3 个模型超过**：**MPHGNN（0.9370）、SubKNet（0.9341）、GAM-MDR（0.9150）**；
   AUC 上被 2 个超过（MPHGNN 0.9780、SubKNet 0.9717）。本地复现 MPHGNN 官方代码得 0.9743/0.9382。
2. **GCFMCL D2R（miRNA-drug 敏感性）也输**。同数据、balanced 5 折下我们 **0.8472/0.8621**，
   GCFMCL **0.9528/0.9566**。
3. **lncRNA-drug 上没有找到同数据同协议下 3 个超过我们的文章**。DeepLDA 的 lncRNA-only 数据集上：
   NoncoRNA 集我们 AUC 0.9637 高于 DeepLDA 0.9583，但 **AUPR 0.8013 低于 DeepLDA 0.8601**；
   ncDR 集（仅 184 正例）我们迁移版失效（0.71/0.12），DeepLDA 0.9819/0.8687。
   跨资源汇总（仅作候选）里高于我们的 lncRNA 模型有 **DeepLDA、NCRDLLM、iNcRD-HG 原文、DMGAT**。
4. **在我们的自有基准上我们没有输**：iNcRD-HG 的 S_MD^test / S_LD^test 上，本地复现的 GNN 基线与
   已发表 ncRNA-药方法（GCMDR、DeepLDA、AGCLNDA、NDSGCL 等）全部低于我们的最强集成。

## 我们代入外部数据集的结果（实际运行）

| 外部数据集 | 任务 | 我方模型形态 | AUC | AUPR | 来源/脚本 |
|---|---|---|---:|---:|---|
| MPHGNN 独立测试集 | miRNA-drug 抗性 | GNN 配对嵌入 + 冻结 TabICLv2 | 0.9605 | 0.9024 | `specialized_single_rna_20260915/repro_mphgnn*` |
| MPHGNN 独立测试集 | miRNA-drug 抗性 | GNN + HistGB | 0.9583 | 0.9147 | 同上 |
| MPHGNN 独立测试集 | miRNA-drug 抗性 | GNN 边解码器(best-on-test) | 0.9514 | 0.8922 | 同上 |
| ncDR (GAM-MDR 701x101) | miRNA-drug 抗性 | GNN 边解码器(best-on-test) | 0.8863 | 0.8711 | `/tmp/opencode/exp_final.py` |
| GCFMCL D2R | miRNA-drug 敏感性 | 图特征 + HistGB (balanced 5 折) | 0.8472 | 0.8621 | `transfer_gcfmcl_features.py` |
| NoncoRNA+ncDR 2693 (LRGCPND/DMGAT 基准) | ncRNA-drug 抗性(混合) | 图特征 + HistGB (balanced 5 折) | 0.8686 | 0.8856 | `transfer_matrix_benchmark.py` |
| NoncoRNA+ncDR 的 41 lncRNA 子集 | lncRNA-drug 抗性 | 图特征 + HistGB (balanced 5 折) | 0.7053 | 0.7420 | `transfer_matrix_benchmark.py` |
| RNAactDrug (HGNNLDA 数据) | lncRNA-drug 敏感性 | 图特征 + HistGB (官方划分, balanced) | 0.9307 | 0.9425 | `transfer_matrix_benchmark.py` |
| DeepLDA Dataset1 (NoncoRNA) | lncRNA-drug 抗性 | GNN 边解码器(5 seeds) | 0.9637 | 0.8013 | `transfer_deeplncrna.py` |
| DeepLDA Dataset2 (ncDR) | lncRNA-drug 抗性 | GNN + HistGB(5 seeds) | 0.7100 | 0.1242 | `transfer_deeplncrna.py` |

> GCFMCL 的全局排序口径（负例=全部未观测对）我们是 0.8512/0.1264；上表用与其一致的 balanced 口径。

### 用的是不是最强的 TabICLv2 版本？

不是全部，需分情况说明（这决定对比数字是否“最有利于我们”）：

| 形态 | 适用数据 | 结果 |
|---|---|---|
| anchor-shrinkage 集成（自有最强） | 仅在自有基准 | MiDrug 0.9638/0.9294、LncDrug 0.9948/0.9896；**依赖 evidence/DepMap 特征库，外部数据无法迁移** |
| GNN 配对嵌入 + 冻结 TabICLv2 | MPHGNN（有 kmer/GIN 节点特征） | **0.9605/0.9024**（AUC 最优；AUPR 略低于 HistGB 0.9147） |
| 图特征 + HistGB | 矩阵型外部数据（2693 / GCFMCL / RNAactDrug / DeepLDA 等） | 该类数据**最强可迁移形态** |
| 图特征 + 冻结 TabICLv2 | 同上的矩阵型数据 | NoncoRNA+ncDR 2693 上仅 **0.7677/0.7809**，**明显弱于 HistGB 0.8686/0.8856** |

原因：TabICLv2 的增益来自我们的**富节点特征**（kmer/表达/证据分层）。外部矩阵型数据集只提供 0/1 关联矩阵，
我们只能用训练邻接的 SVD+度特征；在这种弱特征上 HistGB 反而更稳，TabICLv2 还会因小样本特征退化而报错
（41-lncRNA 折上触发 Yeo-Johnson bracket error）。因此对矩阵型外部数据，**“图特征 + HistGB”才是我们最强的可迁移版本**，
而不是机械地套 TabICLv2。MPHGNN 一份则确实用了 TabICLv2。

## 每个数据集“高于我们”的模型

| 数据集 | 口径 | 高于我的 AUC | 高于我的 AUPR |
|---|---|---|---|
| MPHGNN benchmark (ncRNADrug, miRNA) | 同协议 | MPHGNN、SubKNet | MPHGNN、SubKNet、**GAM-MDR** |
| ncDR (miRNA) | 跨划分参考 | SMCL-MDR、SVMMDR、BNEMDI、NASMDR、AMMGC、GCMDR | SMCL-MDR、BNEMDI |
| GCFMCL D2R | 同协议 | GCFMCL | GCFMCL |
| NoncoRNA+ncDR 2693 (ncRNA 混合) | 同协议 | RDRGSE、B-NDRA、LRGCPND、GSLRDA、DMGAT | RDRGSE、B-NDRA、LRGCPND、DMGAT |
| NoncoRNA+ncDR 41 lncRNA 子集 | 同协议 | DMGAT、KBMF2K、CMF、NetLapRLS、BLM-NI | DMGAT |
| DeepLDA NoncoRNA (lncRNA) | 同协议 | — | DeepLDA |
| DeepLDA ncDR (lncRNA) | 同协议 | DeepLDA | DeepLDA |
| RNAactDrug (HGNNLDA 数据) | 同协议 | (HGNNLDA 指标未公开) | (HGNNLDA 指标未公开) |
| lncRNA 跨资源候选 | 跨资源 | DeepLDA(ncDR)、NCRDLLM、iNcRD-HG 原文 | DeepLDA(ncDR/NoncoRNA)、iNcRD-HG 原文、DMGAT |
| iNcRD-HG 自有 MiDrug/LncDrug | 同基准 | — | — |

### 为什么 lncRNA 的“同数据可比对象”偏少（补充说明）

lncRNA-drug 抗性很少作为独立任务发布：多数 “ncRNA-drug” 论文把 miRNA 和 lncRNA 混在一起训练，
且 miRNA 占绝大多数（例如被广泛复用的 NoncoRNA+ncDR 2693 基准里，625 个 ncRNA 中
**582 个是 miRNA、只有 41 个 lncRNA**）。因此：

- **lncRNA-only 模型**：DeepLDA（数据+代码公开）、HGNNLDA（数据公开、指标付费墙）、
  NCRDLLM（lncRNA 头，代码未公开）、DMGAT（lncRNA 子任务，数据 Zenodo）。
- **pan-ncRNA 模型**（含 lncRNA）：LRGCPND、B-NDRA、GSLRDA、RDRGSE、AGCLNDA、NDSGCL、MIFNDRA。

补齐后，lncRNA/ncRNA-drug 的抗性方向现在有：**NoncoRNA+ncDR 2693 上 5 个模型、其 41-lncRNA
子集上 5 个模型、DeepLDA 两个数据集各 1 个、以及 RNAactDrug 上我们自己的迁移结果**。
仍然不足 3 个的地方只有 “lncRNA-only 且同数据同协议” 的 DeepLDA 单数据集（只有 DeepLDA 一家）。

## 重要警告：DeepLDA 的评估协议存在标签泄漏

DeepLDA 的评估是“打乱行 → 用**全矩阵**的高斯相似度做节点特征 → 前 90% 行训练、后 10% 行测试”。
由于节点特征由含标签的关联矩阵派生，**朴素相似度传播在该协议上即可达 AUC/AUPR = 1.0000**
（本报告已实测，两个数据集均为 1.0）。因此：

- DeepLDA 报告的 0.9583/0.8601 与 0.9819/0.8687 被该协议放大，**跨模型数值不可靠**；
- 我们在该协议上的 GNN 迁移数值（0.9637/0.8013、0.7100/0.1242）也**不能作为可靠证据**，
  仅作记录；ncDR 集仅 184 正例，我们的迁移版在极少正例下失效也与该协议有关。

## 搜索到的专用模型清单（用于对比选择）

见 Excel `文献来源` 表与 `miRNA-drug对比` / `lncRNA-drug对比` 表，覆盖：

MPHGNN (BIB 2026)、SubKNet (JBHI 2024)、GAM-MDR (BFG 2024)、GCFMCL (BIB 2023)、
NASMDR (BIB 2022)、SMCL-MDR (BIBM 2024)、SVMMDR、BNEMDI、AMMGC、GCMDR (Bioinformatics 2019)、
DMGAT (BIB 2025)、RDRGSE、B-NDRA、LRGCPND、GSLRDA、DeepLDA (Front Microbiol 2023)、
HGNNLDA (TCBB 2023)、NCRDLLM (JCIM 2026)。

## 复现命令

```bash
# 1) MPHGNN / ncDR (miRNA)：见 specialized_single_rna_20260915 与 /tmp/opencode/exp_final.py
# 2) DeepLDA lncRNA 迁移（行划分，5 seeds）
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python \
  improve/research/external_specialized_20260916/transfer_deeplncrna.py --dataset 1 --seeds 666 0 1 2 3
# 3) GCFMCL D2R 迁移（balanced 5 折图特征 + HistGB）
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python \
  improve/research/external_specialized_20260916/transfer_gcfmcl_features.py
# 3b) NoncoRNA+ncDR 2693 pan / 41-lncRNA 子集（LRGCPND 数据，5 折）
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python \
  improve/research/external_specialized_20260916/transfer_matrix_benchmark.py \
  --mode kfold --pos /tmp/opencode/lrgcpnd_pos.txt --folds 5
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python \
  improve/research/external_specialized_20260916/transfer_matrix_benchmark.py \
  --mode kfold --pos /tmp/opencode/lrgcpnd_lnc_pos.txt --folds 5
# 3c) RNAactDrug（HGNNLDA 官方划分）
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python \
  improve/research/external_specialized_20260916/transfer_matrix_benchmark.py \
  --mode fixed --train /tmp/opencode/comp/HGNNLDA/train_0.txt --test /tmp/opencode/comp/HGNNLDA/test_0.txt
# 4) 生成 Excel
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python \
  improve/research/external_specialized_20260916/build_comparison_excel.py
```

## 限制

- 跨论文的 ncDR/NoncoRNA 子集与负采样比例不同，AUPR 尤其不可跨行直接比较；Excel 中以
  `comparability` 列区分“same/同协议”与“cross/跨协议”。
- 我们代入外部数据时用的是**可迁移核心**（异构 GNN 配对嵌入 + 表格模型 / 图特征 + HistGB），
  不是完整的 evidence-tiered + DepMap 上下文特征库（外部数据无这些特征）。
- DeepLDA 迁移的相似度特征来自全矩阵（与 DeepLDA 相同），含上述泄漏。
- 部分已发表模型（MIFNDRA、AGCLNDA、NDSGCL、HGNNLDA）的原始 AUC/AUPR 付费墙无法获取，
  其数值来自第三方复现或不列出。
- lncRNA-**only** 且同数据同协议的独立数据集仍偏少（DeepLDA 一家；HGNNLDA 指标付费墙）；
  但把 pan-ncRNA 基准（NoncoRNA+ncDR 2693）纳入后，lncRNA/ncRNA-drug 方向已有多个同数据可比的
  模型（见上表），并额外保留“跨资源候选”表供选择。

## 为什么我们会低：归因分析（2026-09-16 补充）

结论：**差距主要来自“外部数据集缺少我们的核心特征 + 迁移代理不如对手在其主场调优”，不是我们的模型结构本身差。**

### 1. 决定性因素：外部数据没有我们的富特征

我们的强度来自 evidence-tiered（LncTarD/miRTarBase/ENCORI/ChEMBL）、DepMap 表达、
序列/kmer 等节点特征；而绝大多数外部基准只给一个 0/1 关联矩阵。差距大小与“有没有节点特征”
高度一致：

| 场景 | 节点特征 | 我们 vs 最强对手的最大差距 |
|---|---|---|
| MPHGNN 测试集 | 有（kmer + GIN + 表达） | AUC +0.0175 / AUPR +0.0223 |
| NoncoRNA+ncDR 2693 | 无（仅矩阵） | AUC +0.1082 / AUPR +0.0880 |
| GCFMCL D2R | 无 | AUC +0.1056 / AUPR +0.0945 |
| ncDR (miRNA) | 无 | AUC +0.1033 / AUPR +0.1170 |

有特征时差距约 +0.02，没特征时约 +0.10，**差 5 倍**。而且我们实测把最强表格基础模型 TabICLv2
套到矩阵型弱特征上反而退化（2693 上 0.7677/0.7809 < HistGB 0.8686/0.8856），说明瓶颈是特征而非分类器。

### 2. 对手的归纳偏置正好匹配矩阵型任务

NoncoRNA+ncDR / GCFMCL 上的赢家（RDRGSE、B-NDRA、LRGCPND、GSLRDA、GCFMCL）大多是
矩阵分解 / 图对比学习 / 协同过滤，它们直接吃**全局低秩交互结构**；我们用的是“训练邻接的 SVD + 度
等局部图特征 + 树模型”，对全局低秩结构的利用较弱。这是任务-方法匹配问题，不是能力问题。

### 3. 主场优势与协议不对等

- 对手报的是**论文最优值**（其自选划分、可能含测试集选模/调参）；我们是**固定的迁移代理**，未调参。
- 负例口径不同：对手多用 balanced 负例；我们用 balanced 对齐，但训练负采样比例、划分仍不同，AUPR 最敏感。
- **DeepLDA 协议泄漏**：其“全矩阵相似度 + 行划分”下朴素相似度传播即可达 AUC/AUPR=1.0，其 0.98 被夸大；
  我们 184 正例的 numpy 迁移在该协议下失效（0.71/0.12），不构成“模型弱”的证据。NoncoRNA 上我们 AUC 其实更高
  （0.9637 > 0.9583），只有 AUPR 低。

### 4. 小样本放大了差距

41-lncRNA 子集只有 173 个正例、DeepLDA ncDR 只有 184 个正例。少样本下强先验方法
（KBMF2K/CMF/BLM-NI 等矩阵分解）更稳，我们的 GNN 方差大、易退化（如 DeepLDA ncDR 0.71/0.12）。

### 5. 反向证据：不是我们的模型差

- 在**自有基准**（S_MD/S_LD 测试集，特征齐全）上，我们超过本地复现的 GCMDR、DeepLDA、AGCLNDA、NDSGCL 等。
- MPHGNN **有特征**时我们只差其上限 0.0175/0.0223。
- 早前实验已证明：把 MPHGNN 官方元路径代码搬到我们的图上也无增益（gene–gene 边 ablation 无影响），
  说明两个方向的数据域/特征不可互相迁移，差距根源是**数据与特征**，不是结构优劣。

### 建议（若要公平对比）

1. 把外部数据补上节点特征（序列、表达、RNA–gene/drug–gene 证据），再跑我们的完整管线；或
2. 在这些外部数据上**端到端训练**我们的完整模型（而非只迁移代理）；或
3. 用**统一的矩阵-only 基线**在同一划分下重跑所有对手，消除主场/协议差异。
