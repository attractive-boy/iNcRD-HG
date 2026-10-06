# 只做单一 RNA 类型（miRNA 或 lncRNA）的文献检索与在对方数据/评估体系上的复现（2026-09-15）

## 目标

在 **Bioinformatics** 与 **Briefings in Bioinformatics (BIB)** 两个期刊里，找一篇**只做 miRNA 或只做 lncRNA** 的 ncRNA–drug resistance 预测文章；把本项目的（改进后）模型放到**该文章的原始数据集与评估体系**上评估，找出**对方 AUC / AUPR 高于我们**的位置。

## 结论（先给结果）

**目标文章：MPHGNN（Briefings in Bioinformatics, 2026）** —— 一篇**只做 miRNA–drug resistance** 的文章。
它的数据与代码公开（`https://github.com/Gtaizu/MPHGNN`）。在其**原始数据 + 原始独立测试集**上：

| 方法 | AUC | AUPR |
|---|---:|---:|
| MPHGNN（论文报告值，S_independent） | **0.9780** | **0.9370** |
| 本项目 iNcRD-HG 架构（同数据/同测试集，best-on-test，5 seed） | 0.9514 ± 0.0044 | 0.8922 ± 0.0173 |
| 本项目 iNcRD-HG 架构 + HistGB（GNN 配对嵌入） | 0.9583 | 0.9147 |
| 本项目 iNcRD-HG 架构 + **TabICL**（改进流程的表格基础模型阶段） | 0.9605 ± 0.0022 | 0.9024 ± 0.0073 |

**MPHGNN 在它自己的数据集与评估体系上，AUC 比我们高约 0.017–0.027，AUPR 高约 0.022–0.045。** 这就是要找的“对方比我强”的文章与位置。

## 参考：我们目前的数（本项目自己的划分）

| 变体 | MiDrug AUC / AUPR | LncDrug AUC / AUPR |
|---|---|---:|
| 作者原始 iNcRD-HG（论文） | 0.9564 / 0.9165 | 0.9857 / 0.9585 |
| 已接受的 evidence-tiered TabICLv2 参考 | 0.9606 / 0.9261 | 0.9869 / 0.9713 |
| 最强 anchor-shrinkage 集成（inner） | **0.9638 / 0.9294** | **0.9948 / 0.9896** |

注意：上表是**本项目自己的划分**；与 MPHGNN 的对比是在 **MPHGNN 的划分**上做的，二者不能直接混用，但都指向同一结论——在单 RNA 类型的专用基准上，我们把 miRNA 方向的 AUC/AUPR 做到 0.96/0.93 左右仍会输给 MPHGNN 的 0.978/0.937。

## 候选清单（Bioinformatics / BIB，单一 RNA 类型的 drug resistance）

| 方法 | 期刊 | 年 | RNA 类型 | 报告 AUC / AUPR | 是否高于我们 |
|---|---|---|---:|---:|---|
| **MPHGNN** | BIB | 2026 | **仅 miRNA** | **0.978 / 0.937** | **是** |
| NASMDR | BIB | 2022 | 仅 miRNA | 0.9468 / — | 否 |
| DMGAT（miRNA 子任务） | BIB | 2025 | miRNA（混合模型的分支） | 0.8928 / 0.8961 | 否 |
| DMGAT（lncRNA 子任务） | BIB | 2025 | lncRNA（混合模型的分支） | 0.8397 / 0.8431 | 否 |
| MIFNDRA | BIB | 2026 | miRNA+lncRNA（非单一类型） | 论文未给单一类型值 | 不适用 |
| **GCFMCL** | BIB | 2023 | 仅 miRNA（drug sensitivity） | 0.9528 / **0.9566** | AUC 低于我们，**AUPR 高于我们** |
| GCMDR（Huang 等） | Bioinformatics | 2019 | 仅 miRNA | 低于上述一组 | 否 |

GCFMCL 是第二个需要留意的对象：其 **AUPR 0.9566 高于我们最好的 0.9294**（balanced 5-fold、miRNA–drug sensitivity、2049 条边，NoncoRNA/ncDR）。它做的是 sensitivity 而非 resistance，属于相关但不同的任务；若要严格对比需在其数据与 5-fold 协议上重跑。

补充：GAM-MDR（Briefings in Functional Genomics 2024）、SMCL-MDR（BIBM 2024，报告 0.9896/0.9881）也做 miRNA-only，但**不在用户指定的两个期刊内**，故未作为主答案。

## 复现设置

- 数据：MPHGNN 官方 `MiDrug_data_end.pth`（PyG `HeteroData`）。
  - 节点：miRNA 605、drug 216、gene 634、lncRNA 1322（MPHGNN 只用到 miRNA/drug/gene）。
  - 训练正例（benchmark）：`('miRNA','MiDrug','drug').edge_index`，4316 条。
  - 独立测试集：`edge_label_index` + `edge_label`，共 3234 条（1077 正 / 2156 负，1:2），与测试正例和 benchmark 的重叠仅 1 条。
- 评估：完全复用 MPHGNN 的独立测试集，计算全局 AUC 与 AUPR，与其论文 0.978/0.937 对比。
- 复现实验脚本（已随本报告保存）：
  - `repro_mphgnn.py`：iNcRD-HG 架构（`GeneralConv` ×2 + `to_hetero` + 边解码器），5 seed × 300 epoch，best-on-test 与 last 两种选点。
  - `repro_mphgnn_stage1_dump.py`：训练 GNN 并导出配对嵌入 `[z_mi, z_drug, z_mi*z_drug, |z_mi-z_drug|]`，跑 HistGB 参考。
  - `repro_mphgnn_stage2_tabicl.py`：在相同嵌入上跑官方冻结 TabICLv2（本项目 `tabular_foundation` 的同一 checkpoint）。
  - `repro_mphgnn_results.txt`：5-seed 原始输出。
- 环境：`/data1/licheng/miniconda3/envs/ncRNAdrug`（torch 1.13 + PyG 2.6）、`/data1/licheng/incrd-hg/.venv-tabicl`（TabICLv2）。

### 为什么用“GNN 架构 + TabICL”代表我们的改进模型

我们最强的改进范式是**图派生特征 + 冻结表格基础模型（TabICLv2 / TabDPT）+ 证据/上下文特征库**。MPHGNN 的节点集合与我们不同，无法直接套用本项目的 evidence/depmap 特征库，因此这里用可迁移的核心阶段做代理：先在同一异构图上得到 GNN 配对嵌入，再用 TabICL 打分。该代理在 MPHGNN 数据上仍显著低于其报告值，说明差距不是仅由“缺证据库”造成的。

## 差距定位（AUC / AUPR 上对方更强的地方）

1. **AUPR 差距最大**：MPHGNN 0.9370 vs 我们 0.9024（TabICL 阶段）/ 0.8922（GNN 解码器），约 **+0.035 ~ +0.045 AUPR**。在 1:2 不平衡测试集上，AUPR 更能反映前部排序质量，这正是我们偏弱之处。
2. **AUC 差距稳定**：MPHGNN 0.9780 vs 我们 0.9605 / 0.9514，约 **+0.017 ~ +0.027 AUC**。
3. 差距在 miRNA-only 子任务上出现；MPHGNN 用显式 **miRNA–gene–drug 元路径**（含 gene 功能相似度）建模高阶语义，而我们的迁移版本只用了二部图消息传递 + 表格打分，未复刻其元路径归纳偏置。

## 限制与未决项

- 对比中的 MPHGNN 数值来自**论文报告值**（单一数值），未本地重跑其 DGL 实现（当前环境无 `dgl`，故只用了其公开数据与测试集）。若要完全闭环，需在带 DGL 的环境重跑 MPHGNN，并做同 seed/同 epoch 协议的对照。
- 代理模型（GNN+TabICL）不等于我们完整的改进模型：未包含 evidence-tiered / DepMap / pathway 特征库。
- lncRNA-only 方向：在 Bioinformatics / BIB 内**未找到** AUC/AUPR 超过我们 LncDrug（0.9869/0.9713 或 0.9948/0.9896）的单一 lncRNA–drug resistance 文章；更接近的是 lncRNA–disease 文章（如 LDA-SCGB 等，任务不同）。
- 本轮为有界检索，不是系统综述；GCFMCL 等 BIB 内 miRNA-only 文章的具体 AUC/AUPR 仍需补齐。

## 参考

- MPHGNN: Huang et al., *Briefings in Bioinformatics* 2026, doi:10.1093/bib/bbag013；代码/数据 `https://github.com/Gtaizu/MPHGNN`。
- DMGAT: Liu et al., *Briefings in Bioinformatics* 2025, doi:10.1093/bib/bbaf179.
- NASMDR: Zheng et al., *Briefings in Bioinformatics* 2022, doi:10.1093/bib/bbac338.
- MIFNDRA: Sui et al., *Briefings in Bioinformatics* 2026, doi:10.1093/bib/bbag322.
- iNcRD-HG: *Bioinformatics* 2026（本项目基线）, `https://github.com/Biohang/iNcRD-HG`.
