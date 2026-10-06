# MSFFMDI 论文获取尝试与公开信息提取

> **更新（2026-09-20）**：用户已提供论文 PDF，5 个对比基线（LAGCN / GCNNMMA / MFIDMA /
> LDAMAN / DLST-MDA）已在 ncDR / RNAInter 上复现完成。结果见
> [`REPORT_MSFFMDI_BASELINES.md`](REPORT_MSFFMDI_BASELINES.md) 与 `MSFFMDI_baselines.xlsx`。
> 本文档保留最初的公开信息获取记录。

目录：`improve/research/msffmdi_baselines_20260920/`
目标：获取 MSFFMDI 论文全文 → 复现其论文中的对比基线。

## 0. 结论（一句话）

**未能获取论文全文。** 该文为**完全闭源（closed access）**，OpenAlex/Unpaywall 确认
`oa_status = closed`、无任何仓库副本；无公开代码、无补充材料；正文的“对比基线”表格
不在公开页面中。因此**其论文对比基线目前无法忠实复现**。

需要用户提供 PDF（机构订阅）后即可继续。

---

## 1. 论文信息

| 项 | 值 |
|---|---|
| 标题 | Predicting miRNA-Drug Interactions Based on Multi-source Feature Fusion of Heterogeneous Network |
| 作者 | Chenyue Lei, Xiujuan Lei, Lian Liu, Jianrui Chen, Fang-Xiang Wu |
| 期刊 | Interdisciplinary Sciences: Computational Life Sciences, 18(3):986–1009 |
| DOI | 10.1007/s12539-025-00775-7 |
| PubMed | 41083770（2025-10-13 电子发表） |
| 开放状态 | **closed**（OpenAlex `is_oa=false`；Unpaywall 无 OA location；不在 PMC/EuropePMC；无代码） |

## 2. 获取尝试记录（均已失败）

| 途径 | 结果 |
|---|---|
| Springer 文章页 `link.springer.com/article/...` | 200，但仅含摘要 + 19 张公开图；正文 paywalled |
| Springer PDF `.../content/pdf/...pdf` | 返回 HTML 付费页，非 PDF |
| ResearchGate | HTTP 403 |
| Semantic Scholar API | 摘要被出版商 elided，`openAccessPdf.url` 为空 |
| OpenAlex API | `oa_status=closed`，`any_repository_has_fulltext=false` |
| Unpaywall API | `best_oa_location=null` |
| Europe PMC / PMC | `inEPMC=N, inPMC=N, hasPDF=N` |
| PubMed 页面 | 反爬 PoW 挑战；摘要经 Europe PMC 取得 |
| Google Scholar / Web 搜索 | 未发现任何 PDF 副本 |

## 3. 从公开来源提取到的信息

### 3.1 摘要（Europe PMC，已确认）

MSFFMDI = **双通道多源特征融合异质网络**：
- 通道 1（属性）：miRNA = k-mer + word2vec；drug = GIN（分子结构）+ mol2vec（化学/功能序列）。
- 通道 2（拓扑）：集成相似度 + 已知关联建异质网络 → GAT 更新节点 → 多尺度 CNN 提拓扑表示。
- 融合：两通道拼接 → PCA 降维 → 预测。
- 在**两个数据集**上实验，并有 case study。

### 3.2 从公开图（19 张）中读出的论文报告值

> 图均可从 `media.springernature.com` 公开下载，已存 `figures/`。以下为图中标注数值。

| 实验 | 数据集 1 (AUC) | 数据集 2 (AUC) |
|---|---|---|
| 主结果 5 折 ROC（Fig11/Fig12） | **0.9483**（AUPR **0.9461**） | **0.9216**（AUPR **0.9163**） |
| PCA 网格最优（Fig7） | 0.9471 | 0.9214 |
| 负例比 1:5 / 1:10 / 1:20（Fig15） | 0.9303 / 0.9121 / 0.8866 | 0.8839 / 0.8476 / 0.8333 |
| drug 特征消融：GIN / mol2vec（Fig9） | 0.9424 / 0.9286 | 0.9034 / 0.9105 |
| miRNA 特征消融：k-mer / word2vec（Fig10） | 0.9433 / 0.9280 | 0.9211 / 0.8659 |

（Fig8 = 通道消融 Topo/Attri/A&T；Fig13/14 = PCA/t-SNE；Fig16/17 = KEGG/基因-通路；Fig18/19 = case study。）

**注意**：这些是**论文自己的 MSFFMDI 数值**，不是基线数值；基线对比在正文表格中，未公开。

### 3.3 参考文献中的候选基线（39 篇，仅能推测，未证实被对比）

LGCMDS (BIBM 2021)、Huang 2020 GCN-MDR (Bioinformatics)、Dual-channel MDS (JCIM 2022)、
AMMGC (Front Pharmacol 2022)、GAM-MDR (BFG 2024)、Mghstckw (ESWA 2025)、GCFMCL (BIB 2023)、
MSMP-GAE (JBHI 2025)、NASMDR (BIB 2022)、Guo 2020、DLST-MDA (JCIM 2025)、BNEMDI (Front Genet 2022)、
MFIDMA (Biology 2023)、TCN+BiLSTM (FCS 2025)、GCNNMMA (Front Genet 2023)。

**无法确定论文实际对比了哪些、以及它们的报告值。**

## 4. 为什么现在不能复现其基线

要复现“论文里的对比基线”，至少需要三样，且都不可得：
1. **论文实际对比的基线清单** —— 在 paywalled 正文表格中；
2. **论文的两个数据集及其预处理/划分** —— 正文/数据可用性声明未公开（摘要只说“two datasets”）；
3. **论文报告的基线数值** —— 同上。

已有工作（`../new5_repro_20260919/`）是**按摘要忠实重实现的 MSFFMDI 方法本身**，跑在 DLST-MDA
的 1578×156 benchmark 上（AUC 0.9483 / AUPR 0.9391），**不是论文数据、也不是论文数值复现**。

## 5. 下一步（需要用户）

- **首选**：用户提供论文 PDF（机构订阅）或数据集/代码链接 → 我按正文复现其对比基线。
- 备选（可立即做，但为**推测性**）：以公开 ncDR 数据库重建“耐药/敏感”两数据集，按论文方法描述
  复现候选基线（LGCMDS/GAM-MDR/GCFMCL/NASMDR 等）+ MSFFMDI 重实现，做同协议排名。
  该结果只能作为“同协议可比参考”，**不能宣称复现了论文数值**。

## 6. 产物

```
msffmdi_baselines_20260920/
├─ ACQUISITION_REPORT.md   # 本文件
└─ figures/Fig1..Fig19.png # 从 Springer 公开图床下载（论文全部图）
```
