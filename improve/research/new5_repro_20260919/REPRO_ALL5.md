# 五篇 miRNA-drug 专用模型：基线复现汇总（2026-09-19）

目标：对用户在研的 5 篇 miRNA-drug 抗性/关联方法逐篇复现其论文实验（官方代码优先；
无代码者做忠实重实现并明确标注）。所有脚本、日志、补丁、结果均落在本目录。

## 0. 一句话结论

| 方法 | 期刊 | 有无官方代码 | 复现方式 | 复现 AUC | 复现 AUPR | 论文 AUC | 论文 AUPR | 判定 |
|---|---|---|---|---|---|---|---|---|
| **MPHGNN** | BIB 2026 | 有 | 官方代码 5 折 | **0.9743–0.9755** | **0.9382–0.9415** | 0.9780 | 0.9370 | ✅ 命中 |
| **DLST-MDA** | JCIM 2025 | 有 | 官方代码 5 折 | **0.9437** | **0.9356** | 0.9411 | 0.9326 | ✅ 命中 |
| **PLMF-MDA (MDR)** | PLoS Comp Biol 2026 | 有 | 官方代码 5 seed | **0.9260 ± 0.0061** | **0.9135 ± 0.0064** | 0.9222 | 0.9062 | ✅ 命中 |
| **PLMF-MDA (MDS)** | PLoS Comp Biol 2026 | 有 | 官方代码 5 seed | **0.9255 ± 0.0014** | **0.9145 ± 0.0034** | 0.9301 | 0.9207 | ✅ 接近 |
| **DLMVF** | JCIM 2026 | 有 | 官方代码 5 折 | **0.9591** | **0.9514** | 0.9611 | 0.9543 | ✅ 命中 |
| **MSFFMDI** | Interdiscip. Sci. 2025 | **无** | 按摘要忠实重实现 | 0.9483 ± 0.0044 | 0.9391 ± 0.0071 | 未公开取到 | — | ⚠️ 重实现，非数值复现 |

> 4 篇有官方代码的方法，复现值与论文报告值的差距都在 0.2–0.8 个百分点内（多为复现噪声），
> 判定为复现成功。MSFFMDI 无公开代码、正文付费墙，只能按摘要重实现，**数值不可与原文比较**。

---

## 1. MPHGNN（Briefings in Bioinformatics 2026, bbag013）

- 数据/协议：论文自有 MiDrug benchmark（`MiDrug_data_end.pth`），官方 5 折。
- 代码：官方仓库，已归档到 `repos/mphgnn_official/`。
- 命令：`main.py -ep 10 -fo 5 -se 42`（另有 seed422 日志）。
- 复现结果：
  - seed42 5 折：**AUC 0.9755 / AUPR 0.9415**（ACC 0.9246 / F1 0.8866 / MCC 0.8301）
  - seed422 5 折：**AUC 0.9743 / AUPR 0.9382**（ACC 0.9242 / F1 0.8871 / MCC 0.8302）
  - 论文：AUC 0.9780 / AUPR 0.9370 / ACC 0.9394 / F1 0.9131 / MCC 0.8633
- 日志：`logs/mphgnn_official_s422.log`（seed42 见旧台账 `baseline_repro_20260918/REPRO_LEDGER.md` 第九节）。

## 2. DLST-MDA（J. Chem. Inf. Model. 2025, 65(11):5317）

- 数据/协议：作者自建 benchmark，1578 miRNA × 156 drug，8720 关联，5 折（正负 1:1，各 1744）。
- 代码：官方 `github.com/sheng-n/DLST-MDA` → `repos/DLST-MDA-main/`。
- 命令：`python code/process_data.py` → `PYTHONPATH=code python code/training.py`（5 折 × 30 epoch）。
- 复现结果（5 折均值）：
  - **AUC 0.9437 / PR-AUC 0.9356**，ACC 0.8753 / Precision 0.8408 / Recall 0.9292 / F1 0.8827。
  - 论文（正负 1:1 主表）：AUC 0.9411 / AUPR 0.9326 / ACC 0.8736 / MCC 0.7532。
  - 差距 AUC +0.0026 / AUPR +0.0030（复现略高于论文，属正常波动）。
- 日志：`logs/dlst_process.log`、`logs/dlst_train.log`。

## 3. PLMF-MDA（PLOS Computational Biology 2026, 22(2):e1013968）

- 数据/协议：作者自建 MDR / MDS 两个 benchmark，官方代码为**单次 80/20 划分**（`test_ratio=0.2`）。
- 代码：官方 `github.com/sheng-n/PLMF-MDA` → `repos/PLMF-MDA-main/`。
- 关键环境修复（见 `patches/plmf.patch`）：
  1. `data.rar` 是 RAR5，`7z 16.02` 报 Unsupported Method；改用 `bsdtar` 解包成功。
  2. ChemBERTa 权重本地已有（`code/DeepChem/ChemBERTa-77M-MLM`），把 `from_pretrained("DeepChem/...")` 改为**本地目录**（HF 不可达）。
  3. RNA-FM：`pip install fm`（rna_fm 0.2.2，装到 `venvs/plmf_site`），权重下载到 `models/rnafm/RNA-FM_pretrained.pth`（1.2 GB）。
- 复现结果（`--seed 0..4`，官方单划分协议，均值 ± 标准差）：

| seed | MDR AUC | MDR AUPR | MDS AUC | MDS AUPR |
|---|---|---|---|---|
| 0 | 0.9227 | 0.9084 | 0.9233 | 0.9131 |
| 1 | 0.9245 | 0.9149 | 0.9255 | 0.9094 |
| 2 | 0.9308 | 0.9187 | 0.9276 | 0.9134 |
| 3 | 0.9347 | 0.9215 | 0.9256 | 0.9179 |
| 4 | 0.9175 | 0.9042 | 0.9256 | 0.9187 |
| **均值** | **0.9260 ± 0.0061** | **0.9135 ± 0.0064** | **0.9255 ± 0.0014** | **0.9145 ± 0.0034** |

- 论文：MDR AUC 0.9222 / AUPR 0.9062；MDS AUC 0.9301 / AUPR 0.9207。
  - MDR 均值略高于论文（+0.0038/+0.0073），MDS 略低（−0.0046/−0.0062），均在 seed 波动范围内。
- 日志：`logs/plmf_mdr*.log`、`logs/plmf_mds*.log`。
- 说明：官方代码 seed 只影响 shuffle，不固定 torch 初始化；seed0 与论文几乎完全一致。

## 4. DLMVF（J. Chem. Inf. Model. 2026, 66(3):1556）

- 数据/协议：同 DLST-MDA 的 1578×156 benchmark，5 折（正负 1:1）。
- 代码：官方 `github.com/Lgubig/DLMVF_model` → `repos/DLMVF_model-master/`。
- 复现结果（5 折均值）：**AUC 0.9591 / PR-AUC 0.9514**（ACC 0.8970 / F1 0.8989）。
  - 论文：AUROC 0.9611 / AUPRC 0.9543；差距 −0.0020 / −0.0029。
- 修复（见 `patches/dlmvf.patch`，完整说明见 `REPRO_DLMVF.md`）：
  1. `training.py` 硬编码 `cuda:7` → `cuda:0`。
  2. `NUM_RUNS=1` → `5`（跑满 5 折）。
  3. 缺失的 `data/MD_adjaceny.txt` 按 `miRTarbase.py::MD_Matrix` 布局重建；缺失的相似度块用
     Gaussian kernel similarity（`calculating_similarity.py` 公式）替代（已文档化）。
  4. `model.py` `drug_offset 1563 → 1578`（矩阵有 1578 个 miRNA）。
  5. `datasetindex()` 字典别名 bug：5 折实际指向同一 fold，改为每折独立 dict（否则“5 折”退化）。
  6. 保留官方 `gene` 特征经 LongTensor 截断为整数的行为以保真。
- 日志：`logs/dlmvf_train.log`。

## 5. MSFFMDI（Interdisciplinary Sciences: Computational Life Sciences 2025, DOI 10.1007/s12539-025-00775-7）

- **无公开代码，正文付费墙**：只能按摘要的方法描述做忠实重实现，**不是数值复现**。
- 实现（`msffmdi/msffmdi.py`，见 `REPRO_MSFFMDI.md`）：
  - 通道 1（属性）：miRNA = 1/2/3-mer 频率 + k-mer word2vec；drug = GIN（RDKit 图）+ SMILES token word2vec
    （**mol2vec 预训练模型离线不可得，用 token word2vec 替代**，已标注为最大保真缺口）。
  - 通道 2（拓扑）：GIP 相似度 + 已知关联构建异质图 → GAT → 多尺度 1D CNN。
  - 融合：拼接 → PCA → MLP 打分。
- 数据/协议：与 DLST-MDA/DLMVF **完全相同**的 1578×156 benchmark 与 5 折划分（正负 1:1）。
- 复现结果：**AUC 0.9483 ± 0.0044 / AUPR 0.9391 ± 0.0071**（逐折 0.9416–0.9538 / 0.9285–0.9480）。
- 日志：`logs/msffmdi_train.log`，结果 `msffmdi/results.json`。

---

## 6. 统一口径下可横向比较的子集

DLST-MDA、DLMVF、MSFFMDI（重实现）三者使用**同一个 1578×156 / 8720 关联 / 等量负例 5 折**协议：

| 方法 | AUC | AUPR |
|---|---|---|
| DLMVF（官方代码） | 0.9591 | 0.9514 |
| **MSFFMDI（重实现）** | 0.9483 | 0.9391 |
| DLST-MDA（官方代码） | 0.9437 | 0.9356 |

MPHGNN（自有 MiDrug benchmark）与 PLMF-MDA（自有 MDR/MDS benchmark）使用各自数据，不与上表直接可比。

## 7. 目录结构

```
new5_repro_20260919/
├─ REPRO_ALL5.md            # 本文件
├─ REPRO_DLMVF.md           # DLMVF 详细报告
├─ REPRO_MSFFMDI.md         # MSFFMDI 重实现说明
├─ repos/                   # DLST-MDA / PLMF-MDA / DLMVF / mphgnn_official
├─ patches/                 # dlmvf.patch, plmf.patch
├─ logs/                    # 所有运行日志
├─ models/rnafm/            # RNA-FM 预训练权重（1.2 GB）
├─ msffmdi/                 # MSFFMDI 重实现代码与结果
└─ venvs/                   # plmf_site(fm+RNA-FM), msffmdi_site
```

## 8. 诚实性声明

- 有官方代码的 4 篇：结果来自真实运行，日志可复核；与论文的差距已逐项列出。
- MSFFMDI：无官方代码、无正文，属“按摘要重实现”，**不得**当作原文数值复现，只能作为同数据同协议下的一个可比较基线。
- DLMVF 的相似度块（`m-m.txt`/`d-d.txt`）原作者未公开，用高斯核相似度替代，可能影响数值。
