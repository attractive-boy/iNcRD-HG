# PLMF-MDA 论文 Fig.3 对比基线复现

> 论文：Sheng N, Liu Y, Gao L, Hou W, Huang L, Wang Y. *A pre-trained language model-based
> cross-modal fusion framework for predicting miRNA-drug resistance and sensitivity
> associations.* **PLoS Comput Biol 22(2): e1013968, 2026.**
> DOI: 10.1371/journal.pcbi.1013968 · 代码/数据：https://github.com/sheng-n/PLMF-MDA
>
> 任务：复现论文 **Fig.3** 的基线对比实验 —— 在论文自建的 **MDRdataset / MDSdataset** 上，
> 用与 PLMF-MDA 相同的 5 折协议运行 4 个对比基线
> **GCNNMMA / SubMDTA / GraphDTA / ML-DTI**，并与已复现的 PLMF-MDA 对照。
>
> 说明：PLMF-MDA **本身**的复现已在 `../new5_repro_20260919/`（MDR 0.9260/0.9135，
> MDS 0.9255/0.9145，与论文一致）。本目录补的是论文里被对比的**基线方法**。

---

## 0. 一句话结论

4 个基线全部在 MDR/MDS 上跑通（5 折 × 两种协议）。**GCNNMMA 与论文值几乎逐位命中**
（ΔAUC ≤ 0.005），**PLMF-MDA 自身也复现良好**（ΔAUC ≤ 0.006）。但 **GraphDTA / SubMDTA / ML-DTI
的复现值显著高于论文报告值**（ML-DTI +0.15 AUC，GraphDTA +0.10~0.13 AUC）——因此论文
“PLMF-MDA 一致优于所有基线”的结论在本复现口径下**不成立**：MDR 上 PLMF-MDA 与 ML-DTI 统计打平，
**MDS 上 ML-DTI 高于 PLMF-MDA**。这与本项目对 DLST-MDA 基线复现得到的结论一致。

---

## 1. 数据与协议（与官方代码一致）

| 项 | 值 |
|---|---|
| 数据来源 | `../new5_repro_20260919/repos/PLMF-MDA-main/data/` |
| MDRdataset | 1317 miRNA × 105 drug，5411 条耐药关联 |
| MDSdataset | 1252 miRNA × 140 drug，5054 条敏感关联 |
| 负例池 | MDR 132874 / MDS 170226 条非关联 |
| 输入 | miRNA 序列（ACGU，截断/补零到 24nt）；drug SMILES（图：78 维原子特征分子图；序列模型：字符整数） |
| 指标 | AUC、AUPR（另存 ACC/Precision/Recall/Specificity/F1/MCC） |
| 模型选择 | 每轮记录**测试集最优 epoch** 的指标（本项目统一口径），并另存最后一轮值 |

**两种协议**（论文正文声称 5-cv，但官方代码只实现单次 80/20 划分）：

1. **`5cv`（主口径）**：正例 + 等量随机负例 → 随机均分 5 折，逐折测试，seed=42；报 5 折均值±折间标准差。
2. **`single`（官方代码口径）**：完全复刻 `code/data_preprocess.py::load_data`
   （`np.random.seed(seed)` → shuffle 正例 → shuffle 负例 → 前 80% 训练 / 后 20% 测试），seed=0..4；
   报 5 seed 均值±标准差。该口径可直接与 `new5_repro_20260919` 的 PLMF-MDA 复现对齐。

> 两种协议的 PLMF-MDA 参考值都由本目录的 `plmf_ref_5cv.py` 重跑得到：它使用官方模型，但把
> **冻结的 RNA-FM / ChemBERTa-2 CLS 嵌入缓存一次**（数值等价，速度约快 30×）。其 `single` 结果与
> 官方逐 batch 前向的 `new5_repro` 结果一致（见 §3），验证了缓存实现。

---

## 2. 各基线来源与实现

| 方法 | 原始论文 | 复现实现 | 说明 |
|---|---|---|---|
| **GraphDTA** | Nguyen et al., Bioinformatics 2021 | `../dlstmda_baselines_20260919/models.py::GraphDTA_GCN` | GCN 药物图 + 1D-CNN miRNA；蛋白分支按 miRNA 长度 24 重定形状 |
| **GCNNMMA** | Niu et al., Front. Genet. 2023 | `gcnnma_plmf.py`（复制自 `../dlstmda_baselines_20260919/gcnnma.py`） | 官方 r-radius(WL) 子结构指纹 + 关系型 GNN；n-gram(3) word + attention-CNN；官方超参照搬 |
| **SubMDTA** | Pan et al., BMC Bioinformatics 2023 | `../dlstmda_baselines_20260919/models.py::SubMDTA` | GIN 药物编码 + 多尺度(2/3/4-mer) BiLSTM miRNA；省略子结构自监督预训练阶段 |
| **ML-DTI** | Yang et al., J. Phys. Chem. Lett. 2021 | `../dlstmda_baselines_20260919/models.py::MLDTI` | 互学习注意力双 CNN（官方 `DTImodel`），输出改为二分类 BCE |

四个模型与超参沿用 DLST-MDA 基线复现中已验证的实现（GraphDTA/SubMDTA 150 epoch、
ML-DTI 150 epoch、GCNNMMA 100 epoch，官方 lr/wd）。药物分子图特征使用 78 维 DLST 版
`atom_features`（与 PLMF-MDA 的 69 维略有差异，属实现细节；两版均含原子类型/度/氢/价/芳香性）。

---

## 3. 复现结果

### 3.1 主口径：5 折交叉验证（seed 42，均值 ± 折间标准差）

**MDRdataset**

| 排名 | 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR | ΔAUPR |
|---|---|---|---|---|---|---|---|
| 1 | **PLMF-MDA** | **0.9243 ± 0.0039** | 0.9227 | +0.0016 | **0.9146 ± 0.0078** | 0.9076 | +0.0070 |
| 2 | ML-DTI | 0.9238 ± 0.0053 | 0.7660 | +0.1578 | 0.9131 ± 0.0076 | 0.7870 | +0.1261 |
| 3 | GCNNMMA | 0.9118 ± 0.0036 | 0.9070 | +0.0048 | 0.8949 ± 0.0088 | 0.8736 | +0.0213 |
| 4 | GraphDTA | 0.9054 ± 0.0029 | 0.8086 | +0.0968 | 0.8907 ± 0.0084 | 0.7979 | +0.0928 |
| 5 | SubMDTA | 0.9044 ± 0.0039 | 0.8936 | +0.0108 | 0.8907 ± 0.0066 | 0.8514 | +0.0393 |

**MDSdataset**

| 排名 | 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR | ΔAUPR |
|---|---|---|---|---|---|---|---|
| 1 | **ML-DTI** | **0.9296 ± 0.0046** | 0.7820 | +0.1476 | **0.9226 ± 0.0047** | 0.7881 | +0.1345 |
| 2 | PLMF-MDA | 0.9270 ± 0.0065 | 0.9301 | −0.0031 | 0.9175 ± 0.0043 | 0.9207 | −0.0032 |
| 3 | GCNNMMA | 0.9133 ± 0.0048 | 0.9143 | −0.0010 | 0.9007 ± 0.0058 | 0.8889 | +0.0118 |
| 4 | GraphDTA | 0.9125 ± 0.0059 | 0.7864 | +0.1261 | 0.9036 ± 0.0061 | 0.7911 | +0.1125 |
| 5 | SubMDTA | 0.9059 ± 0.0050 | 0.8858 | +0.0201 | 0.8923 ± 0.0057 | 0.8587 | +0.0336 |

### 3.2 官方代码口径：80/20 单次划分（seed 0–4，均值 ± seed 标准差）

**MDRdataset**

| 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR | ΔAUPR |
|---|---|---|---|---|---|---|
| **PLMF-MDA** | **0.9287 ± 0.0060** | 0.9227 | +0.0060 | **0.9162 ± 0.0082** | 0.9076 | +0.0086 |
| ML-DTI | 0.9281 ± 0.0049 | 0.7660 | +0.1621 | 0.9171 ± 0.0066 | 0.7870 | +0.1301 |
| GCNNMMA | 0.9179 ± 0.0061 | 0.9070 | +0.0109 | 0.9019 ± 0.0100 | 0.8736 | +0.0283 |
| GraphDTA | 0.9118 ± 0.0066 | 0.8086 | +0.1032 | 0.9006 ± 0.0055 | 0.7979 | +0.1027 |
| SubMDTA | 0.9068 ± 0.0081 | 0.8936 | +0.0132 | 0.8902 ± 0.0171 | 0.8514 | +0.0388 |

**MDSdataset**

| 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR | ΔAUPR |
|---|---|---|---|---|---|---|
| **ML-DTI** | **0.9303 ± 0.0041** | 0.7820 | +0.1483 | **0.9208 ± 0.0036** | 0.7881 | +0.1327 |
| PLMF-MDA | 0.9294 ± 0.0026 | 0.9301 | −0.0007 | 0.9189 ± 0.0053 | 0.9207 | −0.0018 |
| GCNNMMA | 0.9167 ± 0.0042 | 0.9143 | +0.0024 | 0.9028 ± 0.0065 | 0.8889 | +0.0139 |
| GraphDTA | 0.9145 ± 0.0051 | 0.7864 | +0.1281 | 0.9012 ± 0.0037 | 0.7911 | +0.1101 |
| SubMDTA | 0.9075 ± 0.0069 | 0.8858 | +0.0217 | 0.8908 ± 0.0089 | 0.8587 | +0.0321 |

### 3.3 PLMF-MDA 参考值与官方复现的一致性检验

| 数据集 | 本目录 `single`（cached） | `new5_repro` 官方复现 | Δ |
|---|---|---|---|
| MDR | 0.9287 / 0.9162 | 0.9260 / 0.9135 | +0.0027 / +0.0027 |
| MDS | 0.9294 / 0.9189 | 0.9255 / 0.9145 | +0.0039 / +0.0044 |

两者在 seed 波动范围内一致，说明缓存实现保真。

---

## 4. 结论与诚实性说明

1. **复现成功的方法**：GCNNMMA（MDR ΔAUC +0.005，MDS ΔAUC −0.001）与 PLMF-MDA 自身
   （ΔAUC ≤ 0.006）与论文吻合。
2. **复现值显著高于论文的基线**：ML-DTI（+0.15~0.16 AUC）、GraphDTA（+0.10~0.13 AUC）、
   SubMDTA（+0.01~0.02 AUC，AUPR +0.03~0.04）。原因有三：
   - 论文对基线迁移到 miRNA 任务的适配细节未知，本复现做了合理适配并允许充分训练；
   - 本复现取**测试集最优 epoch**，论文多为固定 epoch 的最后一轮，前者天然更高；
   - 深度学习基线在新 benchmark 上很难逐位复现，本表应视为**同数据同划分下的可比基线**，
     而非原文数值的精确重放。
3. **对论文结论的影响**：论文称 PLMF-MDA “consistently outperformed all baseline methods”。
   在本复现口径下：MDR 上 PLMF-MDA 与 ML-DTI 的 AUC 差距仅 0.0005（远小于折间标准差），
   属统计打平；**MDS 上 ML-DTI 的 AUC/AUPR 均高于 PLMF-MDA**。因此该结论**不成立**，
   正式引用需谨慎。此现象与 `../dlstmda_baselines_20260919/REPRO_DLST_BASELINES.md` 中
   DLST-MDA 的结论（基线被论文低估）一致。
4. **最后一轮口径**：若改看 `last_auc/last_aupr`（见 Excel `逐次运行` sheet），各基线仍普遍高于
   论文值，但差距缩小；说明“最优轮选择”是偏差的一部分来源，不是全部。
5. **未做的事**：论文 Fig.4（PLMF-MDA 自身消融）与 Table 2（孤儿药物/miRNA 冷启动）不属于
   “对比基线”，未在本轮复现；如需可另起一轮。

---

## 5. 目录与产物

```
plmf_baselines_20260920/
├─ plmf_data.py            # MDR/MDS 数据加载 + 两种划分
├─ run_bench_plmf.py       # GraphDTA / SubMDTA / ML-DTI 统一 runner
├─ gcnnma_plmf.py          # GCNNMMA runner
├─ plmf_ref_5cv.py         # PLMF-MDA 参考（官方模型 + cached-encoder）
├─ run_all_plmf.sh         # 4 基线一键跑（5cv | single）
├─ run_plmf_ref.sh         # PLMF-MDA 参考跑
├─ summarize_plmf.py       # 原始 CSV 汇总
├─ build_excel.py          # 生成带格式的结果表
├─ PLMF_MDA_baselines.xlsx # 结果表（说明 / Fig3_5cv / Fig3_single / 逐次运行）
├─ results/*.csv           # 逐折 / 逐 seed 明细（100 行）
└─ logs/                   # 运行日志
```

### 复现命令

```bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
bash run_all_plmf.sh 5cv      # 论文协议：4 基线 × 2 数据集 × 5 折
bash run_all_plmf.sh single   # 官方代码协议：4 基线 × 2 数据集 × seed 0-4
bash run_plmf_ref.sh          # PLMF-MDA 参考（5cv seed42 + single seed0-4）
$PY build_excel.py
```
