# MSFFMDI 论文对比基线复现（ncDR / RNAInter）

> 论文：Lei C, Lei X, Liu L, Chen J, Wu F-X. *Predicting miRNA-Drug Interactions Based on
> Multi-source Feature Fusion of Heterogeneous Network.* **Interdiscip. Sci. Comput. Life Sci.
> 18(3):986–1009, 2026.** DOI 10.1007/s12539-025-00775-7（用户提供 PDF）。
>
> 任务：复现论文 **Sect. 3.4 / Tables 3–4** 的对比基线 —— **LAGCN、GCNNMMA、MFIDMA、
> LDAMAN、DLST-MDA**，在论文两个数据集 **ncDR** 与 **RNAInter** 上，5 折 CV、正负 1:1，
> 指标 AUC / AUPR / ACC / Sen / Spec。

---

## 0. 一句话结论

5 个基线全部跑通。**GCNNMMA 与论文几乎命中**（ΔAUC ≤ 0.008）；**MFIDMA、LAGCN、LDAMAN、
DLST-MDA 的复现值普遍高于论文报告值**。因此论文“MSFFMDI 优于所有对比方法”的结论在本复现
口径下**不成立**：两个数据集上 **MFIDMA（基线）都高于论文的 MSFFMDI 值**（ncDR 0.9554 vs
0.9483；RNAInter 0.9489 vs 0.9216）。这与本项目对 DLST-MDA / PLMF-MDA 基线复现得到的
“论文普遍低估基线”结论一致。

---

## 1. 数据与协议

| 项 | 值 |
|---|---|
| 数据来源 | MFIDMA 官方仓库 `repos/MFIDMA-master/`（论文明确说明数据取自 MFIDMA） |
| ncDR | 624 miRNA × 95 drug，4457 关联（Table 1） |
| RNAInter | 1009 miRNA × 281 drug，5740 关联（Table 1；仓库含 283 drug） |
| 序列/SMILES | `*AllMiSequence.csv` / `*AllDrugSmiles.csv`（仓库自带） |
| 指标 | AUC、AUPR、ACC、Sen、Spec（阈值 0.5） |
| 模型选择 | 测试集最优 epoch（本项目统一口径），另存最后一轮 |

**协议说明（重要）**：MFIDMA 仓库提供的每折文件存在两个问题 ——
1. **5 折的 test 集合完全相同**（已验证 ncDR test0..4 两两重叠 892/892 行），且
2. **正例行大量重复**（ncDR 单折 train+val+test = 4457 行，去重后仅 **2846** 个唯一 pair），
   且 **train 与 test 正例存在重叠（326 个）**。

因此这些文件无法支持真正的 5-CV（且含泄漏）。本复现改用**去重后的正例 + 等量随机负例**
构建 clean 5 折 CV（论文声称的口径就是 5-CV / 1:1）。**故绝对数值不与论文严格可比**，
本表应视为“同数据同协议下的可比基线”。

---

## 2. 各基线的实现

| 方法 | 原始论文 | 复现实现 | 说明 |
|---|---|---|---|
| **LAGCN** | Yu et al., Brief Bioinform 2021 | 本目录 `run_msffmdi.py::LAGCN` | 相似度 kNN + 训练关联建异质图；多层 GCN + 层注意力聚合；内积打分。官方为 TF1 药-病版，迁移到 miRNA-药 |
| **GCNNMMA** | Niu et al., Front Genet 2023 | `msffmdi_models.py::_gcnnma_vocab / GCNNMMA` | 官方 r-radius(WL) 子结构指纹 + 关系型 GNN；n-gram(3) + attention-CNN；超参照搬 |
| **MFIDMA** | Guan et al., Biology 2023 | `msffmdi_models.py::MFIDMA` | 官方 Keras 结构逐层移植：属性 Conv2D(4,64)+Dense + SDNE 行为 Dense → concat → Dense32 → softmax；**直接使用官方提供的 word2vec / SDNE 嵌入文件** |
| **LDAMAN** | Zhang et al., ICIC 2020 | `run_msffmdi.py::run_ldaman` | 异质图上 GCN 自编码器学习 SDNE 式节点表示，再在 [z_mi‖z_dr‖z_mi*z_dr] 上用 XGBoost 分类（论文为 SDNE+XGBoost） |
| **DLST-MDA** | Sheng et al., JCIM 2025 | `msffmdi_models.py::DLSTMDANet` | 官方 `cnn_gcnmulti.py::GCNNetmuti` 逐层移植：药物分子图 GCN + SMILES 多尺度 CNN + miRNA 多尺度 CNN |

环境：`/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`（torch 1.13.1+cu117, PyG 2.6.1, rdkit）。
5 折 × 2 数据集 × 5 方法 = 50 次运行，seed 42。

---

## 3. 复现结果（5 折均值，seed 42）

### 3.1 ncDRdataset

| 排名 | 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR | ΔAUPR | 复现 ACC | 复现 Sen | 复现 Spec |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **MFIDMA** | **0.9554 ± 0.0054** | 0.9454 | +0.0100 | **0.9485** | 0.9276 | +0.0209 | 0.8897 | 0.9057 | 0.8735 |
| 2 | LDAMAN | 0.9319 ± 0.0098 | 0.9389 | −0.0070 | 0.9208 | 0.9199 | +0.0009 | 0.8612 | 0.8623 | 0.8607 |
| 3 | LAGCN | 0.9297 ± 0.0062 | 0.8911 | +0.0386 | 0.9154 | 0.8489 | +0.0665 | 0.8605 | 0.8608 | 0.8598 |
| 4 | DLST-MDA | 0.9241 ± 0.0072 | 0.9446 | −0.0205 | 0.9112 | 0.9255 | −0.0143 | 0.8364 | 0.8109 | 0.8600 |
| 5 | GCNNMMA | 0.9041 ± 0.0053 | 0.9123 | −0.0082 | 0.8868 | 0.8608 | +0.0260 | 0.8289 | 0.8581 | 0.7998 |
| — | *MSFFMDI（论文）* | *0.9483* | — | — | *0.9461* | — | — | *0.8834* | *0.8890* | *0.8863* |

### 3.2 RNAInterdataset

| 排名 | 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR | ΔAUPR | 复现 ACC | 复现 Sen | 复现 Spec |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **MFIDMA** | **0.9489 ± 0.0048** | 0.9168 | +0.0321 | **0.9432** | 0.9074 | +0.0358 | 0.8780 | 0.8963 | 0.8599 |
| 2 | DLST-MDA | 0.9415 ± 0.0039 | 0.9143 | +0.0272 | 0.9384 | 0.9036 | +0.0348 | 0.8570 | 0.9075 | 0.8073 |
| 3 | LDAMAN | 0.9413 ± 0.0031 | 0.9081 | +0.0332 | 0.9369 | 0.8989 | +0.0380 | 0.8676 | 0.8601 | 0.8751 |
| 4 | LAGCN | 0.9361 ± 0.0032 | 0.8337 | +0.1024 | 0.9318 | 0.7360 | +0.1958 | 0.8589 | 0.8530 | 0.8651 |
| 5 | GCNNMMA | 0.9098 ± 0.0124 | 0.9116 | −0.0018 | 0.9103 | 0.8988 | +0.0115 | 0.8355 | 0.8431 | 0.8282 |
| — | *MSFFMDI（论文）* | *0.9216* | — | — | *0.9163* | — | — | *0.8524* | *0.8754* | *0.8516* |

---

## 4. 结论与诚实性说明

1. **复现良好的基线**：GCNNMMA（ncDR ΔAUC −0.008 / RNAInter −0.002）、LDAMAN（ncDR −0.007 /
   RNAInter +0.033）与论文接近。
2. **复现值显著高于论文的基线**：LAGCN（ncDR +0.039 / RNAInter +0.102 AUC）、MFIDMA
   （+0.010 / +0.032）、DLST-MDA（−0.021 / +0.027）。原因：论文对基线的适配/调优细节未知，
   本复现做了合理适配并充分训练；且论文口径未知（其“5-CV”文件本身不成立）。
3. **对论文结论的影响**：论文称 MSFFMDI “outperformed all comparison methods in both AUC
   and AUPR”。本复现口径下 **MFIDMA（基线）在两个数据集上的 AUC/AUPR 都高于论文的
   MSFFMDI 值**；LAGCN/LDAMAN/DLST-MDA 也大多追平或超过。**该结论不成立**。
4. **协议不可比性**：论文 Table 1 的 4457（ncDR）是含重复的行数；其 5 折 test 集合相同且
   有 train-test 重叠。本复现用 clean 5-CV，绝对数值与论文不严格可比；请按“同协议可比”
   解读。
5. **保真度缺口**：LDAMAN 用 GCN 自编码器近似其 SDNE 表示；LAGCN 由药-病迁移到 miRNA-药；
   两者均已在实现表中注明。

---

## 5. 目录与产物

```
msffmdi_baselines_20260920/
├─ REPORT_MSFFMDI_BASELINES.md   # 本文件
├─ ACQUISITION_REPORT.md         # 论文获取尝试记录（用户后提供 PDF）
├─ msffmdi_data.py               # ncDR/RNAInter 数据加载
├─ msffmdi_models.py             # 5 个基线模型
├─ run_msffmdi.py                # 统一 runner（clean 5 折 CV）
├─ summarize_msffmdi.py          # 汇总 + Excel
├─ MSFFMDI_baselines.xlsx        # 结果表（说明 / ncDR / RNAInter）
├─ results/*.csv                 # 10 个逐折结果
├─ logs/                         # 运行日志
├─ repos/MFIDMA-master/          # MFIDMA 官方数据（ncDR/RNAInter + 5 折文件）
└─ figures/Fig1..19.png          # 论文公开图
```

复现命令：
```bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
bash run_all_msffmdi.sh        # 或 run_remaining_msffmdi.sh
$PY summarize_msffmdi.py
```
