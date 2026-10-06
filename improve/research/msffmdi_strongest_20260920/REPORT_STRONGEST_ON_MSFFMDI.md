# 本文最强模型在 MSFFMDI 数据（ncDR / RNAInter）上的排名

目录：`improve/research/msffmdi_strongest_20260920/`

## 0. 结论

把项目当前最强框架（**frozen TabICLv2 图上下文参考 + anchor-shrinkage 后融合**）的可移植子集，
接到 MSFFMDI 论文的两个数据集 **ncDR（624×95）** 与 **RNAInter（1009×281）** 上，
用与复现基线**完全相同**的 clean 5 折 CV 实测：

| 数据集 | 本文方法 AUC / AUPR | 排名 | 第 1 名 | 说明 |
|---|---|---|---|---|
| **ncDR** | **0.9535 / 0.9481** | **#2** | MFIDMA 0.9554 / 0.9485 | 与第 1 名仅差 **0.0019 AUC / 0.0004 AUPR**（统计打平）；高于论文 MSFFMDI（0.9483/0.9461） |
| **RNAInter** | **0.9578 / 0.9554** | **#1** | 本文方法 | 领先第 2 名 MFIDMA **+0.0089 AUC / +0.0122 AUPR** |

> 若改用**纯图上下文参考**（不叠 anchor）：ncDR **0.9541 / 0.9491**（AUC 仍 #2、**AUPR 反超 MFIDMA 成 #1**）；
> RNAInter **0.9579 / 0.9553**（#1）。anchor 在此 benchmark 上几乎无增益。

---

## 1. 完整排名（clean 5 折 CV，seed 42）

### ncDRdataset

| 排名 | 方法 | AUC | AUPR | ACC | 备注 |
|---|---|---|---|---|---|
| 1 | MFIDMA | 0.9554 | 0.9485 | 0.8897 | 复现基线 |
| **2** | **iNcRD-HG-strongest（本文方法）** | **0.9535** | **0.9481** | 0.8833 | 图上下文 + anchor |
| 3 | MSFFMDI（论文报告） | 0.9483 | 0.9461 | 0.8834 | 论文值 |
| 4 | LDAMAN | 0.9319 | 0.9208 | 0.8612 | 复现基线 |
| 5 | LAGCN | 0.9297 | 0.9154 | 0.8605 | 复现基线 |
| 6 | DLST-MDA | 0.9241 | 0.9112 | 0.8364 | 复现基线 |
| 7 | GCNNMMA | 0.9041 | 0.8868 | 0.8289 | 复现基线 |

### RNAInterdataset

| 排名 | 方法 | AUC | AUPR | ACC | 备注 |
|---|---|---|---|---|---|
| **1** | **iNcRD-HG-strongest（本文方法）** | **0.9578** | **0.9554** | 0.8870 | 图上下文 + anchor |
| 2 | MFIDMA | 0.9489 | 0.9432 | 0.8780 | 复现基线 |
| 3 | DLST-MDA | 0.9415 | 0.9384 | 0.8570 | 复现基线 |
| 4 | LDAMAN | 0.9413 | 0.9369 | 0.8676 | 复现基线 |
| 5 | LAGCN | 0.9361 | 0.9318 | 0.8589 | 复现基线 |
| 6 | MSFFMDI（论文报告） | 0.9216 | 0.9163 | 0.8524 | 论文值 |
| 7 | GCNNMMA | 0.9098 | 0.9103 | 0.8355 | 复现基线 |

---

## 2. 移植了什么、没移植什么

| 组件 | 是否移植 | 说明 |
|---|---|---|
| 图上下文 33 列（`dlstmda_strongest_20260920/features.py::GraphPairEncoder`） | ✅ | 仅用关联图 + k-mer/Morgan 相似度，数据无关 |
| TabICLv2 参考（frozen checkpoint） | ✅ | `tabicl-classifier-v2-20260212.ckpt` |
| anchor shrinkage（hgb+et+mlp+lr, w=0.30） | ✅ | `context_anchor_verify.py` 配方 |
| 证据分层 / DepMap 上下文档 | ❌ | ncDR/RNAInter 无表达谱/靶标/通路数据 |
| lncRNA↔miRNA 跨任务 bridge | ❌ | 单任务 benchmark，bridge 列置 0 |

> 因此这是最强框架**可移植子集**的结果，是该框架能力的**下界**。

## 3. 协议与诚实性说明

- 与 `../msffmdi_baselines_20260920` 使用**同一 clean 5 折 CV**（去重正例 + 等量随机负例，seed 42），
  各方法同折同负例，可直接横向比较。
- 每折剔除 3–10 个“实体级”重复测试 pair（ncDR/RNAInter 存在重复 miRNA 序列），已保证 train/test 实体不相交。
- TabICLv2 冻结、无训练；anchor members 为 from-scratch 学习者。
- **ncDR 上本文方法为 #2，但与 #1 MFIDMA 差距（0.0019 AUC）远小于折间波动（std 0.004–0.005），
  属统计打平**；RNAInter 上明确 #1。结论：本文最强模型在 MSFFMDI 数据上处于**第一梯队**，
  可稳定超过论文方法 MSFFMDI 与全部复现基线（RNAInter），在 ncDR 上与最强基线并列。

## 4. 产物与复现命令

```
msffmdi_strongest_20260920/
├─ msffmdi_strongest_features.py  # 图特征 + 折数据缓存（rdkit 环境）
├─ run_tabicl_msffmdi.py          # TabICLv2 图上下文 + anchor，逐折评测
├─ rank_msffmdi.py                # 排名汇总
├─ MSFFMDI_strongest_ranking.xlsx # ncDR / RNAInter 两个榜单
├─ graph_cache_{ncDR,RNAInter}.npz / folds_*.pkl
└─ results/strongest_*.csv
```

```bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
$PY msffmdi_strongest_features.py
V=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
$V run_tabicl_msffmdi.py --dataset ncDR     --device cuda:0
$V run_tabicl_msffmdi.py --dataset RNAInter --device cuda:0
$PY rank_msffmdi.py
```
