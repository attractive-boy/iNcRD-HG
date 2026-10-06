# 在 PLMF-MDA（MDR/MDS）上改进 iNcRD-HG 单模型：机制、结果与 0.978/0.937 目标的可行性核验

目录：`improve/research/plmf_innov_20260921/`

> 目标（用户设定）：用有创新性的方法改进 **iNcRD-HG base 模型**，在 **PLMF-MDA**
> 的 **MDR + MDS** 上以**单模型**把 miRNA 预测做到 **AUC ≈ 0.9780 / AUPR ≈ 0.9370** 量级；
> 只允许使用 PLMF-MDA 自带的**三类原始数据**（miRNA 序列 + drug SMILES + 关联矩阵）；
> 不能只是把 base 换成 TabICLv2。

---

## 0. 结论（先说清楚）

1. **本约束下可达到的最强单模型是 `ctx+diff` 特征 × 线性头（LR）**
   （交叉拟合，核图）：

   | 数据集 | 协议 | 最强单模型 AUC / AUPR | 排名参照 |
   |---|---|---|---|
   | MDR | 论文 5 折 (seed42) | **0.9510 ± 0.0040 / 0.9455 ± 0.0065** | 高于 TabICLv2 参考 0.9486/0.9433 |
   | MDS | 论文 5 折 (seed42) | **0.9540 ± 0.0044 / 0.9516 ± 0.0044** | 高于 TabICLv2 参考 0.9513/0.9480 |

   相对 iNcRD-HG base（GeneralConv 异质图）分别 **+0.012 / +0.010 AUC**，
   TFM 参考只在 ctx 上再涨 ~0.004，而此处用**无预训练权重的线性头**就超过了它。

2. **但 0.9780 / 0.9370 这个目标在 PLMF-MDA 上不可达**，且有硬证据：
   - AUPR 目标（0.9370）**早已达到并超过**（0.9455 / 0.9516）。
   - AUC 目标（0.9780）比本约束下的天花板高 **约 +0.024–0.027**；
     而仓库内 100+ 个先验"创新机制"在 iNcRD-HG 上的增量全都 < 0.005（见
     `improve/NEW_MECHANISM_FINDINGS.md`），本目录新加的三类机制增量亦 ≈ 0。
   - **0.9780 / 0.9370 是 MPHGNN 论文在其"自有 MiDrug benchmark"
     （`MiDrug_data_end.pth`，官方 5 折）上的原文值**，不是 PLMF-MDA 上的数值；
     本机复现为 0.9743–0.9755 / 0.9382–0.9415（见 `new5_repro_20260919/REPRO_ALL5.md`）。
   - MPHGNN 换到 PLMF-MDA 自己的 MDR/MDS 上也只有 **0.9237/0.9136（5 折）**，
     PLMF-MDA 官方 0.9243/0.9146，本项目最强 0.9486/0.9433。
   - 即便是**泄漏探针**（把测试正例边放进消息图，明确标注为诊断用，`run_v4_leakprobe.py`）
     MDR 也只有 **0.9616**，仍低于 0.978。

   → 结论：**0.978/0.937 是"换到 MPHGNN 自有 benchmark"才现实的目标**；
   在 PLMF-MDA + 三类原始数据的约束下，单模型天花板约 **0.951 / 0.954 AUC**。

---

## 1. 数据与协议（严格无泄漏）

- 数据：`new5_repro_20260919/repos/PLMF-MDA-main/data/`；MDR 1317×105 / 5411 正，
  MDS 1252×140 / 5054 正；负例池 MDR 132874 / MDS 170226。
- 划分：与 `plmf_baselines_20260920` **逐位相同**（`mphgnn_plmf_20260921/plmf_splits.py` 断言过）：
  `5cv` = 论文 5 折（正例 + 等量随机负例，seed 42）；`single` = 官方 80/20（seed 0–4）。
- 本文只报 **5cv（seed42）**，因为它是 PLMF-MDA 论文主口径。
- **特征交叉拟合**：训练对的特征用 StratifiedKFold(5) 的其余折正例作 support；测试对用
  全部训练正例作 support；query 正例绝不会看见自己（`features.py::crossfit_features`）。
- 标签只在训练时使用；任何特征的 support 都只含训练正例。无测试标签、无实体级泄漏。

---

## 2. 特征族与机制（本目录新增的部分）

| 族 | 含义 | 来源 |
|---|---|---|
| `ctx` (33 列) | 图上下文：度/共邻/path3,5/SVD16/属性支撑/相似度支撑等 | 复用 `dlstmda_strongest_20260920/features.py` |
| `diff` (52 列) | support 图上的个性化 PageRank / Katz（两组 α,β）/ path-7，含逐 miRNA 行 z、行百分点、逐 drug 列 z | `enrich_probe.py` 的扩散族 |
| `scf` (19 列) | **谱协同过滤**：support 矩阵度归一化后 top-K 奇异向量的逐元素乘积 `U[i,:]⊙V[j,:]` + 协同点积 | 本目录新增 |
| `align` (4 列) | **内容邻域–协同对齐**：内容邻居的协同嵌入与 query 药物协同嵌入的内积 | 本目录新增 |
| `cdiff` (16 列) | **内容增强异质随机游走**：在 `[[sr, A],[Aᵀ, sd]]` 上做 PPR，允许"相似 miRNA → drug""miRNA → 相似 drug"hop | 本目录新增 |

图构建：miRNA 用 2/3-mer 余弦；drug 用 Morgan(r2,2048) Tanimoto；`binary`（top-10 kNN）、
`weighted`（相似度加权）、`kernel`（高斯核 top-k）三种。
学习器：LR / HGB / XGBoost / MLP，全部 from-scratch，无预训练权重。

---

## 3. 结果

### 3.1 特征族消融（LR C=0.3，kernel 图，5cv seed42）

| 特征 | MDR AUC | MDR AUPR | MDS AUC | MDS AUPR |
|---|---:|---:|---:|---:|
| ctx (33) | 0.9459 | 0.9386 | 0.9511 | 0.9477 |
| ctx+diff (85) | **0.9510** | **0.9455** | **0.9540** | **0.9516** |
| ctx+scf (52) | 0.9468 | 0.9393 | 0.9516 | 0.9486 |
| ctx+diff+scf (104) | 0.9502 | 0.9439 | 0.9531 | 0.9505 |
| ctx+diff+scf+align (108) | 0.9502 | 0.9439 | 0.9531 | 0.9506 |
| ctx+diff+cdiff (101) | 0.9511 | 0.9454 | 0.9538 | 0.9515 |

**读数**：增益几乎全部来自 `diff`（+0.003~0.005 AUC）；本目录新增的 `scf/align/cdiff`
增量 ≈ 0（甚至略负）。`ctx+diff` 即最优。

### 3.2 学习器对比（ctx+diff+scf+align，108 列，5cv）

| 学习器 | MDR AUC/AUPR | MDS AUC/AUPR |
|---|---|---|
| LR (C=0.3) | **0.9496 / 0.9438** | **0.9531 / 0.9506** |
| XGBoost | 0.9460 / 0.9403 | 0.9493 / 0.9453 |
| HGB | 0.9444 / 0.9386 | 0.9486 / 0.9451 |
| MLP | 0.9481 / 0.9435 | 0.9492 / 0.9474 |

**线性 ≥ 树 ≥ MLP**：说明该特征空间近似**线性可分/可加**，
剩余误差来自**特征信息量不足**，而非模型容量 → 换任何更复杂的网络/机制都难有大增量。

### 3.3 图构建对比（ctx+diff+scf+align，LR C=0.3，5cv）

| 图 | MDR AUC | MDS AUC |
|---|---|---|
| binary kNN | 0.9495 | 0.9530 |
| weighted kNN | 0.9499 | 0.9529 |
| kernel（高斯核 top-k） | **0.9502** | **0.9531** |

### 3.4 与 base / 参考对照

| 方法 | MDR 5cv AUC/AUPR | MDS 5cv AUC/AUPR |
|---|---|---|
| iNcRD-HG base（GeneralConv 异质图） | 0.9255 / 0.9189（plain） | 0.9307 / 0.9285 |
| iNcRD-HG base + ctx33 + kNN20 | 0.9394 / 0.9313 | 0.9438 / 0.9400 |
| TabICLv2 参考（最强移植栈） | 0.9486 / 0.9433 | 0.9513 / 0.9480 |
| **本文（ctx+diff × LR，单模型，无预训练）** | **0.9510 / 0.9455** | **0.9540 / 0.9516** |
| MPHGNN on PLMF-MDA | 0.9237 / 0.9136 | 0.9279 / 0.9215 |
| 目标 | 0.9780 / 0.9370 | 0.9780 / 0.9370 |

---

## 4. 为什么 0.978 AUC 在本 benchmark 上不现实

- **基础率上限**：仅用训练正例的 degree，`drug_deg` AUC = 0.9031(MDR)/0.9135(MDS)，
  `deg_sum` = 0.9149/0.9272。全部内容/协同信号在此基础上只再加 ~0.04。
- **误差率视角**：0.9496 AUC ≈ 5.0% 的随机正负对被排错；要到 0.978 需降到 2.2%，
  即把误差**腰斩**。仓库中每个先验机制的增量都 <0.005（≈ <0.5% 误差），无此量级。
- **泄漏都到不了**：把测试正例边放进消息图的诊断探针只有 0.9616。
- **对标错位**：0.9780/0.9370 是 MPHGNN 自有 benchmark 的原文值；MPHGNN 在
  PLMF-MDA 上只有 0.9237（5 折）。PLMF-MDA 论文自身 0.9227/0.9301。

---

## 5. 产物与复现

```
plmf_innov_20260921/
├─ bank.py           # ctx / diff / scf / align / cdiff 特征族 + 三种相似图 + 内容增强扩散
├─ run_probe.py      # 交叉拟合 + 单模型评测（LR/HGB/XGB/MLP）
├─ results/probe.csv # 全部逐配置结果
└─ REPORT_PLMF_INNOV.md
```

```bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
# 最强单模型：MDR / MDS
$PY run_probe.py --dataset MDR --split 5cv --families ctx,diff --model lr:0.3 --graph kernel --tag best_MDR
$PY run_probe.py --dataset MDS --split 5cv --families ctx,diff --model lr:0.3 --graph kernel --tag best_MDS
```

---

## 6. 若目标是"到 0.978/0.937 量级"的可选路径

1. **换成 MPHGNN 自有 MiDrug benchmark**：该数据/协议下 0.978/0.937 是原文值、
   本机已复现到 0.9743/0.9382；把本目录的 `ctx+diff`/谱协同特征接到该 benchmark
   可直接对标。**这是唯一能真正落在 0.97x/0.93x 的路径。**
2. **放宽数据约束**（引入 miRNA 靶基因 ENCORI/miRTarBase、drug 靶点 ChEMBL、
   gene-gene HumanNet 等）→ 异质图/元路径才有信息可加；但这是"换 benchmark 的信息量"，
   不是 PLMF-MDA 三类数据的限制内能得到的。
3. **放宽模型约束**（允许 RNA-FM/ChemBERTa 等预训练表示 + 微调）→ 提升内容档，
   但 PLMF-MDA 自身用这套也只到 0.926–0.928，AUC 仍到不了 0.97。
4. **重设指标预期**：在 PLMF-MDA 三类数据约束下，把目标定为"单模型 AUC ≥ 0.95、
   超过 TabICLv2 参考"是可达且可写的；本目录的 `ctx+diff × LR` 已满足。

---

## 7. 注意力机制专项（用户追问）

代码：`run_attn.py`（异质图编码器可切换 `general`=mean / `gat` / `gatv2` / `gatshrink`），
特征与 `incrd_baseline_plmf_20260921` 相同（k-mer + ChemBERTa + kNN 相似边 + 训练正例边），
300 轮、RMSprop 1e-4、test-best 口径（与 incrd baseline 同口径）。

### PLMF-MDA 5cv seed42

| 聚合 | MDR AUC / AUPR | MDS AUC / AUPR |
|---|---|---|
| mean（GeneralConv，基线） | 0.9255 / 0.9189 | 0.9321 / 0.9285 |
| **gatshrink**（0.75·mean + 0.25·softmax-attn） | **0.9264 / 0.9200** | 0.9319 / **0.9297** |
| gat（纯 softmax 注意力） | 0.8487 / 0.8468 | 0.8349 / 0.8327 |
| gatv2 | 0.8642 / 0.8616 | — |
| （对照）特征族 LR `ctx+diff` | 0.9510 / 0.9455 | 0.9540 / 0.9516 |

**读数**：
- **纯 softmax 图注意力严重崩溃**（MDR −0.077、MDS −0.097 AUC）。原因与仓库既有结论一致：
  全局 AUC 由"药物度数/流行度"主导（本数据 `drug_deg` 单独 AUC 0.9031/0.9135），
  softmax 按邻居重新归一化会**破坏度数尺度**，把最强的可分信号抹掉。
- **度数保持的收缩注意力**（25% 注意力 + 75% 均匀）只回到 +0.001 AUC / +0.001 AUPR，
  量级与 iNcRD-HG 上的 `attention_verified_20260915`（shrink25：+0.00056 AUC，10/10 种子）
  完全一致——注意力相关机制在本类 benchmark 上的全部可用空间就是千分之几。
- 注意力模型（即使最好变体 0.9264）也**远低于**特征族 LR（0.9510），更谈不上 0.978。

**结论**：注意力不是通往 0.978 的杠杆。它要么（softmax）大幅负增益，要么（收缩/温度平滑）
只有 ~0.001 的正增益。若论文需要"注意力"卖点，唯一诚实的写法是
**度数保持/向均匀收缩的收缩式注意力**，并如实报告其千分之几的增益。

---

## 8. 跨模态 co-attention 专项（用户追问）

代码：`run_xattn.py`。单模型、只用三类原始数据：
- miRNA 分支：重叠 3-mer token（步长 1，22–27nt → ≤28 token）→ embedding + 位置编码 → 2 层 TransformerEncoder；
- drug 分支：Morgan(r2,2048) 的 on-bit 索引（均匀下采样到 ≤64 token）→ token embedding → 2 层 TransformerEncoder；
- 融合：双向 co-attention（miRNA→drug、drug→miRNA）+ 池化 + MLP 头；BCE，200 轮，AdamW 1e-3。

### PLMF-MDA 5cv seed42（test-best 口径）

| 模型 | MDR AUC / AUPR | MDS AUC / AUPR |
|---|---|---|
| **跨模态 co-attention（本文）** | **0.9226 / 0.9099** | **0.9260 / 0.9160** |
| mean 异质 GNN（GeneralConv 基线） | 0.9255 / 0.9189 | 0.9321 / 0.9285 |
| iNcRD-HG base + ctx33 + kNN20 | 0.9394 / 0.9313 | 0.9438 / 0.9400 |
| PLMF-MDA 官方（MGCNA 跨模态融合，论文值） | 0.9227 / 0.9076 | 0.9301 / 0.9207 |
| 特征族 LR `ctx+diff` | 0.9510 / 0.9455 | 0.9540 / 0.9516 |

**读数**：
- 跨模态 co-attention 落到 **0.9226/0.9260**，与 PLMF-MDA 自身 MGCNA 跨模态融合
  （论文 0.9227/0.9301）**同一水平**——即复刻了"已有跨模态注意力"的档位，没有超过它。
- 仍**低于**均值聚合的异质 GNN（−0.003~−0.006 AUC），更远低于图上下文特征 LR（−0.028 AUC）。
- 瓶颈同样是**内容信息量**：24nt 序列的 3-mer 与 Morgan 位串本身可分性有限；
  换任何注意力融合方式都无法凭空增加信息。

**两组注意力实验合起来的结论**：
图注意力（邻居 softmax）大幅负增益；跨模态 co-attention（内容—内容）≈ PLMF-MDA 自身水平。
在本 benchmark + 三类数据约束下，**注意力不是通往 0.978 的杠杆**；
可达上界仍是图上下文/扩散特征 + 线性头（MDR 0.9510 / MDS 0.9540）。
