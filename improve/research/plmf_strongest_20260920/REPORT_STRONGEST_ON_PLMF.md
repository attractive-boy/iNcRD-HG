# 把「本文最强模型」放到 PLMF-MDA 的 MDR/MDS 上：排名结果

目录：`improve/research/plmf_strongest_20260920/`

## 0. 结论

把项目当前最强框架（**frozen TabICLv2 图上下文参考 + anchor-shrinkage 后融合**）的可移植子集，
接到 PLMF-MDA 自建的 **MDRdataset / MDSdataset** 上实测，与 PLMF-MDA 及 4 个复现基线同榜对比：

| 数据集 | 协议 | 本文方法 AUC / AUPR | 排名 | 第 2 名（差距） |
|---|---|---|---|---|
| **MDR** | 论文 5 折 (seed42) | **0.9486 / 0.9433** | **#1** | PLMF-MDA 0.9243（**+0.0243 AUC / +0.0287 AUPR**） |
| **MDR** | 官方 80/20 (seed0-4) | **0.9514 / 0.9458** | **#1** | PLMF-MDA 0.9287（+0.0227 / +0.0296） |
| **MDS** | 论文 5 折 (seed42) | **0.9513 / 0.9480** | **#1** | ML-DTI 0.9296（+0.0217 / +0.0254） |
| **MDS** | 官方 80/20 (seed0-4) | **0.9522 / 0.9486** | **#1** | ML-DTI 0.9303（+0.0219 / +0.0278） |

**四个榜单全部第 1，且领先第 2 名约 0.022–0.024 AUC / 0.025–0.030 AUPR**，优势远大于折间/seed 波动
（标准差 0.003–0.007）。相对 PLMF-MDA 论文报告值（MDR 0.9227 / MDS 0.9301）分别高 +0.026 / +0.021 AUC。

---

## 1. 移植了什么、没移植什么

项目 README 里“当前最强”是一整套配方：**frozen TabICLv2 tiered reference（图上下文 36 列 ⊕ 证据分层 26 列
⊕ DepMap 上下文 14 列，50/50 融合）+ 任务共享 anchor shrinkage**。PLMF-MDA 的 MDR/MDS 只有
**miRNA 序列 + drug SMILES + 关联矩阵**，因此：

| 组件 | 是否移植 | 说明 |
|---|---|---|
| 图上下文 33 列（`dlstmda_strongest_20260920/features.py::GraphPairEncoder`） | ✅ | 仅用关联图 + 序列/指纹相似度，数据无关 |
| TabICLv2 参考（frozen checkpoint） | ✅ | `tabicl-classifier-v2-20260212.ckpt` |
| anchor shrinkage（hgb+et+mlp+lr, w=0.30） | ✅ | `context_anchor_verify.py` 配方 |
| 证据分层 26 列 | ❌ | MDR/MDS 无 RNA→gene 靶标/通路数据 |
| DepMap 上下文 14 列 | ❌ | 无细胞系/表达谱维度 |
| lncRNA↔miRNA 跨任务 bridge | ❌ | 单任务 benchmark，bridge 列置 0 |

> 因此本结果是**最强框架的“图上下文档 + anchor”**在 MDR/MDS 上的落地，不是含证据档的完整配方；
> 在 iNcRD-HG 自有数据上证据档只再涨约 +0.001 AUC，故这里的数字可视为该框架能力的**下界**。

## 2. 实现细节

- **节点**：miRNA = 2/3-mer 频率（80 维）；drug = Morgan 指纹 r=2, 2048 bit。
- **相似图**：miRNA 用 k-mer 余弦 top-10 kNN；drug 用 Morgan Tanimoto top-10 kNN（二值、无自环）。
- **33 列图上下文**：degree / collab / path3 / path5 / svd16 / neighbor-degree / bridge(0) /
  similarity-degree / attribute-support / feature stats。
- **训练特征 5 折 cross-fit**（support=其余折正例），评测特征用全部训练正例 —— 与 iNcRD-HG 一致，无特征层泄漏。
- **TabICLv2**：`n_estimators=32`，`softmax_temperature=0.9`，`average_logits=True`，冻结权重、无训练。
- **anchor**：`sigmoid(0.7·logit(TabICL) + 0.3·mean_m logit(member_m))`，members={HGB, ExtraTrees, MLP, LR}。
- **口径**：与 `../plmf_baselines_20260920` 完全相同的数据划分（5cv seed42 / single seed0-4）。
- **实体级去重**：MDR/MDS 存在重复 miRNA 序列（MDR 1317→1312 唯一，MDS 1252→1243 唯一；drug 均唯一），
  每折有 3–4 个测试 pair 与训练 pair 在“分子实体”层重合，已剔除，保证 train/test 实体不相交。

## 3. 完整排名

### 3.1 MDRdataset

| 排名 | 方法 | 5cv AUC / AUPR | single AUC / AUPR | 论文 AUC / AUPR |
|---|---|---|---|---|
| **1** | **iNcRD-HG-strongest（本文方法）** | **0.9486 / 0.9433** | **0.9514 / 0.9458** | — |
| 2 | PLMF-MDA | 0.9243 / 0.9146 | 0.9287 / 0.9162 | 0.9227 / 0.9076 |
| 3 | ML-DTI | 0.9238 / 0.9131 | 0.9281 / 0.9171 | 0.7660 / 0.7870 |
| 4 | GCNNMMA | 0.9118 / 0.8949 | 0.9179 / 0.9019 | 0.9070 / 0.8736 |
| 5 | GraphDTA | 0.9054 / 0.8907 | 0.9118 / 0.9006 | 0.8086 / 0.7979 |
| 6 | SubMDTA | 0.9044 / 0.8907 | 0.9068 / 0.8902 | 0.8936 / 0.8514 |

### 3.2 MDSdataset

| 排名 | 方法 | 5cv AUC / AUPR | single AUC / AUPR | 论文 AUC / AUPR |
|---|---|---|---|---|
| **1** | **iNcRD-HG-strongest（本文方法）** | **0.9513 / 0.9480** | **0.9522 / 0.9486** | — |
| 2 | ML-DTI | 0.9296 / 0.9226 | 0.9303 / 0.9208 | 0.7820 / 0.7881 |
| 3 | PLMF-MDA | 0.9270 / 0.9175 | 0.9294 / 0.9189 | 0.9301 / 0.9207 |
| 4 | GCNNMMA | 0.9133 / 0.9007 | 0.9167 / 0.9028 | 0.9143 / 0.8889 |
| 5 | GraphDTA | 0.9125 / 0.9036 | 0.9145 / 0.9012 | 0.7864 / 0.7911 |
| 6 | SubMDTA | 0.9059 / 0.8923 | 0.9075 / 0.8908 | 0.8858 / 0.8587 |

## 4. 诚实性声明

- 移植的是**最强框架的可移植子集**（图上下文档 + anchor）；证据/DepMap 档因数据缺失未启用，故为能力下界。
- anchor 与纯 graph-context 参考基本持平（见 `results/*.csv` 的 `ref_*` 列）：本 benchmark 上增益来自图上下文参考本身。
- 每折剔除 3–4 个“实体级”重复测试 pair（重复序列导致），已在第 2 节说明；这对各方法一致。
- 5cv/single 两种协议下结论一致（均 #1，优势 0.022–0.024 AUC），不是单次划分的偶然。
- TabICLv2 为冻结权重、无训练；anchor members 为 from-scratch 学习者。

## 5. 产物与复现命令

```
plmf_strongest_20260920/
├─ plmf_features.py            # MDR/MDS 节点特征 + 折数据适配（复用 dlstmda_strongest/features.py）
├─ run_tabicl_plmf.py          # TabICLv2 图上下文参考 + anchor，逐折评测
├─ rank_plmf.py                # 排名汇总
├─ PLMF_MDA_strongest_ranking.xlsx   # MDR_5cv / MDR_single / MDS_5cv / MDS_single 四个榜单
├─ graph_cache_{MDR,MDS}.npz   # 节点特征缓存
├─ folds_*.pkl                 # 折数据缓存
├─ results/strongest_*.csv     # 逐折/逐 seed 结果
└─ logs/
```

```bash
# 1) 构建节点特征 + 折缓存（rdkit 环境）
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python plmf_features.py
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -c "import plmf_features as P; P.build_folds_cache()"
# 2) 评测（TabICL 环境）
V=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
for ds in MDR MDS; do
  $V run_tabicl_plmf.py --dataset $ds --split 5cv    --seed 42 --device cuda:0
  for s in 0 1 2 3 4; do $V run_tabicl_plmf.py --dataset $ds --split single --seed $s --device cuda:0; done
done
# 3) 排名
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python rank_plmf.py
```
