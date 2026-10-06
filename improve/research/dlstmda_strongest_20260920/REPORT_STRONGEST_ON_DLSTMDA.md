# 把「目前最强模型」跑到 DLST-MDA benchmark 上：结果与排名

目录：`improve/research/dlstmda_strongest_20260920/`

## 0. 结论

把项目当前最强的融合框架（**TabICLv2 graph-context reference + anchor-shrinkage 后融合**）
接到 DLST-MDA 的 1578×156 miRNA–drug benchmark（作者原始 5 折）上实测：

| 设置 | 本方法 AUC / AUPR | 排名 | 榜首 |
|---|---|---|---|
| **1:1（主设定）** | **0.9606 / 0.9557** | **#1**（无泄漏口径） | 本方法 |
| **1:5** | **0.9636 / 0.8513** | **#1** | 本方法 |
| **1:10** | **0.9635 / 0.7681** | **#1** | 本方法 |
| 冷启动 miRNA | 0.9318 / 0.9154 | #3 | DLST-MDA 论文 0.9394 |
| 冷启动 drug | 0.7123 / 0.7020 | #3 | SubMDTA 0.7637 |

- 在三个正负比设定下均为**第 1**，明显高于论文 DLST-MDA（1:1 0.9411）与所有复现基线。
- 仅当把**转导式（含测试关联）DAESTB** 计入时，1:1 为 **#2**（该口径泄漏、不应计入公平排名）。
- 冷启动两档为 **#3**（未见实体场景泛化一般，drug 冷启动尤其）。

## 1. 移植了什么、没移植什么（重要）

项目 README 里“当前最强”是一整套配方：**frozen TabICLv2 tiered reference（图上下文 36 列
⊕ 证据分层 26 列 ⊕ DepMap 上下文 14 列，50/50 融合）+ 任务共享 anchor shrinkage**。

DLST-MDA benchmark 只有 **miRNA 序列 + drug SMILES + 关联**，因此：

| 组件 | 是否移植 | 说明 |
|---|---|---|
| 图上下文 36 列（`graph_tabm_modules.PairFeatureEncoder`） | ✅ 重实现 | 仅用关联图 + 序列/指纹相似度，**数据无关** |
| TabICLv2 参考（frozen checkpoint） | ✅ 原样 | `tabicl==2.2.0` + `tabicl-classifier-v2-20260212.ckpt` |
| anchor shrinkage（hgb+et+mlp+lr, w=0.30） | ✅ 原样 | `context_anchor_verify.py` 配方 |
| 证据分层 26 列（LncTarD/miRTarBase/ENCORI/Reactome） | ❌ 不可移植 | DLST-MDA 无 RNA→gene 靶标/通路数据 |
| DepMap 上下文 14 列 | ❌ 不可移植 | benchmark 无细胞系/表达谱维度 |
| lncRNA↔miRNA 跨任务 bridge | ❌ | 单任务 benchmark，bridge 列置 0 |

> 因此本结果是**最强框架的“图上下文档 + anchor”**在 DLST-MDA 上的落地，不是含证据档的完整配方。
> 在 iNcRD-HG 自己的数据上，含证据档相对纯图上下文只再涨约 +0.001 AUC / +0.003 AP，故这里的
> 数字可视为该框架能力的下界。

## 2. 实现细节

- **节点**：miRNA＝2/3-mer 频率（80 维）；drug＝Morgan 指纹 r=2, 2048 bit。
- **相似图**：miRNA 采用 k-mer 余弦 top-10 kNN；drug 采用 Morgan Tanimoto top-10 kNN（二值、无自环）。
- **33 列图上下文特征**：degree / collab / path3 / path5 / svd16 / neighbor-degree / bridge(0) /
  similarity-degree / attribute-support / feature stats（`features.py::GraphPairEncoder`）。
- **训练特征做 5 折 cross-fit**（`StratifiedKFold`，support=其余折正例），评测特征用全部训练正例 ——
  与 iNcRD-HG 完全一致，杜绝特征层自泄漏。
- **TabICLv2**：MiDrug 视图＝raw，`n_estimators=32`，`softmax_temperature=0.9`，`average_logits=True`。
- **anchor**：`sigmoid(0.7·logit(TabICL) + 0.3·mean_m logit(member_m))`，members={HGB, ExtraTrees, MLP, LR}。
- **评测**：每折单独 fit/score；AUC / AUPR / ACC。3 个推理种子（666/0/1）确认稳定。

**数据口径修正（必读）**：DLST-MDA 的 1578 行 miRNA 中只有 1567 条**唯一序列**，因此按“分子实体”
建图时相邻重复序列会合并；每折有 5–13 个测试 pair 与训练 pair 在“实体级”重合，已剔除以保证
train/test 在实体层不相交。这是图方法在作者“按行划分”的数据上不可避免的口径差异。

## 3. 结果

### 3.1 1:1（作者原始 5 折；seed 20260914 + 666/0/1 共 4 次推理 × 5 折池化）
本方法 **AUC 0.9606 / AUPR 0.9557**；graph-context 参考本身 0.9606/0.9555（anchor 基本持平）。

| 排名 | 方法 | AUC | AUPR |
|---|---|---|---|
| **1** | **iNcRD-HG-strongest（本方法）** | **0.9606** | **0.9557** |
| 2 | ML-DTI | 0.9443 | 0.9352 |
| 3 | DLST-MDA（官方代码复现） | 0.9437 | 0.9356 |
| 4 | DLST-MDA（论文报告） | 0.9411 | 0.9326 |
| 5 | GraphDTA | 0.9325 | 0.9210 |
| 6 | SubMDTA | 0.9293 | 0.9130 |
| 7 | GCNNMMA | 0.9247 | 0.9074 |
| 8 | DAESTB（无泄漏） | 0.8957 | 0.9018 |
| — | *DAESTB（转导 GIP，含测试关联，仅参考）* | *0.9685* | *0.9642* |

### 3.2 正负比 1:5 / 1:10

| 排名 | 1:5 方法 | AUC / AUPR | 1:10 方法 | AUC / AUPR |
|---|---|---|---|---|
| **1** | **本方法** | **0.9636 / 0.8513** | **本方法** | **0.9635 / 0.7681** |
| 2 | ML-DTI | 0.9522 / 0.8026 | ML-DTI | 0.9531 / 0.7085 |
| 3 | DLST-MDA 论文 | 0.9510 / 0.8026 | DLST-MDA 论文 | 0.9514 / 0.7049 |
| 4 | SubMDTA | 0.9472 / 0.7851 | SubMDTA | 0.9500 / 0.6952 |
| 5 | GraphDTA | 0.9433 / 0.7711 | GraphDTA | 0.9453 / 0.6668 |
| 6 | GCNNMMA | 0.9348 / 0.7299 | GCNNMMA | 0.9441 / 0.6537 |
| 7 | DAESTB(无泄漏) | 0.9024 / 0.7356 | DAESTB(无泄漏) | 0.9072 / 0.6312 |

### 3.3 冷启动（自建划分，见基线报告第 8 节）

| 排名 | 冷启动 miRNA 方法 | AUC / AUPR | 冷启动 drug 方法 | AUC / AUPR |
|---|---|---|---|---|
| 1 | DLST-MDA 论文 | 0.9394 / 0.9277 | SubMDTA | 0.7637 / 0.7306 |
| 2 | ML-DTI | 0.9331 / 0.9184 | DLST-MDA 论文 | 0.7327 / 0.5533 |
| **3** | **本方法** | **0.9318 / 0.9154** | **本方法** | **0.7123 / 0.7020** |
| 4 | GCNNMMA | 0.9264 / 0.9112 | ML-DTI | 0.6997 / 0.6735 |
| 5 | GraphDTA | 0.9221 / 0.9047 | DAESTB(无泄漏) | 0.6757 / 0.6700 |
| 6 | SubMDTA | 0.9199 / 0.9050 | GCNNMMA | 0.6689 / 0.6336 |
| 7 | DAESTB(无泄漏) | 0.5475 / 0.6071 | GraphDTA | 0.6554 / 0.6021 |

## 4. 诚实性声明

- 移植的是**最强框架的可移植子集**（图上下文档 + anchor）；证据/DepMap 档因数据缺失未启用。
- 1:5/1:10 与冷启动划分由本项目自建（论文未公开），数值不与原文严格可比；1:1 用的是作者原始划分。
- 每折剔除 5–13 个“实体级”重复测试 pair（重复序列导致），已在第 2 节说明。
- TabICLv2 推理为冻结权重、无训练；anchor members 为 from-scratch 学习者。3 个推理种子结果稳定
  （1:1 AUC 0.9605–0.9607；4 次推理池化 20 折）。

## 5. 产物

```
dlstmda_strongest_20260920/
├─ features.py                  # 33 列图上下文特征（DLST-MDA 版）
├─ run_tabicl.py                # TabICLv2 参考 + anchor，5 折评测
├─ rank.py / rank_all.py        # 排名汇总
├─ DLST_MDA_strongest_ranking.xlsx  # 1to1 / 1to5 / 1to10 / cold_mirna / cold_drug 五个榜单
├─ graph_cache.npz              # 预计算节点特征
├─ results/strongest_*.csv      # 各设置逐折结果（含 3 个推理种子）
└─ REPORT_STRONGEST_ON_DLSTMDA.md
```

复现命令：
```bash
# 1) 构建节点特征缓存（rdkit 环境）
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -c "import features as F; F.build_cache()"
# 2) 各设置评测（TabICL 环境）
V=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
$V run_tabicl.py --device cuda:0 --n-estimators 32 --out results/strongest_per_fold.csv
$V run_tabicl.py --device cuda:0 --n-estimators 32 --proc ../dlstmda_baselines_20260919/ratios/1to5  --out results/strongest_1to5.csv
$V run_tabicl.py --device cuda:0 --n-estimators 32 --proc ../dlstmda_baselines_20260919/ratios/1to10 --out results/strongest_1to10.csv
$V run_tabicl.py --device cuda:0 --n-estimators 32 --proc ../dlstmda_baselines_20260919/coldstart/mirna --out results/strongest_cold_mirna.csv
$V run_tabicl.py --device cuda:0 --n-estimators 32 --proc ../dlstmda_baselines_20260919/coldstart/drug  --out results/strongest_cold_drug.csv
# 3) 排名
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python rank_all.py
```
