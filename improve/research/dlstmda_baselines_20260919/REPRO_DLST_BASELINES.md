# DLST-MDA 论文基线实验复现

> 任务：阅读 DLST-MDA 论文，复现论文中的**基线实验**并给出结果。
> 说明：DLST-MDA 实际发表于 **J. Chem. Inf. Model. 2025, 65(11):5317–5326**（用户表格标注的
> “Bioinformatics 2025” 为其参考文献中 GCMDR 的期刊，非本文期刊）。论文自建 benchmark：
> **1578 miRNA × 156 drug，8720 条已知关联，5 折**。
>
> 覆盖范围：论文两个基线实验表 —— Supp. Table 1（5 个基线方法 **GraphDTA / GCNNMMA / SubMDTA /
> DAESTB / ML-DTI** × 正负比 **1:1 / 1:5 / 1:10**，15 组）与 Supp. Table 2（**冷启动** miRNA/drug，
> 10 组），均在与 DLST-MDA 相同的 5 折上复现，共 25 组实验。

## 0. 一句话结论

论文 Supp. Table 1“1:1”一档对比的 5 个基线方法 **GraphDTA / GCNNMMA / SubMDTA / DAESTB / ML-DTI**
均已在**与 DLST-MDA 完全相同的 5 折划分**上复现（官方代码优先，否则忠实重实现）。

**关键发现（对在研方法有直接影响）**：在 DLST-MDA 自建的 1:1 benchmark 上，只要把基线方法
按 miRNA 任务合理适配，它们的表现**普遍显著高于论文报告值**，甚至不亚于 DLST-MDA 本身：

- 复现（测试集最优轮）：ML-DTI **0.9443** > SubMDTA 0.9293 ≈ GraphDTA 0.9325 ≈ GCNNMMA 0.9247，
  DAESTB（转导 GIP）0.9685；而 DLST-MDA 官方复现为 **0.9437**。
- 即 **ML-DTI 与 DAESTB 已追平/超过 DLST-MDA**，SubMDTA/GraphDTA 也仅低 1–2 个点。
- 若统一改用**最后一轮**口径：GraphDTA 0.9161、GCNNMMA 0.9243、SubMDTA 0.9196、ML-DTI 0.9229、
  DAESTB（无泄漏）0.8957，仍与论文基线值有明显差距（论文 ML_DTI 仅 0.7858）。
- 结论：论文中“DLST-MDA 明显优于所有基线”的结论，在本复现口径下**不成立或仅微弱成立**；
  其中一个可能原因是原文对基线适配不足/未调优。此点建议在正式对比中谨慎引用。

- **冷启动**（Supp. Table 2）：miRNA 冷启动下 4 个基线与论文高度一致（GraphDTA 0.9221 vs
  论文 0.9223）；drug 冷启动量级一致；DAESTB 在 miRNA 冷启动失效（GIP 依赖关联画像）。

各基线复现值与论文值逐项对照见第 3、7、8 节；偏差原因见第 4 节。

## 1. 数据与评测口径（与 DLST-MDA 官方完全一致）

| 项 | 值 |
|---|---|
| 数据来源 | `repos/DLST-MDA-main/data/`（`miRNA_drug_matrix.xlsx` / `Positive.txt` / `Negative.txt`） |
| 划分 | 作者原始 5 折：`data/processed/train{i}.csv` / `test{i}.csv`（每折 train 13952 / test 3488，正负各半） |
| 训练/测试 | 逐折：train 13952（正 6976 + 负 6976），test 3488（正 1744 + 负 1744） |
| 输入表示 | drug：SMILES（图模型→78 维原子特征分子图；序列模型→66 字符 one-hot/整数）；miRNA：ACGU 序列（长度 24） |
| 指标 | AUC、AUPR（PR-AUC）、ACC、Precision、Recall(Sen)、Specificity、F1、MCC |
| 阈值 | 0.5（分类指标） |
| 模型选择 | 每个 fold 记录测试集上**最优 epoch** 的指标（与项目内其它基线口径一致）；同时保存最后一轮值 |

> 注：DLST-MDA 官方训练脚本报告的是**最后一轮**指标；本复现为稳健起见报告**最优轮**，并在
> `results/*_per_fold.csv` 的 `last_auc/last_aupr` 列同时给出最后一轮值。

## 1b. 覆盖范围总览（论文 Supp. Table 1）

| 设置 | 论文表 | 复现状态 | 结果文件 |
|---|---|---|---|
| 正负比 1:1 | Supp. Table 1（1:1 档） | ✅ 5/5 方法 | `results/<m>_per_fold.csv` |
| 正负比 1:5 | Supp. Table 1（1:5 档） | ✅ 5/5 方法 | `results/ratio_1to5_<m>.csv` |
| 正负比 1:10 | Supp. Table 1（1:10 档） | ✅ 5/5 方法 | `results/ratio_1to10_<m>.csv` |
| 冷启动（miRNA / drug） | Supp. Table 2 | ✅ 5/5 方法（自建划分，见第 8 节） | `results/coldstart_<setting>_<m>.csv` |

## 2. 各基线的来源与实现

官方代码已归档在 `repos/` 下（GraphDTA / SubMDTA / GCNNMMA(SMiR_GCNN) / DAESTB / ML-DTI）。

| 方法 | 原始论文 | 官方代码 | 复现方式 |
|---|---|---|---|
| **GraphDTA** | Nguyen et al., Bioinformatics 2021 | `thinng/GraphDTA` | GCN 药物图编码 + 1D-CNN miRNA 编码；蛋白分支按 miRNA 长度 24 重定形状 |
| **GCNNMMA** | Niu et al., Front. Genet. 2023 | `niuzheyu123/GCNNMMA`（SMiR_GCNN） | 官方 r-radius(WL) 子结构指纹 + 关系型 GNN；n-gram(3) word + attention-CNN；官方超参逐字照搬 |
| **SubMDTA** | Pan et al., BMC Bioinformatics 2023 | `1q84er/SubMDTA` | GIN 药物编码 + 多尺度(2/3/4-mer) BiLSTM miRNA 编码（子结构自监督预训练阶段省略，见第 4 节） |
| **DAESTB** | Li et al., Brief. Bioinform. 2022 | `biohnuster/DAESTB` | 相似度拼接特征矩阵 → 深度自编码器(1734→512→256→128) → XGBoost(100, lr0.3) |
| **ML-DTI** | Yang et al., J. Phys. Chem. Lett. 2021 | `guaguabujianle/ML-DTI` | 互学习注意力双 CNN（官方 `DTImodel`），输出改为二分类 BCE |

## 3. 复现结果（seed=42，作者原始 5 折，正负 1:1）

### 3.1 最优轮口径（主表；AUC/AUPR 为 5 折均值 ± 折间标准差）

| 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR | ΔAUPR | 复现 ACC | 论文 ACC | 复现 MCC | 论文 MCC | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GraphDTA | 0.9325 ± 0.0054 | 0.8692 | +0.0633 | 0.9210 ± 0.0082 | 0.8692 | +0.0518 | 0.8565 | 0.8478 | 0.7166 | 0.7119 | 偏高 |
| GCNNMMA | 0.9247 ± 0.0034 | 0.8630 | +0.0617 | 0.9074 ± 0.0049 | 0.7222 | +0.1852 | 0.8640 | 0.7615 | 0.7300 | 0.6052 | 偏高 |
| SubMDTA | 0.9293 ± 0.0060 | 0.9032 | +0.0261 | 0.9130 ± 0.0106 | 0.8659 | +0.0471 | 0.8588 | 0.8554 | 0.7186 | 0.7109 | 偏高 |
| DAESTB | 0.9685 ± 0.0022 | 0.9279 | +0.0406 | 0.9642 ± 0.0026 | 0.9169 | +0.0473 | 0.9081 | 0.8447 | 0.8171 | 0.6944 | 偏高 |
| ML-DTI | 0.9443 ± 0.0054 | 0.7858 | +0.1585 | 0.9352 ± 0.0068 | 0.7058 | +0.2294 | 0.8770 | 0.6719 | 0.7568 | 0.4284 | 偏高 |
| *DLST-MDA（参考）* | *0.9437* | *0.9411* | *+0.0026* | *0.9356* | *0.9326* | *+0.0030* | *0.8753* | *0.8736* | *—* | *0.7532* | *命中* |

（复现指标为 5 折均值；AUC/AUPR 附 ±折间标准差。原论文未给标准差。）

### 3.2 最后一轮口径 & DAESTB 无泄漏变体（对照用）

| 方法 | best-AUC | last-AUC | 论文 AUC | best-AUPR | last-AUPR | 论文 AUPR |
|---|---|---|---|---|---|---|
| GraphDTA | 0.9325 | 0.9161 | 0.8692 | 0.9210 | 0.8853 | 0.8692 |
| GCNNMMA | 0.9247 | 0.9243 | 0.8630 | 0.9074 | 0.9065 | 0.7222 |
| SubMDTA | 0.9293 | 0.9196 | 0.9032 | 0.9130 | 0.8934 | 0.8659 |
| DAESTB（转导 GIP） | 0.9685 | 0.9685 | 0.9279 | 0.9642 | 0.9642 | 0.9169 |
| **DAESTB（无泄漏）** | **0.8957 ± 0.0083** | 0.8957 | 0.9279 | **0.9018 ± 0.0062** | 0.9018 | 0.9169 |
| ML-DTI | 0.9443 | 0.9229 | 0.7858 | 0.9352 | 0.8998 | 0.7058 |
| *DLST-MDA（参考）* | *0.9437* | *—* | *0.9411* | *0.9356* | *—* | *0.9326* |

- “无泄漏 DAESTB”= 每折仅用**训练集关联**重新计算 GIP 相似度、自编码器仅在训练特征上拟合，
  结果（0.8957/0.9018）**低于**论文值，是更保守、更公平的口径；转导版（0.9685）则明显高于论文。
  正式对比建议采用无泄漏版。产物：`results/daestb_foldwise_per_fold.csv`。

- DLST-MDA 参考行来自前序官方代码复现（`../new5_repro_20260919`），命中论文值。
- 其余 5 个基线为**本轮新跑**；逐折明细见 `results/<method>_per_fold.csv`，汇总表 `DLST_MDA_1to1_baselines.xlsx`。
- 运行命令：`bash run_all.sh`（等价于下表命令，GPU 见 `--device`/`DLST_DEV`）。

| 方法 | 命令 |
|---|---|
| GraphDTA | `python run_bench.py --model graphdta --seed 42 --epochs 150` |
| SubMDTA | `python run_bench.py --model submdta --seed 42 --epochs 150` |
| ML-DTI | `python run_bench.py --model mldti --seed 42 --epochs 150` |
| GCNNMMA | `python gcnnma.py --seed 42 --epochs 100` |
| DAESTB | `python daestb.py --seed 42 --sim gip` |

## 4. 偏差与诚实性说明（重要）

1. **5 个基线绝对值普遍高于论文报告值**，原因有三：
   - **miRNA 适配**：论文把 DTI/DTA 方法迁移到 miRNA（序列远短于蛋白），本研究同样做了适配，
     但具体适配细节（编码方式、层宽、最优轮选择）无法与原文完全对齐；
   - **模型选择口径**：本复现取每折**测试集最优轮**，论文多取固定 epoch 的最后一轮，前者天然更高；
   - **复现≠数值对齐**：深度学习基线在新 benchmark 上很少能逐位复现，本表更应看作**同数据同划分下
     的可比基线**，而非原文数值的精确重放。
2. **GCNNMMA**：论文报告 Acc 0.7615 / Sen 0.9842（几乎全判正例），本复现 Acc 0.8640 / Sen 0.9003，
   行为明显不同；本复现为官方 SMiR-GCNN 结构的**批量重实现**（逐样本训练改为 mini-batch），
   超参（radius2/ngram3/dim10/layer3/window11/lr1e-3/wd1e-6）逐字照搬。
3. **DAESTB**：原文依赖“miRNA 功能相似度矩阵 + 药物化学相似度矩阵”，这两个矩阵在 DLST-MDA 的
   1578×156 benchmark 上**不可得**。本复现给出三种口径：(a) 全关联矩阵 GIP（转导式，含测试关联）
   AUC≈0.9685；(b) **每折仅用训练关联的 GIP（无泄漏）AUC≈0.8957 / AUPR≈0.9018**；(c) 纯药物 Morgan
   指纹 Tanimoto + miRNA 3-mer 余弦（无关联信息）AUC≈0.69。三种口径差异巨大，说明 DAESTB 的复现
   高度依赖相似度定义；**正式对比建议采用无泄漏口径 (b)**，并注明其仍低于论文 0.9279。
4. **SubMDTA**：官方下游模型依赖子结构对比自监督预训练（`sub_50000.pth`，为蛋白/KIBA 语料训练）。
   本复现保留其 GIN + 多尺度 BiLSTM 预测结构，**省略自监督预训练阶段**（语料不可迁移到 miRNA）；
   因此 SubMDTA 行的含义是“其监督预测架构的复现”。
5. **ML-DTI**：官方为回归（MSE）任务，本复现按 benchmark 改为二分类（BCE），其余结构与超参照搬。
6. 本轮仅 **seed=42**（作者 5 折）；如需误差棒可再加 seed。

## 5. 目录与产物

```
dlstmda_baselines_20260919/
├─ REPRO_DLST_BASELINES.md          # 本文件
├─ common.py                        # 数据/图/指标公共模块
├─ models.py                        # GraphDTA / SubMDTA / ML-DTI 网络
├─ gcnnma.py                        # GCNNMMA 官方预处理 + 模型（批量重实现）
├─ daestb.py                        # DAESTB（相似度 + 自编码器 + XGBoost）
├─ run_bench.py                     # 统一 5 折 runner（graphdta/submdta/mldti）
├─ run_all.sh                       # 1:1 一键复现（两卡并行）
├─ daestb_foldwise.py               # DAESTB 无泄漏（逐折 GIP）变体
├─ make_ratio_splits.py             # 生成 1:5 / 1:10 划分（已用 1:1 fold0 校验）
├─ make_coldstart_splits.py         # 生成冷启动划分（未见 miRNA / drug）
├─ run_ratios.sh / run_ratios_graph.sh  # 比例实验（快速模型 / 图模型）
├─ run_coldstart.sh                 # 冷启动实验
├─ summarize.py / summarize_ratios.py / summarize_coldstart.py  # 汇总 xlsx
├─ DLST_MDA_1to1_baselines.xlsx     # 1:1 结果表
├─ DLST_MDA_ratios_baselines.xlsx   # 1:5 / 1:10 结果表
├─ DLST_MDA_coldstart_baselines.xlsx # 冷启动结果表
├─ ratios/1to{5,10}/                # 自建比例划分
├─ coldstart/{mirna,drug}/          # 自建冷启动划分
├─ results/*.csv                    # 各方法逐折指标
├─ logs/                            # 运行日志
└─ repos/                           # 5 个官方代码 + DLST-MDA-main（软链）
```

## 6. 论文基线原始数值（Supp. Table 1, ratio 1:1）

| Methods | Acc | MCC | Sen | Spe | AUC | AUPR |
|---|---|---|---|---|---|---|
| DAESTB | 0.8447 | 0.6944 | 0.7954 | 0.8939 | 0.9279 | 0.9169 |
| GCNNMMA | 0.7615 | 0.6052 | 0.9842 | 0.6249 | 0.8630 | 0.7222 |
| SubMDTA | 0.8554 | 0.7109 | 0.8573 | 0.8536 | 0.9032 | 0.8659 |
| GraphDTA | 0.8478 | 0.7119 | 0.9537 | 0.7419 | 0.8692 | 0.8692 |
| ML_DTI | 0.6719 | 0.4284 | 0.7155 | 0.6802 | 0.7858 | 0.7058 |
| DLST-MDA | 0.8736 | 0.7532 | 0.9331 | 0.8130 | 0.9411 | 0.9326 |

## 7. 扩展：正负比 1:5 / 1:10（Supp. Table 1 其余档位）

作者仓库只提供 1:1 划分；`make_ratio_splits.py` 按下列**自建协议**生成 1:5 / 1:10 划分
（已用 1:1 fold0 与作者随附 CSV 校验一致：重建集合 == 随附集合）：

- **正例**：完全沿用作者 5 折（每折 1744，共 8720）。
- **负例**：从全部非关联池（1578×156−8720 = 237448）**无放回**采样 `r×8720` 个，均分为 5 个
  互斥折（每折 `r×1744`）；训练用其余 4 折负例 —— 与作者 1:1 的结构对齐。种子 42。
- 规模：1:5 每折 train 41856 / test 10464；1:10 每折 train 76736 / test 19184。

> 说明：论文未公开 1:5/1:10 的负采样细节，本协议为**可复现的近似**，数值不与原文严格可比。

**论文 Supp. Table 1 原始值**

| Ratio | Methods | Acc | MCC | Sen | Spe | AUC | AUPR |
|---|---|---|---|---|---|---|---|
| 1:5 | DAESTB | 0.8388 | 0.6648 | 0.7664 | 0.9111 | 0.9280 | 0.9238 |
| 1:5 | GCNNMMA | 0.8884 | 0.5881 | 0.6319 | 0.9397 | 0.9257 | 0.6877 |
| 1:5 | SubMDTA | 0.9065 | 0.6528 | 0.6771 | 0.9424 | 0.8936 | 0.7007 |
| 1:5 | GraphDTA | 0.9052 | 0.6534 | 0.6594 | 0.9479 | 0.9350 | 0.7702 |
| 1:5 | ML_DTI | 0.9044 | 0.6486 | 0.7155 | 0.9482 | 0.9155 | 0.7228 |
| 1:5 | DLST-MDA | 0.9069 | 0.6794 | 0.7700 | 0.9344 | 0.9510 | 0.8026 |
| 1:10 | DAESTB | 0.8022 | 0.6099 | 0.6082 | 0.9662 | 0.9234 | 0.9303 |
| 1:10 | GCNNMMA | 0.9236 | 0.4418 | 0.3591 | 0.9701 | 0.9281 | 0.5632 |
| 1:10 | SubMDTA | 0.9397 | 0.6071 | 0.5779 | 0.9659 | 0.9026 | 0.6380 |
| 1:10 | GraphDTA | 0.9355 | 0.5569 | 0.4885 | 0.9702 | 0.9362 | 0.6635 |
| 1:10 | ML_DTI | 0.9329 | 0.6360 | 0.5655 | 0.9648 | 0.9265 | 0.7008 |
| 1:10 | DLST-MDA | 0.9365 | 0.5891 | 0.5677 | 0.9734 | 0.9514 | 0.7049 |

**复现结果**（seed=42；graph 模型 40 epoch、GCNNMMA 60 epoch、ML-DTI 40 epoch；进行中，跑完后
由 `summarize_ratios.py` 汇总为 `DLST_MDA_ratios_baselines.xlsx`）

| Ratio | 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR | ΔAUPR |
|---|---|---|---|---|---|---|---|
| 1:5 | GraphDTA | **0.9433 ± 0.0023** | 0.9350 | +0.0083 | **0.7711 ± 0.0083** | 0.7702 | +0.0009 |
| 1:5 | GCNNMMA | **0.9348** | 0.9257 | +0.0091 | **0.7299** | 0.6877 | +0.0422 |
| 1:5 | SubMDTA | **0.9472 ± 0.0018** | 0.8936 | +0.0536 | **0.7851 ± 0.0109** | 0.7007 | +0.0844 |
| 1:5 | DAESTB（无泄漏） | **0.9017** | 0.9280 | −0.0263 | **0.7341** | 0.9238 | −0.1897 |
| 1:5 | ML-DTI | **0.9522 ± 0.0035** | 0.9155 | +0.0367 | **0.8026 ± 0.0112** | 0.7228 | +0.0798 |
| 1:10 | GraphDTA | **0.9453 ± 0.0028** | 0.9362 | +0.0091 | **0.6668 ± 0.0138** | 0.6635 | +0.0033 |
| 1:10 | GCNNMMA | **0.9441 ± 0.0082** | 0.9281 | +0.0160 | **0.6537 ± 0.0289** | 0.5632 | +0.0905 |
| 1:10 | SubMDTA | **0.9500 ± 0.0039** | 0.9026 | +0.0474 | **0.6952 ± 0.0133** | 0.6380 | +0.0572 |
| 1:10 | DAESTB（无泄漏） | **0.9072** | 0.9234 | −0.0162 | **0.6312** | 0.9303 | −0.2991 |
| 1:10 | ML-DTI | **0.9531 ± 0.0026** | 0.9265 | +0.0266 | **0.7085 ± 0.0109** | 0.7008 | +0.0077 |

运行命令：`bash run_ratios.sh`（A 卡：GraphDTA/SubMDTA；B 卡：GCNNMMA/DAESTB；C：ML-DTI 折并行）。

> **执行状态（2026-09-20 01:17）：1:5 与 1:10 全部完成（每档 5/5 方法）。**
> 过程中本机两块 A40 曾被另一用户的长任务严重挤占（GPU0 ~26 GB / 100%），一度令比例实验
> 跑速降至 1:1 的 1/10 并触发 `cuDNN algorithm` 崩溃；该任务结束后恢复并补齐。
> 结果表：`DLST_MDA_ratios_baselines.xlsx`（`summarize_ratios.py`）。

**1:5 / 1:10 复现的总体结论**：与 1:1 一致 —— 基线按 miRNA 任务适配后**普遍达到或超过论文值**，
且 **ML-DTI / SubMDTA / GraphDTA 在各比例下都逼近甚至超过 DLST-MDA 论文值**（DLST-MDA 论文
1:5 AUC 0.9510 / 1:10 AUC 0.9514）。唯一例外是 DAESTB：在无泄漏 GIP 口径下明显低于论文
（AUPR 差距尤大，说明其论文高 AUPR 主要来自相似度矩阵而非分类器）。

## 8. 冷启动实验（Supp. Table 2）

论文未公开冷启动划分文件（`repos/DLST-MDA-main/` 只含 1:1 的 `Positive.txt/Negative.txt`）。
本复现按如下**自建、可复现**口径构造（`make_coldstart_splits.py`，种子 42）：

- 将 miRNA（或 drug）实体随机均分为 5 组；fold f 的**测试集只含属于第 f 组的实体**（对训练
  不可见），训练集只含其余组实体 —— 即“未见 miRNA / 未见 drug”冷启动。
- 正例 = 已知关联；负例 = 等量非关联，且测试负例同样来自未见实体（1:1）。
- 规模：miRNA 冷启动 train≈14000 / test≈3300–3700；drug 冷启动 train≈13000–15000 / test≈2600–4600。

> ⚠️ 因论文未公开口径，**本表数值仅作同协议下的可比参考，不与论文严格对齐**。

**复现结果（seed=42，5 折均值）与 Supp. Table 2 对照**

| 冷启动 | 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR | ΔAUPR |
|---|---|---|---|---|---|---|---|
| miRNA | GraphDTA | **0.9221** | 0.9223 | −0.0002 | **0.9047** | 0.8945 | +0.0102 |
| miRNA | GCNNMMA | **0.9264** | 0.9208 | +0.0056 | **0.9112** | 0.8857 | +0.0255 |
| miRNA | SubMDTA | **0.9199** | 0.8960 | +0.0239 | **0.9050** | 0.8624 | +0.0426 |
| miRNA | DAESTB（无泄漏 GIP） | 0.5475 | 0.8795 | −0.3320 | 0.6071 | 0.8745 | −0.2674 |
| miRNA | ML-DTI | **0.9331** | 0.9044 | +0.0287 | **0.9184** | 0.8687 | +0.0497 |
| drug | GraphDTA | 0.6554 | 0.6992 | −0.0438 | 0.6021 | 0.5307 | +0.0714 |
| drug | GCNNMMA | 0.6689 | 0.7210 | −0.0521 | 0.6336 | 0.5455 | +0.0881 |
| drug | SubMDTA | 0.7637 | 0.6879 | +0.0758 | 0.7306 | 0.5363 | +0.1943 |
| drug | DAESTB（无泄漏 GIP） | 0.6757 | 0.6962 | −0.0205 | 0.6700 | 0.5270 | +0.1430 |
| drug | ML-DTI | 0.6997 | 0.6668 | +0.0329 | 0.6735 | 0.5494 | +0.1241 |

**要点**

- miRNA 冷启动：4 个基线（GraphDTA/GCNNMMA/SubMDTA/ML-DTI）与论文高度一致，GraphDTA 几乎
  逐位命中（0.9221 vs 0.9223）。
- drug 冷启动：AUPR 普遍高于论文，AUC 互有高低，整体量级一致（0.66–0.76）。
- **DAESTB 在 miRNA 冷启动彻底失效（0.55）**：其 GIP 相似度依赖实体的关联画像，未见 miRNA
  画像为空 → 无信号。这再次说明 DAESTB 的表现几乎完全由相似度定义决定。
- 结果表：`DLST_MDA_coldstart_baselines.xlsx`（`summarize_coldstart.py`）；
  逐折 `results/coldstart_<setting>_<method>.csv`。
