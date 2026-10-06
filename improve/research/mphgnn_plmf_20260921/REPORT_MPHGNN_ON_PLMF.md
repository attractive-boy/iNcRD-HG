# MPHGNN 放到 PLMF-MDA 的 MDR/MDS 数据集上：能跑到多少

目录：`improve/research/mphgnn_plmf_20260921/`

> **MPHGNN**：Huang G, Peng Y, Wu S, Wei H, Liu S. *MPHGNN: metapath-guided heterogeneous graph neural network for miRNA-drug resistance association prediction.* **Briefings in Bioinformatics 2026**, bbag013. 代码/数据 <https://github.com/Gtaizu/MPHGNN>。原文在自建独立测试集上 AUC 0.978 / AUPR 0.937（本机已用官方代码复现到 0.9743/0.9382，见 `../mirna_sota_20260915/`）。
>
> **PLMF-MDA**：Sheng N 等, *PLoS Comput Biol* 2026, e1013968；MDRdataset 1317 miRNA × 105 drug × 5411 耐药关联，MDSdataset 1252 × 140 × 5054 敏感关联；只提供 miRNA 序列 + drug SMILES + 关联矩阵。

## 0. 一句话结论

| 数据集 | 协议 | MPHGNN 同基线口径（测试集最优轮）AUC / AUPR | MPHGNN 无泄漏口径（验证选轮 + 逐对打分）AUC / AUPR | MPHGNN 末轮(50) AUC / AUPR | PLMF-MDA 复现 | ML-DTI 复现 | 本文最强模型 |
|---|---|---|---|---|---|---|---|
| MDR | 论文 5 折 | **0.9237 / 0.9136** | 0.9205 / 0.9092 | 0.9131 / 0.9032 | 0.9243 / 0.9146 | 0.9238 / 0.9131 | 0.9486 / 0.9433 |
| MDR | 官方 80/20 | **0.9383 / 0.9339** | 0.9248 / 0.9125 | 0.9116 / 0.9038 | 0.9287 / 0.9162 | 0.9281 / 0.9171 | 0.9514 / 0.9458 |
| MDS | 论文 5 折 | **0.9279 / 0.9215** | 0.9246 / 0.9168 | 0.9148 / 0.9107 | 0.9270 / 0.9175 | 0.9296 / 0.9226 | 0.9513 / 0.9480 |
| MDS | 官方 80/20 | **0.9381 / 0.9369** | 0.9283 / 0.9192 | 0.9169 / 0.9084 | 0.9294 / 0.9189 | 0.9303 / 0.9208 | 0.9522 / 0.9486 |

（"测试集最优轮"与 `../plmf_baselines_20260920` 各基线的主口径一致，偏乐观；"无泄漏口径"= 从训练对再切 10% 做早停选轮 + 测试对逐对打分（MPHGNN 官方全局分支会把同一 batch 里的查询对连成图，整批打分时测试集的度结构会泄漏进预测，见 §4）；"末轮"是固定 50 轮。）

## 1. 迁移了什么

MPHGNN 需要 miRNA–gene–drug 异质网络 + 节点属性；PLMF-MDA 数据里一样都没有，全部按 MPHGNN 原文的同一来源重新构建（DrugBank 本机 403，改用 ChEMBL 机制靶点）。

| MPHGNN 原文 | 本次 PLMF-MDA 节点上的构建 | MDR | MDS |
|---|---|---|---|
| miRNA–gene：ENCORI 827 条 | ENCORI miRNATarget API（hg38, mRNA）逐 miRNA 抓取 1620 个 miRNA、154 万行；保留 clipExpNum≥5 且 ≥3 个预测程序支持，每 miRNA 取 CLIP 支持最高的 50 个基因 | 15485 条，453/1317 miRNA 有靶基因 | 15726 条，449/1252 |
| gene–drug：DrugBank 579 条 | SMILES→InChIKey→ChEMBL 分子（159/164 命中）→机制靶点（含盐型子分子）→UniProt→Entrez；115/164 药有靶点 | 436 条，79/105 药 | 584 条，96/140 药 |
| gene–gene：HumanNet 5069 条 | HumanNet v3 FN（97.7 万条 LLS 边），限制在上面两类边覆盖的基因集合内，每基因取 LLS top-10 并对称化 | 3057 基因，40190 条 | 3101 基因，40818 条 |
| miRNA 属性：CCLE 938 细胞系表达 | 主变体 **kmer**：2+3-mer 频率 80 维（所有节点可用）；对照变体 **expr**：CCLE 表达（仅 552/1317、565/1252 个 miRNA 有，其余零向量+缺失标记） | | |
| drug 属性：ChemBERTa 768 维行内 min-max | ChemBERTa-zinc-base-v1 mean-pool 768 维 + 行内 min-max（已核对：MPHGNN 官方 `drug.x` 就是 incrd-hg 的 `drug_features-chemBertA.csv` 行内 min-max，逐位相等） | | |
| gene 属性：gene similarity 200 维 | HumanNet 对称归一化邻接（含自环）的 200 维截断 SVD | | |

模型代码是官方 `model.py` / `layer.py` 原样（`mphgnn_model.py` 只把写死的 605-miRNA 偏移改成参数）；超参照官方：hidden 128、3 层 HeteroGCN（sum 聚合 + PReLU）、层注意力、BiTrans 元路径聚合、MIL 注意力包预测 + top-3 实例预测 + 全局 GCN-GAE 边解码三路平均、k=20（每对 441 条元路径实例）、batch 256、Adam lr 1e-4、pos_weight BCE、无 dropout/BN。

## 2. 协议（与 PLMF-MDA 基线复现逐位相同）

- 划分：`plmf_splits.py` 复制 `../plmf_baselines_20260920/plmf_data.py` 的两种协议并断言逐对相等：**5cv** = 论文 5 折（正例 + 等量随机负例，seed 42），**single** = 官方代码 80/20（seed 0–4）。
- 图：每个 split 的消息传递 miRNA–drug / drug–miRNA 边**只用训练正例**；测试推理用同一张图（不像 MPHGNN 官方 `load_test_data` 那样把测试正边也放进图里）。基因边与标签无关，全图共享。
- 训练 50 轮；每轮在测试集上算 AUC/AUPR/ACC 等。报三种口径：(a) **测试集最优轮**（本项目基线统一口径，偏乐观）；(b) **验证集选轮**：从训练对再切 10%（分层）做验证，取验证 AUC 最高轮的测试指标（无泄漏）；(c) **末轮**。另存训练损失最小轮（官方 `check_metric=loss` 口径，实际就是末轮附近）。
- 消融：`gene-none` 去掉全部基因边（无元路径可走，实例退化为随机基因自环填充）；`gene-shuffled` 随机置换基因 id（边数不变、生物学对应被打乱）。

## 3. 完整榜单（测试集最优轮口径，与 `../plmf_strongest_20260920` 榜单同口径）

### MDRdataset · 论文 5 折 (seed42, 5 折均值±折间SD)

| 排名 | 方法 | AUC | ± | AUPR | ACC | 末轮 AUC / AUPR | 论文值 AUC / AUPR |
|---|---|---|---|---|---|---|---|
| 1 | **iNcRD-HG-strongest (+anchor)** | 0.9486 | 0.0042 | 0.9433 | 0.8801 | - / - | - / - |
| 2 | **MPHGNN (k-mer + CCLE expr)** | 0.9248 | 0.0041 | 0.9162 | 0.8395 | 0.9098 / 0.8992 | - / - |
| 3 | PLMF-MDA | 0.9243 | 0.0039 | 0.9146 | 0.8485 | 0.9208 / 0.9085 | 0.9227 / 0.9076 |
| 4 | ML-DTI | 0.9238 | 0.0053 | 0.9131 | 0.8434 | 0.8976 / 0.8709 | 0.7660 / 0.7870 |
| 5 | **MPHGNN w/o gene graph (no metapath, bipartite only)** | 0.9237 | 0.0035 | 0.9136 | 0.8456 | 0.9159 / 0.9063 | - / - |
| 6 | **MPHGNN (k-mer miRNA feat, ENCORI/ChEMBL/HumanNet gene graph)** | 0.9237 | 0.0035 | 0.9136 | 0.8447 | 0.9131 / 0.9032 | - / - |
| 7 | **MPHGNN, gene ids permuted (shuffle control)** | 0.9237 | 0.0035 | 0.9135 | 0.8448 | 0.9146 / 0.9048 | - / - |
| 8 | **MPHGNN, 10% val hold-out + isolated-pair test scoring (test-best column = batch scoring)** | 0.9235 | 0.0043 | 0.9132 | 0.8467 | 0.9102 / 0.8991 | - / - |
| 9 | **MPHGNN, 10% of training pairs held out for early stopping (test-best column still shown)** | 0.9235 | 0.0043 | 0.9132 | 0.8467 | 0.9102 / 0.8991 | - / - |
| 10 | **MPHGNN (CCLE expr miRNA feat; 42-45% miRNAs covered)** | 0.9227 | 0.0049 | 0.9132 | 0.8464 | 0.9099 / 0.8990 | - / - |
| 11 | **MPHGNN, global branch on fixed training graph (official uses batch-pair graph)** | 0.9174 | 0.0065 | 0.9123 | 0.8288 | 0.8550 / 0.8724 | - / - |
| 12 | GCNNMMA | 0.9118 | 0.0036 | 0.8949 | 0.8392 | 0.9116 / 0.8949 | 0.9070 / 0.8736 |
| 13 | GraphDTA | 0.9054 | 0.0029 | 0.8907 | 0.8279 | 0.8898 / 0.8599 | 0.8086 / 0.7979 |
| 14 | SubMDTA | 0.9044 | 0.0039 | 0.8907 | 0.8309 | 0.8972 / 0.8703 | 0.8936 / 0.8514 |

### MDRdataset · 官方 80/20 (seed 0-4 均值±SD)

| 排名 | 方法 | AUC | ± | AUPR | ACC | 末轮 AUC / AUPR | 论文值 AUC / AUPR |
|---|---|---|---|---|---|---|---|
| 1 | **iNcRD-HG-strongest (+anchor)** | 0.9514 | 0.0051 | 0.9458 | 0.8800 | - / - | - / - |
| 2 | **MPHGNN w/o gene graph (no metapath, bipartite only)** | 0.9384 | 0.0064 | 0.9340 | 0.8575 | 0.9147 / 0.9064 | - / - |
| 3 | **MPHGNN (k-mer miRNA feat, ENCORI/ChEMBL/HumanNet gene graph)** | 0.9383 | 0.0065 | 0.9339 | 0.8590 | 0.9116 / 0.9038 | - / - |
| 4 | **MPHGNN, gene ids permuted (shuffle control)** | 0.9383 | 0.0066 | 0.9339 | 0.8593 | 0.9132 / 0.9051 | - / - |
| 5 | **MPHGNN, 10% val hold-out + isolated-pair test scoring (test-best column = batch scoring)** | 0.9372 | 0.0073 | 0.9327 | 0.8556 | 0.9109 / 0.9018 | - / - |
| 6 | **MPHGNN, 10% of training pairs held out for early stopping (test-best column still shown)** | 0.9372 | 0.0073 | 0.9327 | 0.8556 | 0.9109 / 0.9019 | - / - |
| 7 | **MPHGNN (CCLE expr miRNA feat; 42-45% miRNAs covered)** | 0.9318 | 0.0056 | 0.9245 | 0.8559 | 0.9102 / 0.9017 | - / - |
| 8 | **MPHGNN (k-mer + CCLE expr)** | 0.9311 | 0.0068 | 0.9243 | 0.8506 | 0.9102 / 0.9008 | - / - |
| 9 | PLMF-MDA | 0.9287 | 0.0060 | 0.9162 | 0.8568 | 0.9222 / 0.9076 | 0.9227 / 0.9076 |
| 10 | ML-DTI | 0.9281 | 0.0049 | 0.9171 | 0.8538 | 0.9029 / 0.8810 | 0.7660 / 0.7870 |
| 11 | **MPHGNN, global branch on fixed training graph (official uses batch-pair graph)** | 0.9184 | 0.0073 | 0.9152 | 0.8331 | 0.8504 / 0.8739 | - / - |
| 12 | GCNNMMA | 0.9179 | 0.0061 | 0.9019 | 0.8433 | 0.9175 / 0.9017 | 0.9070 / 0.8736 |
| 13 | GraphDTA | 0.9118 | 0.0066 | 0.9006 | 0.8399 | 0.8924 / 0.8623 | 0.8086 / 0.7979 |
| 14 | SubMDTA | 0.9068 | 0.0081 | 0.8902 | 0.8306 | 0.8996 / 0.8747 | 0.8936 / 0.8514 |

### MDSdataset · 论文 5 折 (seed42, 5 折均值±折间SD)

| 排名 | 方法 | AUC | ± | AUPR | ACC | 末轮 AUC / AUPR | 论文值 AUC / AUPR |
|---|---|---|---|---|---|---|---|
| 1 | **iNcRD-HG-strongest (+anchor)** | 0.9513 | 0.0060 | 0.9480 | 0.8845 | - / - | - / - |
| 2 | ML-DTI | 0.9296 | 0.0046 | 0.9226 | 0.8574 | 0.9067 / 0.8922 | 0.7820 / 0.7881 |
| 3 | **MPHGNN (k-mer + CCLE expr)** | 0.9292 | 0.0065 | 0.9230 | 0.8553 | 0.9144 / 0.9113 | - / - |
| 4 | **MPHGNN (k-mer miRNA feat, ENCORI/ChEMBL/HumanNet gene graph)** | 0.9279 | 0.0080 | 0.9215 | 0.8009 | 0.9148 / 0.9107 | - / - |
| 5 | **MPHGNN, gene ids permuted (shuffle control)** | 0.9278 | 0.0080 | 0.9214 | 0.8005 | 0.9158 / 0.9112 | - / - |
| 6 | **MPHGNN w/o gene graph (no metapath, bipartite only)** | 0.9278 | 0.0081 | 0.9214 | 0.7992 | 0.9205 / 0.9161 | - / - |
| 7 | PLMF-MDA | 0.9270 | 0.0065 | 0.9175 | 0.8616 | 0.9234 / 0.9132 | 0.9301 / 0.9207 |
| 8 | **MPHGNN, 10% of training pairs held out for early stopping (test-best column still shown)** | 0.9269 | 0.0086 | 0.9210 | 0.7097 | 0.9149 / 0.9109 | - / - |
| 9 | **MPHGNN, 10% val hold-out + isolated-pair test scoring (test-best column = batch scoring)** | 0.9269 | 0.0086 | 0.9210 | 0.7097 | 0.9149 / 0.9109 | - / - |
| 10 | **MPHGNN (CCLE expr miRNA feat; 42-45% miRNAs covered)** | 0.9268 | 0.0066 | 0.9228 | 0.8524 | 0.9145 / 0.9113 | - / - |
| 11 | **MPHGNN, global branch on fixed training graph (official uses batch-pair graph)** | 0.9193 | 0.0071 | 0.9200 | 0.8360 | 0.8474 / 0.8781 | - / - |
| 12 | GCNNMMA | 0.9133 | 0.0048 | 0.9007 | 0.8513 | 0.9130 / 0.9004 | 0.9143 / 0.8889 |
| 13 | GraphDTA | 0.9125 | 0.0059 | 0.9036 | 0.8380 | 0.8994 / 0.8750 | 0.7864 / 0.7911 |
| 14 | SubMDTA | 0.9059 | 0.0050 | 0.8923 | 0.8216 | 0.8971 / 0.8721 | 0.8858 / 0.8587 |

### MDSdataset · 官方 80/20 (seed 0-4 均值±SD)

| 排名 | 方法 | AUC | ± | AUPR | ACC | 末轮 AUC / AUPR | 论文值 AUC / AUPR |
|---|---|---|---|---|---|---|---|
| 1 | **iNcRD-HG-strongest (+anchor)** | 0.9522 | 0.0030 | 0.9486 | 0.8854 | - / - | - / - |
| 2 | **MPHGNN w/o gene graph (no metapath, bipartite only)** | 0.9383 | 0.0037 | 0.9371 | 0.8613 | 0.9205 / 0.9147 | - / - |
| 3 | **MPHGNN (k-mer miRNA feat, ENCORI/ChEMBL/HumanNet gene graph)** | 0.9381 | 0.0035 | 0.9369 | 0.8622 | 0.9169 / 0.9084 | - / - |
| 4 | **MPHGNN, gene ids permuted (shuffle control)** | 0.9381 | 0.0035 | 0.9369 | 0.8624 | 0.9175 / 0.9092 | - / - |
| 5 | **MPHGNN, 10% val hold-out + isolated-pair test scoring (test-best column = batch scoring)** | 0.9376 | 0.0030 | 0.9362 | 0.8371 | 0.9159 / 0.9083 | - / - |
| 6 | **MPHGNN, 10% of training pairs held out for early stopping (test-best column still shown)** | 0.9376 | 0.0030 | 0.9362 | 0.8371 | 0.9159 / 0.9084 | - / - |
| 7 | **MPHGNN (CCLE expr miRNA feat; 42-45% miRNAs covered)** | 0.9353 | 0.0028 | 0.9300 | 0.8609 | 0.9174 / 0.9087 | - / - |
| 8 | **MPHGNN (k-mer + CCLE expr)** | 0.9347 | 0.0013 | 0.9302 | 0.8600 | 0.9167 / 0.9090 | - / - |
| 9 | ML-DTI | 0.9303 | 0.0041 | 0.9208 | 0.8641 | 0.9123 / 0.8913 | 0.7820 / 0.7881 |
| 10 | PLMF-MDA | 0.9294 | 0.0026 | 0.9189 | 0.8665 | 0.9264 / 0.9159 | 0.9301 / 0.9207 |
| 11 | **MPHGNN, global branch on fixed training graph (official uses batch-pair graph)** | 0.9244 | 0.0035 | 0.9237 | 0.8407 | 0.8569 / 0.8834 | - / - |
| 12 | GCNNMMA | 0.9167 | 0.0042 | 0.9028 | 0.8570 | 0.9159 / 0.9014 | 0.9143 / 0.8889 |
| 13 | GraphDTA | 0.9145 | 0.0051 | 0.9012 | 0.8410 | 0.8978 / 0.8728 | 0.7864 / 0.7911 |
| 14 | SubMDTA | 0.9075 | 0.0069 | 0.8908 | 0.8345 | 0.9003 / 0.8790 | 0.8858 / 0.8587 |

## 4. 无泄漏口径：验证集选轮 × 逐对打分（主变体 kmer/real）

MPHGNN 官方 `model.forward` 的全局分支（GCN-GAE）用**当前 batch 里的 (miRNA, drug) 查询对**作为 GCN 的 edge_index。测试时整批打分，测试集自身的度结构（正例集中在高频 miRNA/药物、负例在负例池均匀采样）就成了可读的信号：只用"测试 batch 内 miRNA 度 × drug 度"这一个启发式，1024 对一批时 AUC 就有 0.83–0.89，整批更高；逐对打分（batch=1）时该启发式为 0.5。下面同一组训练好的模型分别用两种方式打分：

仅用"打分 batch 内 miRNA 度 × drug 度"这一启发式在测试集上的 AUC（越大说明整批打分时可读的测试集结构越多）：

| 数据集 | 协议 | batch=1 | 8 | 32 | 128 | 256 | 1024（本 runner） | 整个测试集（官方） | 参考：训练正例图的度 |
|---|---|---|---|---|---|---|---|---|---|
| MDR | 5cv | 0.500 | 0.566 | 0.686 | 0.799 | 0.825 | 0.831 | 0.879 | 0.850 |
| MDR | single | 0.500 | 0.615 | 0.760 | 0.842 | 0.862 | 0.873 | 0.885 | 0.854 |
| MDS | 5cv | 0.500 | 0.568 | 0.687 | 0.808 | 0.846 | 0.881 | 0.893 | 0.847 |
| MDS | single | 0.500 | 0.613 | 0.766 | 0.854 | 0.873 | 0.890 | 0.897 | 0.850 |

把全局分支的 edge_index 换成固定的训练正例图（`--global_graph train`，无泄漏的"正常" GAE 做法）反而更差：测试最优轮只有约 0.917（MDR/MDS 80/20 与 5 折都在 0.916–0.922），且 50 轮后掉到 0.85——官方设计事实上依赖 batch 图。下面是同一组模型分别用整批 / 逐对两种方式打分：

| 数据集 | 协议 | 验证选中轮(均值) | 验证选轮·整批打分 AUC / AUPR | **验证选轮·逐对打分 AUC / AUPR** | 测试最优轮·整批 | 测试最优轮·逐对 | 末轮·整批 | 末轮·逐对 | PLMF-MDA 复现 |
|---|---|---|---|---|---|---|---|---|---|
| MDR | 5cv | 4.0 | 0.9217 / 0.9118 | **0.9205 / 0.9092** | 0.9235 | 0.9220 | 0.9102 | 0.9110 | 0.9243 / 0.9146 |
| MDR | single | 3.6 | 0.9353 / 0.9301 | **0.9248 / 0.9125** | 0.9372 | 0.9259 | 0.9109 | 0.9159 | 0.9287 / 0.9162 |
| MDS | 5cv | 2.4 | 0.9260 / 0.9201 | **0.9246 / 0.9168** | 0.9269 | 0.9253 | 0.9149 | 0.9154 | 0.9270 / 0.9175 |
| MDS | single | 4.2 | 0.9346 / 0.9311 | **0.9283 / 0.9192** | 0.9376 | 0.9295 | 0.9159 | 0.9190 | 0.9294 / 0.9189 |

说明：这组运行的训练对比主运行少 10%（切给了验证），所以"测试最优轮·整批"与 §3 主运行的数字略有差异。PLMF-MDA 与 4 个基线都是逐对打分的模型，因此**与它们同口径可比的是"验证选轮·逐对打分"这一列**。

## 5. 基因图 / 元路径到底有没有用（消融，测试最优轮口径）

| 数据集 | 协议 | kmer + 真实基因图 | kmer + 基因 id 置换 | kmer + 无基因边 | CCLE 表达 + 真实基因图 | kmer+表达 + 真实基因图 | 全局分支改用固定训练图 |
|---|---|---|---|---|---|---|---|
| MDR | 5cv | 0.9237 / 0.9136 | 0.9237 / 0.9135 | 0.9237 / 0.9136 | 0.9227 / 0.9132 | 0.9248 / 0.9162 | 0.9174 / 0.9123 |
| MDR | single | 0.9383 / 0.9339 | 0.9383 / 0.9339 | 0.9384 / 0.9340 | 0.9318 / 0.9245 | 0.9311 / 0.9243 | 0.9184 / 0.9152 |
| MDS | 5cv | 0.9279 / 0.9215 | 0.9278 / 0.9214 | 0.9278 / 0.9214 | 0.9268 / 0.9228 | 0.9292 / 0.9230 | 0.9193 / 0.9200 |
| MDS | single | 0.9381 / 0.9369 | 0.9381 / 0.9369 | 0.9383 / 0.9371 | 0.9353 / 0.9300 | 0.9347 / 0.9302 | 0.9244 / 0.9237 |

## 6. 产物与复现

```
mphgnn_plmf_20260921/
├─ fetch_encori.py          # ENCORI miRNA-target 抓取 → cache/encori_targets.tsv.gz
├─ build_drug_targets.py    # ChEMBL 机制靶点 → cache/drug_targets.json
├─ build_gene_ids.py        # HumanNet Entrez↔symbol → cache/entrez_symbol.json
├─ build_graph.py           # 异质图 + 节点特征 → cache/graph_{MDR,MDS}.npz
├─ plmf_splits.py           # 与 plmf_baselines 逐位相同的划分
├─ mphgnn_model.py / mphgnn_layer.py   # 官方模型（605 偏移参数化）
├─ run_mphgnn_plmf.py       # 主 runner；run_mphgnn_plmf_val.py 为 10% 验证早停版
├─ run_all.sh / run_val_after.sh / summarize.py / build_excel.py / write_report.py
├─ MPHGNN_on_PLMF-MDA.xlsx  # 说明 / 4 个榜单 / 逐次运行
├─ results/*.csv, *_curves.json  # 逐折/逐 seed 指标与每轮曲线
└─ logs/
```
外部数据：`/data1/licheng/incrd-hg/data_external/humannet_20260921/`（HumanNet-FN，含 SOURCE.txt 与 sha256）。ChemBERTa 权重经 hf-mirror.com 下载到 `.external/models/hf_home`。

```bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python; V=/data1/licheng/incrd-hg/.venv-dgl/bin/python
$PY fetch_encori.py && $PY build_drug_targets.py && $PY build_gene_ids.py && $V build_drug_chemberta.py && $PY build_graph.py
./run_all.sh && ./run_val_after.sh && $V summarize.py && $V build_excel.py && $V write_report.py
```
