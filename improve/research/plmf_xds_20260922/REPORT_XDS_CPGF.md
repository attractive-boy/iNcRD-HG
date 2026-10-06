# PLMF-MDA 提分/创新点第二轮：跨表型迁移（MDR↔MDS）+ 内容条件化可学习多项式图滤波（CPGF）

日期 2026-09-22 · 目录 `improve/research/plmf_xds_20260922/`
承接 `plmf_frontier_20260922/REPORT_PLMF_FRONTIER.md`（前沿扫描 + 冷启动试点）与 `PROPOSAL_NOVEL_TECH.md`。
本轮做了三件事：① 补一轮针对性 arXiv 检索（多行为推荐 / 旁信息矩阵补全 / 可学习滤波）；
② 把"另一表型的标签矩阵"（预测耐药时用敏感矩阵，反之亦然）当旁信息实测；③ 把报告里一直推荐但没实现的
"HL-GNN 式可学习启发式 + 内容增强异质邻接"做成原型 CPGF，在 warm 与冷药两面实测。

## 0. 结论

| 问题 | 答案 |
|---|---|
| warm 5cv 的 AUC/AUPR 还能不能明显提高？ | **不能。** 今天又试了 3 条新杠杆（跨表型旁信息、可学习多项式滤波、二者与 ctx+diff 堆叠），最大增量 +0.002 AUC / +0.0027 AUPR，且打乱对照能拿回其中一半。加上昨天的 knn/aa/mech/集成（`warm_boost.csv`，≤ +0.0005），warm 面天花板 0.951/0.954 已被 5 条独立路线反复确认。 |
| 跨表型迁移是不是新杠杆？ | **信号存在但被流行度吞掉。** 共享 949 miRNA / 81 药；P(MDR 正 \| MDS 正)=0.34 vs 基率 0.06（5.7×）。可是 `x_pair` 单列在 warm 上不加分（0.9516 vs 0.9510），增益全部来自"另一任务里的度数"（xdsd 0.9532）。冷药：MDR +0.014（打乱对照 +0.004），MDS −0.010。→ 可写成分析节，不能当主创新点。 |
| CPGF（可学习滤波）表现如何？ | 单模型、**12 个参数、线性可解释、warm/cold 统一**：warm MDR 0.940/0.934、MDS 0.940/0.939（高于 PLMF-MDA 0.924–0.929、MPHGNN 0.921–0.928，低于 ctx+diff×LR 0.951/0.954）；冷药 MDR 0.795/0.807（PLMF-MDA 0.65；ctx+knn×LR 0.828）。与 LR 栈堆叠增益 = 0。**它能替代 PLMF-MDA/MPHGNN 一类端到端模型，但打不过仓库自己的特征栈。** |
| 那么论文怎么写？ | 维持上一轮判断：主线 = **归纳式（冷药）MDA 评测 + 度偏差诊断 + 一个统一 warm/cold 的简洁模型**。CPGF 可以做"方法"一节（有可解释系数、天然归纳、k-shot 免重训），但主指标要靠 ctx+knn×LR 栈或把 kNN 标签迁移并入 CPGF；不要再指望 warm 总分。 |

## 1. 增量文献（2026-06 → 2026-09，`lit/delta_results.json`）

新角度：把 MDR/MDS 视为**多行为推荐**（辅助行为 = 敏感，目标行为 = 耐药）。2026 年该方向密集：
Environment-Conditioned MBR（2608.22920：辅助信号缺失/不可靠的两个根本问题）、RMBRec（2601.08705：辅助行为与目标行为语义错位的鲁棒性）、
MCLMR（2603.25126：辅助行为混杂的因果框架）、Dynamic Spectral Denoising for MBR（2606.02417：谱去噪）、Multi-Behavior Multi-Task（2601.07294）。
它们讨论的"辅助行为噪声、错位、被流行度混杂"正是本轮实测看到的现象（§2）。
旁信息矩阵补全：BVSIMC（2603.18957，药物发现里的贝叶斯变量选择归纳矩阵补全）、Sample-efficient IMC with inexact side-information（2605.17189）。
可学习传播算子：Dynamic Spectral Filtering（2607.27891）。ncRNA–药物 2026 新文无增量。

## 2. 跨表型旁信息（`xds_probe.py` / `cold_xds.py`，results/*.csv）

特征 `xds`：另一任务矩阵按 miRNA 名 / DrugBank ID 对齐到当前索引，列 = x_pair、x_deg_r、x_deg_d、x_shared_*、另一矩阵上的 PPR/Katz/path7 及内容平滑（26 列）。
当前任务特征仍严格 support-only、交叉拟合（与 plmf_innov 逐位一致）。

**warm 5cv（seed 42，LR C=0.3）**

| 特征 | MDR AUC / AUPR | MDS AUC / AUPR |
|---|---|---|
| ctx+diff（基线） | 0.9510 / 0.9455 | 0.9540 / 0.9516 |
| ctx+diff+xds | 0.9529 / 0.9471 | 0.9563 / 0.9543 |
| ctx+diff+xds，另一矩阵行列打乱 | 0.9517 / 0.9461 | 0.9546 / 0.9519 |
| ctx+diff+x_pair 单列 | 0.9516 / 0.9458 | 0.9543 / 0.9520 |
| ctx+diff+另一任务度数 | **0.9532 / 0.9472** | **0.9566 / 0.9548** |
| 只用 xds（26 列） | 0.9180 / 0.9096 | 0.9292 / 0.9268 |

读数：pair 身份不加分（正例中 28% 是 MDS 正例，但这些 pair 本来就是高分的热门 pair）；只有"另一任务的度数"给 +0.002，其中一半是打乱也能得到的（列数/正则效应）。

**冷药（PLMF Table-2 协议，度 11–20，5 seeds，fit=warm）**：10 个 MDR 测试药在 MDS 中全部存在（MDS 中位度 14）。

| 特征 | MDR AUC / AUPR / macro | MDS AUC / AUPR / macro |
|---|---|---|
| ctx+knn（基线） | 0.8280 / 0.8175 / 0.839 | 0.7768 / 0.7811 / 0.796 |
| +xds 全部 | 0.8395 / 0.8230 / 0.852 | 0.7672 / 0.7740 / 0.797 |
| +xds 打乱 | 0.8322 / 0.8207 / 0.847 | 0.7678 / 0.7773 / 0.792 |
| +另一任务度数 | 0.8419 / 0.8261 / 0.854 | 0.7677 / 0.7748 / 0.798 |
| x_pair 单列无训练 | 0.549 | 0.542 |
| x_ppr 单列无训练 | 0.655 | 0.630 |

→ 跨表型标签本身对冷药几乎无判别力（0.55），可用的仍是 miRNA 在另一任务里的流行度。MDR/MDS 方向不一致，不能作为可靠杠杆。

## 3. CPGF：内容条件化可学习多项式图滤波（`polyfilt.py`，results/polyfilt.csv）

模型：Â = [[γ_r S_r, A],[Aᵀ, γ_d S_d]]，A = support 正例，S = 内容 kNN 核；度保持非对称归一化 A_n = D^{-p} Â D^{-(1-p)}；
score(r,d) = Σ_k β_k z([A_n^k]_{rd}) + w·[log deg_r, log deg_d, log cdeg_r, log cdeg_d] + b。
可学参数：β(K)、γ_r、γ_d、p、w、b，BCE 端到端，全批训练；训练用与 plmf_innov 相同的 5 折交叉拟合 support（`--fit warm`），
或按药物分组的 support（`--fit drugcv`，查询药在 support 中无边 = 模拟冷药），或二者并集（`--fit both`）。GPU 上每折 12–45 s。

| 设定 | MDR | MDS | 备注 |
|---|---|---|---|
| warm 5cv，K=8，fit=warm | 0.9400 / 0.9340 | 0.9402 / 0.9389 | 学到 p≈0.23–0.28（强非对称归一化），γ_d≫γ_r（药物内容边被放大），β 集中在 k=3（A Aᵀ A，路径-3） |
| warm，无内容边（HL-GNN 原版） | 0.9353 / 0.9324 | – | 内容边 +0.005 |
| warm，CPGF 列 + ctx+diff 堆叠 LR | 0.9515 / 0.9462 | 0.9545 / 0.9523 | 与 ctx+diff×LR 0.9510/0.9540 持平 → warm 无增量 |
| 冷药，fit=warm | 0.5961 / 0.6563 | 0.6973 / 0.7046 | 训练时所有查询药都有边，模型依赖 deg_d → 冷药崩溃 |
| 冷药，fit=drugcv（模拟冷药） | 0.7517 / 0.7637 | 0.7336 / 0.7384 | 权重转到 log cdeg_d 与内容路径，回升 +0.15 |
| 冷药，fit=both | 0.7279 / 0.7253 | – | 混合训练更差 |
| 冷药，fit=drugcv，S_d 打乱 | 0.5112 | – | 内容边是冷药唯一信息（对照通过） |
| 冷药，fit=drugcv，p 固定 0.5（对称归一化） | 0.7056 | – | 可学 p 有用 |
| 冷药，fit=drugcv，**稠密内容图 weighted k=40** | **0.7951 / 0.8070 / macro 0.814** | – | 内容图密度是冷药瓶颈 |
| 冷药，fit=drugcv，weighted k=100 | 0.7765 / 0.7885 | – | 过密反而变差 |
| 参考：ctx+knn×LR | 0.8280 / 0.8175 / 0.839 | 0.7768 / 0.7811 | 昨天冷药最优 |
| 参考：PLMF-MDA 官方模型同划分 | 0.647 / 0.628 | 0.667 / 0.658 | |

读数：
1. CPGF 是一个能同时跑 warm 与 cold、参数可读（β_k = 路径类型权重，p = 度归一化方向，γ = 内容边强度）的极简模型，
   在两面都稳压 PLMF-MDA / MPHGNN 一类端到端网络，但**在两面都不超过仓库的特征栈**，堆叠也无增量——
   这与仓库"该特征空间已线性可分、信息量而非容量是瓶颈"的结论一致。
2. 冷药上的关键不是滤波器的可学性，而是**内容图有多稠密 + 训练时是否模拟冷药**（drugcv 0.75 → k=40 0.795；warm-fit 0.60）。
   这为 R1 提供了两条可写的方法论断：cold-simulated cross-fitting 与 content-graph density 的消融。
3. 若要让 CPGF 成为论文里的"方法"，下一步是把 dense Tanimoto 标签迁移（knn 族）作为第 0 阶算子并入 Â（S_d 用稠密 Tanimoto 而非 kNN 核），
   并加 k-shot 曲线（免重训，只重算 support）。预期上限就是 ctx+knn×LR 的 0.83/0.79，不会更高。

## 4. 对"更有创新点、能提 AUC/AUPR"的最终建议

1. **放弃在 warm 5cv 上找增益**：五条独立路线（100+ 机制、TabICLv2、knn/aa/mech/集成、跨表型旁信息、可学习滤波）全部 ≤ +0.002。
   论文里应当把这件事写成结论（度-only 0.90、泄漏探针 0.96、SOTA 之间 <0.005），引用 [2405.14985]、Bioinformatics 2026 btag608。
2. **指标战场 = 冷药 / leave-drugs-out / k-shot**：这里我们 0.83 vs PLMF-MDA 0.65，差距 +0.18，远超 seed 方差；且没有任何 2026 的 MDA 论文做过。
3. **方法卖点 = 统一 warm/cold 的简洁可解释模型 + cold-simulated 训练 + 内容图密度分析**（CPGF 或其并入 kNN 迁移的版本），
   而不是"新的注意力/多模态融合"。
4. **跨表型（耐药↔敏感）**可作为一节分析：共享实体多、pair 重叠 5.7×，但对预测的贡献几乎全是流行度——本身就是一个值得报告的负结果，
   可与 2026 多行为推荐文献的"辅助行为错位"讨论对接。

## 5. 复现

```bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python; export OMP_NUM_THREADS=8
./run_xds.sh; ./run_cold_xds.sh                                   # 跨表型 warm / cold
CUDA_VISIBLE_DEVICES=0 $PY polyfilt.py --dataset MDR --protocol warm --K 8 --epochs 800 --stack ctx,diff
CUDA_VISIBLE_DEVICES=0 $PY polyfilt.py --dataset MDR --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --fit drugcv --graph weighted --knn 40
```
日志 `logs/`，结果 `results/{xds_probe,cold_xds,polyfilt}.csv`，文献 `lit/delta_results.json`。
