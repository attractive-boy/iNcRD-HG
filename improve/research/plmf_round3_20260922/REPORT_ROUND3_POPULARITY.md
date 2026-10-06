# PLMF-MDA 第三轮：冷启动药物的"流行度 + 兼容性"分解与标签无关的文献关注度先验

目录 `improve/research/plmf_round3_20260922/`　日期 2026-09-22
前两轮结论（`plmf_frontier_20260922`, `plmf_xds_20260922`）：warm 5 折 0.951/0.954 是天花板（5 条独立路线 ≤ +0.002）；
指标战场在冷药（PLMF Table-2 划分我们 0.83 vs PLMF-MDA 0.65），但全药物 leave-drugs-out（LDO）的 pooled AUC 只有 0.69，
而逐药物 macroAUC 0.85 —— 差距来自"冷药物的流行度不可知"。本轮就针对这个缺口。

## 0. 结论（一句话）

冷药物 MDA 预测可分解为 `score = compat(m,d) + λ·pop(d)`：**pop(d) 对 LDO pooled AUC 的贡献是 0.69 → 0.93（oracle）**，
而 pop(d) 可以由**标签无关的 PubMed 文献计数**（药名 × miRNA/耐药/癌症语境）预测（Spearman 0.47–0.63），
把 pooled AUC/AUPR 提到 **MDR 0.815/0.820、MDS 0.858/0.857**（+0.13/+0.15，3 seeds，λ 由内层冷模拟 CV 选定，
打乱对照回落到基线）；若允许用另一表型（MDS↔MDR）的度做旁信息，则 **0.913/0.898、0.918/0.905**（接近 oracle）。
化学结构相似性（Tanimoto/ChemBERTa kNN、Morgan+ChemBERTa 岭回归）和 ChEMBL 靶点数**几乎不预测流行度**（ρ≈0.1–0.2），
这解释了为什么 PLMF-MDA 一类"内容→关联"模型在冷药上只有 0.65。

这是一条 2026 年 ncRNA–drug 文献里没有人做过的线：**冷启动下显式建模并预测实体流行度**，
且与 iNcRD-HG 早先在 warm 面上做过的"bibliometric exposure"（+0.0003，`provenance_next_20260917`）形成对照 ——
同一信息在 warm 面无用（度已知）、在冷面是最大杠杆。

## 1. 试点 1（`cold_pop.py`）：oracle 流行度、结构 kNN 插补、伪支持自训练

LDO（全部药物按度分层 5 折，3 seeds，ctx+knn × LR warm 交叉拟合），pooled AUC / AUPR：

| 变体 | MDR | MDS |
|---|---|---|
| base（ctx+knn × LR） | 0.687 / 0.686 | 0.704 / 0.701 |
| + Tanimoto-kNN 插补度, λ=0.5 | 0.703 / 0.699 | 0.727 / 0.720 |
| + oracle 真实度, λ=1 | 0.894 / 0.879 | 0.912 / 0.904 |
| + oracle 真实度, λ=2 | **0.930 / 0.918** | **0.938 / 0.930** |

- oracle 说明 pooled 指标的缺口**全部**是流行度；逐药物 macroAUC（0.850 / 0.824）对任何 λ 不变（同药物内是常数平移）。
- 结构 kNN 插补 ρ(dhat, deg) 只有 −0.03 ~ 0.41（fold 间不稳定）。
- **伪支持自训练**（每个冷药取 top-q 预测对作为伪边、重算特征、不重训）在 PLMF Table-2 划分上 0.828 → 0.768（2 轮 0.765），
  负结果：LR 的高分对里 4–8% 是测试正例本身，其余伪边引入的是错误的"度"。

## 2. 试点 2（`cold_pop2.py`）：哪种旁信息能预测冷药物的流行度

预测目标 log(1+deg_d)，只用训练药物拟合；ρ 为冷药物上的 Spearman（15 折均值）：

| 预测器 | 来源 | MDR ρ | MDS ρ |
|---|---|---|---|
| tan_knn / cb_knn / content_ridge | Morgan Tanimoto、ChemBERTa 余弦、岭回归 | 0.10 / 0.06 / 0.06 | 0.24 / 0.15 / 0.23 |
| tgt_cnt | ChEMBL MoA 靶点数 | −0.00 | 0.11 |
| pub_all / pub_mirna / pub_resist / pub_cancer | PubMed esearch 计数（药名[tiab] × 语境），`fetch_pubmed_drug.py` | 0.29 / 0.38 / 0.41 / 0.45 | 0.43 / 0.54 / 0.55 / 0.62 |
| combo_lit（4 个文献列岭回归） | 同上 | **0.47** | **0.63** |
| xdeg | 另一表型数据集的度（MDS↔MDR，共享 81 药） | 0.86 | 0.90 |
| combo_all | 全部 | 0.86 | 0.91 |

pooled LDO（3 seeds；`@sel` = λ 由**内层药物分组、冷模拟** 3 折 CV 在 {0.5,…,6} 中选，测试药物不参与）：

| 变体 | MDR AUC / AUPR | MDS AUC / AUPR |
|---|---|---|
| base | 0.687 ± 0.011 / 0.686 | 0.704 ± 0.002 / 0.701 |
| + pub_mirna@sel | 0.803 / 0.811 | 0.835 / 0.836 |
| + **combo_lit@sel（纯文献先验）** | **0.815 ± 0.014 / 0.820** | **0.858 ± 0.006 / 0.857** |
| + xdeg@sel（另一表型度） | 0.913 ± 0.008 / 0.898 | 0.918 ± 0.013 / 0.905 |
| + combo_all@sel | 0.911 / 0.896 | 0.919 / 0.906 |
| oracle λ=2 | 0.930 / 0.918 | 0.938 / 0.930 |
| **打乱对照**（pub/xdeg 跨药物置换）combo_lit@sel | 0.683 | 0.669 |
| 打乱对照 xdeg@sel | 0.687 | 0.687 |

- 文献先验的收益 +0.13 / +0.15 AUC 在 seed 方差（≤0.015）之外，打乱后归零或略低于基线（λ 仍被选为正）。
- xdeg 的注意事项：只对同时出现在另一数据库的药物可用（MDR 105 药中 81 个），论文里应作为"半冷启动"旁信息单列。
- PLMF Table-2 的窄带划分（测试药度 11–20）：流行度先验**无效甚至有害**（MDR 0.829 → 0.81，MDS 0.784 → 0.70），
  因为该划分把流行度方差人为压到零；内层 CV 模拟的是全度分布，λ 被选大。这本身是对 PLMF-MDA 冷启动协议的批评：
  它测的只是 compat，而真实的新药物度分布是重尾的。
- 实现陷阱：内层 CV 若不把留出药物按"冷"重算特征（用 warm 交叉拟合的特征直接切分），λ 总被选成最小值（0.5），
  先验看起来无效；必须在内层折内用内层 support 重算留出药物的特征。

## 3. 文献检索（本轮补充）

arXiv export API 本轮全程 429/503（`logs/lit_arxiv.log`），Europe PMC 与 OpenAlex 可用（`lit/scan3_results.json`, `lit/scan3_openalex_rel.json`），
但宽泛查询噪声很大；与本轮直接相关、改变判断的只有：

| 论文 | 与本轮的关系 |
|---|---|
| On Inherited Popularity Bias in Cold-Start Item Recommendation [2510.11402] | 冷启动模型继承并放大 warm 流行度偏差；本轮反过来说明**显式预测流行度**才是 pooled 指标的杠杆 |
| Implicit degree bias in the link prediction task [2405.14985] | 度-only null 模型接近最优；本轮是它在**归纳式**设定下的对应：度需要被预测 |
| Cold-Start Recommendation towards the Era of LLMs: survey [2501.01945] | 冷启动旁信息分类；文献计数式"外部关注度"先验未在其中出现 |
| BioPathNet, Nat BME 2026-01 | 生物医学 KG 链路预测，同样报告随机划分高估；无冷实体流行度处理 |
| RGCNMDA (Bioinformatics 2026, btag620), LI-DTI (btag428), PCIM-DTA (btag646) | 冷启动生物关联预测的同期工作，均未建模流行度，pooled 指标直接报 |
| iNcRD-HG 自身 `provenance_next_20260917` §3 | 同一 PubMed 计数在 warm 面 +0.0003 —— 与本轮冷面 +0.13 对照，证明"流行度信息在 warm 面已被度吸收" |

2026-05 → 09 的 ncRNA–drug 论文（Europe PMC）无新的方法学论文超出前两轮列表（MIFNDRA、MPHGNN、PLMF-MDA、MVGSCA、HDBI、mDADGAN、DC-MetaMG、NCRDLLM）。

## 4. 对论文的建议（在前两轮方案 A 上更新）

方法部分改为"**流行度–兼容性解耦的归纳式 MDA 模型**"：
1. `compat(m,d)`：现有 ctx+knn 线性读头（或 CPGF），warm 交叉拟合；
2. `pop(d)`：冷药物用标签无关的文献关注度先验预测（PubMed 计数 × 语境，4 列岭回归），warm 药物用观测度；
   可选"半冷"版本加另一表型度；
3. λ 由冷模拟内层 CV 选择；
4. 评估：warm 5cv（持平 0.951/0.954，说明 pop 已被度吸收）、PLMF Table-2 窄带（只测 compat，0.83 vs 0.65）、
   **全药物 LDO pooled + 逐药物 macro**（本轮的主表：0.69 → 0.82/0.86，oracle 0.93）、打乱对照、k-shot 曲线。

这比"新的注意力/多模态融合"更有辨识度：它给出一个可检验的解释（为什么 PLMF-MDA 冷药 0.65、为什么 pooled 与 macro 差 0.16），
并用一个完全外部、可复现、零训练成本的先验把 pooled AUC/AUPR 提高 0.13–0.15。

审稿风险与回答：
- "文献计数是标签泄漏？" —— 计数是实体级、不含 miRNA–药物对，与 MDA 数据库的构建（文献挖掘）同源是**机制解释**而非泄漏；
  打乱对照归零；warm 面无收益说明它不携带 pair 信息。
- "只是给分数加了个常数？" —— 正是要点：在归纳式设定下这个"常数"决定了 0.24 AUC，而所有 2026 方法都忽略了它。
- 下一步可做：把 pop_hat 作为特征进模型（替代 degree_d，冷模拟训练）而非事后加性；文献计数按年份截断做前瞻性划分；
  在 iNcRD-HG 自有 MiDrug/LncDrug cold_drug surface 上复现。

## 5. 复现

```bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python; export OMP_NUM_THREADS=8
python3 fetch_pubmed_drug.py && python3 fix_pubmed_names.py            # data/pubmed_drug_plmf.csv（165 药，1 个无名）
$PY cold_pop.py  --dataset MDR --protocol ldo --seeds 0,1,2            # oracle / 结构插补；--pseudo 伪支持
$PY cold_pop2.py --dataset MDR --protocol ldo --seeds 0,1,2 --tag sel  # 流行度预测器 + 内层 CV λ
$PY cold_pop2.py --dataset MDR --protocol ldo --seeds 0,1,2 --tag shuf --shuffle_side
```
日志 `logs/`，结果 `results/cold_pop2_{MDR,MDS}_{ldo,plmf}_sel.csv`、`results/cold_pop2_rho_*.csv`。
