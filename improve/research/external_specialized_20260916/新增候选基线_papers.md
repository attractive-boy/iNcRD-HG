# ncRNA / miRNA / lncRNA-drug 抗性：新增候选基线论文（重点 BIB / B刊）

> 目标：在现有基线（MPHGNN、SubKNet、GAM-MDR、SMCL-MDR、SVMMDR、BNEMDI、NASMDR、AMMGC、GCMDR、DMGAT、DeepLDA、RDRGSE、B-NDRA、LRGCPND、GSLRDA、HGNNLDA、NCRDLLM）之外，
> 补齐 **同任务（miRNA-drug / lncRNA-drug / ncRNA-drug 抗性或敏感性）** 且来自 **BIB / CCF-B / 中科院二区及以上** 的论文。
> 检索日期：2026-09-18。分区/代码状态如不确定已标注“待确认/待查”，**报告指标为论文值，未本地复现**。

## 一、优先级最高（同任务 + B刊/BIB + 有指标或代码）

| 方法 | 任务 | 期刊/会议 | 分区 | 年份 | 报告指标 | DOI | 代码/数据 |
|---|---|---|---|---|---|---|---|
| **MSMP-GAE** | miRNA-drug 关联 | IEEE JBHI | CCF-B / 中科院1区 | 2026 | AUC 0.9861 / AUPR 0.9823 | 10.1109/JBHI.2025.3558303 | 待确认 |
| **DLP** | drug-miRNA 关联 | IEEE/ACM TCBB | CCF-B / 中科院2区 | 2026 | 未取到 | 10.1109/TCBB.2024.3402215 | github.com/kaizheng-academic/DLP（MATLAB） |
| **SGCLDGA** | drug-gene 关联（方法可迁移） | Briefings in Bioinformatics | CCF-B / 中科院1区 | 2024 | AUC 0.8860 / AUPR 0.9072 | 10.1093/bib/bbae231 | 官方代码（见论文） |
| **BCheM** | disease-drug 抗性 | Front. Bioinformatics | — | 2026 | AUC 0.8890 / AUPR 0.8295 | 10.3389/fbinf.2026.1838904 | 待确认 |
| **LR-GNN** | 分子关联(含 lncRNA/miRNA-disease) | Briefings in Bioinformatics | CCF-B / 中科院1区 | 2022 | 未取到 | 10.1093/bib/bbab513 | 待确认 |

- MSMP-GAE：多源信息 + 元路径增强矩阵的图自编码器，与我们的“异构/元路径”路线最接近，且指标公开，**建议优先复现并纳入同表对比**。
- DLP：子空间分割 + 对偶链路预测，MATLAB 代码公开，属于 TCBB（B刊），**可复现性最好**。
- SGCLDGA：BIB 上的简单图对比学习，论文自身即以 LRGCPND（ncRNA 抗性）为基线，方法可直接迁移到 ncRNA/lncRNA-drug。
- BCheM：其对比基线就含 NASMDR、GCFMCL，可作为 miRNA-drug 抗性方向的近邻对照。
- LR-GNN：BIB 通用图关联基线，适合作为结构类（非专用）基线的补充。

## 二、CCF-B 会议（BIBM）与邻近会议：miRNA-drug 方向

| 方法 | 任务 | 会议 | 年份 | DOI | 备注 |
|---|---|---|---|---|---|
| SMCL-MDR | miRNA-drug 抗性 | IEEE BIBM | 2024 | 已在表中 | — |
| Dual-Stream Heterogeneous GNN (Zero-Shot) | miRNA-drug 敏感性 | IEEE BIBM | 2024 | 待查 | 零样本嵌入的双流异质图 |
| DNMDA | miRNA-drug 交互 | IEEE BIBM | 2024 | 待查 | 深度非负矩阵分解多级融合 |
| LGCMDS | miRNA-drug 敏感性 | IEEE BIBM | 2021 | 10.1109/BIBM52615.2021.9669566 | 轻量图卷积，ncDR 系常引用 |
| LAGCN-MDR | drug-miRNA 抗性 | IEEE BIBM | 2021 | 待查 | 层注意力图卷积 + 多通道特征 |
| Integrating Molecular LLM + Multi-View | miRNA-drug 关联 | IEEE BIBM | 2025 | 待查 | 分子大模型 + 多视图 |
| MHTMG | miRNA-drug 关联 | IEEE HPCC(CCF-C) | 2025 | 10.1109/HPCC67675.2025.00029 | 多头注意力 + GraphSAGE |
| Dual-Channel Signed Bipartite GNN | miRNA-drug 交互 | IEEE Trans. Big Data | 2025 | 待查 | 结构特征 + 符号二部图 |

## 三、其他相关模型（中科院一区/二区，非 BIB）

| 方法 | 任务 | 期刊 | 年份 | DOI |
|---|---|---|---|---|
| IRPCA | miRNA-drug 关联（可解释鲁棒 PCA） | J. Chem. Inf. Model. | 2025 | 10.1021/acs.jcim.4c02385 |
| ALRTGCL | miRNA-drug 关联（低秩 Transformer + 图对比） | IEEE Trans. Consumer Electronics | 2025 | 10.1109/TCE.2025.3593065 |
| VGAE + CMF | miRNA-drug 敏感性 | Eng. Appl. Artif. Intell. | 2026 | 10.1016/j.engappai.2026.113901 |
| Metapath-aware Graph Fusion Encoder | miRNA-drug 关联 | ACM(会议) | 2025 | 10.1145/3766671.3766753 |
| SAGCN | circRNA-drug 敏感性 | IEEE/ACM TCBB | 2024 | 待查 |

## 四、lncRNA-drug 抗性：仍稀缺

现有可比对象依然只有 **DeepLDA**（NoncoRNA / ncDR 两个数据集）与 **NCRDLLM**（lncRNA 头，JCIM 2026）。
本轮检索未发现新的、同数据同协议的 lncRNA **only** 抗性预测论文；lncRNA 相关工作多混在 pan-ncRNA（如 NoncoRNA+ncDR 2693）里，
或落在 lncRNA-disease / 药物敏感性（如 PNAS 2019 lncRNA 药敏预测、SAGCN circRNA-drug）而非 lncRNA-drug 抗性。
→ 建议：lncRNA 方向以 pan-ncRNA 基准（41-lncRNA 子集）上的 DMGAT/KBMF2K/CMF/NetLapRLS/BLM-NI 作为主对比，lncRNA-only 单列 DeepLDA。

## 五、结论与建议

1. **可直接纳入同表对比的最佳新增**：MSMP-GAE（JBHI）、DLP（TCBB，有代码）、SGCLDGA（BIB）。
2. **BIBM 系**可挑 2–3 个数据/协议最接近的（SMCL-MDR 已在，另加 Dual-Stream Zero-Shot、LGCMDS、LAGCN-MDR）。
3. 复现前务必核对：数据集（ncDR / NoncoRNA / MDR 版本）、负采样比例、划分方式；AUPR 尤其不可跨行直接比较。
4. 若只是为了“补齐 BIB 基线”，BIB 上与本任务直接相关的抗性预测论文极少（MPHGNN/GCFMCL/NASMDR/DMGAT 已收录），
   其余多为 ncRNA-disease 或 drug response；可用 **SGCLDGA / LR-GNN** 作为 BIB 侧的方法类基线。

## 附：Excel 位置

新增候选已写入 `improve/results/external_specialized_20260916/外部专用模型对比_论文格式.xlsx` 的
sheet **`新增候选基线(BIB-B刊)`**（脚本：`build_paper_style_excel.py`）。
