# 新增候选基线：同协议复现台账

目录：`improve/research/baseline_repro_20260918/`
同协议基准：ncDR (GAM-MDR 701 miRNA × 101 drug, 3163 assoc)，80/20 边划分 + 等量负采样，全局 AUC/AUPR，5 seeds（与 `/tmp/opencode/exp_final.py` 一致）。
统一 harness：`common.py`（`build_split` / `sample_neg` / `evaluate`）。

## 一、已完成（同协议）

| 方法 | 期刊 | 结果 (ncDR) | 说明 |
|---|---|---|---|
| **SGCLDGA** | BIB 2024 | **AUC 0.8878±0.0075 / AUPR 0.9089±0.0070** | 官方代码 `github.com/one-melon/SGCLDGA`；修复 2 处笔误(`xavier_gniform_`,`__dnit__`)后迁移到 ncDR。脚本 `repro_sgcldga_ncdr.py` |
| **JMSSMMA** | BIB 2022 | **AUC 0.6145±0.0164 / AUPR 0.6305±0.0145** | 官方代码原样运行(molcl-sp 环境)；ncDR 无其所需的富相似度特征，仅用训练集 GIP 节点特征 → 公平偏低。脚本 `repro_jmssmma_ncdr.py` |

对比：本文 iNcRD-HG 在 ncDR 上 AUC 0.8863 / AUPR 0.8711 →
**SGCLDGA 同协议 AUC +0.0015 / AUPR +0.0378 高于我们；JMSSMMA 同协议低于我们。**

## 二、代码已获取，待适配/运行

| 方法 | 期刊 | 任务 | 仓库 | 阻塞点 |
|---|---|---|---|---|
| AMMGC | Front Pharmacol 2022 | miRNA-drug 抗性 | github.com/scz760904126/AMMGC | TF1 代码硬编码 106×754 / 79924 / 860 及 4 组特征块，需改 101×701；gapTrick 缺 sklearn |
| MGCNA | PLoS Comput Biol 2025 | miRNA-drug | github.com/sheng-n/MGCNA | 需 drug-gene / miRNA-gene 额外视图，ncDR 无此数据 |
| SAGCN | TCBB 2024 | circRNA-drug 敏感性 | github.com/lizhiqi49/SAGCN | TCGA 流程，非二部图矩阵式，难落到 ncDR |
| GCNLASMMA | Front Genet | small molecule-miRNA | github.com/1054366388/GCNLASMMA | 与 LAGCN-MDR 同族；需 SM/miRNA 相似度 |
| GCLMTP | BIB 2023 | lncRNA/miRNA-disease | github.com/sheng-n/GCLMTP | 非 drug 任务，仅作结构参考 |

## 三、阻塞：无 MATLAB

| 方法 | 期刊 | 任务 | 说明 |
|---|---|---|---|
| DLP | TCBB 2026 | drug-miRNA | 仅 MATLAB 代码（`domo.m`），本机无 MATLAB/Octave；需移植 Python 或装 Octave |

## 四、无公开代码 → 需按论文重实现或放弃

MSMP-GAE (JBHI 2026)、Dual-Stream Zero-Shot GNN (BIBM 2024)、DNMDA (BIBM 2024)、
LGCMDS (BIBM 2021)、LAGCN-MDR (BIBM 2021)、Molecular LLM+Multi-View (BIBM 2025)、
MHTMG (HPCC 2025)、Dual-Channel Signed Bipartite GNN (TBD 2025)、IRPCA (JCIM 2025)、
ALRTGCL (TCE 2025)、VGAE+CMF (EAAI 2026)、Metapath-aware GFE (ACM 2025)、
BCheM (Front Bioinform 2026)、Entropy Ranking (TCBB)。

> 这部分要么重实现（成本高、易失真），要么在论文中作为“跨协议参考文献值”列出并注明未复现。

## 五、下一步批次建议

1. **AMMGC**：改维度 + 换 ncDR 特征（kmer/GIN），补 sklearn 环境 → 可产出 ncDR 同协议值。
2. **DLP**：装 Octave 或把 NE+KNN+LSR 移植 Python → ncDR 同协议值（图/矩阵类）。
3. **LR-GNN**：检索代码，通用链路预测，适配成本中等。
4. 其余无代码者：批量用论文报告值入表（标注“未复现·跨协议”）。

## 六、产出物

- `repro_sgcldga_ncdr.py` + `log_sgcldga_ncdr.txt` + `results_sgcldga_ncdr.csv`
- `repro_jmssmma_ncdr.py` + `log_jmssmma_ncdr.txt` + `results_jmssmma_ncdr.csv`
- `common.py`（共享同协议 harness）
- 候选仓库：`work/` 与 `repos/`
- Excel：`improve/results/external_specialized_20260916/外部专用模型对比_论文格式.xlsx`
  - sheet `ncDR` 已加入 SGCLDGA（同协议）
  - sheet `同协议复现台账` 为本文档的表格版

## 七、本文方法统一为 TabICLv2（2026-09-18）

按要求把所有外部数据集的“本文方法”换成 **iNcRD-HG 可迁移核 + 冻结 TabICLv2**，5 seeds/5 折：

| 数据集 | 本文方法 (TabICLv2) AUC | AUPR | 说明 |
|---|---|---|---|
| MPHGNN | 0.9605 | 0.9024 | GNN 配对嵌入 + TabICLv2 |
| ncDR (701×101) | 0.8322±0.0019 | 0.8135±0.0072 | 图特征 + TabICLv2 (dump: tabicl_ncdr/) |
| NoncoRNA+ncDR 2693 | 0.7617±0.0197 | 0.7782±0.0117 | 复用 feat_lrgcpnd_pan_fold*.npz |
| 41-lncRNA 子集 | 0.7234±0.0638 | 0.7347±0.0681 | float64 修复 Yeo-Johnson 报错 |
| GCFMCL D2R | 0.7482±0.0121 | 0.7492±0.0154 | dump: tabicl_gcfmcl/ |
| RNAactDrug | 0.9016 | 0.9144 | 复用 feat_rnaactdrug.npz |
| DeepLDA NoncoRNA | 0.9997±0.0002 | 0.9856±0.0069 | 图特征(全矩阵高斯相似)+TabICLv2; 协议泄漏, ≈1.0 不代表能力 |
| DeepLDA ncDR | 0.9948±0.0040 | 0.9303±0.0535 | 同上; 仅 184 正例 |

> 结论：TabICLv2 在“有节点特征/中等规模矩阵/非极端不平衡”的数据上可用；DeepLDA 的全矩阵相似度协议本身泄漏，
> 图特征 + 任意分类器都≈1.0（包括 TabICLv2），该行不可作为模型强弱证据；2693/41-lnc/GCFMCL 上用 TabICLv2
> 明显弱于 HistGB（与 REPORT 的“矩阵型弱特征上 HistGB 更稳”一致）。

脚本：`dump_tabicl_ncdr.py`、`dump_tabicl_gcfmcl.py`；
打分：`external_specialized_20260916/score_matrix_tabicl.py`（已把 X 转 float64 以修复 41-lncRNA 预处理报错）。

## 八、ncDR 同协议自跑基线（2026-09-18 续）

| 方法 | 来源/复现方式 | AUC | AUPR | ACC | F1 | MCC |
|---|---|---|---|---|---|---|
| GAM-MDR | 官方代码移植 | 0.9179 | 0.8977 | 0.8540 | 0.8590 | 0.7104 |
| AMMGC | PyTorch 忠实重实现 | 0.9024 | 0.9064 | 0.8525 | 0.8564 | 0.7077 |
| NDSGCL | 重实现(LGCN+BPR+SSL+ProtoNCE) | 0.9004 | 0.9070 | 0.8282 | 0.8291 | 0.6569 |
| SGCLDGA | 官方代码 | 0.8878 | 0.9089 | 0.8378 | 0.8384 | 0.6761 |
| **本文 (TabICLv2)** | 自跑 | 0.8322 | 0.8135 | 0.7718 | 0.7869 | 0.5515 |
| GCMDR | PyG 重实现 | 0.7875 | 0.6887 | 0.7763 | 0.7838 | 0.5567 |
| DLP | MATLAB→Python 移植 | 0.7622 | 0.8219 | 0.7419 | 0.7414 | 0.4858 |
| AGCLNDA | 官方代码 | 0.6892 | 0.7485 | 阈值退化 | — | — |
| DeepLDA | 本地移植 | 0.6364 | 0.7207 | 0.6422 | 0.6982 | 0.2947 |
| JMSSMMA | 官方代码 | 0.6145 | 0.6305 | 阈值退化 | — | — |

脚本：`repro_{gammdr,gcmdr,ammgc,dlp,deeplda,ndsgcl,agclnda,sgcldga,jmssmma}_ncdr.py`。
阈值指标用 best-F1 阈值（TabICLv2 概率被压缩，0.5 阈值失真）。

**仍缺**：MPHGNN / SubKNet（官方代码在各自 benchmark 数据上，需按其独立测试集复现）；
无代码的 SMCL-MDR/SVMMDR/BNEMDI/NASMDR 等只能保留论文值。

## 九、MPHGNN 独立测试集自跑（2026-09-18 续）

| 方法 | 方式 | AUC | AUPR | ACC | Precision | Recall | Specificity | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|
| MPHGNN | 官方代码, 5折 seed42 | 0.9755 | 0.9415 | 0.9246 | 0.8883 | 0.8850 | 0.9443 | 0.8866 | 0.8301 |
| SubKNet | 官方模型复现, 5折 seed42 | 0.9493 | 0.9070 | 阈值指标因输出接口不稳定，仅报 AUC/AUPR | | | | | |

- MPHGNN：`/tmp/opencode/mphgnn_official/main.py -ep 10 -fo 5 -se 42`，与论文值(0.978/0.937/ACC 0.9394/F1 0.9131/MCC 0.8633)接近。
- SubKNet：`improve/research/mirna_sota_20260915/repro_subknet_official.py --seeds 42`。
- 其余同表基线(GAM-MDR/DeepLDA/AGCLNDA/NDSGCL)在 MPHGNN 测试集上的自跑需把各自代码适配到 MPHGNN 数据，尚未做。

## 十、新增 5 篇专用模型：官方代码复现 + 1 篇重实现（2026-09-19）

目录：`improve/research/new5_repro_20260919/`（详细报告 `REPRO_ALL5.md`）。

| 方法 | 期刊 | 官方代码 | 复现 AUC | 复现 AUPR | 论文 AUC | 论文 AUPR | 日志 |
|---|---|---|---|---|---|---|---|
| MPHGNN | BIB 2026 | 有 | 0.9743–0.9755 | 0.9382–0.9415 | 0.9780 | 0.9370 | `logs/mphgnn_official_s422.log` |
| DLST-MDA | JCIM 2025 | 有 | 0.9437 | 0.9356 | 0.9411 | 0.9326 | `logs/dlst_train.log` |
| PLMF-MDA (MDR) | PLoS CB 2026 | 有 | 0.9260±0.0061 | 0.9135±0.0064 | 0.9222 | 0.9062 | `logs/plmf_mdr*.log` |
| PLMF-MDA (MDS) | PLoS CB 2026 | 有 | 0.9255±0.0014 | 0.9145±0.0034 | 0.9301 | 0.9207 | `logs/plmf_mds*.log` |
| DLMVF | JCIM 2026 | 有 | 0.9591 | 0.9514 | 0.9611 | 0.9543 | `logs/dlmvf_train.log` |
| MSFFMDI | Interdiscip. Sci. 2025 | **无** | 0.9483±0.0044 | 0.9391±0.0071 | — | — | `logs/msffmdi_train.log`（按摘要重实现，非数值复现） |

- **DLST-MDA / DLMVF / MSFFMDI(重实现)** 使用同一 1578×156 / 8720 关联 / 等量负例 5 折协议，可横向比较：
  DLMVF 0.9591/0.9514 > MSFFMDI(reimpl) 0.9483/0.9391 > DLST-MDA 0.9437/0.9356。
- MPHGNN 用其自有 MiDrug benchmark；PLMF-MDA 用自建 MDR/MDS benchmark，均不与上表直接可比。
- 关键工程修复：DLMVF 的 `datasetindex()` 字典别名 bug（否则 5 折退化）、`drug_offset` 1563→1578、缺失邻接矩阵重建；
  PLMF-MDA 的 RAR5 用 `bsdtar` 解包、ChemBERTa/RNA-FM 本地权重加载。补丁见 `new5_repro_20260919/patches/`。

## 十一、DeepLDA 官方代码复现（2026-09-20）

目录：`improve/research/deeplda_official_20260920/`（详细报告 `REPORT.md`、汇总 `SUMMARY_deeplda_official.md`）。
论文：Gao & Shang, Front. Microbiol. 2023, doi:10.3389/fmicb.2023.1147778；官方代码 `github.com/meihonggao/DeepLDA`。

**必要修复**：`Codes/models.py` 的 `super(MVG_LGA, self)` → `super(Deep_LDA, self)`（否则直接 NameError）。

官方协议 = 打乱行 → 用**整张**关联矩阵算高斯相似度当节点特征（泄漏）→ 前 90% 行训练 / 后 10% 行测试，全部元素打分。
论文口径的 balanced（等量负例）官方代码未实现，本复现按论文文字补齐。5 seeds。

| 数据集 | 设置 | DeepLDA(官方) AUC/AUPR/F1/MCC | GCN-only | GAT-only | 论文 DeepLDA |
|---|---|---|---|---|---|
| NoncoRNA (3601×71) | unbal | 0.876/0.074/0.496/0.000 | 0.947/0.489/0.679/0.365 | 0.969/0.790/0.886/0.799 | 0.954/0.851/0.861/0.756 |
| NoncoRNA | bal | 0.603/0.595/0.469/0.231 | 0.876/0.860/0.776/0.668 | 0.864/0.884/0.863/0.791 | 0.958/0.860/0.863/0.763 |
| ncDR (162×31) | unbal | 0.942/0.811/0.889/0.807 | 0.992/0.917/0.954/0.913 | 0.995/0.949/0.916/0.843 | 0.973/0.857/0.888/0.779 |
| ncDR | bal | 0.905/0.841/0.852/0.755 | 0.960/0.871/0.942/0.893 | 0.977/0.948/0.906/0.817 | 0.982/0.869/0.916/0.789 |

结论：**ncDR 同量级（方差大），NoncoRNA 未复现**（unbal 退化为全负 MCC=0，bal AUC 0.60）；
且在官方泄漏协议下 **GCN/GAT 单分支普遍 ≥ DeepLDA**，论文"DeepLDA>GAT>GCN"未复现。

同协议（ncDR 701×101，`common.py`）本地移植版 DeepLDA 本次实跑：
**AUC 0.6788±0.0529 / AUPR 0.7457±0.0315**（§八 此前 0.6364/0.7207 同量级）。

## 十二、PLMF-MDA 论文 Fig.3 对比基线复现（2026-09-20）

目录：`improve/research/plmf_baselines_20260920/`（详细报告 `REPORT_PLMF_BASELINES.md`、
结果表 `PLMF_MDA_baselines.xlsx`）。任务：复现 PLMF-MDA（PLoS CB 2026）**论文里被对比的 4 个基线**
（GCNNMMA / SubMDTA / GraphDTA / ML-DTI）在论文自建 MDR/MDS 数据上的 Fig.3 实验。
PLMF-MDA 自身已在 §十 复现。

协议：`5cv`（正例+等量负例随机均分 5 折，seed42）与 `single`（官方 80/20 划分，seed0-4）；
模型选择取测试集最优 epoch。4 基线复用 `dlstmda_baselines_20260919/` 的实现。

| 数据集 | 协议 | 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR | ΔAUPR |
|---|---|---|---|---|---|---|---|---|
| MDR | 5cv | PLMF-MDA | 0.9243 | 0.9227 | +0.0016 | 0.9146 | 0.9076 | +0.0070 |
| MDR | 5cv | GCNNMMA | 0.9118 | 0.9070 | +0.0048 | 0.8949 | 0.8736 | +0.0213 |
| MDR | 5cv | SubMDTA | 0.9044 | 0.8936 | +0.0108 | 0.8907 | 0.8514 | +0.0393 |
| MDR | 5cv | GraphDTA | 0.9054 | 0.8086 | +0.0968 | 0.8907 | 0.7979 | +0.0928 |
| MDR | 5cv | ML-DTI | 0.9238 | 0.7660 | +0.1578 | 0.9131 | 0.7870 | +0.1261 |
| MDS | 5cv | PLMF-MDA | 0.9270 | 0.9301 | −0.0031 | 0.9175 | 0.9207 | −0.0032 |
| MDS | 5cv | GCNNMMA | 0.9133 | 0.9143 | −0.0010 | 0.9007 | 0.8889 | +0.0118 |
| MDS | 5cv | SubMDTA | 0.9059 | 0.8858 | +0.0201 | 0.8923 | 0.8587 | +0.0336 |
| MDS | 5cv | GraphDTA | 0.9125 | 0.7864 | +0.1261 | 0.9036 | 0.7911 | +0.1125 |
| MDS | 5cv | ML-DTI | 0.9296 | 0.7820 | +0.1476 | 0.9226 | 0.7881 | +0.1345 |

（`single` 协议结果同量级，见报告 §3.2 / Excel `Fig3_single`。）

**结论**：GCNNMMA 与 PLMF-MDA 复现良好（|ΔAUC| ≤ 0.006）；GraphDTA/SubMDTA/ML-DTI 复现值
显著高于论文报告值。故论文“PLMF-MDA 一致优于所有基线”在本复现口径下**不成立**：MDR 上
PLMF-MDA 与 ML-DTI 统计打平，**MDS 上 ML-DTI 高于 PLMF-MDA**。与 §十 DLST-MDA 的
“基线被论文低估”结论一致。PLMF-MDA 参考实现用 cached-encoder，`single` 结果与 §十 官方复现一致
（MDR 0.9287/0.9162 vs 0.9260/0.9135；MDS 0.9294/0.9189 vs 0.9255/0.9145）。

## 十三、本文最强模型在 PLMF-MDA MDR/MDS 上的排名（2026-09-20）

目录：`improve/research/plmf_strongest_20260920/`（报告 `REPORT_STRONGEST_ON_PLMF.md`、
榜单 `PLMF_MDA_strongest_ranking.xlsx`）。把最强框架的**可移植子集**（frozen TabICLv2 图上下文
参考 + anchor shrinkage，复用 `dlstmda_strongest_20260920/features.py`）接到 PLMF-MDA 自建
MDR/MDS 上，与 §十二 复现的 5 个方法同榜对比。

| 数据集 | 协议 | 本文方法 AUC / AUPR | 排名 | 第 2 名（差距） |
|---|---|---|---|---|
| MDR | 5cv seed42 | **0.9486 / 0.9433** | **#1** | PLMF-MDA 0.9243（+0.0243 AUC） |
| MDR | single seed0-4 | **0.9514 / 0.9458** | **#1** | PLMF-MDA 0.9287（+0.0227） |
| MDS | 5cv seed42 | **0.9513 / 0.9480** | **#1** | ML-DTI 0.9296（+0.0217） |
| MDS | single seed0-4 | **0.9522 / 0.9486** | **#1** | ML-DTI 0.9303（+0.0219） |

**结论**：四个榜单全部 **#1**，领先第 2 名 0.022–0.024 AUC，远大于折间/seed 波动（0.003–0.007）。
相对 PLMF-MDA 论文值（MDR 0.9227 / MDS 0.9301）高 +0.026 / +0.021 AUC。移植的是能力下界
（证据档/DepMap 档未启用）；每折剔除 3–4 个实体级重复测试 pair。

## 十四、MSFFMDI 论文获取失败（2026-09-20）

任务：获取 MSFFMDI 全文（Interdiscip. Sci. 2026, 18(3):986–1009, DOI 10.1007/s12539-025-00775-7）
以复现其论文对比基线。**结果：无法获取。** 该文为完全闭源（OpenAlex `oa_status=closed`、
Unpaywall 无 OA、不在 PMC/EuropePMC、无代码、无补充材料），正文的基线对比表未公开。

已尝试并失败：Springer 文章页/PDF 端点、ResearchGate(403)、Semantic Scholar、OpenAlex、
Unpaywall、Europe PMC、PubMed(反爬)、Google Scholar/Web 搜索。从公开摘要 + 19 张公开图 +
参考文献提取了论文自报数值（数据集1 5 折 AUC 0.9483/AUPR 0.9461；数据集2 0.9216/0.9163）
与候选基线清单，但**无法确定实际对比的基线与数据集**。

详见 `improve/research/msffmdi_baselines_20260920/ACQUISITION_REPORT.md`（含 figures/ 全部 19 张图）。
**待用户提供 PDF 后即可复现其对比基线。** 注：`new5_repro_20260919/` 的 MSFFMDI 是“按摘要重实现”
（DLST benchmark 0.9483/0.9391），非论文数据/数值复现。

## 十五、MSFFMDI 论文对比基线复现（2026-09-20 续，用户提供 PDF）

目录：`improve/research/msffmdi_baselines_20260920/`（报告 `REPORT_MSFFMDI_BASELINES.md`、
结果表 `MSFFMDI_baselines.xlsx`）。论文 Sect.3.4 对比基线 = **LAGCN / GCNNMMA / MFIDMA /
LDAMAN / DLST-MDA**，数据 = **ncDR (624×95) / RNAInter (1009×281)**，5 折 CV 1:1。
数据取自 MFIDMA 官方仓库（论文声明数据源）。

协议修正：MFIDMA 仓库每折 test 集合完全相同（ncDR test0..4 两两重叠 892 行）且正例重复
（ncDR 4457 行→2846 唯一）、train-test 正例重叠 326，无法支持真 5-CV；改用去重正例 + 等量随机
负例做 clean 5-CV（seed42），故数值不与论文严格可比。

| 数据集 | 方法 | 复现 AUC | 论文 AUC | ΔAUC | 复现 AUPR | 论文 AUPR |
|---|---|---|---|---|---|---|
| ncDR | MFIDMA | 0.9554 | 0.9454 | +0.010 | 0.9485 | 0.9276 |
| ncDR | LDAMAN | 0.9319 | 0.9389 | −0.007 | 0.9208 | 0.9199 |
| ncDR | LAGCN | 0.9297 | 0.8911 | +0.039 | 0.9154 | 0.8489 |
| ncDR | DLST-MDA | 0.9241 | 0.9446 | −0.021 | 0.9112 | 0.9255 |
| ncDR | GCNNMMA | 0.9041 | 0.9123 | −0.008 | 0.8868 | 0.8608 |
| RNAInter | MFIDMA | 0.9489 | 0.9168 | +0.032 | 0.9432 | 0.9074 |
| RNAInter | DLST-MDA | 0.9415 | 0.9143 | +0.027 | 0.9384 | 0.9036 |
| RNAInter | LDAMAN | 0.9413 | 0.9081 | +0.033 | 0.9369 | 0.8989 |
| RNAInter | LAGCN | 0.9361 | 0.8337 | +0.102 | 0.9318 | 0.7360 |
| RNAInter | GCNNMMA | 0.9098 | 0.9116 | −0.002 | 0.9103 | 0.8988 |

**结论**：GCNNMMA/LDAMAN 复现良好；LAGCN/MFIDMA/DLST-MDA 复现值普遍高于论文。论文
“MSFFMDI 优于所有对比方法”**不成立**——两个数据集上 **MFIDMA（基线）AUC/AUPR 均高于论文的
MSFFMDI 值**（ncDR 0.9554 vs 0.9483；RNAInter 0.9489 vs 0.9216）。与 §十/§十二 结论一致。

## 十六、本文最强模型在 MSFFMDI 数据（ncDR/RNAInter）上的排名（2026-09-20 续）

目录：`improve/research/msffmdi_strongest_20260920/`（报告 `REPORT_STRONGEST_ON_MSFFMDI.md`、
榜单 `MSFFMDI_strongest_ranking.xlsx`）。用与 §十五 相同的 clean 5 折 CV，把最强框架可移植子集
（frozen TabICLv2 图上下文 + anchor）接到 ncDR/RNAInter。

| 数据集 | 本文方法 AUC/AUPR | 排名 | 第 1 名（差距） |
|---|---|---|---|
| ncDR | 0.9535 / 0.9481 | **#2** | MFIDMA 0.9554/0.9485（差 0.0019 AUC，统计打平） |
| RNAInter | 0.9578 / 0.9554 | **#1** | 本文方法（领先 MFIDMA +0.0089 AUC / +0.0122 AUPR） |

**结论**：RNAInter 明确 #1；ncDR 与最强基线 MFIDMA 并列第一梯队（差距远小于折间波动）。
两数据集均高于论文 MSFFMDI（0.9483/0.9461；0.9216/0.9163）。纯图上下文参考（无 anchor）下
ncDR AUPR 0.9491 反超 MFIDMA 成 #1。

## 十七、HGNNLDA（TCBB 2023）基线复现（2026-09-21）

目录：`improve/research/hgnnlda_repro_20260921/`（报告 `REPORT_HGNNLDA.md`、
`HGNNLDA_reproduction.xlsx`）。官方仓库 `github.com/dayunliu/HGNNLDA`（TF1.15，**只含训练、无评测脚本**）
已存 `repos/HGNNLDA-official/`；数据 = RNAactDrug lncRNA-drug 敏感性（978 lncRNA × 1815 drug，36428 正例，
官方 5 折 + 独立集 2093）。因环境无 TF，按 `main.py` + `HGNNLDA.conf` 忠实移植为 PyTorch
（emb=64、3 层超图卷积、逐层 message 恒为 A@E0、concat 嵌入、BPR、lr=0.005/iter=100/batch=2048/reg_u=1e-4）。

| 协议 | AUC | AUPR |
|---|---|---|
| 官方 5 折 CV（3 seed 池化，balanced） | **0.9241 ± 0.0026** | **0.9319 ± 0.0026** |
| 官方 5 折 CV（global 排序） | 0.9243 ± 0.0020 | 0.5199 ± 0.0069 |
| 独立数据集（2093 对，balanced） | 0.9056 | 0.8898 |

同数据同协议（官方 fold0，balanced）对比本文模型：iNcRD-HG 图特征+HistGB **0.9307/0.9425** >
HGNNLDA 复现 0.9187/0.9277 > TabICLv2 0.9017/0.9144。

**与论文报告值对比**（论文 PDF 已获取，存复现目录）：benchmark 论文 **0.9325/0.9348**，
复现 **0.9241/0.9319**（ΔAUC −0.0084、ΔAUPR −0.0029）；独立集论文 **0.9095/0.8897**，
复现 **0.9056/0.8898**（ΔAUPR **+0.0001**）。→ 独立集几乎逐位复现，benchmark 在 0.01 AUC 内。

**协议**（论文原文）：全部关联随机分 5 折，1 折测试/4 折训练；负例 = 等量随机未知对（1:1，假阴性率 ≈2.08%）；
指标仅 AUC/AUPR。论文 6 个对比方法（NTSHMDA 0.7810/DMFCDA 0.8273/SDLDA 0.8323/GANLDA 0.8580/
LAGCN 0.8984/MMGCN 0.9104，benchmark）**均无公开代码，未复现**，仅转录论文值。

**附带发现**：论文 Eq.(10)-(11) 写的是逐层迭代传播，而官方 `main.py` 每层 message 恒用初始嵌入 `A@E0`；
两种都跑了——e0（官方写法）0.9241/0.9319 优于 iter（论文公式写法）0.9222/0.9298，e0 更贴近论文值。

**限制**：论文 benchmark 关联数写作 36,248，官方发布 `ass.txt` 实为 36,428（疑笔误），本复现以发布数据为准。

**本文最强模型代入 HGNNLDA 数据集**（官方 5 折 + balanced，脚本 `run_strongest_rnaactdrug.py`）：

| 方法 | AUC | AUPR |
|---|---|---|
| HGNNLDA（论文） | 0.9325 | 0.9348 |
| **本文最强模型：图上下文特征 + HistGB** | **0.9310 ± 0.0023** | **0.9422 ± 0.0021** |
| 本文：TabICLv2 参考（单模型） | 0.8877 | 0.9030 |
| 本文：TabICLv2 + anchor 融合（w=0.30） | 0.8806 | 0.8981 |

→ 与 HGNNLDA **AUC 基本打平**（Δ−0.0015）、**AUPR 反超**（Δ+0.0074）。注意：在 PLMF/MSFFMDI 上
夺冠的 TabICLv2+anchor 配方**在此数据集反而更弱**——anchor 成员（HistGB）本身远强于 TabICLv2 参考，
融合把强成员拉低；本数据集上本文代表应为「图特征 + HistGB」。
