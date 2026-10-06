# HGNNLDA 论文实验数据整理（TCBB 2023）

来源：Dayun Liu et al., *HGNNLDA: Predicting lncRNA-Drug Sensitivity Associations via a Dual Channel
Hypergraph Neural Network*, IEEE/ACM TCBB 2023, 3547–3555. DOI `10.1109/TCBB.2023.3302468`。
全部数值转录自论文 Fig.2 / Fig.3 / Table 1 / Table 2。

数据：RNAactDrug；benchmark 36,248 关联 / 978 lncRNA / 1,815 drug；负例 1:1 随机未知对；
独立集 2,093 关联（273 lncRNA / 480 drug）。协议：全部关联随机分 5 折，1 折测试 / 4 折训练；
指标 AUC / AUPR；独立集用全部 benchmark 训练。

## 1. benchmark 5 折交叉验证（Fig.2 + Fig.3a,b）

| 排名 | 方法 | AUC | AUPR | 来源 |
|---|---|---|---|---|
| 1 | **HGNNLDA** | **0.9325** | **0.9348** | Fig.3a/b |
| 2 | LightGCN | 0.9132 | 0.9134 | Fig.2 |
| 3 | MMGCN | 0.9104 | 0.9051 | Fig.3a/b |
| 4 | NGCF | 0.9033 | 0.8982 | Fig.2 |
| 5 | LAGCN | 0.8984 | 0.8966 | Fig.3a/b |
| 6 | GANLDA | 0.8580 | 0.8566 | Fig.3a/b |
| 7 | SDLDA | 0.8323 | 0.8210 | Fig.3a/b |
| 8 | DMFCDA | 0.8273 | 0.8014 | Fig.3a/b |
| 9 | NTSHMDA | 0.7810 | 0.7312 | Fig.3a/b |

> 正文 3.3 处 HGNNLDA AUPR 写作 0.9328，但 Fig.3b 图例为 0.9348，此处以图例为准。

## 2. 独立测试集（Fig.3c,d）

| 排名 | 方法 | AUC | AUPR | 来源 |
|---|---|---|---|---|
| 1 | **HGNNLDA** | **0.9095** | **0.8897** | Fig.3c/d |
| 2 | MMGCN | 0.8938 | 0.8792 | Fig.3c/d |
| 3 | LAGCN | 0.8821 | 0.8658 | Fig.3c/d |
| 4 | GANLDA | 0.8429 | 0.8351 | Fig.3c/d |
| 5 | SDLDA | 0.8385 | 0.8221 | Fig.3c/d |
| 6 | DMFCDA | 0.8091 | 0.7743 | Fig.3c/d |
| 7 | NTSHMDA | 0.7748 | 0.7775 | Fig.3c/d |

> 独立集只对比了 6 个相关方法，未含 NGCF / LightGCN。

## 3. 本文复现值（参考，非论文值）

| 评测 | 论文 | 本文复现 | Δ |
|---|---|---|---|
| benchmark AUC | 0.9325 | 0.9241（e0）/ 0.9222（iter） | −0.0084 / −0.0103 |
| benchmark AUPR | 0.9348 | 0.9319（e0）/ 0.9298（iter） | −0.0029 / −0.0050 |
| 独立集 AUC | 0.9095 | 0.9056 | −0.0039 |
| 独立集 AUPR | 0.8897 | 0.8898 | +0.0001 |

## 4. 案例研究（Table 1 / Table 2）

**Table 1 — kalamycin**（去掉 26 个已知关联，952 候选；top-15 中 **10 个**被文献验证）

| # | lncRNA | PubMed | # | lncRNA | PubMed |
|---|---|---|---|---|---|
| 1 | PDCD4-AS1 | Unconfirmed | 9 | MIR210HG | 34897892 |
| 2 | GS1-24F4.2 | Unconfirmed | 10 | EXTL3-AS1 | Unconfirmed |
| 3 | LOXL1-AS1 | 32449981 | 11 | GATA3-AS1 | 34358678 |
| 4 | LINC00477 | Unconfirmed | 12 | ZBED3-AS1 | 29749482 |
| 5 | RHPN1-AS1 | 34917132 | 13 | TOB1-AS1 | 31482275 |
| 6 | MCM3AP-AS1 | 32678686 | 14 | MIR31HG | 30125988 |
| 7 | LINC00937 | 33376751 | 15 | CACNA1G-AS1 | 34238292 |
| 8 | EMX2OS | Unconfirmed | | | |

**Table 2 — bulletyanin**（去掉 43 个已知关联，1772 候选；top-15 中 **7 个**被文献验证）

| # | lncRNA | PubMed | # | lncRNA | PubMed |
|---|---|---|---|---|---|
| 1 | LINC00710 | Unconfirmed | 9 | NR2F1-AS1 | 34128858 |
| 2 | LINC00592 | Unconfirmed | 10 | HOXA-AS2 | 33174119 |
| 3 | LY86-AS1 | 32854616 | 11 | LOXL1-AS1 | 32449981 |
| 4 | THRB-AS1 | Unconfirmed | 12 | LINC00312 | 34336850 |
| 5 | LINC01016 | 31856144 | 13 | IGSF11-AS1 | Unconfirmed |
| 6 | GS1-24F4.2 | Unconfirmed | 14 | UBL7-AS1 | Unconfirmed |
| 7 | ERVH48-1 | Unconfirmed | 15 | PRKCQ-AS1 | Unconfirmed |
| 8 | WT1-AS | 30468780 | | | |

## 5. 参数敏感性（Fig.4）

| 参数 | 取值范围 | 论文结论 |
|---|---|---|
| 高阶连通深度 | 1, 2, 3, 4 | 2 阶最优（原文：the optimal value when using the second-order connectivity） |
| 嵌入维度 | 16, 32, 64, 128, 256 | 64 最优 |
| 网络层数 | 1, 2, 3, 4 | 3 层最优 |

## 6. 产物

- `HGNNLDA_paper_tables.xlsx` — 本文件全部表格（benchmark_5fold / independent / 案例研究 / 参数敏感性）
- `build_paper_data.py` — 生成脚本
- `REPORT_HGNNLDA.md` — 复现报告
- `HGNNLDA_reproduction.xlsx` — 复现明细
