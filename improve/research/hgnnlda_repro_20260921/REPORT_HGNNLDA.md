# HGNNLDA 基线复现（TCBB 2023）

目录：`improve/research/hgnnlda_repro_20260921/`

## 0. 结论

论文 PDF 已获得（存本目录），其报告值与本次复现高度吻合：

| 评测 | 论文报告 | 本文复现（e0，官方代码写法） | Δ |
|---|---|---|---|
| benchmark 5 折 AUC | 0.9325 | **0.9241 ± 0.0026** | **−0.0084** |
| benchmark 5 折 AUPR | 0.9348 | **0.9319 ± 0.0026** | **−0.0029** |
| 独立集 AUC | 0.9095 | **0.9056** | −0.0039 |
| 独立集 AUPR | 0.8897 | **0.8898** | **+0.0001** |

即：**独立集几乎逐位复现（AUPR 差 +0.0001），benchmark 5 折在 0.01 AUC 以内**，
差异主要来自负例抽样随机性与评测实现细节（论文未公开评测脚本）。

同数据同协议（官方 fold0，balanced）对比本文模型：

| 方法 | AUC | AUPR |
|---|---|---|
| iNcRD-HG 图特征 + HistGB | **0.9307** | **0.9425** |
| HGNNLDA（本次复现，fold0） | 0.9187 | 0.9277 |
| iNcRD-HG TabICLv2 | 0.9017 | 0.9144 |

---

## 1. 论文信息与官方代码

- Dayun Liu, Xianghui Li, Liangliang Zhang, Xiaowen Hu, Jiaxuan Zhang, Zhirong Liu, Lei Deng.
  *HGNNLDA: Predicting lncRNA-Drug Sensitivity Associations via a Dual Channel Hypergraph Neural Network.*
  IEEE/ACM TCBB, 2023, pp. 3547–3555. DOI `10.1109/TCBB.2023.3302468`。
- 官方代码：`github.com/dayunliu/HGNNLDA`（已存 `repos/HGNNLDA-official/`）——TF1.15，
  **只含训练过程，`main.py` 没有评测函数**；数据 `ass.txt` + `dataset/train_{0..4}.txt`
  + `dataset/test_{0..4}.txt` + `indenpent_dataset.txt` 全部官方划分可用。

## 2. 数据与论文协议（论文原文）

- RNAactDrug 数据库；去冗余后 benchmark 含 **36,248 个 lncRNA-drug 敏感性关联、978 lncRNA、1,815 drug**
  （论文数值 36248；官方发布 `ass.txt` 实为 36,428 条，应为论文笔误，本复现以发布数据为准）。
- 未知对 = 978×1815 − 36248 = 1,738,822；**负例 = 从未知对随机取等量（1:1）**，论文取 36,248 个，
  假阴性率 ≈ 2.08%。
- **5 折 CV**：全部关联随机分 5 份，轮流 1 份测试 / 4 份训练。
- 指标：**AUC、AUPR**（论文只报这两个）。
- 独立集：PubMed 文献检索构建 2,093 个关联（273 lncRNA、480 drug），与 benchmark 无交叉；用全部 benchmark 训练后测试。

## 3. 方法（忠实复现的细节）

超图：二部矩阵 `A`（lncRNA × drug，训练正例）。
- lncRNA 超图（drug 为超边）：`A_u = D_l^{-1/2} H D_e^{-1} Hᵀ D_l^{-1/2}`（978×978）；
- drug 超图（lncRNA 为超边）：`A_i = D_d^{-1/2} Hᵀ D_l^{-1} H D_d^{-1/2}`（1815×1815）。

网络（`main.py` 逐行照抄）：
- 可学习 lncRNA / drug 嵌入（`emb=64`，stddev 0.005）；
- 3 层：**每层 message 恒为 `A @ E0`**（官方写的是 `sparse_matmul(H, self.lncRNA_embeddings)`，
  用的是初始嵌入变量而非上一层嵌入——已原样保留，称 **e0 变体**）；
  `E_k = leaky_relu(A @ E0 @ W_k + E_{k-1})`，训练时 `dropout(0.1)`，再按行 L2 归一化；
- 最终嵌入 `concat([E0, E1, E2, E3])`（256 维），**内积**打分。

训练：BPR pairwise loss（每个正例采样一个该 lncRNA 未关联的 drug 作负例），
Adam `lr=0.005`、`iter=100`、`batch=2048`、`reg_u=1e-4`（均取自 `HGNNLDA.conf`）。
论文超参敏感性实验结论也一致：嵌入维 64、层数 3 最优。

## 4. 一处值得记录的差异：e0 vs iter

论文 Eq.(10)-(11) 描述的是**逐层迭代传播**（`M^(l) = HGNNConv(H, E^(l))`），
而官方 `main.py` 实现的是**每层都用初始嵌入**（`A @ E0`）。两种都复现了：

| 变体 | benchmark AUC | benchmark AUPR |
|---|---|---|
| **e0（官方代码写法）** | **0.9241** | **0.9319** |
| iter（论文公式写法） | 0.9222 | 0.9298 |
| 论文报告 | 0.9325 | 0.9348 |

e0 更接近论文报告值，故作为主结果；两者差距很小（0.002 AUC）。

## 5. 本文最强模型代入 HGNNLDA 数据集（RNAactDrug，官方 5 折，balanced）

用与 HGNNLDA 完全相同的官方 5 折 + balanced 负例协议，把本文模型代入（图上下文特征 = 训练邻接的
SVD+degree，leak-free；脚本 `run_strongest_rnaactdrug.py`）：

| 方法 | AUC | AUPR |
|---|---|---|
| HGNNLDA（论文） | 0.9325 | 0.9348 |
| **本文最强模型：图上下文特征 + HistGB** | **0.9310 ± 0.0023** | **0.9422 ± 0.0021** |
| 本文：TabICLv2 参考（单模型） | 0.8877 ± 0.0026 | 0.9030 ± 0.0020 |
| 本文：TabICLv2 + anchor 融合（w=0.30） | 0.8806 ± 0.0022 | 0.8981 ± 0.0019 |

**结论**：在 HGNNLDA 自己的数据上，本文最强的可迁移形态（**图上下文特征 + HistGB**）与 HGNNLDA
**AUC 基本打平**（0.9310 vs 0.9325，Δ−0.0015），**AUPR 反超**（0.9422 vs 0.9348，Δ+0.0074）。
注意：本文在 PLMF/MSFFMDI 上夺冠的 **TabICLv2+anchor** 配方在此数据上明显更弱（0.8806/0.8981）——
因为这里 anchor 成员（HistGB）本身远强于 TabICLv2 参考，融合反而把强成员拉低；
**本数据集上应直接用「图特征+HistGB」作为本文代表**。

## 6. 论文对比方法（论文报告值）

**benchmark 5 折（Fig. 2 / Fig. 3a,b）**：

| 方法 | AUC | AUPR |
|---|---|---|
| **HGNNLDA** | **0.9325** | **0.9348** |
| LightGCN | 0.9132 | 0.9134 |
| MMGCN | 0.9104 | 0.9051 |
| NGCF | 0.9033 | 0.8982 |
| LAGCN | 0.8984 | 0.8966 |
| GANLDA | 0.8580 | 0.8566 |
| SDLDA | 0.8323 | 0.8210 |
| DMFCDA | 0.8273 | 0.8014 |
| NTSHMDA | 0.7810 | 0.7312 |

**独立集（Fig. 3c,d）**：

| 方法 | AUC | AUPR |
|---|---|---|
| **HGNNLDA** | **0.9095** | **0.8897** |
| MMGCN | 0.8938 | 0.8792 |
| LAGCN | 0.8821 | 0.8658 |
| GANLDA | 0.8429 | 0.8351 |
| SDLDA | 0.8385 | 0.8221 |
| DMFCDA | 0.8091 | 0.7743 |
| NTSHMDA | 0.7748 | 0.7775 |

> 这 6 个对比方法（NTSHMDA/DMFCDA/SDLDA/GANLDA/LAGCN/MMGCN）**均无公开代码/数据**，本次未复现，
> 仅转录论文值供对比表使用。

## 7. 复现命令

```bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd improve/research/hgnnlda_repro_20260921
for S in 1 2 3; do
  CUDA_VISIBLE_DEVICES=1 $PY run_hgnnlda.py --seed $S --device cuda:0            # 5 折, e0
done
CUDA_VISIBLE_DEVICES=1 $PY run_hgnnlda.py --seed 1 --device cuda:0 --propagate iter \
  --out results/hgnnlda_iter_seed1.csv                                          # iter 变体
CUDA_VISIBLE_DEVICES=1 $PY run_hgnnlda_independent.py --seed 1 --device cuda:0
$PY summarize_hgnnlda.py
```

## 8. 产物

```
hgnnlda_repro_20260921/
├─ HGNNLDA ...pdf / paper_text.txt   # 论文与提取文本
├─ repos/HGNNLDA-official/           # 官方代码 + 数据（全部 5 折）
├─ hgnnlda_model.py                  # PyTorch 超图卷积 + BPR（e0 / iter）
├─ hgnnlda_data.py / run_hgnnlda.py / run_hgnnlda_independent.py
├─ summarize_hgnnlda.py / HGNNLDA_reproduction.xlsx
├─ run_strongest_rnaactdrug.py / HGNNLDA_paper_tables.xlsx / PAPER_TABLES.md
└─ results/hgnnlda_seed{1,2,3}.csv, hgnnlda_iter_seed1.csv, hgnnlda_independent_seed1.csv, strongest_rnaactdrug*.csv
```

## 9. 局限

- 论文未公开评测脚本，负例抽样与论文不完全一致，故 benchmark 有 ~0.008 AUC 偏差；独立集几乎一致。
- 论文 6 个对比方法无公开代码，未复现。
- 论文 benchmark 关联数写作 36,248，官方发布数据为 36,428（疑为笔误），本复现以发布数据为准。
