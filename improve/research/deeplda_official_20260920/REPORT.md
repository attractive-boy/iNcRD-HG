# DeepLDA (Front. Microbiol. 2023) 基线复现报告

- 论文：Gao M, Shang X. *Identification of associations between lncRNA and drug
  resistance based on deep learning and attention mechanism.* Front. Microbiol.
  14:1147778, 2023. doi:10.3389/fmicb.2023.1147778
- 方法：DeepLDA = GCN（多属性特征提取）+ GAT（注意力聚合）→ 内积解码。
- 官方代码：`github.com/meihonggao/DeepLDA`（本目录 `official_codes/`，原始数据
  `Datasets/` 来自官方仓库）。
- 复现日期：2026-09-20。

## 0. 一句话结论

**官方代码在 ncDR（Dataset2）上能复现到论文同量级（AUC/AUPR 略低、方差很大），
但在 NoncoRNA（Dataset1）上无法复现论文数值（DeepLDA 明显劣化，反而 GAT 单分支
接近论文值）；且在官方"全矩阵高斯相似度 + 行划分"的泄漏协议下，GCN/GAT 单分支
普遍 ≥ DeepLDA，论文"DeepLDA > GAT > GCN"的结论未能复现。**

## 1. 官方代码与必要修复

官方仓库当前版本有一处会直接报错的笔误：

- `Codes/models.py:10`：`super(MVG_LGA, self).__init__()`（类名是 `Deep_LDA`，
  `MVG_LGA` 未定义）→ 已改为 `super(Deep_LDA, self).__init__()`。

其余按官方默认超参运行：`epochs=1050, lr=1e-3, weight_decay=5e-4, nhid=50,
nclass=200, nheads=8, dropout=0.0, alpha=0.2`，种子 `[72, 666, 0, 1, 2]`。

## 2. 官方评测协议（`official_codes/utils.py::load_data`）

1. 读取 lncRNA×drug 0/1 关联矩阵；
2. 打乱 lncRNA（行）顺序；
3. 用**整张关联矩阵**（含测试行）算 lncRNA / drug 的高斯相似度作为节点特征，
   再按 `>0.5` 二值化成 GAT 的稠密邻接（这一步导致标签泄漏）；
4. 单次行划分：前 9/10 行为训练，后 1/10 行为 val=test，测试时对该批行的
   **全部元素**打分；
5. 指标：AUC、AUPR、F1（`np.rint(score)` 即 0.5 阈值，macro）、MCC。

论文口径另有 **balanced**（按正例数等量抽样负例后再训练/测试）与
**unbalanced**（用全部负例）两种设置，均为"10 折"。官方发布的代码只实现了
单次 90/10 行划分，未实现平衡抽样。本复现按论文定义补上 balanced 分支
（正例 + 等量随机负例，训练/测试各自抽样），其余与官方一致。

## 3. 复现结果（5 seeds，官方协议）

论文 DeepLDA 值见每节首行；`paper-avg` 为论文 Table 3 中 GCN/GAT 的四指标均值。

### 3.1 NoncoRNA / Dataset1（3601×71，3802 关联）

| 设置 | 变体 | AUC | AUPR | F1 | MCC |
|---|---|---|---|---|---|
| unbalanced | **DeepLDA（官方）** | 0.8763±0.0041 | 0.0744±0.0056 | 0.4963±0.0001 | 0.0000±0.0000 |
| unbalanced | GCN-only | 0.9465±0.0602 | 0.4892±0.3848 | 0.6793±0.2241 | 0.3646±0.4466 |
| unbalanced | GAT-only | 0.9694±0.0580 | 0.7902±0.3319 | 0.8862±0.1751 | 0.7994±0.2988 |
| — | *论文 DeepLDA (unbal.)* | *0.9536* | *0.8511* | *0.8612* | *0.7562* |
| balanced | **DeepLDA（官方）** | 0.6032±0.2189 | 0.5949±0.1077 | 0.4687±0.0142 | 0.2310±0.0204 |
| balanced | GCN-only | 0.8759±0.1789 | 0.8599±0.1624 | 0.7758±0.2519 | 0.6683±0.3616 |
| balanced | GAT-only | 0.8635±0.2708 | 0.8836±0.2305 | 0.8626±0.2178 | 0.7913±0.3056 |
| — | *论文 DeepLDA (bal.)* | *0.9583* | *0.8601* | *0.8625* | *0.7628* |

**结论：NoncoRNA 上未复现。** 官方 DeepLDA 在 unbalanced 下退化为"几乎全判负"
（F1≈0.50、MCC=0），balanced 下 AUC 仅 0.60。相反，**只用 GAT 分支**在
unbalanced 下 AUC 0.969 / AUPR 0.790，反而接近论文 DeepLDA 的值。

### 3.2 ncDR / Dataset2（162×31，184 关联）

| 设置 | 变体 | AUC | AUPR | F1 | MCC |
|---|---|---|---|---|---|
| unbalanced | **DeepLDA（官方）** | 0.9416±0.0902 | 0.8113±0.1778 | 0.8888±0.1115 | 0.8069±0.1803 |
| unbalanced | GCN-only | 0.9922±0.0082 | 0.9172±0.0695 | 0.9536±0.0353 | 0.9132±0.0650 |
| unbalanced | GAT-only | 0.9951±0.0046 | 0.9488±0.0414 | 0.9164±0.0654 | 0.8425±0.1234 |
| — | *论文 DeepLDA (unbal.)* | *0.9728* | *0.8572* | *0.8876* | *0.7792* |
| balanced | **DeepLDA（官方）** | 0.9050±0.0980 | 0.8409±0.1175 | 0.8517±0.1522 | 0.7545±0.2258 |
| balanced | GCN-only | 0.9597±0.0347 | 0.8709±0.1068 | 0.9422±0.0651 | 0.8927±0.1174 |
| balanced | GAT-only | 0.9771±0.0209 | 0.9483±0.0732 | 0.9064±0.0618 | 0.8174±0.1186 |
| — | *论文 DeepLDA (bal.)* | *0.9819* | *0.8687* | *0.9163* | *0.7885* |

**结论：ncDR 上部分复现。** unbalanced 下 AUC/AUPR 略低于论文（0.942/0.811 vs
0.973/0.857），F1 基本一致、MCC 更高，但 5 个种子方差很大（AUC 0.76–0.998）；
balanced 下整体偏低。GCN/GAT 单分支同样 ≥ DeepLDA。

## 4. 主要发现（对本文对比表的意义）

1. **官方代码本身有一处必崩的笔误**，修复后才能运行；发布版只实现单次行划分，
   与论文所述"balanced/unbalanced + 10 折"不完全一致。
2. **复现一致性**：ncDR 达到论文同量级；NoncoRNA 明显失败（尤其 balanced）。
3. **论文主结论未复现**：论文称 DeepLDA > GAT > GCN，但在其自身的泄漏协议下，
   GCN/GAT 单分支在两张数据集上都 ≥ DeepLDA；说明该协议的分数主要由
   "全矩阵高斯相似度"泄漏驱动，而非 DeepLDA 的 GCN+GAT 组合。
4. **阈值退化**：官方 F1/MCC 用 0.5 阈值，在极度不平衡（NoncoRNA 正例 1.5%）
   下会把高 AUC 输出压成"全负"，MCC=0；论文报告的 0.76 需换阈值/平衡口径才可能达到。

## 5. 在本文同协议（ncDR 701×101）下的 DeepLDA 基线

用 `baseline_repro_20260918/common.py` 的同一协议（701 miRNA × 101 drug、3163 关联、
`RandomLinkSplit(test=0.2)`、等量均匀负采样、5 seeds），本地移植版 DeepLDA
（`repro_deeplda_ncdr.py`，GCN+GAT 点积解码）本次实跑：

| 方法 | AUC | AUPR | ACC | F1 | MCC |
|---|---|---|---|---|---|
| DeepLDA（本地移植，同协议） | 0.6788±0.0529 | 0.7457±0.0315 | 0.6851±0.0952 | 0.7124±0.0380 | 0.3714±0.1906 |

该值与台账 §八 中此前的 0.6364/0.7207 同量级（种子/实现细节略有差异）。

## 6. 产物与复现命令

```
# 官方代码 + GCN/GAT 单分支，unbalanced
bash run_all_official.sh <gpu> <dataset 1|2>

# balanced（按论文定义补的等量负例抽样）
bash run_balanced.sh <gpu> <dataset 1|2>

# 汇总
python analyze_deeplda_official.py   # -> SUMMARY_deeplda_official.md

# 本文同协议（ncDR 701×101）
python ../baseline_repro_20260918/repro_deeplda_ncdr.py
```

文件：
- `official_codes/`（官方代码，含 1 行修复）、`Datasets/`（官方数据）
- `repro_deeplda_official.py`、`run_all_official.sh`、`run_balanced.sh`
- `results_{deeplda,gcn,gat}_ds{1,2}.csv`（unbalanced）
- `results_bal_{deeplda,gcn,gat}_ds{1,2}.csv`（balanced）
- `SUMMARY_deeplda_official.md`
- 日志：`/tmp/opencode/deeplda_official_ds{1,2}.log`、`/tmp/opencode/deeplda_bal_ds{1,2}.log`

## 7. 说明与局限

- 官方发布数据只有 raw（unbalanced）矩阵，balanced 矩阵由按论文文字抽样重建，
  与作者实际使用的不完全一致，故 balanced 数值仅供参考。
- 论文称 10 折，官方代码为单次 90/10 行划分；本复现用 5 次随机行划分近似，
  未做完整 10 折平均。
- NoncoRNA 的失败伴随很强的种子不稳定，说明该实现/协议在大而稀疏的图上不稳健；
  这不排除作者用未公开的预处理/超参获得了论文值，但**用官方代码 + 官方数据无法复现**。
