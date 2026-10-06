# iNcRD-HG 自有数据 MiDrug 上的 query-hollow 模块试点（去掉 lncRNA/circRNA）

目录：`improve/research/own_midrug_pilot_20260928/`　日期：2026-09-28
问题：把在 PLMF-MDA 上开发的 M1 QMP / M2 SP-RNCN / M3 MC-EASE（`../plmf_nolqo_20260923/`）
迁到论文自有 MiDrug 数据上做消融，是否同样有提升？

## 0. 结论

**有。四个 board 全部为正，8/8 个 split×model 单元 AUC 与 AUPR 全胜。**

主口径：valsel（按自有 val 划分选 epoch，报该 epoch 的测试值）；8 个单元 = 发布划分 seed 666 × 3 个模型种子
+ 本轮重划分 rs 0–4 × 1 个模型种子；epoch 300、RMSprop 1e-4、隐藏 256，与发布模型一致。

| 配置 | 单元数 | valsel AUC | valsel AUPR | last AUC | last AUPR |
|---|---|---|---|---|---|
| plain | 8 | 0.9536 | 0.9129 | 0.9499 | 0.9048 |
| + M1 QMP | 8 | 0.9553 | 0.9151 | 0.9546 | 0.9140 |
| + M2 SP-RNCN | 8 | 0.9577 | 0.9179 | 0.9559 | 0.9150 |
| + M3 MC-EASE | 8 | 0.9558 | 0.9170 | 0.9533 | 0.9136 |
| + M1+M2 | 8 | 0.9578 | 0.9185 | 0.9579 | 0.9182 |
| + M1+M3 | 8 | 0.9577 | 0.9213 | 0.9573 | 0.9210 |
| + M2+M3（不用 M1 的严格版） | 8 | 0.9586 | 0.9199 | 0.9569 | 0.9175 |
| **+ M1+M2+M3（full）** | 8 | **0.9597** | **0.9233** | **0.9596** | **0.9228** |
| 同样通道/参数量，全部不做掩蔽（对照，仅官方划分 3 个种子） | 3 | 0.9487 | 0.9150 | 0.6052 | 0.4706 |

论文表格式（MiDrug = 自有数据，去掉 lncRNA/circRNA；valsel，8 个 split×model 单元均值）：

| 模块 / 变体 | MiDrug AUC | MiDrug AUPR | ΔAUC vs 参考 | ΔAUPR vs 参考 |
|---|---|---|---|---|
| iNcRD-HG | 0.9536 | 0.9129 | — | — |
| iNcRD-HG + M1 QMP 查询掩蔽消息传递 | 0.9553 | 0.9151 | 0.0016 | 0.0022 |
| iNcRD-HG + M2 SP-RNCN 简单路径关系型共同邻居 | 0.9577 | 0.9179 | 0.0041 | 0.0050 |
| iNcRD-HG + M3 MC-EASE 掩蔽单元闭式协同回归 | 0.9558 | 0.9170 | 0.0021 | 0.0041 |
| iNcRD-HG + M1 + M2 | 0.9578 | 0.9185 | 0.0042 | 0.0056 |
| iNcRD-HG + M1 + M3 | 0.9577 | 0.9213 | 0.0041 | 0.0084 |
| iNcRD-HG + M2 + M3 | 0.9586 | 0.9199 | 0.0050 | 0.0070 |
| iNcRD-HG + M1 + M2 + M3（full） | 0.9597 | 0.9233 | 0.0061 | 0.0104 |

配对差（同 split 同 model seed，对 plain）：

| 配置 | ΔAUC valsel | ΔAUPR valsel | 胜折 | ΔAUC last | ΔAUPR last |
|---|---|---|---|---|---|
| + M1 QMP | +0.0016（se 0.0003） | +0.0022 | 8/8 · 8/8 | +0.0047 | +0.0092 |
| + M2 SP-RNCN | +0.0041（se 0.0004） | +0.0050 | 8/8 · 8/8 | +0.0061 | +0.0102 |
| + M3 MC-EASE | +0.0021（se 0.0003） | +0.0041 | 8/8 · 8/8 | +0.0034 | +0.0088 |
| + M1+M2 | +0.0042（se 0.0004） | +0.0056 | 8/8 · 8/8 | +0.0081 | +0.0135 |
| + M1+M3 | +0.0041（se 0.0004） | +0.0084 | 8/8 · 8/8 | +0.0074 | +0.0162 |
| + M2+M3 | +0.0050（se 0.0004） | +0.0070 | 8/8 · 8/8 | +0.0071 | +0.0127 |
| + M1+M2+M3 | **+0.0061（se 0.0003）** | **+0.0104** | **8/8 · 8/8** | +0.0097 | +0.0180 |

单个模块按 valsel 排序：M2（+0.0041/+0.0050）> M3（+0.0021/+0.0041）> M1（+0.0016/+0.0022），全部 8/8；
三者两两组合与 full 也都是 8/8 为正。M1 的价值主要在稳定性：它单独看 valsel 提升最小，但
"No M1 的严格版"（M2+M3）last 0.9569 明显低于 valsel 0.9586，而 M1+M2+M3 的 last≈valsel
（0.9596/0.9597）；M2+M3 再叠 M1 的最后一跳为 +0.0011/+0.0034。

逐单元结果见 `tables_pilot.md`。对照：自有 MiDrug 发布值 test 0.9564/0.9165（`repro_multiseed.py` 记录的论文口径），
9.bench 多种子重划分 0.9576/0.9222；本试点 plain（valsel 口径）0.9536/0.9129，同口径 full 已到 0.9597/0.9233。

增益来自"查询空心"结构本身，不是参数量：**保留完全相同的通道与参数量、只把查询掩蔽关掉**（对照，官方划分 3 个种子），
valsel 0.9487/0.9150（AUC 反而低于 plain 0.0043），训练后期崩溃（last 0.6052/0.4706，三个种子 0.49–0.74）；
同一 3 个单元上 full 比它高 +0.011/+0.008。这与 PLMF 轮的结论（`REPORT_NOLQO.md` §4.3）一致：
空心性是这些配对通道能稳定训练的前提。

## 1. 迁移怎么做

`run_own.py` 从 `run_nl.py` 原样复制方法与训练循环（M1/M2/M3、`--check_enc`、`--check_inv` 全保留），
只替换数据与划分层：

- 图：只保留 miRNA / drug 两类节点与 MiMi / MiDrug / DrugDrug 三种关系（lncRNA、circRNA 全部切掉）。
  节点特征用发布图里的行 min-max 表达谱（938 维）与 ChemBERTa（768 维）。
- 训练对/验证对/测试对：发布划分直接读 `3.heter_data/MiDrug_{train,val,test}_data.pth`；
  重划分 `--split rs` 从 5394 条正例里按 60/20/20 重排、每个划分独立抽样 2 倍负例（与
  `RandomLinkSplit(num_val=.2, num_test=.2, neg_sampling_ratio=2.0)` 同义）。
- 消息图：train/val 用训练正例，test 用训练+验证正例（发布协议如此，训练图与测试图相差 1078 条边）。
- 忠实复刻发布图的细节：发布图里 MiMi/DrugDrug 的自环被计了两次（原始矩阵一次 + `AddSelfLoops` 一次），
  且训练图有 2 对重复正例边（3238 条边 / 3236 唯一）。本实现把同配邻接与关联邻接都按**重数矩阵**处理
  （`Sr1 = Sr + 2I`，M1 的 q 取重数），因此 dense 编码器与 PyG 编码器逐位一致。

## 2. 空心性与等价性验收

| 检查 | 结果 |
|---|---|
| dense 编码器 = PyG 编码器（`--check_enc`） | max diff 7.6e-5 / 1.5e-4（float32 + CUDA scatter 噪声） |
| 掩蔽状态 = 显式删掉查询边重算（`--check_enc`） | max diff 7.6e-6 / 3.0e-7 |
| full 训练完：logit(整图+掩蔽) − logit(删边重算)（`--check_inv 100`） | max 3.3e-6，mean 8.0e-7 |
| M3 掩蔽 EASE = 删边后 EASE（`--check_enc`） | 一致（`ease_side` 为空，无外部核） |

## 3. 与 PLMF 轮的差异（如实说明）

1. **M4 SeqSide 没有做**：自有发行只有 miRNA 名字与 Ensembl 前体序列，没有 PLMF-MDA 的成熟序列，
   seed7/seed6 与 k-mer 核无法按同一定义复现；M4 的 pre/fam 可只从名字派生，属下一步。
2. **基础特征不同**：PLMF 用成熟序列 k-mer；这里用发布表达谱，属自有基准的原始输入。
3. **相似图不同**：PLMF 是 kNN(10) 余弦图；这里是发布 |Kendall|>0.3 阈值图（去掉自环后
   miRNA 平均度 ~25、drug ~2）。
4. **epoch 选择**：这里用自有 val 划分选 epoch（valsel），也报 last（与论文 last-epoch 口径可比）；
   plain 的 last 0.9499/0.9048 低于发布 0.9564/0.9165，说明原实现的最优 epoch/口径另有差异，
   但四个配置与发布值都对同一套试验做配对比较，结论不受此影响。
5. **规模**：test 只有 3234 对（1078 正例），单折噪声大；本报告已用 8 个单元配对，
   8/8 同向（符号检验 p = 3.9e-3）。

## 3.5 为什么自有数据上的增幅小于 PLMF（量化）

1. **查询边泄漏的规模小得多**。同一 9 维结构探针（度/3-路径/内容桥接路径 → LR）：
   含查询边计算 vs 简单路径口径，own MiDrug 0.895→0.958 / 0.900→0.961 / 0.904→0.956（rs 0-2，差 0.055–0.065），
   PLMF MDR 是 0.79→0.936（差 0.146，`REPORT_NOLQO.md` §1）。模块要修的"训练看得见答案"问题在自有数据上只有约四成规模。
2. **低度正例占比**：自有测试正例里消息图度 0 的 miRNA 只占 1.8%、度 ≤1 占 6%，度 ≥8 占 66%；
   PLMF MDR 对应是 11% / 21% / 34%。M1 的收益本来就集中在低度（PLMF §5 零度桶 rank loss 0.131→0.051），
   自有数据几乎没有这块失分（plain 零度桶 0.359，但只占 1.8%，单模型种子下仅 19 个正例）。
3. **基线本身更强、图更密**：own plain 0.9536（PLMF 0.9242）；节点特征是 938 维表达谱而非 k-mer 的 80 维，
   MiMi 是稠密 Kendall 图（平均度 25）而非 kNN(10) 稀疏图。M2/M3 都属于"在图上补结构证据"的通道，
   图稠密、特征强时增量自然小。逐桶 rank loss（官方划分 s666 m0）：full 在 8-15 桶 0.1145→0.1089、
   16+ 桶 0.1485→0.1204；而 M1 单独在 1 度、2-3 度桶反而变差（低度正例太少，救不回来）。
4. **测试集规模**：own 每折 1078 正例 / 3234 对，PLMF 每折约 2000+ 对；小测试集上同样的效应更容易被噪声吃掉，
   所以可见 delta 更小，但 8/8 个单元同向说明方向稳定。

## 3.6 对照实验：把 MiMi 从稠密 Kendall 图换成 kNN(10) 稀疏图

`--rel_knn 10`：把 MiMi 换成 |Kendall| top-10 的 kNN 图（平均度 16.1，与 PLMF MDR kNN(10) 的 14.1 同级；
DrugDrug 保持发布图不变）。同样 8 个 split×model 单元、同样阶梯（`grid_knn.txt`，`tables_knn.md`）：

| 配置 | ΔAUC valsel（稠密图，参考） | ΔAUC valsel（kNN10） | ΔAUPR valsel（稠密图） | ΔAUPR valsel（kNN10） |
|---|---|---|---|---|
| plain 绝对值 | 0.9536 | 0.9531 | 0.9129 | 0.9120 |
| + M1 | +0.0016 | +0.0018 (7/8) | +0.0022 | +0.0020 |
| + M2 | +0.0041 | **+0.0048** | +0.0050 | **+0.0059** |
| + M3 | +0.0021 | **+0.0028** | +0.0041 | **+0.0048** |
| + M2+M3 | +0.0050 | **+0.0054** | +0.0070 | **+0.0077** |
| full | +0.0061 | **+0.0072** | +0.0104 | **+0.0120** |

读法：**稀疏化几乎不动基线（0.9536→0.9531），但所有模块的增量都变大**（full +0.0061→+0.0072 AUC、
+0.0104→+0.0120 AUPR，8/8）。方向与"稠密相似图本身已经包含了一部分 M2/M3 要补的配对结构证据"一致；
但量级只解释差距的一小部分（≈0.001），剩下的差距来自低度正例占比（自有 2% vs PLMF 11%）与
基线本身（0.953 vs 0.924，特征更强、每 miRNA 关联更多）。换言之：自有数据上增益小，主要是"可修的问题少"，
不是模块失效。

## 4. 复现

```bash
cd improve/research/own_midrug_pilot_20260928
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
# 单次运行（plain / full）
$PY run_own.py --split official --seed 666 --model_seed 0 --tag P_plain --epochs 300
$PY run_own.py --split official --seed 666 --model_seed 0 --tag P_full --epochs 300 \
    --enc mask --exact --qis --qis_ext --qncn --ncn_rel co,knn,dknn --ease 10,50
# 验收
$PY run_own.py ... --check_enc           # dense==PyG；掩蔽==删边
$PY run_own.py ... --check_inv 100       # full 的端到端空心性
# 全部运行 + 汇总（grid_final.txt = 四配置主表；grid_modules.txt = M1/M2/M3 单独与两两组合；
#                       grid_knn.txt = kNN(10) 稀疏图对照）
./run_grid.sh grid_final.txt && ./run_grid.sh grid_modules.txt && ./run_grid.sh grid_knn.txt
$PY summarize.py P_ | tee tables_pilot.md && $PY summarize.py K_ | tee tables_knn.md
```

产物：`results/{tag}_{split}_s{seed}_m{model_seed}.csv`（+ `_scores.csv`、`_curves.json`、`_args.json`）、
`logs/`、`tables_pilot.md`。文件：`run_own.py`（唯一入口）、`grid_final.txt`、`run_grid.sh`、`summarize.py`。
