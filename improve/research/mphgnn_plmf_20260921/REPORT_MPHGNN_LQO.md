# LQO 放到 MPHGNN 上：留出查询边消息传递是不是通用协议改进

目录：`improve/research/mphgnn_plmf_20260921/`　日期：2026-09-23
脚本：`run_mphgnn_plmf_lqo.py`（在 `run_mphgnn_plmf_val.py` 基础上加 `--lqo K`）、`run_lqo_experiment.sh`

## 0. 目的

在 iNcRD-HG 本体上，LQO（leave-query-out 消息传递）把 base 从 0.9227 提到 0.9348（MDR 5cv）。
本实验回答一个公平性问题：**LQO 是不是只对 iNcRD-HG 有效的"调参"，还是任何用 miRNA–drug 消息图的
GNN 都能受益的通用协议修正？** 选同图、同划分、同基线的 **MPHGNN**（DGL，官方 model/layer 原样）。

## 1. 设置

- 数据 / 划分：PLMF-MDA MDR / MDS，论文 5 折 seed 42（`plmf_splits.get_split`，与基线复现逐位相同）。
- 无泄漏协议（与 iNcRD-HG 口径对齐）：`--val_frac 0.1`（从训练对分层切 10% 选轮）+ `--global_graph train`
  （全局 GCN-GAE 分支的 edge_index 固定为训练正例图，去掉官方按 batch 建图带来的"批内度结构"泄漏）。
- 唯一变量：`--lqo 5`（训练正例分 5 组，第 j 个 pass 的消息图只含其它组正例，损失只取第 j 组；测试用全部训练正例图）。
- 模型 / 超参照官方：hidden 128、3 层 HeteroGCN、BiTrans、k=20、batch 256、Adam 1e-4、50 epoch。

## 2. 结果（5 折 seed 42，均值）

| 数据集 | 口径 | base（无 LQO）AUC / AUPR | + LQO AUC / AUPR | Δ |
|---|---|---|---|---|
| MDR | valsel | 0.9124 / 0.9073 | **0.9321 / 0.9230** | **+0.0197 / +0.0156（5/5）** |
| MDR | test-best | 0.9129 / 0.9077 | **0.9324 / 0.9234** | +0.0196 / +0.0157（5/5） |
| MDS | valsel | 0.9144 / 0.9163 | **0.9397 / 0.9362** | **+0.0253 / +0.0199（5/5）** |
| MDS | test-best | 0.9146 / 0.9163 | **0.9401 / 0.9365** | +0.0255 / +0.0201（5/5） |

逐折 valsel AUC：
- MDR base `[0.9051, 0.9195, 0.9101, 0.9170, 0.9104]` → LQO `[0.9297, 0.9361, 0.9307, 0.9379, 0.9262]`
- MDS base `[0.9138, 0.9134, 0.9205, 0.9210, 0.9033]` → LQO `[0.9438, 0.9362, 0.9449, 0.9435, 0.9299]`

## 2b. 稳健性：研究分组划分（study-grouped）

随机 5 折会让同一篇论文报道的关联可能分到训练和测试两侧，"共现在某研究里"在测试时也部分成立。
为排除"LQO 的增益只是利用了同源传递性"，把划分换成 **primary-study grouped**（每个正例归到其最具体的
PMID，同研究对绝不跨训练/测试边界；`plmf_splits.py::study_split_primary`，与 `plmf_xrel_20260922/study_split2.py --group primary` 同构造）。

| 数据集 | 口径 | base AUC / AUPR | + LQO AUC / AUPR | Δ |
|---|---|---|---|---|
| MDR | valsel | 0.8230 / 0.8392 | **0.8659 / 0.8666** | **+0.0429 / +0.0273（5/5）** |
| MDR | test-best | 0.8249 / 0.8409 | **0.8672 / 0.8672** | +0.0423 / +0.0263（5/5） |
| MDS | valsel | 0.8579 / 0.8746 | **0.8912 / 0.9005** | **+0.0333 / +0.0259（5/5）** |
| MDS | test-best | 0.8588 / 0.8752 | **0.8970 / 0.9019** | +0.0382 / +0.0267（5/5） |

- 换成更难的研究分组划分后，base 从 0.912/0.914 掉到 0.823/0.858（同源传递性被切断），
  但 **LQO 的增益反而从 +0.020/+0.025 扩大到 +0.043/+0.033**，两数据集仍 5/5 折、AUPR 同向。
- valsel 选轮：base 停在 epoch 1–9，LQO 到 epoch 19–50；base 末轮掉到 0.755/0.783，LQO 末轮 0.866/0.891。
- 结论：LQO 的增益**不来自同研究传递性**，而是消除"目标边在图里"的捷径；划分越难、捷径越有害，去掉它收益越大。

## 2c. 稳健性：冷药物（cold-drug / inductive）划分

再换到更难的 **cold-drug** 划分（`plmf_splits.py::cold_drug_split`，与 `plmf_frontier_20260922/cold_probe.py --protocol ldo` 同构造）：药物按正例度分箱、`StratifiedKFold(5)` 切成 5 折，**测试药物在训练集中零正例**；test = 测试药物的正例 + 等量 (miRNA, 测试药物) 零采样，train = 其余药物的正例 + 等量零采样。

| 数据集 | 口径 | base AUC / AUPR | + LQO AUC / AUPR | Δ |
|---|---|---|---|---|
| MDR | test-best | 0.7269 / 0.7247 | **0.7331 / 0.7298** | +0.0062 / +0.0051（4/5） |
| MDR | valsel | 0.7045 / 0.7000 | **0.7074 / 0.7074** | +0.0029 / +0.0074（4/5） |
| MDR | last(50) | 0.4776 / 0.5269 | **0.7076 / 0.7076** | +0.2300 / +0.1807（5/5） |
| MDS | test-best | 0.7153 / 0.7078 | **0.7363 / 0.7395** | +0.0210 / +0.0318（5/5） |
| MDS | valsel | 0.6878 / 0.6859 | **0.7049 / 0.7076** | +0.0171 / +0.0216（4/5） |
| MDS | last(50) | 0.5008 / 0.5470 | **0.7073 / 0.7092** | +0.2065 / +0.1621（5/5） |

- 冷药物上 base 的选轮在 epoch 1–9（val 是暖药物，选得偏早），用测试最优轮口径：MDR +0.0062（4/5）、MDS +0.0210（5/5），**仍非负**。
- 最刺眼的是稳定性：base 把暖药物的邻居捷径背下来后，在冷药物上 **末轮塌到 0.48 / 0.50（接近随机）**；LQO 末轮稳在 0.71（MDR 5/5、MDS 5/5）。说明 LQO 学到的是可迁移的结构，而非"记住训练药物邻域"。
- 读数：真·未见实体的划分下增益会缩小（符合预期），但**不会变负**；随机 5 折 +0.02~0.025、研究分组 +0.033~0.043、冷药物 +0.006~0.021（test-best），三种划分方向一致。

## 3. 读数

1. **LQO 在完全不同的 GNN（DGL / 元路径 / MIL）上同样显著正向**：随机 5 折 MDR +0.0197、MDS +0.0253 AUC，
   两数据集 5/5 折，AUPR 同向；研究分组划分下增益更大（+0.0429 / +0.0333，§2b），冷药物划分仍非负（+0.006~+0.021，§2c）。
   说明它是**协议层的通用修正**，不是 iNcRD-HG 特有的技巧，也不是同源传递性或暖药物邻域记忆，论文里可以作为"目标边泄漏"的独立贡献写。
2. **选轮 epoch 的对比最能说明机制**：base 的 valsel 轮在 epoch 5–8 就早停（消息图含查询边 ⇒ 立刻记住"边在图里⇒正"），
   LQO 则在 epoch 42–50 仍持续改善（`last` 与 `best` 几乎相同：MDR 0.9323 / MDS 0.9400）。base 的末轮掉到 0.85 左右
   （last Δ 高达 +0.08/+0.09），正是目标边捷径导致的训练/测试分布偏移。
3. 与 `REPORT_MPHGNN_ON_PLMF.md` 的旧口径对照：MPHGNN 论文 5 折口径 0.9237/0.9136（MDR）、0.9279/0.9215（MDS）；
   无泄漏口径 0.9205/0.9092、0.9246/0.9168。本目录 `--global_graph train` 的 base（0.9129/0.9146）与之同量级，
   LQO 后 0.9324/0.9401 明显高于旧口径——**同一份外部图，仅改变"查询边是否进消息图"就能拿到这么多**。

## 4. 复现

```bash
cd improve/research/mphgnn_plmf_20260921
PY=/data1/licheng/incrd-hg/.venv-dgl/bin/python
# base
$PY run_mphgnn_plmf_val.py --dataset MDR --split 5cv --seed 42 --val_frac 0.1 --global_graph train --device cuda:0
# + LQO
$PY run_mphgnn_plmf_lqo.py  --dataset MDR --split 5cv --seed 42 --val_frac 0.1 --global_graph train --lqo 5 --device cuda:0
# 两数据集 4 个作业一起：bash run_lqo_experiment.sh
# 研究分组划分：bash run_study_experiment.sh
# 冷药物划分：  bash run_cold_experiment.sh
```
产物：`results/mphgnn_kmer_val_gg-train_{MDR,MDS}_5cv_seed42.csv`（base）、
`results/mphgnn_kmer_val_gg-train_lqo5_{MDR,MDS}_5cv_seed42.csv`（+LQO），日志 `logs/lqo_{base,5}_*.log`；
study / cold 划分同名 `_study_` / `_cold_` 文件与 `logs/lqo_{study,cold}_*.log`。

## 5. 注意

- LQO 需要 `--global_graph train`：官方默认的 batch 图把查询对自身连成边（`mphgnn_model.py:98`），
  那一支无法"留出查询边"，只能用固定训练图替换后 LQO 才干净。论文里应写明这一点。
- 只跑了 seed 42；若要写进正文，建议补 seed 43/44 与官方 80/20（`single`）协议。
