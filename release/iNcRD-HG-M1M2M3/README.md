# iNcRD-HG + M1 + M2 + M3

三数据集最终模型代码（MiDrug / MDR / MDS）。在原始 iNcRD-HG 编码器上叠加三个
query-hollow 模块，训练图只含共享训练正例，验证集 AUC 选轮，测试只在模型选定后
打分。本目录是干净、自包含的复现包：模型、训练/评测入口、显式删边核验脚本和冻结
输入（`data/frozen/`）都在这里，不依赖仓库其余实验目录。

## 模型

**底座**：原始发布的 iNcRD-HG，提交 `eb8de624`（`provenance/SOURCE.json`）。
`GeneralConv` 两层、hidden=256、`to_hetero(aggr='mean')` 关系平均、leaky ReLU(0.1)。
MiDrug 保留发布的 lncRNA / miRNA / drug 三类节点、表达谱 / ChemBERTa 特征与全部
辅助关系（9 类）；MDR/MDS 没有发布表达谱与 lncRNA 辅助图，使用 80 维 2+3-mer、
原版 ChemBERTa-zinc 768 维、对称余弦 kNN(10)（4 类关系）。

**M1 QMP — query-edge message passing 修正**：对每个查询对 (m, d)，两层消息传递
的端点表示等于「删除该查询边后重算」的结果。实现是精确修正而非近似：第 1 层预激活
减去被删边消息，再将该位移沿第 2 层传播，包括同类关系的自环重数与目标节点所有
`lin_self` 项；关系缩放系数使用目标节点的实际入关系数。

**M2 SP-RNCN — 简单 2-hop 路径的中间节点汇聚**：co（共享邻居的 miRNA 侧与 drug
侧）、knn、dknn 三组中间节点状态按路径权重汇聚，权重中扣除查询边；中间节点状态同
样修正到删边后的取值（利用 M1 给出的第 1 层位移）。`ncn_dim=64`。

**M3 MC-EASE — masked-cell EASE**：两侧掩蔽单元 EASE 协同分数，λ=10,50，
`sign(x)·log1p(10|x|)`；查询单元按 rank-2 Woodbury downdate 从 Gram 逆中剔除。
无 M4 序列关系，无外部核。

**融合头**：`[端点(512) | M2 汇聚(4×64) | 查询不变统计 qis(17) | EASE(4)] → MLP(1280→512→256→1)`。
其中 qis 含简单的 3-/5-path、桥接关系统计、掩蔽邻居度与内容余弦等查询不变特征。

训练配方：RMSprop(lr=1e-4, alpha=0.99)、BCE、全批量、300 epochs、无早停；
checkpoint 由验证集 AUC 选择，测试指标只计算一次。

## 数据与协议

| 数据集 | 划分 | 种子 | 正负比 | 目标图 |
|---|---|---|---|---|
| MiDrug | 发布 train/val/test 固定划分 | 42/43/44 | 1:2 | 训练正例 |
| MDR | seed=42 外层 5 折（训练内再分层留 10% 验证） | 42 | 1:1 | 训练正例 |
| MDS | 同上 | 42 | 1:1 | 训练正例 |

所有目标关联图、GIP 与结构统计只从共享训练正例构建；验证/测试正例不进入消息图。
指标为 AUC 与 AP（`average_precision_score`），阈值 0.5。冻结输入逐文件 SHA256 校验
（`data/frozen/manifest.json`），构建来源见 `data/frozen/provenance/`。

## 运行

环境：Python 3.8 / torch 1.13.1+cu117 / PyG 2.6.1（见 `requirements.txt`）。

```bash
# 0) 模块核验：dense 编码器 == PyG；显式删边 == query-hollow（M1/M2/M3/最终 logit）；
#    留出标签扰动不改图与 pair 通道；梯度有限
python check_modules.py --device cuda:0

# 1) 单次运行（默认输出到 results/）
python run.py --dataset MiDrug --fold 0 --seed 42 --device cuda:0
python run.py --dataset MDR --fold 3 --seed 42 --device cuda:1

# 2) 论文固定 13 次运行（MiDrug×3 seeds；MDR/MDS×5 folds），双卡，断点续跑
python launch.py
```

`run.py` 产出 `{tag}_predictions.npz`（val/test 概率）、`{tag}_curves.json`、`{tag}.pt`
与 `{tag}.json`（选轮、指标、配置、划分哈希）。`launch.py` 跳过已有结果的组合。

## 参考结果

`incrd_m123`（本包模型），运行均值 ± 样本标准差：

| 数据集 | AUC | AUPR | 原始 iNcRD-HG AUC / AUPR | ΔAUC / ΔAUPR |
|---|---|---|---|---|
| MiDrug（3 seeds） | 0.9577 ± 0.0002 | 0.9225 ± 0.0003 | 0.9534 / 0.9137 | +0.0043 / +0.0088 |
| MDR（5 folds） | 0.9483 ± 0.0057 | 0.9437 ± 0.0083 | 0.9243 / 0.9171 | +0.0240 / +0.0266 |
| MDS（5 folds） | 0.9509 ± 0.0055 | 0.9490 ± 0.0042 | 0.9317 / 0.9286 | +0.0192 / +0.0205 |

注：GPU 稀疏算子存在运行间非确定性；同一配置重复运行 AUC 波动约 1e-4 量级，
最佳轮次波动数轮。上表是固定 seeds/folds 的均值，测试集不参与任何选择。

## 目录

```
iNcRD-HG-M1M2M3/
├── incrd_hg_m123/
│   ├── config.py         固定模型配置（hidden/layers/ncn_dim/ease lambda）
│   ├── encoder.py        原始 GNNEncoder + 稀疏稠密化 + M1 精确删边修正
│   ├── pair_channels.py  ease_masked / PairStruct（M2 qncn、M3 ease、qis）
│   ├── model.py          融合头
│   ├── adapter.py        数据 → 图 → pair 通道 → 模型
│   ├── data.py           冻结输入加载与训练图构建（哈希校验）
│   └── engine.py         验证选轮 + 测试评测
├── check_modules.py      显式删边核验（M1/M2/M3 验收）
├── run.py                单次训练/评测入口
├── launch.py             固定 13 次运行
└── data/frozen/          冻结图、划分与 provenance（SHA256 见 manifest.json）
```

## 复现与验收

- 网络结构逐一对照被锁定实现：同一 state_dict（99 个参数张量）可直接互载，
  前向 logits 差 ≤ 3e-6（float32 噪声）；pair 通道完全相同。
- `check_modules.py` 在三个数据集上对 6 个训练正例做显式删边对照，编码器端点、
  M2 汇聚、qis/EASE、最终 logit 全部通过（容差 3e-4）；另检查留出标签扰动不改图。
- 冻结输入的 SHA256 在 `load_fold` / `load_graph` 中强制校验，防止静默换数据。

## 来源

- 原始 iNcRD-HG 代码与数据：`attractive-boy/iNcRD-HG` 提交 `eb8de624`。
- 模块定义与超参来自本项目 2026-09 的 PLMF-MDA 实验（QMP / SP-RNCN / MC-EASE）
  以及 2026-10-04 的三数据集基线统一评测。
- 冻结输入的文件哈希、节点/关系规模、每个划分的正负样本数见
  `data/frozen/manifest.json`；原始 manifest 与序列 provenance 见
  `data/frozen/provenance/`。
