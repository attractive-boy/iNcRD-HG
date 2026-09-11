# 首轮：三个独立特征实验

**当前筛选条件已按用户最新要求改为：原 GNN、seed=666，只要求两个数据集的 val AUC/AUPR 全部提升，test 不参与判断。PCA64＋三个表达统计量已符合条件。**

后续[表达尺度机制核验](results/expression_scale_mechanism_seed666_20260909/FINDINGS.md)已完成：原始均值、标准差能数值反演完整归一化表达，但 PCA-only 也能取得双数据集 val 提升；当前消融不支持把组合收益全部归因于统计量补回尺度。
见 [当前 val 对照表](results/first_round_features_20260909/VAL_SEED666.md)。下文保留已完成首轮的原始实验协议供溯源。

本轮已完成并核验 150/150 次运行。
[结果解读](results/first_round_features_20260909/FINDINGS.md)、
[完整报告](results/first_round_features_20260909/REPORT.md)、
[Excel 汇总](results/first_round_features_20260909/首轮三项特征实验.xlsx)。

入口为 `python -m improve.first_round_features`。本轮只改变输入信息，使用原两层
GeneralConv GNN、一个 64 维隐层的 MLP 和 ExtraTrees；不增加 AE、Pair-GNN、
注意力层、额外投影层、训练损失或 DropEdge 变体。

| 实验 | 对照 | 处理 | 保持固定 |
|---|---|---|---|
| 药物池化 | 同一 ChemBERTa 的普通 token 均值 | 按 attention mask 排除 padding 后取均值 | checkpoint、tokenizer、截断、padding 长度、维数、RNA 特征、图 |
| 表达压缩与统计量 | 原 RNA 表达输入 | PCA64 + 原表达均值、标准差、零值比例，共 67 维 | 药物特征、图、网络结构与训练参数 |
| 连续边权 | 现有边全部权重为 1 | LncLnc/MiMi/DrugDrug 的绝对相似度，按目标节点保留每种关系的原入边总质量 | 边集合、节点向量、其他关系的单位边权、网络结构与训练参数 |

两个表达/边权实验共用 `base`。池化必须单独比较 `drug_masked − drug_unmasked`：
本机可用模型为 ChemBERTa-77M-MLM，输出 384 维，而历史原始药物特征为 768 维。
因此共有五个运行臂、三个配对对照，不计算 `drug_masked − base` 的池化结论。
表达臂同时包含压缩与统计量，首轮不能拆分二者的贡献。

## 固定数据与公平输入

沿用 `incrd-run/3.heter_data` 的验证/测试样本及标签。真实数据检查发现：

- LncDrug 训练与验证有一个正例重叠；只删除其训练副本。
- MiDrug 训练中有两条重复正例；每个训练 pair 只保留第一次出现。

这些清理只写入新的实验缓存，不修改原始数据文件；每条删除记录、原始行号、
pair、标签及原因保存在 `inputs/<task>/audit.json`。清理后所有实验一次性冻结
同样的训练样本、负例和图。训练目标关系的正向/反向边同步处理，其他关系保持原样。

三个模型在验证和测试时都只使用训练图。原发布测试图额外包含验证正边，本轮不使用
这些边。原始划分的指纹和本轮实际图的指纹分别保存，避免与历史结果混用。

所有模型读取同一份节点特征缓存和同一组现有边权：

- MLP/ExtraTrees：拼接完整目标 RNA 向量、完整药物向量，以及 14 个固定的训练图统计。
  不将新特征压缩为几个余弦分数后再交给简单模型。
- GNN：直接读取所有节点类型的向量和异构训练图，沿原来的两层消息传播使用边权。
- 图统计只来自当前训练图，包含度、RNA/药物邻域支持、三跳路径、跨 RNA 支持等；
  不读取阈值之外的稠密相似度。边权实验在简单模型中也有二值/连续两个版本。
- 训练图统计固定用五折交叉构造，去掉待预测样本所在折的正边；验证和测试统计
  使用完整训练图。自环不会在标签派生的 pair 统计中读到待预测边自身。
- 表达输入标准化、PCA、PC 标准化及统计量标准化只拟合训练节点；pair 标准化
  只拟合训练样本。保存全部变换参数、拟合节点 ID 和指纹。

简单模型和 GNN 对结构信息的表示不同；本轮以每个模型内的配对增益判断输入作用，
不把两个模型的绝对分数差当作纯架构收益。

## 训练与判读规则

两个任务：LncDrug、MiDrug。固定种子：666、0、1、2、3。
矩阵为 2 个任务 × 5 个运行臂 × 3 个模型 × 5 个种子 = **150 次运行**。

GNN 保留原设置：LncDrug 为 64 通道/min 聚合/RMSprop 0.0005/100 轮，
MiDrug 为 256 通道/mean 聚合/RMSprop 0.0001/300 轮。MLP 为一个 64 维 ReLU
隐层，Adam 0.001，weight decay 0.0001，全批训练，轮数与同任务 GNN 一致。
两者使用普通 BCEWithLogits，无提前停止；从所有训练轮次中按
`(validation AUC + validation AUPR) / 2` 最大值选一个检查点，同分取最早轮次。
两个测试指标均来自这同一个检查点，测试只在选模结束后运行。

ExtraTrees 固定 500 棵树、min_samples_leaf=3、max_features=sqrt，其他设置
写入协议，单次拟合，不搜索超参数。种子也不根据测试成绩挑选。

预先固定的描述性筛查规则为：平均配对验证选模分数增益 > 0，且至少 4/5 个种子
为正。分别记录简单模型与 GNN 是否通过；这不是统计显著性检验。报告所有三项
实验的验证与测试差值、每个种子的结果、配对标准差及正向种子数。
同一划分的多种子不等于多次独立数据验证，本数据也已用于历史实验。

## 执行

使用已有 `ncRNAdrug` 环境，在仓库根目录运行。以下 checkpoint 是本机现有缓存，
不会下载或部署新模型：

```bash
python -m improve.first_round_features prepare \
  --out-dir improve/results/first_round_features_20260909 \
  --model-dir /home/zhourongxian/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MLM/snapshots/ed8a5374f2024ec8da53760af91a33fb8f6a15ff \
  --device cuda:0

python -m improve.first_round_features run \
  --out-dir improve/results/first_round_features_20260909 \
  --task LncDrug --device cuda:0

python -m improve.first_round_features run \
  --out-dir improve/results/first_round_features_20260909 \
  --task MiDrug --device cuda:1

python -m improve.first_round_features summarize \
  --out-dir improve/results/first_round_features_20260909
```

两个任务的 `run` 可在不同终端同时执行。`--models` 和 `--seeds` 只用于执行冻结矩阵
的一个子集，不能追加协议之外的种子。每次运行有独立锁和完成标记，重启会核验
已完成的文件再跳过。代码或缓存与冻结协议不一致时拒绝混入已有结果。

执行检查应在另一个输出目录用 `prepare --smoke` 创建协议：保留所有任务/模型/臂，
只用种子 666、神经模型各 2 轮和 ExtraTrees 8 棵树。它不用于解释性能。

```bash
python -m unittest improve.test_first_round_features improve.test_network_feature_ablation -v
```

`summarize` 要求矩阵完整，并重新核验输入/产物哈希、样本对、标签、保存预测所算
指标、最大验证分数对应的检查点和固定轮数。结果为 `REPORT.md`、
`首轮三项特征实验.xlsx`、`summary.csv`、`paired_deltas.csv`、`per_seed.csv`、
`evidence.csv` 和 `VALIDATION.json`，同时保留源代码快照、模型和训练轨迹。
