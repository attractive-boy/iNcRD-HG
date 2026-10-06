# 前沿机制核读：SubKNet 与 MPHGNN（2026-09-15）

本轮问题的对象是"目前 miRNA 方向较强的两个做法"：SubKNet 的**逐链接子图 + graph kernel 分解**，
以及 MPHGNN 的**元路径引导高阶语义路径**。下面给出两者可实现的机制要点，以及在 incrd-hg 上的可迁移点与边界。
这不是系统综述；来源与阅读深度见 `sources.json`。

## 1. SubKNet（IEEE JBHI 2024, DOI 10.1109/JBHI.2024.3390092）

**问题**：生物网络链路预测中，同一节点在不同链接里角色不同，且已有子图方法忽略子图间共享信息。
官方代码 `github.com/mengluli-web/SubKNet`（本轮已浅克隆核读 `code/model.py`、`code/subgraph_extraction.py`）。

**机制（逐链接子图）**

1. 把异构网络拼成一张二部/同构邻接：`constructNet` 把 `<相似度, 关联; 关联^T, 相似度>` 堆叠。
   对候选对 `(i,j)` 抽取 **h-hop enclosing subgraph**（`subgraph_extraction_labeling`）。
2. 抽取时区分"纯边"（同类相似度）与"混合边"（跨类关联），把 miRNA 侧邻居标为 `4h-2 / 4h`，
   disease 侧标为 `4h-1 / 4h+1`，得到双半径式的 hop 标签 `node_labels`。
3. 每个候选对得到：子图邻接 `adj`、子图节点特征 `feat`、两端节点在子图内的位置 `sub_nodes`、标签。

**机制（graph kernel 分解）**

`RW_layer` 是可训练图核：

- 学一组 **graph filters**：特征滤波器 `z ∈ R^{F×d_in×d_out}`（`features_hidden`）与
  邻接滤波器 `adj_hidden`（`adj_hidden_norm`，上三角参数对称化后 ReLU）。
- 对每个子图：`zx = einsum("mcn,abc->ambn", z, x)`；在 `max_step` 步里同时传播
  `x ← adj·x` 与 `z ← adj_hidden_norm·z`，`t = zx * t`，对滤波器和节点取均值。
  即"把子图分解为可训练图滤波器的组合"。
- **diversity 正则** `L_diversity`：对特征滤波器均值与邻接滤波器 flatten 后做
  `relu(2 - pairwise_L2)` 的惩罚，防止滤波器塌缩。
- 同时用整图 GCN 得到全局节点嵌入，拼两端 → `outs_grl`；最终 `preds = fcs([h, h2])`，
  `softmax` 二分类。即"子图核表示 + 全局拓扑表示"。

**在 MPHGNN 同一基准上的报告值**：AUC **0.9717**、AUPR **0.9341**（MPHGNN 表 1，k=1, r=20, lr=0.01, batch=128）。
只比 MPHGNN 的 0.978/0.937 低约 0.006/0.003，说明"子图 + 图核"本身就能接近该基准上限。

**可迁移点**：incrd-hg 当前没有"逐链接局部子图"结构表示，36 列图特征只有全局度数/协作/路径计数量。
本轮的迁移实现见 `metapath_subgraph_experiment.py` 的 subgraph-kernel 分支。

## 2. MPHGNN（Briefings in Bioinformatics 2026, DOI 10.1093/bib/bbag013）

**机制**

1. **异构网络**：miRNA–gene（ENCORI 827 条）、drug–gene（DrugBank 579 条）、
   gene–gene 功能连接（HumanNet 5069 条）、miRNA–drug 抗性（ncRNADrug 7540 条）；
   节点特征为 miRNA 表达（CCLE 938 细胞系）+ 药物 SMILES（ChemBERTa）。
2. **gene similarity feature**：先在 gene–gene 图上做 GCN，把 miRNA–gene 与 drug–gene 关联
   投影到统一的 gene 空间，构造潜在 miRNA–drug 连接。
3. **元路径**：`<miRNA ⇔ gene ⇔ gene ⇔ drug>`，即 miRNA→gene、gene~gene（功能相似）、gene→drug。
   从邻居集合采样具体路径实例。
4. **Node aggregator**：沿路径实例按节点特征**拼接**，分两个方向（M→g1→g2→D 与 D→g2→g1→M），
   各自 mean 聚合，再拼接成实例表示 `z_{m,d}`。
5. **Metapath embedding projector**：对实例做 **attention** 加权聚合，得到该对的元路径级表示；
   与 drug 表示内积 + 线性层给元路径分数。
6. **Global-level**：整图 GCN 编码 + MLP（Hadamard 积）解码，给全局分数。
7. 最终 `A = f(元路径分数, 全局分数)`（式 18），BCE 训练。消融显示 GAE 与 Meta 模块都不可缺。

**注意**：其基准划分是 `S_all` 5 折（4 训练 1 验证，正负 1:1）+ 独立测试集（正负 1:2）。
报告 AUC 0.978 / AUPR 0.937 来自**独立测试集**。其代码/数据公开（`github.com/Gtaizu/MPHGNN`）。

**可迁移点**：incrd-hg 的图**没有 gene 节点**；证据库已有 RNA→gene（functional/binding/association）、
drug→gene（ChEMBL 645 边）、gene→pathway（Reactome），但**没有 gene–gene 功能连接**，
也没有显式元路径实例聚合。这正是本轮要补的两块。

## 3. 与 incrd-hg 现状的差距定位

上一轮（`specialized_single_rna_20260915/REPORT.md`）已确认：在 MPHGNN 自己的数据与独立测试集上，
本项目架构（GeneralConv×2 + `to_hetero` + 边解码器）5 seed **AUC 0.9514 / AUPR 0.8922**，
加 HistGB 0.9583/0.9147，加 TabICL 0.9605/0.9024，均低于 MPHGNN 0.978/0.937 与 SubKNet 0.9717/0.9341。

三个可能来源，本轮逐一检验：

| 差距来源 | 现状 | 本轮的检验 |
|---|---|---|
| 缺 gene 中介的高阶语义路径 | incrd-hg 图无 gene 节点、无 gene–gene 功能边 | 用 pathway 共现构造 gene–gene 功能相似图，显式做 `<RNA→gene⇔gene→drug>` 元路径特征 |
| 缺逐链接局部子图表示 | 36 列只有全局度数/路径计数 | 构造 h-hop enclosing subgraph 的图核式结构特征 |
| 评测协议不同 | 对方独立测试集 1:2、报告单值；本项目用强 TabICLv2 + 内部门槛 | 不混用数字；在 incrd-hg 自己的 inner/resplit 协议上，检验新通道是否 MiDrug↑ 且 LncDrug 不↓ |

## 4. 方法边界

- 元路径/gene 中介本身已有先例（MPHGNN、MIFNDRA、MVGSCA），"加 gene 节点"不构成创新；
  本轮目标是**评估 incrd-hg 能否借此达到同等级**，不是宣称首创。
- SubKNet 的图核是**可训练**的；若本轮先用固定图核特征（受既有"固定特征 + TabICLv2"范式与样本量限制），
  需在报告中明确这是适配版，不等价于其端到端可训练版本。
- 所有对比在同一划分内进行；不把推断 seed 当作独立生物重复。
