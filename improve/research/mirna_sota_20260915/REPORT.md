# incrd-hg 能否提升到 SubKNet / MPHGNN 同等级？（2026-09-15，修订版）

## 结论（修订）

前版本把"MPHGNN 复现不到 0.978"写成了不可复现，这是错的。**安装 dgl 后运行 MPHGNN 官方代码，
在它自己的数据与独立测试集上复现成功：AUC 0.9743 / AUPR 0.9382（seed 422）、0.9739 / 0.9364（seed 0），
论文报告 0.978 / 0.937。** 之前的 0.948 是本项目用 PyG 自写的结构近似版，不是官方实现。

**SubKNet 本版本地适配版未复现其报告的 0.9717：** 用 SubKNet 官方模型代码（`model.py` 的
`RW_layer` 图核 + `DenseGCNConv` 全局分支）、在 MPHGNN 同一基准上，seed 666/0 五折分别为
**0.9474 / 0.9060** 与 **0.9491 / 0.9054**。差距需谨慎归因：本轮用的是表达式/结构余弦相似度 + one-hot
特征、训练轮数 25（论文默认 200，本研究接了 early stop），不是作者原始 MDA 数据管线，因此
**不能据此断言 SubKNet 不可复现**。前版本的"0.570 固定子图核"也不是 SubKNet，只是固定特征近似。

**对 incrd-hg 的结论（含端到端实验）：目前做不到同等级。** 两条路都试了——(1) 把 gene–gene 功能元路径
作为固定特征通道加入已接受的 TabICLv2；(2) 把 MPHGNN **官方模型代码**端到端接到 incrd-hg 的图上
（gene 中介消息传递 + 元路径实例）。两者在 incrd-hg 自己的 inner 划分上都没有同时提升 MiDrug 且保住
LncDrug；端到端纯 GNN 本身（MiDrug 0.9559、LncDrug 0.9863）也低于已接受参考（0.9626 / 0.9938）。
已接受 TabICLv2 参考保持不变。

## 关键数字

| 方法 / 实现 | 数据与评测 | AUC | AUPR | 与报告 |
|---|---|---:|---:|---|
| MPHGNN 论文报告 | 自有独立测试集 | 0.9780 | 0.9370 | — |
| SubKNet 论文报告（MPHGNN 表 1） | 同一独立测试集 | 0.9717 | 0.9341 | — |
| **MPHGNN 官方代码（dgl 1.1.2+cu117）seed 422** | 同数据/同测试集，5 折 | **0.9743** | **0.9382** | 复现 |
| **MPHGNN 官方代码 seed 0** | 同上 | **0.9739** | **0.9364** | 复现 |
| 本项目 PyG 结构近似版（3 seed×5 折） | 同上，泄漏安全 | 0.9481 | 0.9057 | 差 0.03（实现差异） |
| **SubKNet 官方模型本地适配（seed 666/0，5 折）** | 同上，余弦相似度输入 | 0.9483 | 0.9057 | 差 0.023（适配差异未排除） |
| 本项目早期弱代理 GNN+TabICL | 同上 | 0.9605 | 0.9024 | — |
| **incrd-hg gene–gene 元路径通道（固定特征）** | incrd-hg 历史 inner | MiDrug −0.000007 | MiDrug −0.000105 | 联合门槛不通过 |
| **incrd-hg 端到端 gene 中介元路径 GNN（官方模型）** | incrd-hg 历史 inner | 纯 GNN MiDrug 0.9559 / LncDrug 0.9863 | 参考 0.9626 / 0.9938 | 联合门槛不通过 |

## 为什么官方 MPHGNN 能复现、我们的 PyG 版不能

官方实现与 PyG 近似版的关键差别（读 `layer.py`/`model.py`/`load_data.py` 后逐项对照）：

1. **关系图只有 4 类边**：官方 message-passing 图是 `gene-drug / miRNA-drug(benchmark) / drug-miRNA /
   miRNA-gene`，**gene–gene 只用于元路径实例生成**，不参与 GCN 传播。PyG 版把 gene–gene 也放进传播。
2. **元路径实例生成不同**：官方 `meta_path_instance` 会做"miRNA 靶基因 × drug 靶基因"的**全配对**，
   并用 `k*(k+2)+1=441` 条做补齐；PyG 版做了同构实现但训练/选模细节不同。
3. **选模与报告协议**：官方按 **训练损失** 逐 epoch 存"最优"模型，5 折里最后一折的最优模型拿去测；
   我们按验证 AUC 选模并对 5 折取平均。两者不可直接比。
4. **测试期图**：官方 `load_test_data` 用**测试正例边**建图（transductive）。PyG 版的 transductive
   变体在本机不稳定（塌陷到 0.54），说明该协议对实现高度敏感。

因此 0.948 vs 0.974 主要是实现/协议差异，而不是论文不可复现。

## incrd-hg 侧实验（不变）

gene–gene 功能相似由 Reactome 通路共现构造；`<RNA→gene1~gene2→drug>` 双向、分证据层、
通路程序版共 14 列**无标签**特征，加入已接受 TabICLv2 分支，候选 = 0.5×graph + 0.5×(graph+tiered+元路径)。

| 任务 | 参考 AUC/AP | 通道 AUC/AP | ΔAUC | ΔAP | 门槛 |
|---|---:|---:|---:|---:|---|
| MiDrug | 0.962593 / 0.927204 | 0.962586 / 0.927095 | −0.000007 | −0.000105 | 否 |
| LncDrug | 0.993750 / 0.987141 | 0.993654 / 0.986870 | −0.000096 | −0.000270 | 否 |

单特征 AUC（inner val）：固定通路共现相似度没有提供超出 RNA 基因度数/精确共享靶点的配对级信息，
把 gene–gene 打乱后基本不变。`improve/results/metapath_channel_20260915/ASSESSMENT.json`。

### 结果 B2：端到端 gene 中介元路径 GNN（MPHGNN 官方模型，incrd-hg 自身划分）

把 MPHGNN **官方模型代码**（`HeteroLinear + 3×HeteroGraphConv + 层注意力 + BiTrans 元路径聚合 +
MIL/Instance + GCN/MLP`）直接接到 incrd-hg 的图上：节点 = {RNA, drug, gene}，边 =
RNA–gene 证据、drug–gene、train 正例 RNA–drug、gene–gene（仅用于元路径实例），
在历史 inner 上训练（3 seed × 30 epoch，验证集选点）。

| 任务 | 参考 AUC/AP | GNN AUC/AP | 0.9×参考+0.1×GNN | 联合门槛 |
|---|---:|---:|---:|:---:|
| MiDrug | 0.962593 / 0.927204 | 0.955925 / 0.916714 | 0.962870 / 0.927648（3/3 正） | — |
| LncDrug | 0.993750 / 0.987141 | 0.986284 / 0.960839 | 0.993535 / 0.986628（3/3 负） | **否** |

- **纯 GNN 明显低于已接受参考**（MiDrug 0.9559 vs 0.9626；LncDrug 0.9863 vs 0.9938），
  即端到端元路径模型在 incrd-hg 自己的划分上到不了参考水平。
- 低权重融合时 MiDrug 有极小正增益，但 LncDrug 在每个 seed、每个权重上均下降；
  四个权重（0.1/0.2/0.3/0.5）联合门槛全部不通过。脚本 `repro_incrd_metapath_gnn.py`，
  指标 `incrd_metapath_gnn_metrics.csv`。

### 结果 B3：gene–gene 通道消融（诊断，MiDrug，seed 666，30 epoch）

对同一个端到端模型只改动 gene–gene 功能边（元路径的 `gene1~gene2` 部分）：

| gene–gene 设置 | 纯 GNN AUC/AP |
|---|---:|
| real（通路共现 top-10） | 0.9566 / 0.9179 |
| shuffled（随机邻居、保持条数） | 0.9566 / 0.9179 |
| none（去掉 gene–gene，仅直接 RNA 靶基因 × drug 靶基因配对） | 0.9568 / 0.9181 |

**打乱或删除 gene–gene 边对结果没有影响**，说明该模型在我们数据上完全靠"RNA 靶基因 × drug 靶基因
直接配对 + 实体特征"，gene–gene 高阶语义贡献为零。指标 `metrics_shuffled.csv`、`metrics_none.csv`。

## 诊断：是数据问题还是结构问题？

**都不是单一方面的问题，而是"该信息在我们的基准上已被现有特征库覆盖"。**

1. **不是结构问题。** 用的是 MPHGNN **官方代码**，同一套结构在他们基准上能到 0.974；
   在我们的图上也能正常训练（纯 GNN 0.956），只是低于我们自己的参考 0.9626。
2. **不是"缺数据"问题（数据类型层面）。** gene 节点、RNA–gene、drug–gene、gene–gene 都补上了，
   端到端融合仍然无效；B3 消融进一步显示 gene–gene 边完全不起作用。
3. **真正的差距点是冗余与数据质量：**
   - 我们的已接受参考是 **TabICLv2 + 36 图特征 + 26 证据分层 + DepMap 上下文**，其中
     `gene_cosine / drug_program_cosine / rna_pathway_*` 等已经把"RNA 靶基因与 drug 靶点的关系"
     编码进去了；元路径 GNN 提供的主要也是这条直接配对信息，边际增量≈0。
   - 对方用的 gene–gene 是 **HumanNet 金标准功能连接**；我们用 **Reactome 通路共现**近似，信号粗、
     噪声大，且与 RNA 基因度数高度相关（B 节单特征探针：打乱后 AUC 基本不变）。
   - drug–gene 覆盖只有 **93/216** 个药物，元路径在高覆盖药物之外基本无处可走。
4. **协议/数据划分差异也放大了观感差：** 对方 0.974 是**单次划分 + 测试正例入图的 transductive**
   协议、benchmark 正例更多（4316）；我们的 0.9626 是多次用于开发的内部划分上的强参考。

一句话：**结构已验证有效（在对方基准上 +0.014 AUC），数据通道也补齐了，但在 incrd-hg 的基准上
这些信息与已接受特征库高度冗余、且 gene–gene 代理质量不足，所以边际收益≈0，无法同等级提升。**

## 我们自己的结果有没有提升？

**没有。** 本轮所有改动相对已接受参考（MiDrug 0.962593/0.927204，LncDrug 0.993750/0.987141）都未通过
联合门槛：固定元路径通道使两任务同时略降；端到端 GNN 纯模型低于参考，融合后 MiDrug 至多
+0.0002 AUC / +0.0005 AP，但 LncDrug 每个 seed 都下降。已接受模型保持不变。

## 回答"两篇能不能复现 / incrd-hg 能否到同等级"

- **MPHGNN：能复现。** 安装 `dgl==1.1.2+cu117`（独立 venv `/data1/licheng/incrd-hg/.venv-dgl`）后
  跑官方 `main.py` 即得 0.974/0.937，与报告一致。
- **SubKNet：未完全复现，但差距可归因于本地适配。** 官方模型代码给出 0.947/0.906；
  要给出"不可复现"的结论，需要在作者原始数据/管线或其 5 个网络上跑通原实验。
- **incrd-hg 到同等级：目前做不到。** 固定特征通道（结果 B）与端到端 gene 中介元路径 GNN
  （结果 B2）在 incrd-hg 自己的划分上都没有同时提升 MiDrug 且保住 LncDrug；
  纯元路径 GNN 本身也低于已接受参考。已接受 TabICLv2 参考保持不变。

## 复现命令

```bash
# MPHGNN 官方（需 dgl cu117）
cd /tmp/opencode/mphgnn_official
/data1/licheng/incrd-hg/.venv-dgl/bin/python main.py -ep 30 -se 422 -nl 3 -lr 0.0001 -bs 256   # AUC 0.9743/0.9382
/data1/licheng/incrd-hg/.venv-dgl/bin/python main.py -ep 30 -se 0   -nl 3 -lr 0.0001 -bs 256   # AUC 0.9739/0.9364
# SubKNet 官方模型（本地适配到 MPHGNN 基准）
/data1/licheng/incrd-hg/.venv-dgl/bin/python \
  improve/research/mirna_sota_20260915/repro_subknet_official.py --seeds 666 0 1 --epochs 25 --folds 5
# incrd-hg gene–gene 元路径通道
/data1/licheng/incrd-hg/.venv-tabicl/bin/python \
  -m improve.research.mirna_sota_20260915.metapath_channel run --variants metapath --tasks MiDrug,LncDrug
```

## 限制

- SubKNet 本地输入是余弦相似度 + one-hot，非作者原始相似度；轮数 25 而非 200。
- MPHGNN 复现只跑了 seed 422/0，未穷尽 seed 与 epoch。
- 全部 incrd-hg 结果为历史 inner 开发划分复用，未访问原始测试集。
- 未实现 incrd-hg 上的端到端可学习元路径/GCN 通路，故"固定特征无效"不排除端到端版有效。

## 参考

SubKNet: JBHI 2024, 10.1109/JBHI.2024.3390092。MPHGNN: BIB 2026, 10.1093/bib/bbag013, PMC12853127。
机制核读见 [NOTES.md](NOTES.md)、[sources.json](sources.json)。
