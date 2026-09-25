# 已下载数据的共享证据与表达通路适配器

本轮将用户提供的 DepMap 25Q2、LncTarD 2.0 和 human MTI 表接入已有图上下文 TabICLv2。改动位于 RNA–基因–通路特征适配器；TabICLv2 神经权重保持冻结。最终结果见 [实验结论](TABICL_PATHWAY_CONTEXT_FINDINGS.md)。

## 数据输入

原始下载文件位于 `/data1/licheng/incrd-hg/data_external/pathway_20260913/`，不覆盖它们。构建代码为 [pathway_context_data.py](pathway_context_data.py)，新输入单独冻结在 [tabicl_pathway_context_inputs_20260913](results/tabicl_pathway_context_inputs_20260913/INPUTS.json)。

|实体|上一轮严格筛选后的基因证据覆盖|本轮任一证据覆盖|本轮功能证据覆盖|有变化的匹配表达谱|
|---|---:|---:|---:|---:|
|miRNA|372 / 605|594 / 605|501 / 605|605 / 605|
|lncRNA|64 / 1,322|914 / 1,322|317 / 1,322|1,321 / 1,322|
|Drug|93 / 216|93 / 216|沿用已核验 ChEMBL 机制靶点|不适用|

本轮 lncRNA 覆盖增加同时来自已下载 ENCORI 的较低支持证据和新上传的调控表。不能把全部覆盖增量归因于新下载数据。ENCORI 单独提供 858 条 lncRNA 的结合证据；NEAT1 的低于 2 支持数据和 MALAT1 的低于 3 支持数据仍未完整取得。

数据包含 21,318 个基因和沿用 Reactome 97 定义的 1,663 个通路。DepMap 文件有 1,684 个模型、19,205 个基因列，其中 19,197 列映射到蛋白编码 Ensembl 基因；一对列映射到同一基因后平均，得到 19,196 个表达基因。原 RNA 表包含相同但顺序不同的 938 个 ACH ID，实际匹配 937 个；缺失的 `ACH-000534` 未找到官方别名映射。所有矩阵按 ID 重新排序，不按行位置拼接。

三类证据的定义如下，两任务共用同一层级：

- `functional`：miRTarBase 的 `Functional MTI`；LncTarD 中 `SearchregulatoryMechanism` 不为 `expression association` 的机制记录。后者可包含间接调控，不能一概称为直接结合。
- `binding`：ENCORI 结合/相互作用；miRTarBase 弱功能记录中有 CLIP/CLASH 的记录。
- `association`：LncTarD 表达关联；miRTarBase 其他弱功能记录。该层不宣称已验证直接靶向。

`hsa_MTI.csv` 共 4,004,967 行，其中仍含 488 条非人类 miRNA–人类靶基因记录。物种两端均限制为 hsa，并排除 4,666 条人类配对的 Non-Functional 记录。没有把非功能性记录加入正相互作用，也没有用它们补充 RNA–drug 负标签。去重按项目 RNA、Ensembl 基因、证据层进行，重复位点/重复行不会增加模型边权。

LncTarD 共 8,360 行，按机制字段正好分为 5,531 条功能机制记录和 2,829 条表达关联。`positively-E` 中的 E 不用于判断证据强弱。原文件存在 139 个非法 UTF-8 字节；原压缩包保留并校验 CRC，解析用 surrogateescape 保留原始字节，优先用稳定 ID 映射，无法确定的对象排除。17 条 regulator ID 冲突、3 条 target ID 冲突等分别计数。

`Drugs`、耐药临床应用、自由文本 Evidence 未进入模型特征，数据构建未读取原项目 RNA–drug 标签。Model 表仅用于身份和 OncotreeLineage。额外上传的全物种 MTI、Target Sites、TCGA/GEO/CTC 差异表达表未重复或混入本轮输入。

重复表达列为 `DDI2 (84301)` 和历史名称 `RSC1A1 (6248)`。冻结 HGNC 中，RSC1A1 已列为 DDI2 的 prev_symbol，两列因此对应同一个 Ensembl ID，按事先定义的规则平均。逐列映射与原文件复核见 [EXPRESSION_GENE_MAPPING.tsv](results/tabicl_pathway_context_20260913/EXPRESSION_GENE_MAPPING.tsv) 和 [EXTERNAL_SOURCE_AUDIT.json](results/tabicl_pathway_context_20260913/EXTERNAL_SOURCE_AUDIT.json)。

完整覆盖与排除计数见 [COVERAGE.json](results/tabicl_pathway_context_inputs_20260913/COVERAGE.json)。用户确认 DepMap 三文件为 25Q2；原始 RNA 表 release 尚不明确，miRTarBase 的确切下载版本未单独核实，两者均通过原文件哈希定位输入。

## 共享结构

设 F、B、E 分别为功能、结合和关联的二值 RNA–基因矩阵。证据适配器使用预先固定的共同权重：

```text
A[r,g] = max(F[r,g], 0.25 × B[r,g], 0.10 × E[r,g])
```

同一边的多个来源取最大值，不按观察次数相加。通过 gene→pathway 矩阵投影，并按基因参与通路数和通路大小的平方根降权，得到 L2 归一化 RNA/药物通路向量。

表达分支将 937 个细胞中的每个 RNA/基因转为平均秩，中心化并做 L2 归一化。第一视图计算全细胞秩相关；第二视图先在 OncotreeLineage 内中心化，减少癌种之间的表达差异对相关性的影响。表达通路分数是加权基因秩的归一化聚合，属于表达派生分数，不等于实测通路活性。

最终比较四个结构，前三个可参加选择：

|结构|新增列数|总列数|含义|
|---|---:|---:|---|
|`annotation`|14|50|原 direct 配对特征定义，使用本轮证据的二值并集|
|`tiered`|26|62|带共同置信权重的 14 列＋每证据层 4 列独立统计|
|`context`|40|76|tiered＋14 列全局/癌种内表达相关、靶点聚合和证据通路一致性|
|`context_permuted`|40|76|保留 tiered，按固定随机种子打乱整块 RNA 表达特征；仅为等维度负对照|

本轮最终选择 `tiered`、alpha=0.5。DepMap 表达特征完成了实际比较，但没有进入接受的 26 列新特征，因此不能把最终增量归因于额外的表达通路分支。

另做了复用历史预测的固定 14 列核对：原 direct 与新 annotation 的特征名、上下文、成员数、种子、权重摘要及基线指标一致。在 alpha=0.5 的内部结果中，新 annotation 相对原 direct 的 LncDrug AUC/AP 增加 0.000286/0.000619，MiDrug 则减少 0.000125/0.000436。该比较说明新增/放宽的证据有助于修复 LncDrug 的限制，但增加数据本身未同时改善所有方向。它同时改变了下载注释和 ENCORI 阈值，不能把两者分开归因；见 [DATA_ONLY_COMPARISON.json](results/tabicl_pathway_context_20260913/DATA_ONLY_COMPARISON.json)。

前 36 列图特征和现有图上下文组装方式保持匹配。新增分支预测与强图上下文基线按以下公式组合：

```text
score = (1 − alpha) × graph_baseline + alpha × augmented_graph_model
```

两任务使用同一个新结构和 alpha。没有新增按任务身份绕过分支的规则，也没有按任务单独学习新融合系数。

## 实验协议与验证

协议在正式指标产生前冻结：三个推理种子 666/0/1，32 成员；alpha 为 0.001、0.005、0.01、0.025、0.05、0.1、0.25、0.5、1。共 27 个候选配置以及 9 个表达错配对照配置；同一分支的多个 alpha 复用同一组拟合预测，不能视作独立训练重复。

本轮验收比只检查均值更严格：LncDrug 每个种子的 AUC、AP 均不下降，无负容差；MiDrug 两项均值提高，且每项至少两个种子提高。按药物宏 AP 单独报告，不进入硬约束。内部通过后固定一个候选，再访问历史外层；若没有候选通过则停留在内部，不访问外层。若选择 context，其匹配 alpha 的错配对照也在外层报告，但不改变选择。

13 项工程检查覆盖物种过滤、Non-Functional 排除、ID 冲突、LncTarD E/F 误读、常量表达、癌种内中心化、细胞/基因重排、两任务规则一致、错配对照及外层访问闸门。28 个小样本状态重载误差为 0，换序误差为 0，分批最大差异为 4.083e−6，在登记容差内。

历史内外层均已多次参与开发；推理种子不是独立数据划分。当前协议不运行原始测试集。多个分支列数不同，TabICL 的列置换也随之变化；错配对照仅控制 context 信息的对应关系，不支持把所有变化都归因于生物机制。

代码入口：[tabicl_context_program.py](tabicl_context_program.py)、[特征模块](tabicl_context_program_modules.py)、[检查](test_tabicl_context_program.py)。[正式协议](results/tabicl_pathway_context_20260913/protocol.json)与[工程验证](results/tabicl_pathway_context_20260913/ENGINEERING_VALIDATION.json)已保存。

复现时用新的输出目录，避免覆盖冻结输入与原结果：

```bash
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
  /data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.pathway_context_data \
  --out-dir improve/results/pathway_context_inputs_reproduction --depmap-release 25Q2

/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.tabicl_context_program prepare \
  --out-dir improve/results/pathway_context_reproduction \
  --inputs-dir improve/results/pathway_context_inputs_reproduction

/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.tabicl_context_program run \
  --out-dir improve/results/pathway_context_reproduction --task MiDrug --device cuda:0
/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.tabicl_context_program run \
  --out-dir improve/results/pathway_context_reproduction --task LncDrug --device cuda:1
/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.tabicl_context_program report \
  --out-dir improve/results/pathway_context_reproduction --phase inner
```

仅当新目录 `SELECTION.json` 中 `eligible=true` 时，用相同两个 run 命令加 `--phase outer` 生成外层预测，再运行 `report --phase outer`。
