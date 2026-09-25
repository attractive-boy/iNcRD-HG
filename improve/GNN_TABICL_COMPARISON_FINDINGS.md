**实测结论：增加表征不必然削弱 GNN；当前 SA-HGT 没有超过 TabICLv2 的全局 AUC/AP，增加训练步数也没有弥补差距。**

2026-09-13 完成两个任务、三个输入条件、三个种子的对照。训练了 36 个 HGT/MLP 模型，保存并复核了 72 个检查点；TabICL 使用同一批样本和证据对应的已验证预测。完整数据见 [对照报告](results/gnn_tabicl_compare_20260913_logits/REPORT.md)、[逐种子结果](results/gnn_tabicl_compare_20260913_logits/metrics.csv)、[配对变化](results/gnn_tabicl_compare_20260913_logits/paired_differences.csv)。

**“表征变多”分成两种情况。**

- `graph36 → tiered62`：在 36 个图配对特征上增加 26 个靶点／通路证据特征。
- `tiered62 → expanded62`：仍为 62 维，扩充药物靶点证据；覆盖药物从 93/216 增至 156/216。这里改变的是证据覆盖，并未继续增加维度。

HGT 保留原有节点输入及 13 种关系，新证据进入配对解码器。本轮没有加入新的 gene/pathway 节点，也没有把 miRBind2/PARNET 小规模试点表征用于全量预测。因此，结果适用于这版融合方法，不能外推到任意高维多模态 GNN。

**原训练预算下的平均 AP。** 输入依次为“36 维 → 旧 62 维 → 扩展 62 维”，均为同一内部开发集的 3 个种子均值。

|任务|模型|36 维|旧 62 维|扩展 62 维|
|---|---|---:|---:|---:|
|MiDrug|HGT|0.922161|0.920905|0.919795|
|MiDrug|MLP|0.923240|0.921070|0.918293|
|MiDrug|TabICLv2 全支持模型|0.926921|0.926926|0.925702|
|LncDrug|HGT|0.981270|0.982367|0.980800|
|LncDrug|MLP|0.966051|0.962892|0.962799|
|LncDrug|TabICLv2 全支持模型|0.984681|0.988465|0.988294|

LncDrug 加入旧的 26 个功能特征后，HGT 的 AUC 在 3/3 个种子上改善，AP 在 2/3 个种子上改善；MiDrug 的平均 AP 则下降。继续扩充证据后，LncDrug 的 HGT 和 TabICL 都在 3/3 个种子上出现 AUC/AP 回落。因此，证据增加后的退步也会发生在表格模型中。

在最新扩展输入下，MiDrug 的平均 AUC 分别为 TabICL 0.962581、HGT 0.959536、MLP 0.957304；LncDrug 分别为 0.994010、0.991471、0.981477。TabICL 在这轮两个任务、三档输入中均有更高的全局 AUC/AP。这里的 TabICL 是未与其他预测混合的全支持模型；当前已接受的混合模型另列在完整报告中。

**训练更久对 HGT 的影响大于这次特征增量。**

检查点在运行前固定：MiDrug 为 74 和 450 步，LncDrug 为 24 和 150 步。每步抽样一半训练配对。最新扩展输入下，HGT 的全局 AP：

- MiDrug：0.919795 → 0.880709，下降 0.039086；训练损失均值同时从 0.2328 降至 0.0254。
- LncDrug：0.980800 → 0.967434，下降 0.013366；训练损失均值同时从 0.1471 降至 0.0439。

两任务的全局 AUC/AP 均在 3/3 个种子上随长训练下降。这与过拟合相符，也需要考虑训练时隐藏查询边、推理时使用完整支持图带来的分布差异；本轮不能单独识别具体原因。LncDrug 的 MLP 在长预算下 AP 升至 0.982364，超过同预算 HGT 的 0.967434，仍低于 TabICL 的 0.988294。MiDrug 的 MLP 长训练 AP 也下降至 0.909456。

**评价目标会改变局部排序。**

最新扩展输入、原预算下，MiDrug 的逐药物平均 AP 为 MLP 0.745322、HGT 0.732874、TabICL 0.720522。HGT 的全局 AP 较低，同时在这一分组指标上较高；不能概括为所有指标均不如表格模型。

另一个诊断是在同时含两类样本的药物子集上平衡药物频率，保持该子集正例比例。LncDrug 长预算、扩展输入的 HGT 在这项诊断中 AUC/AP 为 0.803823/0.948296，TabICL 为 0.791131/0.944438；但 HGT 的全局指标和逐药物平均 AP 仍较低。该诊断只覆盖 23 个药物、268 个配对，加权有效样本量约 114；MiDrug 对应 82 个药物、1,099 个配对，有效样本量约 274。这些加权 AP 不能直接与全量 AP 比较，也不构成切换模型的依据。详见 [分组诊断](results/gnn_tabicl_compare_20260913_logits/diagnostics.csv)。

图分支实际参与了预测：扩展输入、原预算下，乘门控后的图 logit 标准差与配对分支之比，MiDrug 均值为 0.442，LncDrug 为 1.258。MiDrug 的小门控值不能直接解释为“没有使用图”。这些幅度指标也不证明图分支具有生物学因果意义。

**当前取舍。**

继续以已接受的 TabICL 方案作为全局 AUC/AP 主线。现有证据不支持仅因表征变多就换回 GNN，也不支持简单延长 HGT 训练。后续若继续研究 GNN，优先检验新增证据的筛选、融合和训练正则，并事先明确是否要优化逐药物表现；新基因节点或 RNA 多模态表征需要另外设计对照。

本轮没有重新选超参数、挑最佳检查点或替换当前模型，没有进行外层或原始测试集评价。训练／验证配对分别为 MiDrug 7,769/1,943、LncDrug 3,040/760。历史内部数据已经多次参与开发，初始化种子和 TabICL 推理种子不代表独立数据划分，不能据此声称独立泛化或统计显著性。

HGT 和 MLP 使用相同的配对分支、初始化、训练查询及支持集拟合的归一化；三档输入均保持 62 列参数形状，36 维条件的其余列置零。训练正边在每次查询时双向隐藏，36 维特征沿用冻结的五折交叉拟合。HGT 还访问原始图和节点输入，TabICL 使用 32 个预训练成员，并沿用其已有预处理；完整模型容量、预训练和算力并未匹配。因此这是现有方案的对照，不能完全隔离架构本身的效果。

**实现与复核。**

入口为 [gnn_tabicl_compare.py](gnn_tabicl_compare.py)，神经网络与查询隐藏实现在 [gnn_tabicl_compare_modules.py](gnn_tabicl_compare_modules.py)。16 项测试和 24 个短训练检查点通过；72 个正式检查点的恢复预测最大 logit 误差为 2.862e-6。比较使用原始 logit 计算 GNN/MLP 的排序指标，避免 float32 sigmoid 饱和导致排序丢失。此前采用概率分数的未完成运行已标记废弃，未用于模型选择。

结果目录为 `improve/results/gnn_tabicl_compare_20260913_logits`。准备和汇总使用 `/data1/licheng/incrd-hg/.venv-tabicl/bin/python`，GNN 训练使用 `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`；两个冻结环境均未安装或修改依赖。

重新汇总已有结果：

```bash
/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.gnn_tabicl_compare report
/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.render_gnn_tabicl_compare
```

第二步修正冻结报告中 pandas `Series.view` 与输入列同名的展示问题，原始展示另存，数值文件、模型和训练源码保持冻结。验证记录包含独立渲染器的源码哈希。重新训练时先以新目录执行 `prepare`，再分别对两个任务执行 `run --task MiDrug --device cuda:0` 和 `run --task LncDrug --device cuda:1`，均传入同一新目录的 `--out-dir`。
