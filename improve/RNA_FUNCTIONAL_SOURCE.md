本轮把 [下一步研究建议](research/frontier_next_steps_20260913/REPORT.md) 的阶段 A 实现为两个互补实验。结果与判断见 [RNA_FUNCTIONAL_SOURCE_FINDINGS.md](RNA_FUNCTIONAL_SOURCE_FINDINGS.md)。

功能表示使用已核验的公开 miRBind2 binding-site 和历史 PARNET 权重；通用表示使用同一套冻结 RNA-FM。关联预测继续使用已接受的 TabICLv2 图上下文和分层证据方案。来源评价另外使用数据库方向描述任务，不能将它的 AUC 与关联任务直接比较。

|实现|职责|
|---|---|
|[rna_functional_inputs.py](rna_functional_inputs.py)|扩展真实 RefSeq 靶位点，保留全部来源 PMID；提取与 RNA-FM 输入匹配的功能表示|
|[rna_functional_modules.py](rna_functional_modules.py)|训练支持实体上拟合标准化/PCA，保留缺失；按覆盖、支持实体成员、长度/GC/位点数分层打乱|
|[rna_functional_experiment.py](rna_functional_experiment.py)|冻结 TabICLv2 和五组适配器预算，运行历史内部开发对照及覆盖/频率诊断|
|[context_source_eval.py](context_source_eval.py)|原始记录回连、语义分诊、PMID 分组、侧内标签汇总和方向频率/身份基线|
|[context_functional_eval.py](context_functional_eval.py)|去除验证及留出同源位点，再进行相同结构的上下文交互对照与研究来源重采样|

五组新适配器是 `simple`、`generic`、`functional_mean`、`functional_moments`、`functional_shuffled`。所有适配器保留同样的 8 个简单统计和 16 个表示坐标；`simple` 的 16 个坐标为零。缺失实体一直保留。`generic` 和功能分支使用相同实体覆盖，不能靠独占注释覆盖获得优势。

miRNA 的通用分支平均编码与 miRBind 完全相同的成熟链和真实 50 nt 靶窗口，再拼接两端表示。功能分支使用 30 维局部隐藏向量。均值与均值加标准差读取同一批位点；这是 RNA 层面的摘要，还未实现当前药物条件下的具体基因交互。miRBind 的 binding-site 输出不能冒充另一模型的转录抑制输出。

lncRNA 的 PARNET 和 RNA-FM 使用相同的至多 510 nt 编码窗口、8 nt 池化区间和覆盖权重。窗口沿完整注释转录本选择，因此保留了长转录本，但并非全长全碱基表示。当前公开 PARNET 历史权重的 eCLIP 背景和论文最终模型的等价性仍是限制。

每个支持上下文单独拟合压缩器：每个有覆盖的训练 RNA 只计一行，避免样本频率直接改变 PCA；标准化、PCA、方差缩放均不使用查询专属 RNA 或标签。压缩为 16 维，方差下限为 `1e-10`，坐标截取至 `[-10,10]`。所有 PCA 状态与训练实体索引保存到每个拟合目录。

打乱将整个功能向量在分层内循环错配，保留覆盖、支持侧成员关系和训练实体推导的长度/GC/位点数量三分位层。原实体的简单统计不变；单例层不强行错配。它是一次有限打乱，不能解释成完整随机化显著性检验。实际移动比例逐支持上下文保存。

关联实验保留 36 个图特征和 26 个分层证据，再加上述 24 个数值列，所有新分支总计 86 列。每组使用相同样本、32 个推理成员、666/0/1 三个种子与参考图上下文规则；分数始终是 `0.5 × 冻结图分支 + 0.5 × 新分层证据分支`。没有搜索混合系数，也没有重新训练神经参数。简单统计对照用于区分表示信息和额外维度/覆盖带来的改变。

来源任务先把两个 RNA 任务的原始记录按共享 PMID 联合分组，按合并原始记录数量将来源组分配至目标 60%/20%/20% 的训练、验证和留出分区；由于研究组大小及两个任务分布不同，每个任务的实际比例不完全相同。缺失来源进入隔离区。分组后才展开细胞/癌种并在各侧汇总无冲突三元组，避免用验证标签决定训练共识。主评价还排除任何训练侧原始记录中出现过的三元组身份，包括被判为冲突的训练身份。

`Effect` 始终表示数据库的 resistant/sensitive 描述。检测方法、Support、PMID、年份、Condition、原始行号及细胞展开关系均保留；自动规则只提供待核查标记，所有 `intervention_verified` 均为 false。来源 ID、证据文本和实验后信息不作为预测特征。

来源功能对照先删除任一 PMID 与验证或留出记录重叠的 miRNA 位点，再重新计算两类表示和简单统计；对于 miRTarBase 中 `23622248.0` 这类整数序列化形式，仅精确去除 `.0` 后对齐。该过滤审计的是本地辅助位点来源，不能证明公共编码器的预训练数据完全来源独立或时间独立。

上下文适配器是固定 C=1 的逻辑回归：RNA/药物/癌种身份、24 个数值特征，以及数值特征与药物/癌种身份的交互。五组设计矩阵宽度一致，不搜索超参数。对固定验证预测做 500 次研究来源连通组配对重采样，三元组涉及多个来源时先连接相应来源再抽样。区间只描述这次划分条件下的来源不确定性。

在 `iNcRD-HG` 目录复现，使用新的输出目录。已完成的目录及旧参考产物禁止覆盖。RNA-FM 的 `generic_sites` 使用已有 `ncRNAdrug` 环境；其余命令使用 `.venv-tabicl`，没有修改冻结环境依赖。

```bash
../.venv-tabicl/bin/python -m unittest improve.test_rna_functional_inputs improve.test_rna_functional_modules improve.test_context_source_eval improve.test_context_functional_eval improve.test_rna_site_windows

../.venv-tabicl/bin/python -m improve.rna_functional_inputs windows --out-dir NEW_INPUTS
../.venv-tabicl/bin/python -m improve.rna_functional_inputs parnet --out-dir NEW_INPUTS --device cuda:1 --batch-size 16
../.venv-tabicl/bin/python -m improve.rna_functional_inputs mirbind --out-dir NEW_INPUTS --device cuda:0 --batch-size 64
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.rna_functional_inputs generic_sites --out-dir NEW_INPUTS --device cuda:1 --batch-size 64
../.venv-tabicl/bin/python -m improve.rna_functional_inputs assemble --out-dir NEW_INPUTS

../.venv-tabicl/bin/python -m improve.context_source_eval build --out-dir NEW_SOURCES
../.venv-tabicl/bin/python -m improve.context_source_eval evaluate --out-dir NEW_SOURCES
../.venv-tabicl/bin/python -m improve.context_functional_eval prepare --out-dir NEW_CONTEXT --source-dir NEW_SOURCES --inputs-dir NEW_INPUTS
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 ../.venv-tabicl/bin/python -m improve.context_functional_eval run --out-dir NEW_CONTEXT

../.venv-tabicl/bin/python -m improve.rna_functional_experiment prepare --out-dir NEW_EXPERIMENT --inputs-dir NEW_INPUTS
../.venv-tabicl/bin/python -m improve.rna_functional_experiment run --out-dir NEW_EXPERIMENT --task MiDrug --device cuda:0
../.venv-tabicl/bin/python -m improve.rna_functional_experiment run --out-dir NEW_EXPERIMENT --task LncDrug --device cuda:1
../.venv-tabicl/bin/python -m improve.rna_functional_experiment report --out-dir NEW_EXPERIMENT
```

正式实验前另用新目录执行 `prepare --smoke` 和两个任务的 `run`，检查重载、查询顺序及分批不变性。正式报告只在五组所有状态完成后生成；没有根据中途成绩删减分支。历史外层、原始关联测试集以及新来源留出分区均不在本轮评分范围。
