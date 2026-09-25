# 类型约束的结构程序搜索

**2026-09-12 调参补充：** 后续同预算 12 组配置、152 次训练已完成。MiDrug 的度数校正版本相对原 TabM 的 AP 增加约 0.123 个百分点，AUC/宏 AP 也提高；LncDrug 仍未超过原基线。见 [调参结论](results/structure_program_tuning_20260912/FINDINGS.md) 与 [复现说明](STRUCTURE_PROGRAM_TUNING.md)。下面保留第一轮冻结实验的记录。

**2026-09-11：56 次正式训练、全部检查点及 24 个支持图的特征重建核验已完成，当前原型未通过双任务性能筛选。** LncDrug 搜索规则的 AP 下降约 0.50 个百分点；MiDrug 度数校正的 AP 提高约 0.023 个百分点，但 AUC 和按药物宏 AP 下降。现有 TabM 仍为推荐方案。见 [完整结论](results/structure_program_20260911/FINDINGS.md) 与 [数值报告](results/structure_program_20260911/REPORT.md)。

本轮将 [近期研究方案](research/recent_alternatives_20260911/PROPOSAL.md) 的第一项实现为一个有明确预算的原型：枚举关系程序，用内层拟合数据选择少量规则，交给已有 TabM；另以保留目标关联度数的随机图检验结构支持是否超出度数预期。

这不是 AntEvolve 或 SLogic 的复现，也没有调用外部 LLM。原 36 维结构特征、TabM、优化器、数据来源和两档学习率预算沿用既有实现。

**四组对照**

| 组别 | 输入 | 要检验什么 |
|---|---|---|
| base | 原 36 维特征 | 当前 TabM 在同一运行流程中的对照 |
| random | 原特征 + 随机 12 个程序值 | 增加特征、参数和图证据本身的收益 |
| search | 原特征 + 搜索选择的 12 个程序值 | 训练数据选择的规则是否优于随机规则 |
| excess | 原特征 + 同一批 12 个搜索程序的度数校正值 | 条件空模型校正是否优于原始程序值 |

后三组均为 48 维输入，参数量相同。excess 将新增的 12 个原始值替换为校正值，不额外扩大模型。各组独立使用相同的两档学习率选择预算。

**实际实现的搜索空间**

将目标 RNA、另一类 RNA 和药物记为 r/o/d。A 为 r→d，B 为 o→d，C 为 r→o；小写 a/b/c 为逆向；S/T/D 为三类节点的相似关系。

- 程序必须从 r 开始并在 d 结束，类型不匹配直接报错。
- 枚举长度 2–4 的关系链，排除立即使用逆关系的语法组合。这个限制比“只排除立即返回同一节点”更严格，不能把所有这些链统称为精确非回走路径。
- 每条关系链使用计数、逐关系行归一化、端点度数对称归一化三种运算；移除若干与原特征相同的计数模板。
- 另加入目标二部图上长度五的精确非回走计数，以小图逐条路径枚举核验。
- 共 61 个候选程序。其余 36 维特征及 circRNA 辅助信息仍沿用原基线；新程序只在上述三类实体上运算。

选择过程仅使用内层拟合数据。先对五折生成的候选值取 log1p，用前三个度数特征的逻辑回归残差计算关联强度，再减去固定的长度、已有特征冗余和已选程序冗余惩罚，贪心取 12 个。随机对照从相同的非恒定、非重复候选池抽取 12 个，随机种子固定为 20260911。

两个任务共享语法和选择算法，各自选择程序；本轮没有联合学习一套跨任务规则，也没有进行迁移评估。

**度数校正与遮边**

训练样本继承原 TabM 的五折编号，每一行的程序只读取其整折正边已删除的目标支持矩阵。目标图在 `ProgramEncoder` 中由显式支持边重新构造，不读取图对象中储存的目标正向或反向边。其他任务的关联沿用原单任务协议。

对每个支持图独立生成 8 个条件空模型，每个空模型目标为接受 `5 × 边数` 次二部双边交换：

- 严格保留目标关系的 RNA 端和药物端度数。
- 所有查询配对都禁止被交换加入，包括两类标签的查询；该约束只使用配对位置，不使用查询标签。
- 其他关联、相似图和跨 RNA 桥保持固定。因此不依赖目标 A/a 的程序会得到恒定空模型，直接回退为原值。
- 标准差设置相对下限，标准化值裁剪到 ±10；方差退化时保留原程序值。
- 保存接受次数、目标次数、尝试次数、实际换边比例、度数误差、查询边数和回退比例。达到交换次数不等于已证明充分混合。

该空模型条件于整批查询位置；更换查询集合可能改变有限随机样本下的校正值。本原型是固定评价候选集合上的实验，尚未提供任意批次等价的线上推理接口。

**冻结实验**

先以 seed 666 为每组选择学习率和最早的最优连续五轮 `mean(AUC, AP)`；再运行 seed 0/1；用三次内层选中轮数的中位数固定外层重训轮数。程序 ID 在内层拟合后固定，外层不重新搜索规则，也不按外层指标选择轮数。

两任务、四组、两档初筛配置、三个确认/重训初始化，共 56 次正式训练。工程短跑另存目录，不作为效果证据。外层是历史开发验证集，未读取测试集。每组保存检查点、逐轮轨迹、预测和哈希，报告程序重放所有检查点并重算指标。

预设性能筛选：同一任务上 AUC/AP 均值超过 base，两个指标各至少 2/3 种子提高，宏 AP 不下降；两个任务都通过才称双任务通过。搜索机制另外比较 search 对 random，校正机制比较 excess 对 search。工程核验通过不代表这些性能筛选通过。

**复现入口**

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
OUT=improve/results/structure_program_repeat

$PY -m unittest improve.test_structure_programs -v
$PY -m improve.structure_program_search prepare --out-dir "$OUT"
$PY -m improve.structure_program_search run --out-dir "$OUT" --stage screen --device cuda:0
$PY -m improve.structure_program_search select --out-dir "$OUT" --stage screen
$PY -m improve.structure_program_search run --out-dir "$OUT" --stage confirm --device cuda:0
$PY -m improve.structure_program_search select --out-dir "$OUT" --stage confirm
$PY -m improve.structure_program_search run --out-dir "$OUT" --stage outer --device cuda:0
$PY -m improve.structure_program_search report --out-dir "$OUT" --device cuda:0
```

`run` 支持 `--task` 和 `--arm` 分配独立作业。特征创建使用任务锁，各训练作业使用独立锁；相同作业不会被重复写入。正式冻结后不修改源码、输入或程序选择文件。

实现：[程序与空模型](structure_program_modules.py)、[实验入口](structure_program_search.py)、[关键正确性检查](test_structure_programs.py)、[实际特征重建核验](audit_structure_programs.py)。论文来源和创新边界见文首研究方案。
