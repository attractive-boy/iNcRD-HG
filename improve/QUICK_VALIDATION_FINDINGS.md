**已完成两轮快速验证，当前参考模型保持不变。** 2026-09-14。按“先做能快速验证的改进”的要求，暂停逐篇实验语义整理，复用现有预测、RNA 表示和数据划分。

第一轮的 4 个固定关联候选均未通过既定共同门槛。第二轮发现 RNA-FM 加来源平衡的局部信号，但逐药物 AP 下降；预先指定的功能表示主比较也未通过。没有根据这些结果继续搜索混合系数或改换主比较。

**复用预测的三类改进。** 计算、读取和核验合计 6.95 秒，无新增神经训练或推理。参考是已接受的 `0.5 × graph + 0.5 × tiered`。下表为 seed 666/0/1 的均值差，单位为百分点。

|固定方案|MiDrug ΔAUC / ΔAP|LncDrug ΔAUC / ΔAP|判定|
|---|---:|---:|---|
|等权 logit 融合|+0.0059 / +0.0079|+0.0026 / +0.0028|均值略升，但 LncDrug seed 0 的 AP 下降，未通过|
|按 RNA/药物证据覆盖调整融合|+0.0182 / +0.0666|−0.0372 / −0.0818|MiDrug 三种子均升，LncDrug 三种子均降|
|RNA 训练频率轻度校正|−0.0003 / +0.0395|−0.0057 / −0.0188|存在指标取舍，未通过|
|药物训练频率轻度校正|−0.0587 / −0.1663|+0.0359 / +0.1211|LncDrug 升、MiDrug 降，未通过|

logit 融合最接近通过，但 LncDrug seed 0 的 AP 实际下降 **0.000006261694176**（0–1 指标单位），不能用四舍五入或均值掩盖。MiDrug 的药物宏 AP 还从 0.719361 降至 0.717978。

覆盖门控使用共同规则：`alpha = 0.25 + 0.5 × drug_target_present × RNA_evidence_strength`；RNA 强度沿用功能、结合、关联层的 1 / 0.25 / 0.1。覆盖随机打乱只作对照，不能参加选择。

频率校正的平滑先验质量固定为 20，logit 修正强度固定为 0.25，均未搜索。它使用训练中的数据库正例/未标注比例，不是生物学真实发生率。**药物校正不改变同药物内部的 RNA 排名**，所以其药物宏 AP 完全不变；全局变化来自跨药物重排。RNA 校正虽提高两任务宏 AP，却未满足全局 AUC/AP 条件。

共同门槛沿用上一轮：LncDrug 每个种子的 AUC/AP 均不下降；MiDrug 两项均值提高且各至少 2/3 种子提高。详见 [完整关联结果](results/quick_score_20260914/REPORT.md)、[逐种子判定](results/quick_score_20260914/ASSESSMENT.json)、[逐药物结果](results/quick_score_20260914/per_drug.csv)和[频率匹配诊断](results/quick_score_20260914/diagnostics.csv)。

**按论文来源平衡训练。** 共完成 30 次逻辑回归拟合：2 个任务 × 3 种表示 × 5 种权重。拟合累计 2.93 秒，含读取、核验、评价及重采样共 11.45 秒；计时不含代码编写。RNA-FM、功能表示、PCA 和上下文设计均复用上一轮冻结产物。

每个三元组的 k 个来源各获得 1/k 的计数，按来源总计数倒数赋予训练权重，并将总权重归一化为训练行数。另比较保留原类别权重总量的版本，以及“仅改类别比例”“类别内打乱来源权重”两个对照。主评价仍是未见三元组的来源验证，MiDir 1,885 行、LncDir 256 行。

唯一在同表示、同权重规则下同时提高两任务 AUC 和 resistant AP 的描述性候选是 **RNA-FM＋来源平衡**：

|任务|指标|RNA-FM 均匀权重|RNA-FM 来源平衡|差值（百分点）|
|---|---|---:|---:|---:|
|MiDir|AUC|0.552199|0.573954|+2.1755|
|MiDir|耐药描述 AP|0.502483|0.504428|+0.1945|
|MiDir|药物宏 AP|0.600681|0.570395|−3.0286|
|LncDir|AUC|0.625302|0.640994|+1.5691|
|LncDir|耐药描述 AP|0.721646|0.722698|+0.1052|
|LncDir|药物宏 AP|0.857180|0.853192|−0.3988|

这是预先列出的次要比较，**不能替代未通过的主比较，也不是全面提升**。LncDir 加权 RNA-FM 的 AP 仍低于原身份上下文基线的 0.755209。仅 6 个 LncDir 药物满足宏 AP 的双类纳入条件。

预先指定的主比较是“功能均值＋SD 表示＋保留类别比例的来源平衡”对同表示均匀权重。MiDir / LncDir 的 AUC 分别下降 0.8122 / 0.2484 个百分点，AP 下降 2.2415 / 1.4963 个百分点；来源组配对 bootstrap 区间均包含零。没有事后为次要比较追加置信区间并包装成主结果。

来源平衡也改变了训练分布：MiDir 加权 resistant 比例从 46.41% 变为 31.04%，LncDir 从 67.29% 变为 83.04%；保留类别比例对照消除这项变化后，RNA-FM 在 LncDir 的 AP 不再提高。有效样本量分别从 5,693 / 1,709 降至约 1,575 / 452，完整逆来源频率加权会明显减少大研究提供的训练信息。

因此，下轮快速研究可优先检验**较温和的来源权重与正则化**，并先在训练侧按来源交叉验证；本轮没有把它们追加为新的调参轮次。logit 融合作为接近持平的工程候选留档，覆盖门控和直接频率扣除暂不扩大搜索。

完整来源结果见 [报告](results/quick_source_balance_20260914/REPORT.md)、[所有对照及判定](results/quick_source_balance_20260914/ASSESSMENT.json)、[主比较区间](results/quick_source_balance_20260914/BOOTSTRAP.json)和[权重审计](results/quick_source_balance_20260914/AUDIT.json)。

**验证边界与工程交付。** 新增 14 项机制/边界测试通过。关联分数适配器保存 2 个状态，6 组种子预测通过重载、换序、分 7 批检查，原参考分数逐元素完全复现。30 个来源模型通过重载、换序、分 5 批检查，最大重放误差为 0；均匀权重重现上一轮预测。训练与验证的来源组、PMID、原始记录交集在两个任务联合后仍为 0。

所有结果属于已参与开发的内部/来源验证，不是独立测试确认。来源方向标签仍为数据库 resistant/sensitive 描述，不能当成 RNA 干预药敏；关联的 0 标签仍是未标注配对。历史外层数组、原始关联测试数组和新来源留出分区均未评价，旧模型和结果未修改。

代码：[分数组合](quick_score_modules.py)、[关联实验入口](quick_score_eval.py)、[来源权重](quick_source_balance_modules.py)、[来源实验入口](quick_source_balance.py)。每轮输出都保存协议、精确源代码快照、输入哈希、模型状态、预测和完成清单。

在 `iNcRD-HG` 目录中复现，使用新输出目录：

```bash
QUICK_PY=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$QUICK_PY" -m unittest improve.test_quick_score improve.test_quick_source_balance -v
"$QUICK_PY" -m improve.quick_score_eval prepare --out-dir improve/results/quick_score_repeat
"$QUICK_PY" -m improve.quick_score_eval run --out-dir improve/results/quick_score_repeat
"$QUICK_PY" -m improve.quick_source_balance prepare --out-dir improve/results/quick_source_balance_repeat
"$QUICK_PY" -m improve.quick_source_balance run --out-dir improve/results/quick_source_balance_repeat
```

测试、结果完整性与本轮汇总见 [完成清单](results/quick_validation_20260914/COMPLETION.json)。两个实验都没有外层或测试集运行入口。
