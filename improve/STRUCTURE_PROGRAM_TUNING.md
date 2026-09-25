# 固定结构程序的同预算调参

**已完成：152 次正式训练和全部核验。MiDrug 的度数校正版本 AP 从 0.921693 升至 0.922689，同时超过原 TabM 与同预算调参 TabM；LncDrug 仍未超过原基线。** 见 [完整结论](results/structure_program_tuning_20260912/FINDINGS.md) 和 [全部数值](results/structure_program_tuning_20260912/REPORT.md)。没有获得双任务一致提升。

2026-09-12。对上一轮结构程序原型进行有限的网络参数搜索，回答两个问题：调参能否补回原方案的下降，以及新增规则能否超过获得相同搜索预算的原始 TabM。

本轮的输出目录为 `results/structure_program_tuning_20260912/`。最终数值和核验记录分别写入 `REPORT.md`、`VALIDATION.json`；解释性结论写入 `FINDINGS.md`。上一轮的源码、协议、特征与结果保持冻结。

四个方案沿用 [结构程序实验](STRUCTURE_PROGRAM_SEARCH.md)：base 为 36 维图特征；random、search、excess 分别追加随机规则、搜索规则、同一搜索规则的度数校正值，均为 48 维。规则数固定为 12，不重新选择规则，也不修改空模型。

每个任务、每个方案均比较以下笛卡尔积，共 12 组配置：

| 参数组 | 候选 |
|---|---|
| 宽度 / 深度 | 128 / 3、64 / 2 |
| Dropout / AdamW 衰减 | 0.1 / 0.0001、0.3 / 0.01 |
| 学习率 | 0.0005、0.001、0.003 |
| 固定参数 | 16 个内部成员；batch size 1024；恒定学习率 |

容量参数、正则化参数各成组改变，结果只能解释为配置的效果，不能将差异单独归因于某个参数。梯度累积方式沿用旧实现，每轮为一次全训练集平均 BCE 的 AdamW 更新。

为给较低学习率足够的拟合时间，内层训练上限随学习率改变；各方案使用同一个上限表。

| 任务 | LR 0.0005 | LR 0.001 | LR 0.003 |
|---|---:|---:|---:|
| LncDrug | 400 | 250 | 200 |
| MiDrug | 900 | 600 | 450 |

内层先以 seed 666 跑全部 12 组，按最优连续五轮的 `mean(AUC, AP)` 排名前二。两组都追加 seed 0/1，以三个种子的选中分数均值确定唯一配置，分数相同则按配置 ID 排序。外层训练轮数固定为胜出配置三个内层轮数的中位数。每个检查点同时提供 AUC/AP，不分别挑选两项指标的最好轮数。

正式预算为 96 次初筛、32 次确认、24 次外层重训，共 152 次。四个方案获得相同的配置数、种子数和各学习率上限。不同方案最终可能选择不同容量和正则化，因此这是同预算流程比较。

预设比较包括：各方案对其上一轮版本、各方案对上一轮 base、三个扩展方案对本轮同预算 base、search 对 random、excess 对 search。性能门槛要求 AUC/AP 均值同时提高、两个指标各至少 2/3 种子提高、宏观药物 AP 均值不下降。机制比较不要求最后一项。

特征加载验证原缓存及依赖哈希，沿用此前 24 个支持图的精确重建核验。新流程在自己的 `REFIT.json` 冻结前拒绝加载外层数组；外层循环不计算验证指标。报告重放全部检查点、重新计算两阶段选择和配对指标，并比较 16 个旧配置的训练轨迹前缀，检查训练器复用是否改变行为。

历史验证已被多次用于开发，本次也是基于之前的结果提出的后续实验；三个初始化不构成三个独立数据划分。没有测试集评估，也没有改变原空模型回退率高、依赖跨任务侧信息等机制限制。

复现需在仓库根目录执行，使用一个新的输出目录：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
TUNING_PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
TUNING_OUT=improve/results/structure_program_tuning_repeat

"$TUNING_PY" -m unittest improve.test_structure_program_tuning -v
"$TUNING_PY" -m improve.structure_program_tuning prepare --out-dir "$TUNING_OUT"
"$TUNING_PY" -m improve.structure_program_tuning pipeline --out-dir "$TUNING_OUT"
```

`pipeline` 使用两张 GPU、每张两个进程；进程按方案分工，阶段之间等待全部完成再选择。单 GPU 可按阶段使用 `run --stage screen/confirm/outer --device cuda:0`，在 screen 和 confirm 后分别执行对应的 `select`，最后执行 `report`。正式运行前可用另一新目录加 `prepare --smoke` 做 40 次三轮工程短跑，短跑不作为效果证据。

实现与关键检查：[调参入口](structure_program_tuning.py)、[选择逻辑检查](test_structure_program_tuning.py)。
