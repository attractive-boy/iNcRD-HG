# 同一种平滑注意力能否同时改善 MiDrug 与 LncDrug

本轮补齐 LncDrug 上的验证，直接沿用已经在 MiDrug 冻结的两种机制：
温度 T=4 的路径注意力，以及 25% 注意力 + 75% 均匀权重的收缩方案。
不针对 LncDrug 修改温度、混合比例、路径集合或投影宽度。

入口 `cross_task_path_attention.py` 复用上一轮 `smooth_path_attention.py` 的训练函数和模型。
LncDrug 沿用其原模型的 64 维、关系 min、学习率 0.0005、100 个 epoch；
MiDrug 对照来自已经核验的 256 维、关系 mean、学习率 0.0001、300 个 epoch 结果。
“同一机制”指相同注意力公式与新模块设置，两任务分别对照各自原模型。

## 对照与判断

LncDrug 的 5 组为 base、path_uniform、path_attention、temperature4、shrink25。
种子 666、0、1、2、3，共 25 次训练；每个 base 再作一次同设备重复，共 30 次。
另用 3 个 epoch 的 10 次短跑检查保存、重放和确定性，不用短跑分数作性能判断。

模型选择仍只使用验证集 AUC/AUPR 均值，完全相同则取最早 epoch。
同时检查测试 AUC/AUPR 的配对均值与提升种子数。
需区分超过原模型、超过均匀池化、超过原注意力三个不同的比较。
不能把不同任务各自最好的不同模块组合后称为同一种注意力的双任务提升。

## 核验与输出

- 两任务使用上一轮清理后的固定划分；所有编码均用训练图。
- 检查源码、协议、图、路径资产和 checkpoint 状态哈希。
- 重新计算全部预测及干预指标，并核对选模轨迹与权重约束。
- 20 个共同对照运行须完整复现此前 LncDrug 的曲线、状态与预测。
- 首个种子的 5 个模型在 CPU 重建并重放所有干预。
- `joint_comparisons.csv` 将已有 MiDrug 结果与新增 LncDrug 配对比较放在一张表中。
- `VALIDATION.json`、`CONTROL_REPLAY.json` 记录核验与复现。

在仓库根目录，用 `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python` 执行：

```bash
python -m improve.cross_task_path_attention init --out-dir improve/results/smooth_path_attention_lnc_20260908
python -m improve.cross_task_path_attention run --out-dir improve/results/smooth_path_attention_lnc_20260908 --device cuda:0 --shard-index 0 --shard-count 2
python -m improve.cross_task_path_attention run --out-dir improve/results/smooth_path_attention_lnc_20260908 --device cuda:1 --shard-index 1 --shard-count 2
python -m improve.cross_task_path_attention summarize --out-dir improve/results/smooth_path_attention_lnc_20260908
```

两个 worker 写入不重叠的运行目录，保持与此前 LncDrug 相同的种子/GPU 分配。
已冻结目录用 `run` 恢复，不重复 `init`。
正式结果目录为 `results/smooth_path_attention_lnc_20260908`。

两任务的跨任务一致性仍不能代替新数据划分验证。
这是对既有固定划分和预先冻结注意力机制的验证，没有通过本轮测试结果重新搜索设置。
