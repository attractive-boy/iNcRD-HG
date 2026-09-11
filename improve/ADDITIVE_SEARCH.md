# 加法配对模型的参数与种子选择

2026-09-10。按用户要求继续寻找更好的参数和单次运行。以已完成的 [矩形试验](results/rectangle_completion_20260910/FINDINGS.md) 为参照，重点优化表现较好的加法配对版，同时给原模型相同搜索次数。

已完成 56 次正式训练、全部检查点重放和四个选中模型的导出。完整结论见 [正式结果](results/additive_search_20260910/FINDINGS.md)，可直接查看 [参数与指标汇总](results/additive_search_20260910/selected_summary.csv) 和 [五种子统计](results/additive_search_20260910/five_seed_summary.csv)。模型结构、数据和历史源码保持原有版本；新的训练入口是 [additive_search.py](additive_search.py)。

加法版内部验证选中 LncDrug seed 666（AUC/AP `0.990489/0.978624`）和 MiDrug seed 0（`0.958779/0.922009`）。对应外层重训结果分别为 `0.982377/0.948502`、`0.955150/0.917390`；两者的外层 AP 都低于各自按内部标准选出的原模型。相同种子比较显示部分指标有小幅收益，尚不构成稳定全面提升。LncDrug 未刷新上一轮 AP `0.980390` 的最好单次记录。

以下为实际执行并保留的冻结设计；没有根据外层结果改选种子或追加搜索。

**参数搜索**

| 项目 | 原模型 | 加法配对版 |
|---|---|---|
| 中心基础学习率 | LncDrug 0.0005；MiDrug 0.00003 | 相同 |
| 中心新分支学习率 | 无 | LncDrug 0.0005；MiDrug 0.00009 |
| 六个恒定学习率配置 | 基础 LR × 0.5/1/1.5，与 WD 0/1e-4 组合 | 两组 LR 同时 × 0.5/1/1.5，与修正上限 1/2 组合 |
| 两个余弦配置 | 中心 LR，WD 0/1e-4 | 中心 LR，修正上限 1/2 |
| 总配置数 | 8 | 8 |

余弦曲线从初始学习率降到其 5%。其他设置固定：RMSprop alpha=0.99、普通 BCE、查询比例 0.5、前 20% 轮数逐步启用修正、分支宽度 16、两层。LncDrug 150 轮，MiDrug 450 轮。

**参数与种子分两步选择**

先用 seed 666 跑完所有配置：两个任务 × 两组模型 × 八套配置，共 32 次内部训练。每次保留最早达到最高连续五轮 mean(AUC,AP) 的单个检查点；以同样分数选择一个配置。

将参数写入 `PARAMETERS.json` 后，固定该配置跑 seed 0、1、2、3，再与原 seed 666 合并，得到五种子分布；这一阶段增加 16 次内部训练。按同一内部标准选最佳种子，写入 `SELECTION.json`。平分时按配置编号、种子整数依次排序；AUC 和 AUPR 始终来自同一检查点。

最终分别报告验证集选出的最佳单次结果、固定参数下全部五种子的均值/标准差，以及全部尝试记录。最大值经过验证集筛选，不能作为无偏平均值，也不等于新结构贡献。

**固定方案的外层验证**

两个选择文件冻结后，取每个任务两组模型各自最佳种子的并集，双方都用这些种子重训。这样既能比较双方各自验证选出的最佳单次，也能比较相同种子的结果。最多增加八次正式重训，总预算最多 56 次。

外层重训使用完整原拟合集，训练轮数由对应内部运行提前确定。学习率曲线和修正 ramp 的时间尺度沿用原内部上限，不因提前选到较小 epoch 而压缩曲线。训练循环不评价外层标签，完成固定轮数后评价一次。

准备阶段只复制外层文件字节和核验来源哈希，内部训练只加载 `inner.npz`。外层数组在参数与种子选择冻结之后才加载。测试集不参与本轮。内部与外层验证都曾用于历史开发，不能把这次外层结果称为全新独立测试。

**检查与产物**

四项协议检查覆盖配置预算与去重、衰减曲线及重训时间尺度、同一指标与平分规则、外层结果不能进入选择函数。短跑覆盖完整选择、重训和重放流程。正式结果逐个重放检查点，并核对学习率轨迹、轮数、输入、源码和选择文件。

完成后导出每个任务、每组模型的最佳种子配置，内部与外层检查点、预测文件和各自指标，位于 `best/<task>/<arm>/`。五种子统计另存在 `inner_mean_std.csv`，避免与最佳单次混用。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m unittest improve.test_additive_search -v
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.additive_search prepare --out-dir improve/results/additive_search_20260910
```

每个阶段都可由两张 GPU 分担：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.additive_search run --out-dir improve/results/additive_search_20260910 --phase screen --shard 0 --shards 2 --device cuda:0
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.additive_search run --out-dir improve/results/additive_search_20260910 --phase screen --shard 1 --shards 2 --device cuda:1
```

screen 完成后运行 `parameters`，再将 phase 改为 `seeds`。seeds 完成后运行 `select`，再执行 `outer` 阶段。最后依次运行 `report` 和 `export`。所有命令使用同一个 `--out-dir`；report 可指定 `--device cuda:0`。不要重新 prepare 或改写已有冻结目录。
