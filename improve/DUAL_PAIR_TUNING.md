# 双流结构的参数补救实验

接续 [双流结构实验](DUAL_PAIR.md)。任务、图、节点特征、双流拓扑和监督目标保持原设置，检查上一轮的不利结果能否通过更合适的训练参数改善。上一轮 MiDrug 主模型选择了 0.0003 的统一学习率，而原结构选择 0.00003；两者内部验证差距很小。这提供了将骨干与新增分支学习率分开的动机。

**本轮已完成，结论为部分补救。** MiDrug 相对旧双流参数的 AUC/AUPR 均值增加 0.001264/0.001741，但 AUPR 仍略低于重新调参的原结构；LncDrug 的 AUPR 比旧双流降低 0.004035。整体性能与结构互补筛选未通过，全部 122 次正式训练及检查点重放已通过记录核验。详见 [最终结论](results/dual_pair_tuning_20260910/FINDINGS.md)。

本轮不重新选模型方向，也不新增监督标签。输出仍为节点骨干分数加双流配对修正分数：

\[
s(r,d)=s_{node}(r,d)+\alpha\,s_{pair}(r,d).
\]

两层、每层 16 维的双流结构保持不变。新调整包括骨干学习率、分支相对学习率、L2 权重衰减、配对输出缩放系数 α，以及配对层已有可学习残差步长的初值。步长不是固定权重，训练时仍会更新。分组 RMSprop 的形式与原优化器一致；原查询采样比例 0.5 和目标正边双向遮蔽规则继续使用。

**预先固定的搜索范围**

原结构获得 12 个配置：学习率倍率 0.1、0.3、1、3，分别配权重衰减 0、1e-4、1e-3。

双流获得 12 个配置：骨干学习率倍率 0.1、0.3、1，分别与以下四组分支设置搭配。这里是预先选定的组合搜索，不是所有变量的完整笛卡尔积。

| 分支配置 | 分支 LR / 骨干 LR | L2 | 配对输出 α | 可学习步长初值 |
|---|---:|---:|---:|---:|
| 原分支设置 | 1 | 0 | 1 | sigmoid(-1) ≈ 0.269 |
| 缓慢、收缩的分支 | 0.3 | 1e-4 | 0.3 | 0.1 |
| 较快、较大步长的分支 | 3 | 1e-4 | 1 | 0.5 |
| 快速、收缩并正则化的分支 | 10 | 1e-3 | 0.3 | sigmoid(-1) ≈ 0.269 |

学习率倍率的基数为 LncDrug 0.0005、MiDrug 0.0001。L2 同时作用于骨干和分支参数。所有搜索配置的最大轮数统一提高到 LncDrug 150、MiDrug 450，以允许较小学习率完成学习。原结构和双流使用相同试验次数、初始化种子和最大轮数，但搜索空间按各自参数定义设置。

**选择与验证流程**

1. 沿用原训练标签的固定 80/20 内部分割，分割种子 20260912。每个任务的原结构与双流各训练 12 个配置，初筛种子 666，共 48 次训练。
2. 用内部 mean(AUC, AP) 的连续五轮平均值排名，每组取前三个配置。
3. 为入围配置增加种子 0、1，共 24 次训练。对每个配置，先逐轮平均三个种子的指标，再计算连续五轮平均。由此选择一个配置及其最高平滑指标对应的末轮；相同指标取最早轮数，配置平局按 ID 顺序。
4. 冻结配置和轮数后，用全部原训练标签按 666、0、1、2、3 五个种子重训。已有外层验证只在每次固定轮数训练完成后评价。

外层重训包含五组：重新调参的原结构、旧双流参数参照、新双流参数、沿用新双流全部设置的双自身交互对照，以及双邻域传播对照。两任务合计 50 次重训，本轮正式训练总数为 122。

两种结构对照使用主模型的学习率、衰减、缩放、步长、轮数和初始化，检查固定训练条件下的算子作用。本轮没有为它们单独调参，因此不应称为“分别最优的同规模模型”。三组分配参数相同；关闭关系仍可能使辅助类型的部分参数不影响输出，有效容量不一定相同。

预设检查分别回答：新参数是否优于旧双流参数、是否优于同时重新调参的原结构、是否优于两种固定配置结构对照。各项要求两个任务的 AUC/AP 均值都增加，并且每项任务至少 4/5 个种子的平均指标增加。原默认基线的结果不会被用来代替重新调参后的原结构对照。

**内部验证已选定的参数**

以下设置在五种子外层重训前冻结，完整记录见正式输出目录的 `SELECTION.json`。两个任务的主模型均保留输出缩放 α=1，因此不能将后续收益归于输出缩放。

| 任务 | 模型 | 骨干 LR | 配对分支 LR | L2 | 步长初值 | 轮数 |
|---|---|---:|---:|---:|---:|---:|
| LncDrug | 重新调参的原结构 | 5e-4 | — | 0 | — | 93 |
| LncDrug | 旧双流设置 | 1.5e-3 | 1.5e-3 | 0 | 0.269 | 83 |
| LncDrug | 新双流设置 | 1.5e-4 | 1.5e-4 | 0 | 0.269 | 148 |
| MiDrug | 重新调参的原结构 | 3e-5 | — | 1e-4 | — | 271 |
| MiDrug | 旧双流设置 | 3e-4 | 3e-4 | 0 | 0.269 | 272 |
| MiDrug | 新双流设置 | 1e-4 | 3e-4 | 1e-4 | 0.5 | 160 |

这些是联合选出的参数组合，不能单凭新旧配置之差确定某一个参数独立贡献了多少收益。

**复现和证据**

- [训练与冻结入口](dual_pair_tuning.py)：新增独立入口，旧实验源码和结果不改动。
- [新参数与选择逻辑检查](test_dual_pair_tuning.py)：默认参数与原实现一致、输出缩放梯度、优化器参数分组、匹配结构及多种子平滑选择。
- [分析与独立重放](analyze_dual_pair_tuning.py)：重建初筛和最终选择，检查冻结设置、图支持隔离、检查点、预测分解与排名指标数值误差。
- 正式输出目录：`improve/results/dual_pair_tuning_20260910`。
- 小规模检查目录：`improve/results/dual_pair_tuning_20260910_smoke`，14 次训练及重放已通过。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.dual_pair_tuning prepare --out-dir improve/results/dual_pair_tuning_20260910
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.dual_pair_tuning run --stage screen --out-dir improve/results/dual_pair_tuning_20260910
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.dual_pair_tuning shortlist --out-dir improve/results/dual_pair_tuning_20260910
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.dual_pair_tuning run --stage confirm --out-dir improve/results/dual_pair_tuning_20260910
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.dual_pair_tuning select --out-dir improve/results/dual_pair_tuning_20260910
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.dual_pair_tuning run --stage outer --out-dir improve/results/dual_pair_tuning_20260910
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.analyze_dual_pair_tuning --out-dir improve/results/dual_pair_tuning_20260910
```

已有协议目录应直接使用 `run` 续跑，不再次执行 `prepare`。训练阶段可分别用 `--shard 0 --shards 2 --device cuda:0` 和 `--shard 1 --shards 2 --device cuda:1` 分配到两张 GPU；决策步骤需等待本阶段所有分片结束。

本轮没有打开测试划分。已有内外层验证均被多轮查看，参数空间也受到此前结果启发，属于探索性补救。初始化种子不代表新的数据划分。更换参数后涨点可以证明该配置更有效；要声称结构互补仍需结构对照支持。
