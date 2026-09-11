**支持案例驱动的药物预测器已完成性能初筛，没有涨点。**

后续已完成 [局部案例误差适配与参数补救](LOCAL_SUPPORT.md)：恢复完整原模型后，初版跌点基本补回，LncDrug 的整体 AP 有小幅增加，但尚未实现双任务、双指标共同提升。下文专门记录最初预测器的否定结果。

在保留现有 ncRNA–drug 预测任务和输入数据的前提下，新结构将当前目标关联移出编码器，作为显式支持集输入可微正则化求解层；每个药物得到一个由先验与支持案例共同决定的线性预测器。模型设计、公式及近邻文献见 [研究方案](research/association_memory_20260910/PROPOSAL.md)，解码器见 [support_operator_modules.py](support_operator_modules.py)。

本次 27 次正式训练覆盖两个任务、三个初始化，以及原模型、仅先验、均匀支持检索和学习支持检索。MiDrug 另包含采用之前选定优化参数的原模型对照；LncDrug 的此前最优原模型参数与当前主对照一致，因此不重复运行同一设置。27 个检查点均独立重放通过。LncDrug、MiDrug 相对同设置原模型的平均 AP 分别下降 1.83、3.26 个百分点。

详细结果和解释见 [FINDINGS.md](results/support_operator_20260910/FINDINGS.md)。本轮采用新的训练内部划分，未进行外层验证或测试评价；当前结果只评价未经调参的固定原型。

复现入口为 [support_operator_pilot.py](support_operator_pilot.py)。输入、配置和源码在准备阶段冻结；已有结果不会被训练程序覆盖。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.support_operator_pilot prepare \
  --out-dir improve/results/support_operator_repeat

/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.support_operator_pilot run \
  --task LncDrug --device cuda:0 --out-dir improve/results/support_operator_repeat

/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.support_operator_pilot run \
  --task MiDrug --device cuda:1 --out-dir improve/results/support_operator_repeat

/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.support_operator_pilot report \
  --device cuda:0 --out-dir improve/results/support_operator_repeat
```

两个任务的 `run` 可以并行。短跑使用另一个目录并在 `prepare` 时加 `--smoke`，覆盖全部版本各三个训练轮次。
