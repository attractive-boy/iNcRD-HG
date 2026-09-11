# 路径注意力平滑：固定的 MiDrug 追加实验

入口是 `smooth_path_attention.py`，正式目录是 `results/smooth_path_attention_20260908`。
这轮以边属性/路径实验中验证集表现最好的 MiDrug 路径方案为起点。
原注意力在多个路径间的权重较集中，且等权干预在验证与测试上的方向不一致，
因此检验更弱的选择能否改善稳定性。

| 实验组 | 路径权重 |
|---|---|
| base | 原模型，无路径分支 |
| path_uniform | 路径均匀池化 |
| path_attention | 原 softmax 注意力，T=1 |
| temperature4 | softmax(score / 4) |
| shrink25 | 0.25 × softmax(score) + 0.75 × uniform |

两个新方案的数值在训练前固定，不扫描温度或混合比例。
零路径输出、单路径权重、路径集合、查询/值投影、路径计数、解码器及训练日程均与对应原方案一致。
它们不增加新的训练参数。

仍用种子 666、0、1、2、3，每组 300 个 epoch、RMSprop、学习率 0.0001、无权重 BCE。
5 组 × 5 种子 = 25 次训练；每个 base 另作同设备严格重复，共 30 次。
只按验证集 AUC/AUPR 均值选一个 checkpoint，最早的完全相同值优先。
每组的两个指标来自同一个模型。

先通过 `test_smooth_path_attention.py` 的权重约束、平滑性质、对照梯度/更新等价性及梯度有效性测试。
短跑目录 `results/smooth_path_attention_20260908_smoke` 对 5 组各重复两次，共 10 次。
短跑用于检查执行与保存，不用于模型选型。

正式汇总同时核对原模型、均匀池化和原注意力与上一轮的完整曲线、模型参数及预测是否完全一致，
并重建第一个预设种子的 5 个 CPU checkpoint 及干预预测。
`CONTROL_REPLAY.json` 记录对照复现，`VALIDATION.json` 记录全部检查。

在仓库根目录使用 `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`：

```bash
python -m unittest improve.test_smooth_path_attention -v
python -m improve.smooth_path_attention init --out-dir improve/results/smooth_path_attention_20260908
python -m improve.smooth_path_attention run --out-dir improve/results/smooth_path_attention_20260908 --device cuda:0 --shard-index 0 --shard-count 2
python -m improve.smooth_path_attention run --out-dir improve/results/smooth_path_attention_20260908 --device cuda:1 --shard-index 1 --shard-count 2
python -m improve.smooth_path_attention summarize --out-dir improve/results/smooth_path_attention_20260908
```

两个 worker 可同时运行，写入互不重叠的目录。已初始化目录使用 `run` 续跑。
源码和输入通过哈希冻结，源码副本保存在 `source_snapshot`。

本轮是在观察上一轮验证/测试结果后提出的追加探索；当前测试集不是一个未被观察过的独立确认集。
若获得候选，需要用新的数据划分检验，不能将同划分多种子的提升称为已证实的泛化优势。
