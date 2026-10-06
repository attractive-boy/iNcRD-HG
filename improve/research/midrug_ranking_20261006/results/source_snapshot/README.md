# MiDrug 同药物排序训练（不使用 lncRNA）

只保留发布数据中的 miRNA/drug 节点、原始表达谱/药物特征、MiMi/DrugDrug 相似关系和训练正例目标边。M1/M2/M3 与发布复现包相同，修改训练目标为 BCE + λ·BPR。原 `release/iNcRD-HG-M1M2M3` 包不修改。

负例只从原训练负例中按药物 ID 配对。比较 BCE、混合均匀/中等分数负例 BPR（λ=0.05/0.1/0.2）、均匀负例 BPR（λ=0.1）；每配置固定初始化种子 42/43/44，共 15 次训练。前 30 轮 BCE，后 30 轮线性加权，随后固定 λ，总计 300 轮。其他训练参数不变。

“中等分数”定义：每个正例每轮有放回均匀抽取 8 个同药物训练负例，按当前模型分数排序，从中间 4 个随机取一个；再搭配一个均匀候选。它是采样启发式，不宣称为实验确认的真阴性，也不同于基于正例 margin 定义的 semi-hard mining。

仅根据平均验证 AUC 选择 BPR 配置，测试阶段只评估同图 BCE 与这个配置。测试结果不用于重新选择权重；如果 AUC/AP 不能同时提升，明确报告失败。

环境沿用 `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`，两张 A40 每卡一次训练。

```bash
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/midrug_ranking_20261006
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
$PY check_ranking.py
$PY experiment.py --phase train
$PY experiment.py --phase select
$PY experiment.py --phase test
$PY summarize.py
```

`experiment.py` 校验源文件/数据/配方签名后可断点续跑。单次训练：

```bash
$PY run_ranking.py --config bpr010_mixed --seed 42 --device cuda:0 --output results_reproduce
```

所有 checkpoint 按验证 AUC 选轮。训练阶段 `validation/*_predictions.npz` 只保存验证预测，测试输入不送入 Adapter。`selection.json` 保存全部验证结果哈希；测试阶段加载已选 checkpoint，并校验验证预测重现。训练和测试均只使用训练正例构图。

运行后见 `REPORT.md`、`per_run.csv`、`paired_delta.csv`、`validation_summary.csv`、`test_summary.csv` 和 `results/verification.json`。三个初始化种子的标准差不能解释为独立划分的不确定性。
