# 原训练集内部嵌套折外残差实验

最终目标保持：固定 MiDrug 测试 seeds 42/43/44 均值 AUC ≥0.9733929274189016、
AP ≥0.933683373732677，且 PLMF-MDA 的 MDR/MDS 不下降。
仅用现有 miRNA、药物和原训练标签，不用 lncRNA 或外部数据。

本实验落实 [先前的折外残差设计](../midrug_subgraph_20261006/NEXT_EXPERIMENT.md)。
动机来自此前联合分支收益很小，以及主干训练/内部留出之间存在拟合差距。
它是本项目的待验证训练方案，不声称来自 OCN 论文或保证提升。
近期文献与 OCN 适配结果分别见
[文献笔记](../midrug_literature_20261006/RESEARCH_NOTES.md)和
[OCN 消融](../midrug_ocn_20261006/REPORT.md)。

## 数据与训练

1. 原 9,712 个训练对的三折继续作为外层开发评价。
2. 每个外层训练部分再做 5 折 StratifiedKFold，random_state=20261006+外层折。
3. 每个内层基模型只在四折训练，固定 RMSprop 1e-4、300 epochs，最后一轮
   预测第五折。没有用被预测折标签选轮、早停或调参。共 15 个内层基模型。
4. 每个外层训练部分再训练一个固定 300 轮的完整基模型，预测外层留出，共 3 个。
   全部 18 个基模型均为纯 miRNA M123。
5. 残差训练输入为 15 个内层模型产生的 OOF logit，和相应内层训练图生成的
   子图；每个外层训练对恰好有一次 OOF 预测。原外层留出标签不参与残差拟合。
6. 子图复用已核验的纯结构 SubgraphGNN：1-hop、每类 16 节点、hidden 32、3 层，
   特征为类型/角色/距离/度数。缓存全部按对应内层图重新生成，没有冒用旧外层图缓存。
7. 残差推理用完整外层训练基模型及其训练图。训练和推理图密度不同是本方案的一部分，
   结果不能解释为只改变 offset 的单因素证据。

三种残差配置：offset_l001、offset_l01 分别优化
`BCE(OOF_logit + delta, y) + lambda * mean(delta^2)`，lambda 为 0.01/0.1；
independent25 独立训练子图 BCE，推理固定使用 0.75×base_logit+0.25×subgraph_logit。
均为 AdamW lr=0.001、weight_decay=0.001；最多 800 轮、至少 300，最优轮后
200 轮无改善停止。checkpoint 只按外层内部留出 `(AUC+AP)/2` 选择。

主要比较对象是相同 300 轮的完整外层基模型；进入原验证前还要求相对上一轮
优化过轮次的纯 miRNA M123 对照，两指标平均也都增加 ≥0.002。
这些开发规则不能替代最终测试 +0.02 要求。

## 复现与审计

解释器 `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`，设置
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`。
顺序：`launch_base.py` → `audit_base.py` → `launch_residual.py` → `audit_residual.py` → `summarize.py`。
已有同名训练结果不覆盖；新运行使用独立输出目录。

`base_verification.json` 记录 18 个 checkpoint 重放、内部索引不交叠、OOF 恰好一次
覆盖、每模型四条独立重建子图以及零残差/梯度检查。
`metadata_repair.json` 记录一次仅元数据修正：三个完整基模型缓存的
prediction_rows 最初误写为训练行范围，改为对应的预测行范围；
训练输入、全局索引、预测 logit、图数组及 checkpoint 均未变。
原生成代码及 plan 保存在 source_snapshot_base 和 bases/plan_before_metadata_fix.json。
