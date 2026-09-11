# 节点倾向与配对交互分解：可检验的大幅改动

本轮把单条边的二分类扩展为“边分类＋配对交换排序”，同时重写解码器和训练图采样。目标是检验 RNA–drug 配对信息能否独立于加性的节点倾向发挥作用。它不保证涨分，也不将四元组、矩阵分解或排序损失称作首次提出。

## 模型

输入沿用已清理训练图及训练节点拟合的 PCA64；保留现有异构 GNN 编码器，以便把实验集中在解码和监督方式上。新的解码器为：

\[
s(r,d)=a(z_r)+b(z_d)+h(r,d),\qquad
h(r,d)=\frac{(u_r-\bar u)^\top(v_d-\bar v)}{\sqrt{32}}.
\]

- a、b 是两个独立的小 MLP，表达 RNA 和药物各自的倾向。
- u、v 是 LayerNorm 后的 32 维线性投影。
- 两个均值分别只在当前拟合样本覆盖的 RNA、药物上计算。因此 h 在这些节点形成的笛卡尔积上具有零行均值、零列均值。
- 这一分解是函数上的约束，不意味着 a、b 就是纯粹的研究热度，也不意味着 h 已经是生物学或因果效应。

## 配对交换目标

从当前查询集取两条已知训练正边 (r,d)、(s,e)，要求端点不同，且交叉配对 (r,e)、(s,d) 不是当前拟合集正边。定义：

\[
\Delta=s(r,d)+s(s,e)-s(r,e)-s(s,d),\qquad
L_{swap}=\operatorname{softplus}(-\Delta/2).
\]

任何附加的 RNA 项 a(r) 和 drug 项 b(d) 都会在差分中抵消。相比只要求一条正边高于随机负边，这个目标比较的是两种端点出现次数完全相同的配对方式。测试覆盖了这种不变性，以及辅助损失对独立边际头的零梯度。

总损失为 BCE＋0.1×渐增系数×辅助损失，渐增到完整权重所需轮数为原训练预算的 20%。每轮最多 2,048 组交换。

未知交叉配对可能包含尚未发现的真实关联。因此这是弱排序目标，不把未知对称作已证实的阴性或药物敏感关系，也没有正例—未标注学习的一致性保证。采样仅排除当前拟合集正边，不查看内部留出、外层验证或测试标签。

仅用既有抽样负例构造完整四元组，LncDrug 只有 48 组，覆盖 57/1,266 条训练正边；MiDrug 有 3,892 组，覆盖 2,018/3,236 条训练正边。这是允许未知配对参与弱排序的实际原因。

## 查询图与对照

每个阶段先把目标关系限定为该阶段的拟合正边。除原模型组外，每轮分层抽取一半训练行作为查询，随后从两个方向移除查询正边。交换中的两条正边都来自这一查询集，因此它们在消息传递图中不可见。其余关系和输入属性保持冻结。

| 方案 | 解码器 | 每轮查询与图 | 辅助目标 |
|---|---|---|---|
| pca_full | 原拼接 MLP | 全部拟合行；完整拟合图 | 无 |
| pca_query | 原拼接 MLP | 一半拟合行；移除查询正边 | 无 |
| factor_bce | 边际＋居中双线性交互 | 同 pca_query | 无 |
| factor_pairwise | 同 factor_bce | 同 pca_query | 同样四个分数形成的四项普通两两排序 |
| factor_exchange | 同 factor_bce | 同 pca_query | 端点次数相同的联合交换排序 |

后三组具有相同初始参数及查询/交换采样种子。pca_query 与 pca_full 的对比同时改变了查询量和图中的可见边，不能单独解释为边屏蔽的因果作用；factor_bce 提供了新机制共享训练方式下的直接对照。

## 参数搜索与评估边界

1. 仅将原训练行按固定种子 20260911 分为 80% 内部拟合、20% 内部验证；内部验证正边同时从目标关系正反方向移除。
2. 每个方案、每个任务都搜索同样 6 个配置：学习率为原值的 0.3、1、3 倍，权重衰减为 0、1e-4。网络宽度、交互秩、辅助权重、查询比例不再搜索。
3. 用 seed 666 完成每个配置的完整原轮数预算：LncDrug 100、MiDrug 300。每条训练轨迹只选首次严格最大化平均 AUC/AUPR 的一个检查点；方案内再用相同准则选择一个配置。
4. 配置和轮数写入带哈希的 SELECTION.json 后冻结。使用原训练集、seed 666/0/1/2/3 重训，只跑选定轮数；过程中不计算外层验证指标。
5. 重训结束后评估既有外层验证集一次，同时报告全局指标、按药物和按 RNA 的组内宏平均，以及边际/交互分项诊断。

正式预算：60 次内部调参训练＋50 次重训。完整原模型也享有同样调参机会。五个初始化种子不等于五个独立划分；外层验证此前已被多次使用，所以本轮不是独立泛化确认。测试集没有被加载或评估。

### 提前规定的判断

- 性能目标：交换模型相对本轮调参后的 pca_full，在两个任务上平均 AUC 与 AUPR 都上升，且每个任务至少 4/5 个种子的平均 AUC/AUPR 上升。
- 机制目标：相对分别调参后的 factor_bce、factor_pairwise，在两个任务上按药物宏平均 AUPR 都上升，且每个任务、每个对比至少 4/5 个种子上升。
- 两项分别报告。全局涨分不能替代机制对照；组内变化也不能改称全局性能提升。上述门槛不是显著性检验。

## 实现与复现

运行环境为 ncRNAdrug 的 Python 3.8；主模块兼容该版本。

```bash
python -m unittest improve.test_pair_exchange -v
python -m improve.pair_exchange prepare --out-dir improve/results/pair_exchange_20260909
python -m improve.pair_exchange tune --task LncDrug --device cuda:0 --out-dir improve/results/pair_exchange_20260909
python -m improve.pair_exchange tune --task MiDrug --device cuda:1 --out-dir improve/results/pair_exchange_20260909
python -m improve.pair_exchange select --out-dir improve/results/pair_exchange_20260909
python -m improve.pair_exchange refit --task LncDrug --device cuda:0 --out-dir improve/results/pair_exchange_20260909
python -m improve.pair_exchange refit --task MiDrug --device cuda:1 --out-dir improve/results/pair_exchange_20260909
python -m improve.analyze_pair_exchange --replay --out-dir improve/results/pair_exchange_20260909
```

正式运行前先使用独立目录和 --smoke 完成整条流程。旧版短训在 Python 3.8 上发现一个字符串 API 兼容问题；失败现场保留在 pair_exchange_20260909_smoke，修正后的短训在 pair_exchange_20260909_smoke_v2。正式目录仅在修正后创建。

文献证据与不能宣称的贡献见 [新颖性核对](research/pair_exchange_20260909/NOVELTY.md)。
