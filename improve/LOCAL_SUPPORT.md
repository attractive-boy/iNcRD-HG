**支持案例适配的补救实验：保留完整原模型，用配对相关案例修正预测误差。**

**结果已完成：32 次正式训练与全部重放通过。相对本轮调好的原模型，LncDrug 的 AP 增加约 0.19 个百分点、AUC 略降；MiDrug 的 AUC 略升、AP 略降。两个任务均未通过预设的双指标提升门槛，未追加正式结构对照。完整结论见 [FINDINGS.md](results/local_support_tuning_20260910/FINDINGS.md)。**

上一版固定原型的 AP 在 LncDrug、MiDrug 分别低于原模型 1.83、3.26 个百分点；每个药物内置换支持标签几乎不影响结果，见 [原型初筛](results/support_operator_20260910/FINDINGS.md)。本轮针对这两个观察调整结构与参数，不把前一版作为已经有效的方法。

本版保留原 PCA GNN、目标关联消息边和拼接 MLP。每个训练批次仍将 query 正例及其反向边移出消息图；所有 query 的正、负标签同时从显式案例库排除。可用的支持标签只来自当前拟合数据。

预测配对 q 时，用原始 PCA/药物输入特征的可学习投影计算 RNA 与 drug 两端的余弦相似度均值，选取 top-k 案例，再在所选案例内做 softmax。这里相同药物、不同 RNA 的查询可以选择不同案例，区别于上一版只按药物选择支持。top-k 的成员选择是离散操作，梯度通过被选分数、权重和案例特征传递。

原模型给出基准 logit `l0(q)`。对支持案例 j，令 `p_j=sigmoid(l0(j))`、`e_j=y_j-p_j`；支持先验预测停止梯度。RNA 与药物特征生成 rank=16 的配对基底，并附加截距。求解局部正则化 logistic Newton 步：

\[
\Delta w_q=\left(X_q^\top\operatorname{diag}\{a_{qj}p_j(1-p_j)\}X_q+\lambda I\right)^{-1}
            X_q^\top\operatorname{diag}(a_{qj})(y-p),
\]

\[
\ell(q)=\ell_0(q)+\eta\tanh(x_q^\top\Delta w_q).
\]

因此案例只对当前预测作有界修正，每个 logit 的变化不超过 η。空支持集返回当前检查点的基准分支输出。这个分支经过联合训练，其参数不一定等于独立训练原模型的参数；空支持回退和幅度限制都不保证验证性能不会下降。

案例特征与检索参数通过求解层反向传播；基准模型接受查询损失的梯度，但不通过支持置信度 p 反向传播。支持案例的基准预测可能使用包含该案例正边的上下文图，因此这里是条件拟合误差，**不是交叉拟合的校准残差**。当前 query 标签始终不进入图或案例库。零标签仍表示数据集中的采样未标记/负例，不能解释成实验确认的药物敏感性。

**冻结的调参范围**

原模型和主候选各 4 组配置，LncDrug 最多 150 轮、MiDrug 最多 450 轮。原模型搜索基础学习率倍率 `{0.3,1}` 与权重衰减 `{0,0.0001}` 的四种组合。基础学习率分别为 0.0005、0.0001，优化器仍为 RMSprop，损失为普通 BCE。

| 主候选配置 | 基础学习率倍率 | 权重衰减 | 分支额外学习率倍率 | top-k | λ | η |
|---|---:|---:|---:|---:|---:|---:|
| c0 | 0.3 | 0 | 3 | 16 | 0.1 | 0.25 |
| c1 | 0.3 | 0.0001 | 3 | 64 | 1 | 0.5 |
| c2 | 1 | 0 | 1 | 16 | 0.1 | 0.25 |
| c3 | 1 | 0.0001 | 1 | 64 | 1 | 0.5 |

检索温度固定为 0.2；修正项在训练前 20% 轮次从零线性升至全幅，验证始终使用全幅。四组是预先定义的参数组合，不能据此单独归因某一个超参数的作用。

先以 seed 666 筛选全部配置，共 16 次训练；每个任务、每个搜索模型保留前两组，以 seed 0、1 确认，再训练 16 次。每次以内部 AUC/AP 均值的连续五轮滑动均值首次最大处选择一个检查点。最终按三个种子的滑动峰值均分选择配置；各次运行保留自己的选定检查点。所有筛选、确认配置和失败结果均保留。

当前内部划分复用上一版，属于自适应探索。只使用原训练内部标签，不读取外层验证或测试标签。三个初始化不代表三个独立数据划分。

若某个任务的候选在平均 AUC、AP 上同时超过本轮调好的原模型和上一轮原模型参考，且对每个参考至少 2/3 个种子的均值指标提高，才为该任务运行额外结构对照：`local_mlp`（不读取案例结果的非线性修正）及 `case_drug`（只用药物端检索案例）。它们使用主候选选中的参数和同样训练/检查点预算，未独立调参。总参数量相同不代表有效容量相同。

固定检查点另检查关闭修正、按源药物置换支持标签、只用 drug 检索和所选案例均匀加权。仅恢复到原模型水平不能称为案例机制的创新增益；需要性能和机制对照共同支持。

**实现与验证入口**

- [local_support_modules.py](local_support_modules.py)：两端检索、有界局部误差适配、原 GNN 预测器封装。
- [test_local_support.py](test_local_support.py)：8 项数值与信息边界检查，包括有限差分梯度、查询正负标签排除、同药物不同 RNA 的案例选择、支持置信度停止梯度和输出幅度限制。
- [local_support_tuning.py](local_support_tuning.py)：冻结协议、四配置筛选、多种子确认、条件结构对照和全部检查点重放。
- [正式配置](results/local_support_tuning_20260910/protocol.json) 与 [8 次短跑核验](results/local_support_tuning_20260910_smoke/VALIDATION.json)。

复现时使用新的输出目录，以免覆盖已有协议和选择结果。以下各命令使用 `ncRNAdrug` 环境，设置 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`。每个 `run` 阶段可按 `--task LncDrug --device cuda:0`、`--task MiDrug --device cuda:1` 分别并行执行。

```bash
python -m improve.local_support_tuning prepare --out-dir improve/results/local_support_repeat
python -m improve.local_support_tuning run --stage screen --out-dir improve/results/local_support_repeat
python -m improve.local_support_tuning finalists --out-dir improve/results/local_support_repeat
python -m improve.local_support_tuning run --stage confirm --out-dir improve/results/local_support_repeat
python -m improve.local_support_tuning select --out-dir improve/results/local_support_repeat
python -m improve.local_support_tuning run --stage controls --out-dir improve/results/local_support_repeat
python -m improve.local_support_tuning report --out-dir improve/results/local_support_repeat
```

短跑在另一个目录的 `prepare` 中使用 `--smoke`。短跑强制执行所有结构对照以检查代码，即使其三轮分数未通过性能筛选。
