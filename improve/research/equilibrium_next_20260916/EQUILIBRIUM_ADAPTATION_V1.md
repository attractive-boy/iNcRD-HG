# 多关系通道隐式平衡传播 v1：性能评价前的适配声明

## 依据与创新边界

候选依据 *Multi-Channel Equilibrium Graph Neural Network for Multi-View Semi-Supervised Learning*，TPAMI 2025，DOI [10.1109/TPAMI.2025.3587216](https://doi.org/10.1109/TPAMI.2025.3587216)，PMID 40690351。Crossref、OpenAlex、Europe PMC 与 Semantic Scholar 的元数据／摘要已经核验；**未取得期刊全文**。作者公开仓库为 [EricWang-CS/MEGNN](https://github.com/EricWang-CS/MEGNN)，固定提交 `eb47cba61fdf0e4f7f8e7ee3fa12153da89008b8`。

已经读取固定提交的 `layers.py`、`solvers.py`、`models.py`、`train.py`、`main.py`、`DataLoader.py` 与 `utils.py`。Git blob 下载及哈希见 `MEGNN_SOURCE_FETCH_V3.json`、`MEGNN_SOURCE_FETCH_V4.json` 与先前来源记录。以下是基于公开代码的独立适配，不能称为全文复现。首次 raw 下载超时、第二次因缺少 curl 失败的记录保留。

主假设为：在分别保留物理关系语义的通道上，学习稳定耦合并传播到平衡点，比同参数的两跳传播更能补充现有强参考遗漏的 MiDrug 信号。所测核心是隐式平衡传播；普通 attention 融合、强参考融合和精度／来源审计设施不算创新。它与已有有限层传播、谱滤波和图动力学方向存在联系，若没有区别于这些方向的机制证据，不能仅凭换模型名称计为独立新技术。

## 方程与本地实现

作者代码采用 `Z = X + gamma * A @ Z @ G`，其中 `G = F.T @ F / (||F.T @ F||_F + 1e-12)`，默认 `gamma=0.8`。每个视图独立传播、线性映射后由节点级 attention 融合。作者按属性视图构造 kNN 图，并使用 BN、dropout 和节点分类头；其最多 200 步、绝对阈值 1e-6 的求解器未收敛时仍可返回末值。

本地固定为九个物理关系通道，未添加外部关系。每次使用某查询折之前，先从原关系中删除该折所有目标事实；随后将各关系构造为二元无向图，加自环并行归一。无连接节点因此只保留自环。原属性逐行 RMS 归一，与同一 masked 图的关系度数拼接，经节点类型专属投影、LayerNorm 和 GELU 得到共享的 32 维注入量 X。各通道独立学习 F。通道输出线性映射、attention 加权并做 LayerNorm；RNA／药物端点的向量、乘积和绝对差输入配对残差头，最后一层零初始化。没有作者预训练权重。

主臂 `full` 求线性平衡点；唯一必要对照 `two_hops` 从 X 开始恰好迭代两次，即只保留 0、1、2 跳 Neumann 项。两臂参数、初始化、输入、图、attention、损失、优化器、训练预算与选择规则相同。对照用于隔离更远距离平衡项的效应，不代表所有有限深度网络的最优表现。

平衡核心采用 FP64，前向与后向均最多 200 次迭代，并显式检查 `||X + gamma*A@Z@G - Z||_F / (1 + ||X||_F) <= 1e-8`；失败直接抛错。外部投影与配对头为 FP32，关闭 TF32／AMP。隐式后向独立求解 `U = upstream + gamma*A.T@U@G.T`，使用 `dX=U`、`dG=gamma*(A@Z).T@U`，再由 autograd 经过归一 Gram 求 F 的梯度。只支持一阶导数。

数学依据由本地独立检查：G 为半正定，谱范数不超过 1；A 为行随机矩阵，谱半径为 1；gamma 小于 1，因此线性固定点唯一，Neumann 迭代收敛。这不是对未取得期刊全文中定理的复述，也不保证相对原始属性获得性能收益。转置伴随迭代使用同一谱半径条件，并另行核验实际残差。

## 数据边界与预算

训练使用严格 nested OOF 强参考偏置的 BCE，预测为原评价强参考 logit 加上 0.1 倍五张支持图的平均原始残差。参考保持 `0.7*logit(tiered_reference) + 0.3*mean(logit(HGB, ExtraTrees, MLP, LogisticRegression))`，四个经典成员沿用冻结 context76。参考设施不计创新。

主臂／对照均预先使用 lr=0.001、0.003，seed=666，各 400 次参数更新、每 20 步评价。每臂由共同标准选择一个学习率；筛选全部门槛通过后才追加 seed=0、1。随后固定步数、学习率验证三个预定开发分区；全部通过后才进入冻结 outer 重查。历史划分、三个追加划分与历史 outer 均有开发暴露，不称独立测试。本候选尚未读原始保留测试源。

LncDrug 预先指定精确强参考恒等，不建新图、不训练参数。恒等记录不算拟合，也不支持跨任务泛化的主张。主臂 MiDrug AUC/AP 必须同时正增益，每项至少 2/3 初始化／分区为正，且两项均值胜过必要机制对照；LncDrug 任一指标下降不得超过 0.001。零残差必须严格恢复参考。任何阶段失败即停止该冻结版本。最终成功数只有在全部阶段通过后才可增加。

## 性能前核验

五项单元检查已通过：与稠密 Kronecker 解一致；隐式梯度与稠密求解和有限差分一致；两跳截断及远距离反例；未收敛／非法配置拒绝；查询屏蔽、相同初始化、非零读出换序分批和 LncDrug 恒等。真实 MiDrug 两臂各五步训练探针与两条恒等重放通过，最大重放误差 1.49e-9，峰值显存 0.137 GiB；40 个任务／折边界检查通过。探针没有计算性能指标。参见 `EQUILIBRIUM_SMOKE_V1.json` 和 `EQUILIBRIUM_UNIT_V1.json`。

正式执行必须经过 `anchor_equilibrium_experiment_v1.py`，保存所有前后向求解诊断；不能绕到普通执行入口而遗漏检查。所有输入、源码、协议和正式结果保留哈希。此文档在任何本候选正式性能评价前冻结，不预先宣称有效。
