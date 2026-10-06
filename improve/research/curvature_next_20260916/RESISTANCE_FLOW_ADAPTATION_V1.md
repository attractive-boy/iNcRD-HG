# 阻力曲率流：来源、边界与预先确定的适配

本候选尚未观察本地评价分数。它检验的是：在训练查询整折删除后，
由实体隐表示形成的亲和图，能否通过全路径阻力曲率的可微演化，
提供超过强参考及必要机制对照的 MiDrug AUC/AP 增益。
成功计数仍为 0；实现正确、来源核验及短训练均不算性能成功。

## 检索与排除

- 保存 arXiv 与 Crossref 原始响应、查询、访问时间及全文来源。
  搜索记录见 `ARXIV_DISCOVERY_V1/V2.json`、`CROSSREF_DISCOVERY_V1.json`、
  `CURVATURE_FULLTEXT_V1.json`，全部使用公开接口，没有可用学术 MCP。
- *Depth-Adaptive Graph Neural Networks via Learnable Bakry-Émery Curvature*，
  arXiv:2503.01079v1；Crossref 核验 KDD 2025，DOI:10.1145/3711896.3736905。
  核读正文 §3–6，并固定 BEC-GNN commit
  `cc95da1cc32239d9ec25d0af3ecd0bae740c3363`。
  完整模块导入因本环境缺少 torch_geometric 失败，未运行作者完整 GNN。
  V2 审计只执行列明的未经修改 AST 定义：发现 kappa[N,1] 与 gamma[N]
  广播成 N×N；作者硬 mask 对独立下游聚合的曲率参数梯度全部为 None；
  GraphSAGE 的普通 Python list 为静态注册风险。
  两节点检查显示作者 Gamma2 函数符合算子定义，论文展开式少 1/2。
  详见 `BEC_AUTHOR_AUDIT_V2.json`，这些有限证据不构成作者完整实验失败。
  本轮不采用 BEC 路径。
- *Geometric Evolution Graph Convolutional Networks: Enhancing Graph
  Representation Learning via Ricci Flow*，arXiv:2603.26178v2。
  已读 §4 及附录 B；未核验作者代码，作为备用来源，未实现或评价。

## 选用来源与数学契约

*Dynamic Graph Structure Learning via Resistance Curvature Flow*，
arXiv:2601.08149v1，2026-01-13 预印本。核读正文 §3–4。
论文主仓库 cqfei/RCF 存在但为空，commits/HEAD 返回 HTTP 409；
原响应保存在 `rcf_commit_v1.json`。
这里采用论文脚注链接的独立库 cqfei/resistance_curvature，固定 commit
`30aad3daf194c26359146e25a5c193d1066f02c1`；README、核心 Python、许可证
均经 Git blob 校验，清单为 `RESISTANCE_AUTHOR_SOURCE_V1.json`。
这不是主仓库的应用实验复现，不继承论文速度、规模或复杂度结论。

每个实体轴的邻接支持仅取固定属性的对称 10NN 并集并保留边界同距点。
学习投影后的单位向量 z 产生边权 w_ij=exp(-||z_i-z_j||²)。
对实际正边计算三次以下步骤（FP64，步长 0.1）：

1. L=diag(W1)-W，Q=(L+0.001I)^(-1)，
   R_ij=Q_ii+Q_jj-Q_ij-Q_ji。这是带 ridge 的有效电阻计算；
   字段 minimum_true_resistance 表示全路径分支，不能解释为未经
   正则化的精确 Laplacian 伪逆电阻。
2. p_i=1-0.5 sum_j w_ij R_ij；
   kappa_ij=(p_i+p_j) min_edge(R)/R_ij。
3. 触及无权叶节点的负曲率边取反号。
4. d_ij=1/w_ij，以 d 加权的曲率均值为中心 c；
   d'_ij=d_ij[1-0.1(kappa_ij-c)]，w'_ij=1/d'_ij。

上述尺度（原曲率乘 min_positive_R/2）、叶规则、距离加权中心均跟随
固定版本的库。论文 Eq.5 写普通边平均；二者差异明确保留。
每次逆检查相对残差 ≤1e-10，正边电阻与距离乘子必须严格正且有限；
若失效就停止，不裁剪、不动态调步长。

本地只在实际正边做除法，并采用非原地更新。
作者前向一致性最大误差 7.0144e-12；其原地反传报错原文已保留。
本地梯度检查通过，不声称作者反传通过。
独立 SciPy 全流核验最大误差 2.0429e-14。
证据为 `RESISTANCE_FLOW_UNIT_V2.json`；V1 失败源码、日志与结果保留。

## 本地任务适配与对照

普通两层 typed attention，width=32、heads=4、dropout=0.1。
只处理 miRNA 与 drug 两种实体；metric width=16、固定带宽 1。
演化后的 W 加自环并作对称归一化，传播实体表示，残差线性层加 LayerNorm。
配对读出使用两端、逐维乘积与绝对差；最后线性层零初始化。
对 MiDrug 拟合嵌套 OOF 强参考上的 BCE 残差，预测权重固定 0.1。
LncDrug 预先声明精确恒等，不新增神经训练。

- full：使用上述带 ridge 的全路径有效电阻。
- local_resistance：以 1/w_ij 替换 R_ij，去除替代路径信息。
- no_flow：输出初始亲和图，取消曲率演化。

三臂使用完全相同的参数模块、初始化和随机流。两个对照仍计算同等数量的
逆及流步骤，未用的分支乘零保持计算图；不以此宣称各臂有同样的有效梯度。
数值诊断计数相同，实际梯度由相应机制决定。
正式每个模型 400 次优化、每 20 步选点、两个学习率 0.001/0.003；
seed 666 筛选合格后，才运行 seeds 0/1。

既有 DGM/cDGM 已探索学习亲和图。因此本候选归于图结构学习相关家族，
不可仅因曲率名称或 2026 日期重复计为全新家族。可检验的新贡献限于
全路径电阻驱动的可微图度量演化，必须同时胜过 no_flow 及 local_resistance。
借用 nonlinear_pair_field_v1.entity_neighbors 仅为固定属性构图基础设施，
没有采用或重新计数此前非线性配对场。

## 验证顺序与声明限制

先做三臂各五步训练专用 GPU 短测及 40 个任务/划分/支持折边界审计。
训练查询回放只使用删除其整折的支持图，不在其余可能暴露目标边的图上平均。
短测不读评价标签、不算 AUC/AP。之后冻结源码、协议、输入及队列哈希。

历史、三个固定开发分区、另行冻结 outer 必须依次通过联合门槛：
MiDrug 两指标平均增量均正、各至少 2/3 重复为正，且均值胜过全部机制对照；
LncDrug 单指标绝对损失不超过 0.001。失败即归档停止该版本。
历史评价、固定分区及历史 outer 均已开发暴露，不称独立测试。
不读取原始保留测试源，不覆盖旧失败，不在看到结果后改流步数或融合重试。
