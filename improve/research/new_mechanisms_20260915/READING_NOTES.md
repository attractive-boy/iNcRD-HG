# 新机制查新与核读（2026-09-15）

本轮目标是至少三条**不同于既有成功方向**、实际验证能提高 MiDrug 全局 AUC/AP 且 LncDrug 至多扰动级回退的技术。以下是候选登记，不是成功清单。

## 排除的既有成功方向

- TabM、TabICL、TabDPT 或同类表格模型替换、上下文选择及模型输出融合。
- 36/62/76 列图特征、证据分层、通路及 DepMap 表达列的继续扩展。
- 共享经典学习器锚收缩；logit、rank、成员子集或权重变体只能算同一工程方向。
- REIFM 冻结图表示加属性核：已有开发证据，不能重新记为本轮新技术。

已有失败方向包括路径注意力、配对场、矩形闭环、sheaf、连续/离散扩散、BioPathNet 残差。此次审阅了已完成的 `discrete_flow_20260915`（30 次拟合）及 `biopath_head_recovery_20260915`（6 次拟合）；二者均未通过，不重新训练、不计成功。

## 1. TGSBM：变分重叠群组

Yang et al. **TGSBM: Transformer-Guided Stochastic Block Model for Link Prediction**. Proceedings of the ACM Web Conference 2026. DOI: [10.1145/3774904.3792497](https://doi.org/10.1145/3774904.3792497)。作者预印本 [arXiv:2601.20646v1](https://arxiv.org/abs/2601.20646)。Crossref 验证题名、四名作者、会议和出版日 2026-04-12；作者预印本日期为 2026-01-28。

已取得并核读作者预印本正文、式 (2)–(16)、训练设置和消融。PDF 位于 `tgsbm_export.pdf`，文本 `tgsbm_full.txt`；下载与哈希见 `alternate_source_manifest.json`。ACM 直链返回机器人验证页，未重试该页面；改用公开作者预印本，不把验证页当作论文。未找到可确认的官方 GitHub 实现；同名搜索中的硬件仓库无关。

核心是三项协同组件：真实图边加稀疏 expander 全局连接；Bernoulli 群组成员关系和 Gaussian 群组强度；Beta stick 先验及概率重构。重叠群组容许一个节点参与多个潜在角色，既不是单一聚类标签，也不是手工共同邻居计数。

必须保留原文/实现差别：

1. 原文式 (12) 将 Binary-Concrete 噪声写为单个 Gumbel；标准二元 Concrete 使用 Logistic 噪声（两个独立 Gumbel 之差）。本地实现用 `log(u)-log(1-u)`，有频率测试。
2. 原文式 (11) 印为排他 stick 权重 `v[k] * product(1-v[:k])`，训练设置又采用 Beta(10,0.1)。本地采用明确登记的 IBP 风格累积乘积 `product(v[:k+1])`、Beta(10,1) 先验；不宣称原样复现。学习的 Beta 通过可微 Gamma 比值采样。
3. 本任务是异类型二部关系：使用 RNA 和药物各自的读出，原文是对称的共享读出。保留相同类型度输入作为所有消融的共同工程部分，不把该输入的贡献归于群组机制。
4. 查询事实在计算类型度、反向边和全部注意力之前删除。原文常规链接重构协议不同；本地预测只能使用当前支持事实。
5. ELBO 各 KL 项按变量平均，并施加固定 beta=0.001；这是加权变分训练，不是未加权的精确 ELBO。后验预测平均 Bernoulli/Gaussian/Beta 抽样后的概率，不把均值隐变量的预测当作后验积分。

主候选 `full`；对照 `deterministic`（同一稀疏 Transformer 和读出，无变分群组）、`no_stick`（固定 0.5 群组先验）、`no_expander`（真实支持图，不加入全局连接）。两学习率给予全部机制相同预算。若不能超出这些对照，不支持群组机制贡献。

## 2. Hypformer：双曲表示与聚合

Yang et al. **Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space**. KDD 2024. DOI: [10.1145/3637528.3672039](https://doi.org/10.1145/3637528.3672039)；[arXiv:2407.01290](https://arxiv.org/abs/2407.01290)。作者维护代码在 2025 年补充可复用模块。

作者 README、代码及开放 PDF 已取得：`hypformer_readme.md`、`hypformer_author.py`、`hypformer_layers.py`、`hypformer_lorentz.py`、`Yang_2024_Hypformer.pdf`。来源与哈希见 `hypformer_source_manifest.json` 和 `HYPFORMER_PDF_MANIFEST.json`。

核读重点是 Lorentz 时间坐标重建、负 Lorentz 距离注意力和双曲 midpoint。只给欧式 embedding 再附一个时间坐标不自动建立有效的双曲推理；需要检查哪些运算实际使用对应几何，并对照相同输入、容量、训练预算的欧式模型。本地是固定曲率、稀疏边上的改造，不是完整 Hypformer 复现。

完整 Lorentz 模型没有胜过历史对照。根据 seed 666 筛选、在另外两个初始化之前登记的主候选为 `attention_only`：**双曲注意力和聚合，欧式解码器**。历史三种子相对共享强参考的 MiDrug AUC/AP 增量约 +0.000982/+0.001425，但超过欧式对照的额外增量仅约 +0.00013/+0.00019。不能将全部增益归于双曲几何。新划分验证已完成：MiDrug 平均 +0.000209/+0.000247，超过欧式对照 +0.000181/+0.000209，但略低于完整 Lorentz 变体，因此未通过全部登记对照门槛。增益区间包含零，不能列入最终成功清单。

## 3. 多尺度 Gromov–Wasserstein 传输：已完成，未通过

Crossref 检索得到 2025 年 *Multi-modal Knowledge Graph Link Prediction via Neural Optimal Transport*（10.1007/978-981-96-6599-0_12）及 *Cross-Network Relationship Learning via Optimal Transport for Link Prediction*（10.1109/IJCNN64981.2025.11227850）。目前只有元数据，未读全文，尚不据其名称决定实现，更不记作已验证技术。

随后取得并核读 **FALCON: Unsupervised Multi-Scale Gromov-Wasserstein Hypergraph Alignment**，[arXiv:2608.29635v1](https://arxiv.org/abs/2608.29635)，2026-08-30 预印本。材料见 `multiscale_hypergraph_ot.pdf` 及对应文本、发现记录。另取得 `multisource_wasserstein.pdf`（arXiv:2608.19914v4），只核读部分图信号/协方差章节，没有实现后一篇的方法。

本地将药物邻域作为超边，以累计度过滤建立多尺度，学习跨 RNA 类型的矩形耦合；多个尺度共享 GW 计划，代价还包括共享药物谱及 LncMi 关系。使用 degree+1 边际、epsilon=0.1、alpha=0.5、40 次阻尼迭代。全部传输与计数均在查询事实删除后构建。所有臂使用相同三列训练内校准读出；对照为尺度合并、仅属性、无容量约束、各尺度独立计划。这不是原文多图对齐任务的直接复现。

`multiscale_transport_20260915` 完成 10/10 正式拟合，两个验收门槛都未通过。完整模型的 MiDrug AUC/AP 约 0.962708/0.927827，略低于强参考；仅属性对照还略优于完整模型。LncDrug 约 0.993943/0.987284。四项数学检查和短训练成功只说明实现可运行，不能替代这些负面性能结果。

## 4. 选择性图状态空间：新划分未通过

核验并取得两篇同名近似、但不同的 Graph Mamba 论文：

- Wang 等，**Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces**，[arXiv:2402.00789v1](https://arxiv.org/abs/2402.00789)，`graph_mamba_wang.pdf/.txt`。
- Behrouz 与 Hashemi，另一篇 **Graph Mamba**，[arXiv:2402.08678v2](https://arxiv.org/abs/2402.08678)，KDD 2024 DOI [10.1145/3637528.3672044](https://doi.org/10.1145/3637528.3672044)，`graph_mamba_behrouz.pdf/.txt`。
- Dao 与 Gu，**Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**，[arXiv:2405.21060v1](https://arxiv.org/abs/2405.21060)，`ssd_mamba2.pdf/.txt`。官方 `state-spaces/mamba` Listing 1 代码已阅读并保存为 `ssd_minimal_author.py`，没有直接执行作者代码。
- 近期综述 DOI [10.32604/cmes.2026.084644](https://doi.org/10.32604/cmes.2026.084644)，2026-08-28 出版，但覆盖文献截至 2025-10；不能把综述出版时间说成原始架构的提出时间。

本地机制是类型化局部稀疏注意力加双向 SSD 记忆，节点依赖的 Delta/B/C、每头标量负 A、8 个状态坐标；按屏蔽后的支持度排序，平度用固定种子打破平局。不含序列卷积或随机重排增强，使用稠密块间收缩，不是官方融合线性时间 CUDA 实现。`static`、`no_state`、`random_order` 对照分别检验输入选择性、跨节点记忆和度序。

`state_space_graph_20260915` 已完成 32/32 正式历史拟合。在筛选阶段确定的学习率上汇总 seed 666/0/1，完整模型 MiDrug AUC/AP 约 0.963167/0.928179，增量约 +0.000429/+0.000310，均略高于三个对照；LncDrug AUC 约下降 0.000047、AP 上升约 0.000117。完整模型在筛选种子上的 AP 略低于 static，三种子均值才超过；机制优势很小，不应夸大。

固定步数取历史三个最佳检查点步数的中位数。新划分执行器 `state_space_validate.py` 单独封存；没有修改仅支持 block/geometry 的旧验证器。新划分 24/24 拟合已完成，MiDrug 平均 −0.000134/−0.000421，AP 三个划分均下降；LncDrug 平均约 −0.000273/−0.000355。性能及机制门槛都未通过。完整汇总见 `improve/NEW_MECHANISM_FINDINGS.md`。

## 5. 图掩码重建与教师目标：新候选

Hou 等，**GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner**，WWW 2023，DOI [10.1145/3543507.3583379](https://doi.org/10.1145/3543507.3583379)，[arXiv:2304.04779](https://arxiv.org/abs/2304.04779)。已取得作者 PDF、README、`models/edcoder.py` 和损失函数；核读 2.1–2.2、式 (2)–(6)、Figure 2 说明及对应代码。文件以 `graphmae2_` 开头，原 PDF 和下载哈希保留。另核验 2026 年 **Rethinking link prediction: A multi-scale graph masked autoencoder**（10.1016/j.neucom.2025.132115），但只有元数据，不能据其题名宣称复现。

本地采用监督 BCE 加固定 0.1 辅助损失，辅助部分同时重建被掩蔽的原始节点属性并预测 EMA 教师的隐表示；四个节点类型分别有 mask token 和重建头。与原工作先预训练再迁移不同，这里在查询事实屏蔽后的五个支持图上联合训练。两个随机重掩码视图取平均，类型损失等权平均；本地 latent loss 按论文式 (4) 覆盖所有节点，采用平方 cosine error，不含作者实现的额外 predictor。作者 full-batch 代码实际对 `keep_nodes` 使用指数为 1 的 latent loss，不能把两者说成完全相同。EMA 动量固定 0.99，每步更新。

主比较预先登记为完整联合目标对纯 BCE；`no_teacher`、`fixed_remask` 为组件诊断，不分别计新技术。相同优化步数不等于相同算力，辅助分支多次编码/解码，运行时间必须单列。四项数学/梯度检查及八个双任务短拟合通过，32 个正式历史拟合和 24 个固定分区拟合均完成。历史三初始化平均 MiDrug +0.000221/+0.000163，仅略胜 BCE；新分区变为 −0.000057/−0.000248，相对 BCE 的 AUC 也略低，性能和主机制门槛均失败。历史完整模型低于 no-teacher/fixed-remask，固定分区的组件差值转正，证据不一致。训练图上的教师表示诊断显示非零方差、平均配对 cosine 约 0.80/0.83；这只排除了该检查点的完全常量塌缩，不是性能证据。

## 6. 无折返边消息：新候选及同输入对照修订

Park 等，**Non-backtracking Graph Neural Networks**，TMLR 2024-09，[arXiv:2310.07430v2](https://arxiv.org/abs/2310.07430)，[OpenReview](https://openreview.net/forum?id=64HdQKnyTc)。DataCite 核验六名作者和版本时间，PDF 首页核对发表信息；已读第 3 节式 (3)–(5)、初始化与双向池化、度一节点的 begrudging return、第 5.3 节折返/无折返消融，以及复杂度说明。不能套用该论文的长距离节点任务数值到 RNA–药物链接任务。

本地使用 64 维有向边状态、四层残差更新、入边/出边分别池化。以节点入射求和和分组相减实现 cavity mean，不显式构建边×边邻接矩阵。项目中 CircCirc/CircHost 有平行端点关系；已检测并处理所有反向关系消息，不能仅删除一条反向物理边。度一端点组回退为该组反向消息均值。五项运算、梯度、平行边与标签边界检查，固定训练的数据入口检查，以及八个双任务短拟合均通过。

第一版 `cavity_graph_20260915` 完成 16 个筛选拟合，额外把采样 0 标签当作独立**观察关系**参与传播；它们不是生物学负关联。在 seed 666 中，`positive_only` 明显优于完整零标签传播方案。随后在任何新增初始化前，建立 `cavity_positive_v2_20260915`：所有臂只通过已收录正关联及既有物理关系传递消息，采样 0 的边际计数仍等量提供给全部臂。完整无折返、允许折返、仅逐边 MLP 使用相同输入和初始参数；主机制比较为前两者。原来的四个 positive_only 拟合原样复用并重放，补做新同输入对照，不能把复用记录再算新训练。这个修订仍只是一条无折返方向，不增加创新计数。

新版已完成 24/24 正式结果，其中 4 个复用。MiDrug 平均 +0.000514/+0.000576，LncDrug −0.000031/−0.000027；但对允许回退的匹配对照，MiDrug AUC −0.00002385，AP +0.00018415，未通过两项机制门槛。`HISTORICAL_ASSESSMENT.json` 已逐分数核验，不推进为成功方案。

## 7. 支持侧观测似然：原家族的实质修订

在旧候选分区失败后，封存三个 residual 版本。先在每个支持折排除该折全部正/零查询标签，拟合有固定 Gaussian 正则的行列 logistic MAP，再固定这个边际 score，替换神经模型的度数头，让图网络学习交互项。该经典 nuisance 拟合不是前沿技术，也不计创新。新方案必须超过相同 0.1 权重的仅 offset 分数，以及各自预先登记的主机制对照；其余组件结果照常报告。这是针对原失效原因的修订，不追改原判定。

`residual_graph_experiment.py` 和 `residual_graph_validate.py` 分别承担历史选择和固定步数分区拟合。新分区的六个 offset 缓存均只用支持侧标签，代码、输入和缓存已封存；两项入口检查、五项判定检查及六次双任务短拟合通过。无折返和掩码教师的单独执行器 `offset_graph_candidates.py` 也复用相同 offset，另有八次短拟合。

残差几何已完成历史 32 次拟合及固定分区 24 次拟合。新分区 MiDrug 平均 +0.000447/+0.000484，均为 3/3 正增益；LncDrug 平均 +0.000078/+0.000155，单项最差损失 <0.000084。相对欧式和仅 offset 的主对照都通过，但 95% 实体重采样区间包含零。此后又完成 24 次冻结历史 outer 拟合：主候选 attention_only 的 MiDrug +0.000002295/+0.000212304，AUC 仅 1/3 初始化增加；对 euclidean −0.000057225/+0.000023204，性能和机制门槛失败。LncDrug −0.000098736/−0.000438523，容差内。没有事后改选 full 或 distance_only，最终不计成功；该 outer 历史上已用于开发，不是独立保留测试。残差群组历史 +0.000750/+0.001134，换分区变为 −0.000770/−0.000615，再次失败。残差无折返历史平均 MiDrug +0.000398/+0.000601，但对回退对照 −0.000149/−0.000183，故未通过机制门槛。残差状态空间历史通过，后续 24 次分区拟合的 MiDrug −0.000023852/+0.000056438，对 static −0.000142316/−0.000272961，性能及机制门槛失败。残差掩码教师的 16 次筛选拟合已完成：完整模型相对 BCE 的 MiDrug AUC/AP −0.000044126/−0.000432044，暂不扩展确认种子。

## 8. 跨关系结构重建：图自监督家族的另一实质修订

Li、Wu 等，**What's Behind the Mask: Understanding Masked Graph Modeling for Graph Autoencoders**，KDD 2023，DOI [10.1145/3580305.3599546](https://doi.org/10.1145/3580305.3599546)，[arXiv:2205.10053v2](https://arxiv.org/abs/2205.10053v2)。2022 年 v1 的题名为 *MaskGAE: Masked Graph Modeling Meets Graph Autoencoders*。已核对 DataCite 版本、会议 DOI、作者 README，取得 v2 PDF，核读 §5.1–5.4、式 (6)–(11) 及 Figure 2。原框架采用边或路径掩码、边重建和被掩结构的度数回归；不等同于 GraphMAE2 的节点属性重建与教师目标。

本地 `masked_relation_graph.py` 只以**非当前目标关系**提供辅助监督，先删除本任务的拟合查询事实，再遮盖 20% 其他关系边，所有平行端点弧都从传播前删除。各关系等权解码遮盖边与类型匹配的未收录配对，并回归 log 度数；未收录配对只是结构重建零值。主 BCE 与固定 0.1 辅助目标联合训练，并使用已封存的支持侧 offset。辅助内度数损失权重为 0.01；五个支持图上的掩码固定。没有实现原论文的路径掩码，也没有原样复现其预训练/迁移流程。

主比较预先登记为完整模型同时优于 `bce` 和 `shuffled_aux`：后者逐关系置换辅助正边的目标端点、保留每个端点的边际计数，检验结构信息是否超过度数监督。`visible_aux` 仅诊断可见边重建；所有臂仍删除本任务查询。三个结构/梯度测试、两个执行入口/offset 测试及八个双任务短拟合通过。16 次正式筛选已完成：完整模型相对强参考的 MiDrug AUC/AP +0.000726287/+0.000275341，但相对 BCE +0.000097792/−0.000655486、相对 shuffled_aux +0.000351814/−0.000107075，AP 机制门槛失败，暂不扩展确认种子。它与掩码教师属于同一图自监督家族，不额外增加家族数。

还取得并核读 **Hi-GMAE: Hierarchical Graph Masked Autoencoders** 的 2026 年 v2（WWW 2026，DOI [10.1145/3774904.3792515](https://doi.org/10.1145/3774904.3792515)，[arXiv:2405.10642v2](https://arxiv.org/abs/2405.10642v2)）。§4.1–4.2 使用图粗化、从粗到细掩码和逐步恢复，§4.3 描述局部 GNN 与粗图 Transformer。首个预印本在 2024 年，2026 年为修订和会议发表。此处只作后续设计参考；当前跨关系重建版本不含这些层次模块，不能引用其图分类结果声称本项目链接预测提升。

## 9. 定向谱方法的阅读边界

已取得 **MagNet: A Neural Network for Directed Graphs**（[arXiv:2102.11391v2](https://arxiv.org/abs/2102.11391v2)）并核读 §2–3、式 (1)–(6)：相位编码方向的 Hermitian 磁拉普拉斯、Chebyshev 递推、复数激活及读出。该工作发表于 2021 年。另查到 LightDiC 作者仓库并读取 README，未把仓库说明算作全文方法核验。两者在本轮没有训练结果，不列作有效技术。来源、失败的网络请求和成功的定版本 PDF 下载分别保留。

## 10. 神经共同邻居：配对结构解码的新家族

Wang 等，**Neural Common Neighbor with Completion for Link Prediction**，ICLR 2024，[arXiv:2302.00890v4](https://arxiv.org/abs/2302.00890v4)。已核读 §4.1–4.2、式 (6)–(9)，以及补全式 (10)–(11)，并核对 GraphPKU/NeuralCommonNeighbor 作者仓库 README。PDF 为 `ncn_paper.pdf`，SHA256 `ec77791cb31cd2aeecc0a0015e84a50ef6b5c9f045beba141eb8690e115022df`。

本地 `neural_common_neighbor.py` 先用类型化稀疏注意力编码支持图，再对每个候选配对汇聚其共同邻居的可训练隐表示，和端点 Hadamard 交互共同解码。邻接集合为去除查询事实后的对称布尔支持，平行边去重、不含自环；不把节点编号作为数值特征。为控制异质图尺度，使用平方根计数归一化和类型常量，区别于原文简单求和。固定支持侧观测似然只负责边际项。这是 observed NCN 的改造，**没有实现 NCNC 软补全**，因为本任务解码器不能预测任意关系类型。

主候选 full 必须同时超过 count_only（只有可训练类型常量、没有邻居隐状态）及 node_only（仅端点）两个对照，union 为诊断。它是查询条件化的神经结构读出，与添加静态共同邻居列到表格特征库不同。四项交集/梯度/置换/查询删除检查、两项执行入口和 offset 检查、八次双任务两步拟合通过；最大重放误差约 6.30e−7。16 次正式筛选已完成：full 的 MiDrug 相对强参考 +0.000206318/−0.000327888，对 count_only −0.000791879/−0.002016531，对 node_only −0.000723901/−0.001965730，所有主门槛失败。

原 union 诊断给出较好结果，随后独立封存 `neural_neighborhood_union.py` 和 v4 执行器：主候选改为并集池化，并补上同并集 count_only 以及节点类型内固定置换隐状态的 shuffled 诊断。主对照为 count_only 和 node_only。明确承认本次主候选选择利用了第一轮结果，原 NCN 不追改为成功。32 个历史结果中复用 8 个完全相同的筛选拟合，全部重放误差为 0，其余 24 个为新训练。历史三初始化 MiDrug +0.001182651/+0.002222729，LncDrug −0.000015592/−0.000017825，主门槛通过。但追加的 24 次固定分区拟合中，MiDrug −0.001108313/−0.000947013，三个划分均下降；对 count_only −0.000897224/−0.001171866，对 node_only −0.001059814/−0.001635041。LncDrug 最坏 AP 下降 0.001371798，超过容差。三个门槛全部失败；这仍是同一配对结构解码家族，不新增家族计数。

## 11. PolyFormer：逐节点谱滤波的新候选

Ma、He 与 Wei，**PolyFormer: Scalable Node-wise Filters via Polynomial Graph Transformer**，KDD 2024，DOI [10.1145/3637528.3671849](https://doi.org/10.1145/3637528.3671849)，[arXiv:2407.14459v1](https://arxiv.org/abs/2407.14459v1)。已核对 Crossref、DataCite 元数据，取得 14 页公开技术报告，核读 §3.1–3.3、式 (6)–(10)、Algorithm 1–2 与 Appendix C；另读 air029/PolyFormer 的 README、块与主模型代码，不执行作者代码。PDF SHA256 为 `582e2bd69396d412c320b64c473429515a3f160bdefc5f328856f08f8f2ece0c`。

原模型对每个节点的多项式阶数 token 进行注意力，使用 tanh 而非 softmax，以支持随节点变化的正、负滤波系数。作者实现加入了除以平方根 head dimension 的缩放，并学习不限制符号的 bias_scale；论文伪代码省略该缩放，证明中又假设 bias 为正，不能混同。代码的 Chebyshev 算子为 L−I，本文适配明确使用规范化邻接 +A，奇数阶的符号相反。

本地 `polynomial_graph_filter.py` 在删除查询事实后对物理邻接去重、对称规范化，建立谱半径不大于 1 的算子。先将各类型原始属性和支持侧关系度投影到共享 64 维，再形成四阶 Chebyshev token，经过两层、四头 PolyAttn，最后以独立 RNA/药物读出打分。typed projection 可训练，因此每轮重新构建 token；不宣称原论文离线预计算的速度。用 1/(k+1) 阶数衰减和残差输出 dropout，不在注意力分数上 dropout；末端额外 LayerNorm。这是异质链接预测适配，不是原节点分类复现。

主候选 full 对照 type_shared：后者在完整支持图内对同类型节点的 Q/K 取均值，共享一个类型内滤波器，其他参数和预算相同。softmax、no_graph 分别诊断符号自由度与图传播，不单独计数。观测 offset 和融合沿用共同固定配方。四项谱/递推/符号/查询删除测试、两项执行入口检查和八次双任务短拟合通过，最大重放差 <6e−8。32 个历史拟合中，MiDrug +0.000574033/+0.000534000、LncDrug −0.000340425/−0.000661531，历史门槛通过；后续 24 次固定分区拟合中，MiDrug −0.000308483/−0.000654642，两指标均 0/3 正增益，对 type_shared −0.000327167/−0.000562794。LncDrug −0.000467760/−0.000884304，最坏 AP 降幅 0.001579610。追加分区三个门槛均失败，不计成功。

同次近期检索发现 2025 年 **Restricted Global-Aware Graph Filters Bridging GNNs and Transformer for Node Classification**（10.52202/085713-5423）和 **ArnoldiGCL**（10.1145/3711896.3736847）；目前只核对元数据，没有取得和验证其机制。检索及失败的 PDF 请求均保存，不把年份较新或题名相近当作已阅读、已实现的证据。

## 12. ED-HNN：成员条件化超图算子

Wang et al. **Equivariant Hypergraph Diffusion Neural Operators**，ICLR 2023，[arXiv:2207.06680v3](https://arxiv.org/abs/2207.06680v3)。公开定版本 PDF 为 `edhnn_paper.pdf`，25 页；核读 §3.1–3.4、式 (1)–(5)、Algorithm 1 和实验设置。原 arxiv.org PDF 请求返回 406，合法公开 export.arxiv.org 路径成功，记录见 `EDHNN_ALTERNATE_SOURCE.json`。作者仓库 README 已取得，模型源文件下载失败，没有声称读取作者模型实现。

论文用“成员状态＋集合汇聚”表示等变超图扩散消息，同一超边内的不同节点可接收不同输出。本地由查询屏蔽后的每条关系构造中心及邻居的星形超边，共享 phi/rho/update 算子迭代三次。phi 求和按超边基数平方根规范化，rho 接收目标节点状态、汇聚状态和 log 基数；更新使用入射消息均值、初始/当前状态和 log 入射度。宽度 64、dropout 0.1，使用关系和中心类型 embedding，不使用节点数值 ID。超边是建模分组，不是实测高阶生物关联。

主对照 invariant 以超边均值替代 rho 中的接收节点状态，从而为同一超边各成员发送相同消息，保持其余模块和预算一致。pairwise 将组拆为二元关系，no_hypergraph 删除组传播，均为诊断。GELU、LayerNorm、残差与缩放为本地适配；没有显式执行能量最小化或 ADMM，不将论文的收敛/普适表示证明套用到当前有限网络。

四项入射/等变性/梯度/查询删除测试、两项入口检查及八次短拟合通过，重放、倒序和分批误差均为 0。初筛未胜过 invariant 后，完成原先登记的另外两个初始化，共 32 个历史拟合。MiDrug 三初始化平均 +0.000360957/+0.000048055，均 3/3 为正；但对 invariant −0.000002783/−0.000050461、对仅 offset 的 AP −0.000007066。LncDrug −0.000057171/−0.000153980，最大 AP 损失 0.000319814。机制和 offset 门槛仍失败，停止扩大分区；未改主候选或主对照。

## 13. DGM：任务梯度驱动的连续邻接

Kazi et al. **Differentiable Graph Module (DGM) for Graph Convolutional Networks**，TPAMI 2023，DOI [10.1109/TPAMI.2022.3170249](https://doi.org/10.1109/TPAMI.2022.3170249)，公开稿 [arXiv:2002.04999v4](https://arxiv.org/abs/2002.04999v4)，修订于 2022-05-13。Crossref 与 DataCite 核对记录一致。PDF `dgm_paper.pdf` 已验证，核读 §3–4、式 (1)–(8)、Algorithm 1 及实验设置；作者 `lcosmo/DGM_pytorch` 的 README、`layers.py` 和 `layers_dense.py` 已取得并阅读相应类，未执行远程代码。

该工作区分连续 cDGM 与离散 dDGM。前者直接对邻接和任务损失求导，后者使用 Gumbel-Top-k 并另设图奖励损失。本次只适配 cDGM，不能声称实现了离散采样或其图损失。PDF 的连续公式打印为随距离增大的 sigmoid；作者 DGM_c 的实际代码是 `sigmoid(temperature*(abs(threshold)-distance))`。本地使用与作者代码相同的递减方向，并以单调性及梯度测试核验。

本地模型交替两次物理稀疏注意力和类型内稠密潜在邻接。64 维任务状态投影为 16 维图度量，单位规范化后计算距离，温度限制为正、阈值限制在 [0,4]，删除对角并按行规范化。原作者使用初次前向估计的中心/尺度、没有这些约束和最终行规范化；均属明确的本地适配。既有物理边保持独立通道，潜在边只连接同类型节点，不能直接补回被删除的 RNA–药物查询事实。潜在边不解释为新增生物证据。

主比较为完整模型同时胜过 detached、static、no_rewire：分别阻断邻接梯度、固定为原属性相似邻接、只保留逐节点值变换。全部参数模块相同，度量表示还共同进入值分支，因此 detached 对照的度量网络仍可训练。相同参数下 full/detached 前向完全相同，梯度路径不同。四项机制测试、两项执行入口检查及八次两步短拟合通过。16 个正式筛选拟合中，完整模型选定 LR 0.003，MiDrug +0.001314233/+0.000374506、LncDrug −0.000280656/−0.000094846；但 MiDrug AP 比 detached、static、no_rewire 分别低 0.000697847、0.000285944、0.000574400，机制门槛失败，停止追加确认。

## 14. GraphMETRO：扰动监督的对齐专家混合

Wu et al. **GraphMETRO: Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts**，NeurIPS 2024，[arXiv:2312.04693v3](https://arxiv.org/abs/2312.04693v3)。DataCite 核对作者及版本，公开 PDF 首页确认会议，已核读 §3、式 (1)–(3) 和专家架构设计。第三方 Zenodo 同名记录没有作为论文或作者来源使用。作者 GitHub `Wuyxin/GraphMETRO` 的模型、训练和损失文件已取得；raw 下载超时的两个文件改用同仓库固定 blob API 成功读取，见 `GRAPHMETRO_BLOB_MANIFEST.json`，未运行远程代码。

论文将复杂分布变化表示为若干图变换的组合。门控受变换标签监督，各专家对不同扰动建立相对于共同参考专家的表示，随后聚合并预测；任务/对齐梯度不进入门控。原作者 `MoEModel.losses` 保留参考专家两侧梯度，没有 EMA 教师；而当前 `train_moe.py` 在 ablation 注释下无条件执行 `z_dist_loss = 0`。本地实现遵循论文式 (3) 的 lambda=1，对齐损失不置零，另设 no_alignment 对照，不声称原样复现当前训练脚本。

本地使用共享 64 维 GNN、三个配对 MLP 专家和独立 32 维门控 GNN。变换为 identity、删除 15%–35% 物理边、0.1 倍节点 RMS 的 Gaussian 属性噪声及两者组合，四种等概率。删除发生在查询屏蔽后、反向边生成前；变换标签与任务标签无关。原作者每步采样三种变换、门控使用 pos_weight=4 和可选硬聚合；本地每步一种变换、无加权 BCE、软聚合、没有 warmup，均明确登记。Frobenius 对齐按训练配对数归一化，参考专家共同训练。合成扰动近似保持关联标签是建模假设，不是已观测的真实生物分布偏移或因果不变性证明。

主对照为 no_alignment、uniform、erm：依次删除对齐、固定均匀路由、只用清洁支持图训练参考专家。全部模块同样初始化，但 erm 的未使用参数不算额外训练容量。四项扰动边界/双侧梯度/门控梯度隔离/查询次序检查、两项执行入口检查及八次短拟合通过，最大重放差 <4.2e−8。32 次正式拟合已完成。初筛通过，三初始化平均 MiDrug +0.000816128/+0.000820178，LncDrug −0.000007796/−0.000020801；但对 no_alignment 的 MiDrug AUC −0.000073941，对 uniform 的 MiDrug AP −0.000037904，声明的机制门槛失败，没有追加分区。该协议承认其他已冻结实验此前已访问历史 outer 标签，自身训练仍只读取开发输入。

## 15. SWAN：空间与权重双重反对称动力学

Gravina et al. **On Oversquashing in Graph Neural Networks Through the Lens of Dynamical Systems**，AAAI 2025，DOI [10.1609/aaai.v39i16.33858](https://doi.org/10.1609/aaai.v39i16.33858)，[arXiv:2405.01009v2](https://arxiv.org/abs/2405.01009v2)。Crossref、DataCite 及作者 README 已交叉核对；2024 年为首个预印本，2025 年为本次读取的 v2 和会议发表。PDF `swan_paper.pdf` 为 18 页，SHA256 `90c195d4f60528c733c034ba12be5cc8bb905626364f81a6fa75fc17c577e437`。已核读 §2–3、式 (5)–(20)、Appendix A.1–A.4 和 E.1–E.5；作者 `gravins/SWAN` 的两份模型文件通过固定 Git blob 取得，仅阅读，没有运行远程代码。

先读取了 **Anti-Symmetric DGN: a stable architecture for Deep Graph Networks**（ICLR 2023，arXiv:2210.09789v3）的 §2、式 (1)–(7)。该方法约束逐节点权重矩阵，而 SWAN 进一步使整个线性图场反对称：对称空间算子乘反对称通道矩阵，加反对称空间算子乘对称通道矩阵。还读取了 **Non-Dissipative Graph Propagation for Non-Local Community Detection**（arXiv:2508.14097v1，元数据称 IJCNN 2025）的 §III 与式 (4)–(6)，它是 ADGN 的属性/邻接联合重建适配，本地没有实现其聚类流程，不能算第二个动力学方向。

论文的预定义算子为 Â=D⁻¹ᐟ²AD⁻¹ᐟ²、P=D⁻¹A，空间项为 P−Pᵀ。作者 `get_adj` 却在反对称分支中使用 `edge_weight[col]`，其中 col 是节点编号而非反向边索引。在四节点不规则图上，该表达式的矩阵反对称误差为 0.5，论文定义的误差为 0；见 `SWAN_OPERATOR_AUDIT.json`。本地遵循数学定义。论文 Appendix A.3 的 Euler 稳定性不等式写成 >1，与 A.1 的单位圆内条件相反；本地不把该印刷错误当作稳定性保证。

本地 `antisymmetric_graph_dynamics.py` 对已删除拟合查询事实的物理关系建立无向 Boolean 并图，关系类型仍以支持侧计数和类型化属性进入 64 维编码器。使用 W−Wᵀ、V−Vᵀ、Z+Zᵀ，16 次共享权重 tanh Euler 更新，epsilon=0.1、gamma=0.1、beta=0.5；LayerNorm 与 dropout 只置于动力学之前/之后。作者 V/Z 参数化使用上三角元素，本地使用完整原始矩阵的差/和；解码、固定支持侧 offset 和链接任务均为独立适配。没有实现 SWAN-LEARN，也没有新增表格列或预训练特征核。

主对照 no_space 删除空间反对称分支，free_channels 保留相同两种空间算子但解除三组通道约束；两者均必须在 MiDrug AUC/AP 上被超过。shallow 的 2 步传播仅作诊断。no_space 未使用的 Z 不计为训练容量，所有臂的原始参数初始化相同。四项稠密算子/Jacobian/置换与梯度/查询删除测试，两项执行入口检查及八次双任务两步拟合通过，最大重放误差 <2e−8。32 次正式拟合已完成：初筛通过，三初始化平均 MiDrug +0.000721914/+0.000835609，但对 free_channels −0.000307688/−0.000406931，对 no_space 的 AP −0.000012337，机制门槛失败。LncDrug −0.000150723/+0.000186896，最大单项损失 0.000296248，容差内；没有追加分区。反对称恒等式不等于完整训练系统的信息流恒定：有限步长、阻尼、时变 Jacobian 及编码解码均限制理论外推。

## 16. ConFIG：双任务梯度冲突的独立优化机制

Liu、Chu 与 Thuerey，**ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks**，ICLR 2025，[arXiv:2408.11104v3](https://arxiv.org/abs/2408.11104v3)。DataCite 与 PDF 首页核对会议，不能按记忆写成 NeurIPS 2024。36 页 PDF `config_paper.pdf` 的 SHA256 为 `e9230aabfeb4f86fa76ac4de6370d2417fc9802fa3e598b536bf43ba1c9b574e`；已核读 §3.1–3.3、式 (1)–(5)、Algorithm 1、Appendix A.3–A.5 中的秩条件/Algorithm 3，以及 A.12。作者 `tum-pbs/ConFIG` 的梯度和长度算子从固定 Git blob 取得并核读，没有运行远程代码。

一般形式对归一化任务梯度矩阵作伪逆，使更新与各任务梯度形成相等的方向投影，再用所有梯度在该方向上的投影和确定长度。两个非零梯度时，本地使用等价式 g=(‖g₁‖+‖g₂‖)(u₁+u₂)/2，uᵢ=gᵢ/‖gᵢ‖，并以通用伪逆定义验证。严格反向梯度返回零，因为不存在同时严格下降的方向；某个梯度为零时保留另一个。原文主文的“参数维度更大即可”需要 Appendix A.3 的秩条件；不同损失在一点的梯度仍可能相关。本地不将其扩展为无条件保证。原文还称两损失 PCGrad 与 ConFIG 方向相同，但不同梯度范数下不成立；单元测试包含实际投影定义的反例。

本地 `conflict_free_graph_training.py` 使用相同的普通 64 维、两层类型化图编码器，对 MiDrug 与 LncDrug 拟合观察分别求 BCE 梯度，再按 ConFIG、直接求和或 PCGrad 更新；main_only 对照只用当前主任务梯度。所有臂实际计算两个梯度，图、初始化、步数和参数完全相同，不把简单共享模型计作创新。每个主任务仍训练自己的模型；其原始属性基底供两个损失共用。检查发现两套历史数据的 lncRNA 属性不完全相同，因此没有把两个 PCA 数组拼接或互换。

支持图先把另一任务的历史边集替换为该阶段的拟合正例，再同时删去两个任务的当前查询折；两个观测 offset 都复用原封存拟合。另一任务的验证标签与验证配对不参与构造；配对 ID 仅在外部边界审计中用于确认重叠为零。两个历史任务和六个开发分区共 8 个输入单元、每单元 5 折均通过审计。四项数学/边界测试、两项入口测试与八次真实五步短拟合通过，重放误差为 0；短拟合没有评价性能。两项任务梯度各有 2/5 步出现负余弦，ConFIG 合成方向没有冲突；短流程本身不是最终机制证据。

本地采用论文 Algorithm 3 的“合成梯度再交给 Adam”路径，使用共同 AdamW、固定学习率候选和范数裁剪；没有实现 M-ConFIG 的交替动量加速。原始合成方向的非负内积不保证经 AdamW 后仍成立，所以正式轨迹分别保存原始梯度冲突、合成梯度冲突及实际参数步的冲突。主候选必须同时超过 sum、pcgrad 和 main_only 的 MiDrug AUC/AP，LncDrug 遵守共同容差。16 次正式单种子筛选已完成；MiDrug 较强参考 +0.000269525/−0.000162039，相对 main_only −0.000983886/−0.000940072。LncDrug −0.000109144/−0.000342797。性能、机制和 offset 门槛全部失败，停止追加种子与分区。梯度的几何性质没有转化成所需的验证增益。

## 数值复现与当前反面证据

重叠群组的硬后验抽样会放大 CUDA 非确定性图归约差异。仅改用 FP64 概率累计没有解决问题；确定性推理加 FP64 sigmoid/累计后，重载、倒序和分批预测才一致。封存的 v2 执行保留原检查点选择策略，复用 21 个已核验权重，仅补训 11 个未完成模型。历史 MiDrug 增量约 +0.000752/+0.001053，不能掩盖新划分的失败：三个新 MiDrug 划分 AP 均下降，已不能通过新划分性能门槛。

强参考的 MLP 原生 float32 推理也存在很小的批次次序差异。修正只以 FP64 执行既有权重的仿射运算，末层 logit 转回原 float32 后使用原 sigmoid，保留饱和/量化行为。没有重拟合或改融合系数；六个新参考已完成，并逐项保留原始分数与修正审计。

## 搜索边界

本轮 MCP 学术接口未挂载，采用技能允许的公共 API。OpenAlex discovery 及随后批量 arXiv API 返回 429；保留 `discovery.json` 的失败记录，转用 Crossref、arXiv 题名检索和公开作者 PDF。不是系统综述，也不证明全球首次新颖性。引用论文的原始指标不作为本项目预期增益。

## 验证口径

比较原 GNN、冻结 evidence-tiered TabICL 和最新共享锚收缩参考。后者由原训练数据重新拟合既定四成员，再逐项与原存档指标核对。完整开发结果只有在原始预测重算、重载和查询次序复现后才汇报。

历史内部划分、42/2026/31415 重划分均已反复开发，不能再称为独立测试。本轮新候选通过内部筛选后，应先冻结架构/参数/轮数，再建立尚未用于本轮选型的配对划分；这些仍是同一数据池的开发敏感性分析。最终保留测试应在模型冻结后使用。初始化或推理种子不替代独立生物验证。
