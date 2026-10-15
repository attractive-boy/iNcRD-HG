# 表示集中度与近期图自监督方法

检索于北京时间2026-10-07；范围截至2026-10-06，主要关注2025–2026年。
nature-academic-search / multi-source-search。学术MCP未挂载，改用公开
Europe PMC、Crossref、arXiv及GitHub API；三个预检端点均可达。
四个查询返回33条记录，按DOI/arXiv标识及标题合并。这些是发现条目，
并非33篇相关或已读论文。原查询、响应、时间、版本和代码commit均保存。

## 已核对的主要来源

| 来源 | 阅读范围 | 适用边界 |
|---|---|---|
| Tinghe Zhang等，[NodeJEPA: Structure-Conditioned Latent Prediction for Node-Level Graph Self-Supervised Learning](https://arxiv.org/abs/2608.04381v1)，2026-08-05预印本 | arXiv元数据；公开全文§3、§5方差协方差消融、附录N/P；作者仓库损失及编码器定义 | 无标签GCN的结构遮蔽、EMA目标、潜在预测、方差/协方差和SIGReg联合目标，主要评估节点分类。它没有证明本任务AUC/AP增益。 |
| Rui Chen等，[Energy-Balanced Hyperspherical Graph Representation Learning via Structural Binding and Entropic Dispersion](https://arxiv.org/abs/2512.24062v2)，2025-12-30首发、2026-04-07更新 | 元数据；公开全文§3–4及结论限制；元数据称投稿Knowledge-Based Systems，未核实录用 | HyperGRL以邻域均值对齐加归一化表示均值平方范数，配合自适应权重。其后者只约束一阶矩，不能单独保证各向同性或避免低维集中。 |
| Adrien Bardes、Jean Ponce、Yann LeCun，[VICReg](https://arxiv.org/abs/2105.04906v3)，ICLR2022 | 元数据/摘要及作者main_vicreg.py中的具体损失 | 较早的公式出处，不能称作2026年新方法。原任务为双视图图像自监督，另有投影器和不变性损失。 |
| Yunyin Li等，[ALRTGCL](https://doi.org/10.1109/TCE.2025.3593065)，IEEE Transactions on Consumer Electronics，2025 | Crossref题名/作者/卷期元数据 | Crossref没有摘要，Europe PMC定向查询没有返回该文；没有充分方法证据，因此不声称已读全文或可直接复制。 |

DC-MetaMG（2026，10.1109/JBHI.2026.3717803）摘要提到多源特征、双粒度分解，
并分别评估两类RNA；AGCLNDA（2025，10.1109/JBHI.2025.3531112）摘要提到
LightGCN、可学习增强/去噪视图。仅核对摘要，未采用其数据入口或混合RNA监督。
检索中明确的lncRNA–miRNA、miRNA–gene–disease、drug–microbe任务均排除。

## 原文、公开代码与本地方案的差异

NodeJEPA固定commit `4cb95b7988310621627478f6bf4d76585223edd9`：
- 原文§3.2写方差和协方差权重均为.2，SIGReg为.02。
- 公开variance.py用 `clamp(std, min=1e-4)`，协方差去掉对角线后对D×D元素求均值。
- 编码器在GCN层后有可学习LayerNorm。论文主体EMA趋向1，附录N写终点.999，
  本轮没有采用EMA，因此不需要自行消解这项日程差异。

VICReg固定commit `4e12602fd495af83efd1631fbe82523e6db092e0`：
- 标准差为 `sqrt(unbiased_variance+1e-4)`；边界比直接std更平滑。
- 每一视图的协方差损失是非对角平方和除以D，而不是D²；双视图协方差相加，
  两视图方差损失平均。作者原图像配方权重不等于NodeJEPA的.2。

本地适配采用VICReg的平滑方差公式和每D归一化协方差，分别计算两端查询删边后的
M1表示，然后两端等权平均；每行加无参数LayerNorm只用于辅助损失。两个系数固定
为.2是待检验的工程选择，数值上不等价于NodeJEPA代码的.2（同D下协方差项相差D倍）。
没有声称复现上述任一完整方法，也不引入新数据、教师、投影器或推理参数。
逐行LayerNorm使行和为零，因此协方差不可能严格为单位阵；目标是软正则。

HyperGRL式(10)的局限有直接反例：在二维空间，把全部点等分放在+e1和−e1上，
或均分放在±e1、±e2上，两者均值平方范数均为0，但前者秩1、后者秩2。
因此没有把该均值约束当作解决当前低有效秩的充分机制。这是对具体目标的数学
边界检查，并不是对HyperGRL全部实验结论的否定。

## 本地证据与实验

[训练表示预检](../midrug_representation_geometry_20261007/PRECHECK.md)只读取
三折初始模型及六个已验证模型，对原训练查询计算中心化协方差。未前向原验证/测试，
所有参数与缓冲保持不变。unmasked10的M1 miRNA/drug平均参与率有效秩约2.70/1.54，
第一方向方差占比约.571/.813；这些数据说明集中度，但不证明造成预测错误。

[方差—协方差实验](../midrug_variance_covariance_20261007/README.md)设置
varcov/var/cov三臂×三折，仅varcov可推进；须比原control、unmasked10及两个单项
消融的平均AUC/AP都高.002。检验范数、方差/协方差、物理删边和推理等价后才能
冻结选型。最终原测试两指标绝对+.02要求仍未达到。

所有作者源码仅阅读，不导入；没有下载作者训练数据或权重。已获取的公开文本
与元数据按manifest.json保留哈希。不是系统综述或新颖性穷尽检索。
