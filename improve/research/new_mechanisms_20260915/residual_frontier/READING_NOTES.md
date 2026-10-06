# 强参考残差阶段的全文核读与实现边界

## BS-SubGNN，AAAI 2026

Fang et al., *Subgraph Encoding with Bicentric Sphere Node Labeling and Pooling for Link Prediction*, DOI 10.1609/aaai.v40i17.38490，2026-03-14，pp. 14711–14719。Crossref、AAAI 正文及作者 README 一致。公开 PDF SHA256：ea5c66d63274d33a19f5fb5890c7c8a86faa8be2031d2dfd16add4cab4a11ea8。首次直链连接被远端关闭；同一公开文章页给出的下载链接添加其下载参数后成功，未遇到或绕过登录与反机器人挑战。

已核读 Preliminaries、Bounding Distance Node Labeling、Overall Framework、式 (3)–(13) 和实验设置；作者 fang98/BSSubGNN 的属性图 model.py、utils.py、main.py 与 README 按固定 Git blob 取得，只阅读、未执行。作者节点标签为论文式 (6) 的零基形式。h=1 时为两个中心、u 独有邻居、共同邻居、v 独有邻居五类。BSSP 的 exp(tanh(score)) 是未归一化权重求和，不能擅自解释成 softmax 注意力。

本地独立适配使用查询正事实先删除的物理 Boolean 并图，在每个查询诱导一跳子图，再进行三层 32 维 tanh GCN 传播。类型化原属性线性投影与可学习球壳标签相加，三个独立汇聚支路之外加入中心状态旁路，使孤立端点仍能使用属性。输出是一个二元残差 logit，末层零初始化；作者属性图代码实际输出四类，而训练标签是二类，本地没有沿用未使用类别。层内 dropout、类型化投影、中心旁路和强参考残差似然均为本地适配，不宣称精确复现。

主候选 full 必须同时超过 no_mark、flat_pool、no_message。no_mark 只对传播隐藏球壳身份，汇聚仍看到球壳；flat_pool 的三个门控各自汇聚整个邻域并集；no_message 用恒等空间算子替代归一化邻接而保留线性模块。它与先前 NCN/邻域并集有亲缘性，机制区别在查询条件化子图传播，并以 no_message 检验；不会把标签、汇聚、传播三个组件拆算三项创新。若最终通过，也只计一个相关家族。

作者关于距离标签复杂度及随机图表达能力的定理有特定条件，本地没有验证本数据满足这些条件，不借用理论作性能保证。四项数学/查询删除/节点重编号/梯度/查询分批检查通过。真实历史双任务全五折审计覆盖每个拟合查询及评价配对，图中目标事实重叠全部为 0，边界审计未读取评价标签。已完成 32 次历史拟合并通过三初始化确认：MiDrug 对强参考 AUC/AP +0.000376858/+0.001033750，两项均为 3/3 提高；LncDrug +0.000023388/+0.000091122，没有负变化。MiDrug 相对 no_mark、flat_pool、no_message 分别为 +0.000073543/+0.000816758、+0.000019081/+0.000616886、+0.000114489/+0.000910495。LncDrug 不是在每个消融臂上都领先。固定分区和 outer 尚待验证，不计作最终成功。

## ARDiff，AAAI 2026：取得全文，暂不实施

Chen et al., *ARDiff: Anisotropic Residual Diffusion for Heterogeneous Graph Learning*, DOI 10.1609/aaai.v40i17.38477，2026-03-14。PDF SHA256：fbdf620f32f6d43cc1a5392ef1f55af34734294b2bd0a5a374f1045b2f76bf5c。已核读 Methodology、式 (1)–(15) 与实验协议，取得作者 s35lay/ARDiff 的 LP Model.py、params.py、README，仅阅读。

它沿 Chain-of-Semantics 从 View、Favorite、Cart 到 Buy 预测相邻关系表示的残差；噪声按邻域均值/方差与均值减本节点表示的轴分解为平行/正交分量，由学习权重混合。RWPE 差值作为去噪条件。公开实现使用 variance * noise + mean（没有平方根），语义混合参数是未约束实数，损失为对关系残差的 SNR 加权 MSE。其链接实验主要报告购买推荐的 Recall@20 和 NDCG@20，不能当成本项目 AUC/AP 增益证据。

当前 RNA/药物/相似性关系不存在已证实的由弱到强语义链。任意按编号或预测效果排序关系会改变关键假设；本轮暂不把 ARDiff 直接套入，未训练或计作创新。只替换噪声而没有其核心语义残差，也不能声称实现整个方法。

## 训练基础设施：不计作技术创新

nested_anchor_v1 在一致性检查处停止，两个度数列比历史缓存恰少 2；原因是物理图删除了历史图重复自环。v2 按只读原图审计恢复相似关系每节点两条自环、CircHost 一条自环，随后两任务全量特征及全五折 OOF 重建误差均 <4e−16。没有修改旧输入或源码。

每个神经查询折 Q 完全从强参考拟合支持 S 中删除，再在 S 内做五折图特征交叉拟合。Q 的结果标签不传入构造或参考拟合，预处理只拟合 S 的 OOF 特征。参考配方保持既定 TabICL 图上下文/静态证据和四经典成员，训练 offset 固定用种子 20260914。新图候选训练 BCE(OOF_anchor + residual)，评价为已存 anchor + 0.1 × 平均原始残差 logit。该变化和固定缩放不计入创新数；此前各家族的失败不改写，几何主候选仍为 attention_only。

历史训练缓存完成后，几何 attention_only 与变分群组 full 各完成 16 次拟合并通过三初始化确认。MiDrug 对强参考分别为 +0.000516391/+0.000460533 和 +0.000286619/+0.000474538；对必要机制对照分别为 +0.000298942/+0.000480154 和 +0.000031405/+0.000063764。LncDrug 最大单项损失分别为 0 和 0.000081395。状态空间完成 12 次筛选拟合，但 AUC 比 static 低 0.000341080，停止。上述数值是历史开发结果；后续固定分区使用冻结的历史中位训练步数，仍需通过分区和 outer 门槛。

固定分区参考的 v2 执行在 LncDrug split 104729/fold 2/tiered full 的训练输入核验处停止：库自身已按训练数据删除常量列，检查却比较未过滤的 62 列与保存模型的 61 列。`CONSTANT_FILTER_ACTUAL_DATA_AUDIT_V1.json` 记录第 60 列（零基）是唯一删除列，2,432 行过滤后与模型完全相同，没有生成或评价查询预测。v3 只修正这一检查，验证掩码仅由训练行决定；既有完整结果精确复制、中断模型原样续算。原源码、原缓存和原候选保持封存，验证候选以 `_cachev3` 后缀另存修正，学习率、步数、种子、主候选、对照、评价参考和门槛均未变。新旧训练/报告函数主体逐字核对一致。缓存版本和继承结果不算额外技术或独立重复。

## 追加核读：规则引导与拓扑特征的适用边界

Wang et al., **Rule-Guided Graph Neural Networks for Explainable Knowledge Graph Reasoning**, AAAI 2025，39(12):12784–12791，DOI 10.1609/aaai.v39i12.33394，出版社日期 2025-04-11。已读取 8 页正文中的 KG Encoding and Decoding、Rule Encoding and Decoding、式 (5)–(8)、Tables 1–4 及实验协议。公开出版社 PDF SHA256 为 `e961919f955e4c349df87a4c674a965ae955a0b533d6e9a999c5dc6f54a0cc95`；来源页的 gzip 字节与解压 HTML 分别保存，没有覆盖原下载。作者代码地址为 `bohemianc/rule-guided-gnns`，本轮没有取得或执行代码。

该模型把实体和实体对变成推理图节点，用非负、[0,1] 有界权重和 ClippedReLU 保持单调性，自连接矩阵对角线固定为 1。参数与同一配对的关系包含/逆关系、类型包含、domain/range 及部分长度 2 合取规则对应；根据训练 KG 的 standard confidence 对参数添加铰链式正则。正文明确说不支持直接编码 relation chains。Table 3 的 “AUC” 被正文定义为 precision–recall 曲线面积，不能当作这里的 ROC AUC；也未核实其积分实现等同于 sklearn average_precision_score。论文比较十次重新采样的负例，不能直接对照本项目固定配对、三初始化的协议。

本数据物理关系只有四种互斥实体类型；目标 RNA–drug 的每种有向类型配对只有一个目标关系。对目标关系，跨关系同配对蕴涵基本受类型排除，逆边和 domain/range 主要是已知编码约束；相似关系间的重叠出现在 circRNA 类型内。这里最有意义的跨 RNA 中介关系链却不是论文规则注入支持的形式。因此暂不实现：这不是宣称该模型无效，而是尚无有信息量、且不同于已知类型约束的目标规则作为其核心监督。该方法还使用配对重化推理图，与本项目已有表示工作有关，不能仅靠改名称宣称全新方向。

Hoppe, **Target-Edge Leakage Inflates Topological Link Prediction: A Quantification and a Corrected Protocol**，Zenodo 预印本 v1，2026-07-04，DOI 10.5281/zenodo.21181978；概念 DOI 21181977 已去重。公开 PDF 7 页，SHA256 `fb90869ea56f7c2625787ed4df8c4987a45f3ea291852abab5fbb947a4660933`。已读 §3–7、Tables 1–2、数据代码声明。文章比较完整图先算持久同调特征与每折移除测试正边后重算的协议，强调目标边泄漏可能改变方法排序；这一边界与本项目已有逐查询折删除原则一致。它是未核实同行评审的预印本，本文报告的跨图 AUC 增量和显著性没有独立复现，也不能用作本项目的预期增益。其拓扑表示后接 logistic/GBM 的路线属于显式结构特征扩展，本轮不将其计作满足用户排除条件的新技术。网页列出的分析脚本和汇总附件未下载，完整特征构造与评估 harness 声称可向作者取得；本轮未联系作者。

追加检索记录为 `DISCOVERY_REMAINING_V1.json`。六个有界 Crossref/DataCite 查询中，一个 Crossref 查询返回 HTTP 429，其余请求保留原始结果，未循环重试。文献详情及主文下载记录见 `LITERATURE_DETAIL_V1.json`、`REMAINING_PDF_MANIFEST_V1.json`；只下载上述两篇选定主文，未批量下载搜索结果或补充材料。


## 2026-09-16：固定分区的最终结果

强参考残差几何、群组、双中心共 48 次固定分区拟合及全部对照已完成。MiDrug 相对强参考的平均 ΔAUC/ΔAP 分别为 −0.000217846/−0.000475504、+0.000029417/+0.000071380、+0.000020274/−0.000021216。几何低于欧式对照，群组低于确定性对照的 AUC，双中心低于 flat_pool/no_message 的两项指标，故三项都未通过联合门槛。三者 LncDrug 均在原 0.001 容差内。三个 MiDrug 配对实体区间均包含零。

外层队列的 eligible_families 为空，未启动新 outer 拟合。30 个参考折完整，27 个为精确复用，训练 OOF 分数对齐及公式核验通过。历史正向结果和缓存修正不改变当前失败判定，最终新增成功技术仍为 0。完整分数与哈希见 ../../residual_next_20260916/PARTITION_CONSOLIDATION_V1.json；新的文献检索单独存于 ../../residual_next_20260916/。
