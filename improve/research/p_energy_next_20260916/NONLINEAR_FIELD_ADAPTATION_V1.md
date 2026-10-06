# 非线性配对残差场：来源、适配与待检验假设

本候选尚未通过性能验证。当前只有数学单测与训练专用构图检查；只有历史、三个固定分区和另行冻结的 outer 全部通过后，才可能计入最终技术数。冻结前没有查看本候选的评价分数。历史评价、固定分区及历史 outer 已有开发暴露，不能称独立测试。

## 来源选择与去重

检索采用 arXiv、Crossref、OpenAlex 公开 API，时间上界为 2026-09-16。原始响应、查询、访问时间和 SHA-256 保存在本目录 `P_ENERGY_T1_DISCOVERY_V1.json`、`P_ENERGY_T1_DISCOVERY_V2.json` 及所引用文件中。没有可用学术 MCP；没有把 API 检索结果当作已阅读全文。全文下载证据见 `P_ENERGY_FULLTEXT_V1.json`。

1. Anuj Kumar Sirohi、Subhanu Halder、Kabir Kumar、Sandeep Kumar，*Enhancing Robustness of Graph Neural Networks through p-Laplacian*，arXiv:2409.19096v1（2024-09-27）与 arXiv:2511.06143v1（2025-11-08）。标题和作者完全相同，合并为同一工作，不能计作两篇独立证据。2025 记录自述获 AAAI-26 GCLR workshop 接收，不是 AAAI 主会论文的出版核验。两份全文均已取得；未采用其中存在系数表述不一致的 QP 作为运行算法。
2. Oren E. Livne，*A Near-Linear-Time Solver for Graph p-Laplacian Semi-Supervised Learning via Continuation in p*，arXiv:2607.03503v1（2026-07-03）。记录自述投稿 SIAM Journal on Scientific Computing；本地只标注预印本。取得并核读全文第 2–3 节的 p-law、驻点、grounded-node reduction、阻尼 Newton、conductance floor 和 continuation。

第一项作者仓库 `anujksirohi/pLAPGNN` 固定提交为 `15742f8901306641241773f2cdf59f30a322c54e`。Git tree 只有 README 和输出 `ANUJ` 的 `ex.py`，README 所述 `train.py` 不存在；两文件 Git blob 已核验。第二项正文所链 `https://github.com/orenlivne/np` 在本次访问返回 404。分别保留 tree/blob 与 HTTP 证据，没有运行或宣称复现作者求解器。

本地以第二项的数学能量和数值路线作适配依据。它不同于作者的 AC/LAMG 内层引擎、层级复用及 Anderson 加速，因此**不继承作者近线性复杂度结论**，也不声称重现其 SSL 数据集结果。p-Laplacian 和 continuation 本身已有更早文献；本候选的性能与创新程度需分开判断，不能凭论文年份宣称全新算法。

## 本地对象和训练边界

MiDrug 全配对空间有 `605 × 216 = 130680` 个节点，每个节点是一对 miRNA 和药物。本地学习该空间上的一个标量参考残差场 `f`，不是训练另一套实体嵌入或替换参考模型。

两实体轴分别只使用固定输入 `x_1`、`x_2`：按行单位归一后取 10 个最近邻，并保留距离阈值处的所有同值（容差 `1e-12`）。保留有向边的正平方距离中位数为 Gaussian 带宽；再取对称并集，将每轴的平均加权度归一为 1。配对图为两轴图的 Cartesian sum。全部目标边、辅助物理关系、标签和评价分数均不进入构图。原特征处理遵循既有固定 PCA 协议，不重新拟合标签相关变换。

真实训练专用构图得到 4736 条 miRNA 无向边、1616 条药物无向边，Cartesian 图有 2,000,656 条无向边，最大度 92。该结构是基于属性的相似性假设，不是生物因果图。全配对属性是已知的转导信息；不据此宣称对全新未见实体的归纳泛化。

拟合只接收训练数组与逐行已审计的 nested-OOF 强参考 logit。评价标签只在场固定和保存后进入独立评分步骤。目标边不影响构图，因而其他训练行不会借评价目标边改变相似性核。已读取的开发数据边界仍如实披露，不把缓存称为未见测试。

## 能量、控制与求解

对训练配对集合 T、固定参考偏置 a 和二元标签 y，求唯一最小值：

```text
E_p(f) = Σ_{i∈T} BCEWithLogits(a_i + f_i, y_i)
       + (1/p) Σ_{undirected (u,v)} w_uv |f_u - f_v|^p
       + (0.001/2) Σ_all_nodes f_i².
```

完整臂为 `p=3`，主对照为 `p=2`。两臂使用完全相同的图、标签、偏置、正则系数、ridge 和评分规则；p=2 的解只作为 p=3 的初值。这里比较的是非线性边流相对二次平滑的贡献，不搜索 exponent、k、权重或融合系数。

边流为 `w |g|^(p−2) g`。全节点正 ridge 使能量强凸且 coercive；即使有不含标签的连通分量也有唯一解。它们不需从图删除。原论文的平方标签保真项改为 Bernoulli 似然，其 Hessian 对角质量随当前预测变化；本地没有直接照搬原线性 ground-source 等式或 p=2 单次线性求解结论。没有测量相似性流形维数，也不声称 `p=3` 满足论文连续极限中的 `p>d` 条件。

本地用 FP64 阻尼 Newton 与 Jacobi-PCG：

- stationarity 为全空间梯度无穷范数 `≤1e-8`，最多 60 个 Newton 步；
- CG 相对容差 `1e-9`、绝对容差 `1e-12`，最多 1500 步，并重新计算真实线性残差；接受阈值在源码中为 `max(4 × target, absolute_tolerance)`；
- Armijo 系数 `1e-4`，最多 40 次二分回退；
- `1e-6 × max(w)` 的 conductance floor 只进入 Newton 线性化矩阵，原能量和梯度仍用精确 p-law。

预测为固定强参考 `+ 0.1 × clip(f_query, −2, 2)`。不额外校准，不按评价指标选系数。LncDrug 从登记开始采用逐值强参考恒等。参考、融合、恒等及作为对照的二次平滑均不另计创新。

## 检查、复用与判定

`NONLINEAR_FIELD_UNIT_V1.json` 保存五项通过证据，包括独立 incidence/autograd 导数、SciPy trust-exact 最优解、同值邻居全保留、边与标签不影响图、节点重编号、方向与查询换序/分批、无边图独立标量根。独立最优解最大差 `2.30e-11`，Hessian-vector 最大差 `1.78e-15`。

下一步训练专用真实求解必须验证两臂全空间 stationarity、保存后重放、单调能量与无标签节点 maximum principle，且不计算 AUC/AP。若数值失败，保留失败源码与证据，修复另立版本；不能看过性能后再修配方。

若训练专用数值检查通过，正式协议可按哈希原样采用该次历史训练解。这样历史阶段新增数值求解为 0、采用已有数值解为 2；不能把采用/重放当成新增拟合。三份历史参考初始化只是同一方法解的三个评分上下文，不是三次独立方法重复。每个新固定分区需重新训练两臂并分别计数。

历史门槛：MiDrug AUC/AP 对强参考的平均差都必须为正、每项至少 2/3 参考上下文为正，且两项均值均胜过二次对照；LncDrug 恒等。历史通过后才运行三个固定分区，并沿用同一联合门槛。分区通过后另行冻结 outer；在此之前不计最终成功。所有源码、输入、状态、分数与精确计数需封存和只读复算。

此家族与线性图扩散、解析核和既有图能量方法有关联。只有非线性配对能量本身通过机制对照且完整验证，才可讨论是否构成独立方向；一个正的参考增量或不同 p 值不够。
