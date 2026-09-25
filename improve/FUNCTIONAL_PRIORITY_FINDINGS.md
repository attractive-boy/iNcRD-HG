**第一轮功能证据优先实验已完成：保留原基线，新增 RNA 功能表示管线已通过试运行。**

日期：2026-09-13。任务承接[前沿文献评估](research/frontier_reassessment_20260913/REPORT.md)的顺序：负采样偏差审计 → 功能证据与局部交互 → RNA 功能表示。
本轮完成实际推理和对照。新增药物证据的 8 个候选均未满足预先冻结的共同验收条件；没有读取新一轮外层数组，没有评价原始测试集，也没有替换已接受模型。

**主要结果。**

ChEMBL 37 的 9,485 条候选记录经人类单蛋白、直接高置信度结合实验等筛选，保留 6,108 条活动记录。药物靶点覆盖从 93/216 增加到 156/216，边数从 645 增加到 2,227。新增结合证据未推断激活／抑制方向。
模型维度始终是 36 个原图特征＋26 个证据特征。expanded 完全复用旧 26 列定义，只扩充 drug–gene 矩阵；target_local 保留前 14 列及各证据层的 RNA 覆盖／计数，将两个相似度改为逐靶点通路匹配的均值与 top-3 均值。
错配对照保留全部旧边和新增图两侧的度数。1,582 条新增边经过 31,640 次成功交换，76.17% 的新增对应关系发生改变。该对照是一次有限混合，不是均匀随机图或显著性检验。
正式阶段验证 84 个拟合状态；每个状态保存后重载并核对全部查询预测。此前 28 个 smoke 状态也通过重载、查询反序和分批检查。

**以下都是同一个历史内部开发集上的三种推理种子均值。** 三个种子不等于三个独立数据划分。alpha 表示替换原分层证据分支的比例；图分支权重始终为 0.5。

|任务|方案|AUC|AP|药物宏 AP|ΔAP（百分点）|
|---|---|---:|---:|---:|---:|
|MiDrug|baseline|0.962592863|0.927203623|0.719360585|+0.000000|
|MiDrug|expanded_0p10|0.962623075|0.927223826|0.719364262|+0.002020|
|MiDrug|target_local_0p10|0.962617510|0.927212290|0.719364262|+0.000867|
|MiDrug|expanded_1p00|0.962676742|0.926991032|0.718181599|-0.021259|
|MiDrug|target_local_1p00|0.962724445|0.927047070|0.719293524|-0.015655|
|LncDrug|baseline|0.993750211|0.987141085|0.938628852|+0.000000|
|LncDrug|expanded_0p10|0.993758007|0.987150346|0.945604905|+0.000926|
|LncDrug|target_local_0p10|0.993760606|0.987154584|0.945604905|+0.001350|
|LncDrug|expanded_1p00|0.993672251|0.986994613|0.930942593|-0.014647|
|LncDrug|target_local_1p00|0.993674850|0.986996024|0.930372156|-0.014506|

alpha=0.1 时有很小的平均收益，但 LncDrug 仍有个别种子的 AUC/AP 下降。更强替换时 MiDrug 的 AP 下降，故不能通过“LncDrug 每种子两项不下降、MiDrug 两项均值提高且各至少 2/3 种子提高”的共同条件。本轮未继续细扫更小系数，也未对 LncDrug 设置绕过分支。
完整结果与每候选验收见 [REPORT_inner.md](results/tabicl_functional_20260913/REPORT_inner.md)、[inner_ASSESSMENTS.json](results/tabicl_functional_20260913/inner_ASSESSMENTS.json)。正确对应关系相对错配对照没有呈现稳定优势，不能把新增覆盖写成经验证的功能增益。

**频率偏差与药物内排序。**

已完成的[采样审计](results/sampling_bias_audit_20260913/REPORT.md)显示，只用支持集药物正例度数的 AUC：MiDrug 约 0.936、LncDrug 约 0.977。它说明随机配对评价含强烈的实体频率信号，不能据此认定该信号全来自发表偏差。
下表仅在同时有两类标签的药物内，重加权未知配对，使正负类药物边际分布一致。它保留筛选后样本的正例比例；同集合未加权对照和覆盖分组另存于 diagnostics。未知配对不是真实实验阴性。

|任务|方案|匹配 AUC|匹配 AP|保留配对|保留正例|
|---|---|---:|---:|---:|---:|
|MiDrug|baseline|0.665200|0.736505|1099|591|
|MiDrug|expanded_1p00|0.662339|0.732485|1099|591|
|MiDrug|target_local_1p00|0.662291|0.732414|1099|591|
|MiDrug|expanded_rewired_1p00|0.663348|0.733978|1099|591|
|LncDrug|baseline|0.751182|0.929930|268|212|
|LncDrug|expanded_1p00|0.751182|0.929881|268|212|
|LncDrug|target_local_1p00|0.750480|0.929620|268|212|
|LncDrug|expanded_rewired_1p00|0.753536|0.930622|268|212|

这些 AP 只能在相同匹配分布内比较，不能与上表全体配对 AP 直接作差，也不能跨任务比较绝对高低。新增靶点证据在 MiDrug 药物内排序上没有稳定收益。

**RNA 功能表示已跑通，尚无下游提分结论。**

miRBind2：对 35 个真实 50 nt 窗口完成推理，覆盖 27 个 miRNA、15 个药物靶基因，输出结合预测及 30 维局部特征。窗口通过已记录位点与当前 RefSeq NM 转录本的唯一正向精确匹配生成；没有用 N 补齐 mRNA 侧翼。成熟 miRNA 按官方训练规则补齐至 28 nt。当前 RefSeq 版本不一定等于原实验异构体。
PARNET：对 32 个完整 lncRNA 转录本完成推理，长度 430–5588 nt，输出 512 维全局平均表示及五段局部表示。公共权重含 223 个 eCLIP 轨道，模型参数 7,534,815。所有输入均为完整序列，未截断。
两者重载误差均为 0，查询顺序检查通过，权重与运行状态未被修改。[全部 pilot 产物](results/rna_functional_pilot_20260913/)已保留协议、实体索引、序列来源、特征和验证记录。它们是技术可用性验证，不是独立准确性评测，不是新的耐药标签。
PARNET 当前主分支与公开旧权重类名不兼容；使用发布该权重时的提交 ea68bd62861a0658fa6c1a81c8c4f8736e2ed2ed。执行原始 RBPNet/序列编码定义及原始层代码，只排除无关训练导入。当前提交、兼容提交和文件 Git blob/SHA256 均归档。该公共历史权重不能直接宣称等同于论文每项最终实验；其 K562/HepG2 训练背景也不能证明广泛细胞背景迁移。

**关于更多表征与 GNN。**

表征增多本身不会决定 GNN 变差。新增高质量关系可能帮助消息传播；噪声关系、冗余拼接和训练数据不足也可能扩大过拟合。本轮没有增加列数，也没有进行同信息 GNN／TabICL 架构比较，因此只支持对本轮证据与编码的判断。后续合格的架构对照应使用同一批证据和同一划分，分别限制输入投影与模型容量、审计训练图泄漏，并报告各任务和药物内排序。

**复现。** 在仓库根目录，使用 /data1/licheng/incrd-hg/.venv-tabicl/bin/python。已完成目录是冻结产物；重新下载／构建须使用新数据目录，重新实验须使用新实验目录，不能覆盖后重新选模。复用现有证据时可跳过下载。

```bash
python -m unittest improve.test_functional_evidence_data improve.test_tabicl_functional improve.test_rna_site_windows
python -m improve.functional_evidence_data all --out-dir NEW_DRUG_INPUTS
python -m improve.tabicl_functional prepare --out-dir NEW_EXPERIMENT --drug-inputs-dir NEW_DRUG_INPUTS
python -m improve.tabicl_functional run --out-dir NEW_EXPERIMENT --task MiDrug --device cuda:0
python -m improve.tabicl_functional run --out-dir NEW_EXPERIMENT --task LncDrug --device cuda:1
python -m improve.tabicl_functional report --out-dir NEW_EXPERIMENT --phase inner
```

RNA 依赖使用独立的 pilot_dependencies 目录，没有修改冻结 TabICL 环境；版本与下载来源保存在 data_external/functional_priority_20260913/rna_models/pilot_dependency_report.json。公共代码和权重由 rna_functional_sources.py 固定提交并验证 Git blob；窗口由 rna_site_windows.py 构建；表示由 rna_functional_pilot.py 生成。完整命令见各脚本 --help。
截至本轮，缺少可直接使用的记录级 RNA 干预／药敏方向标签和已下载 CRISPR 依赖性矩阵。结合强度、RNA 调控方向、细胞上下文和耐药效应应保持语义区别。下一轮优先检验功能表示的同信息、同维度对照，或补真正的干预证据；本轮不支持继续盲目增加拼接维度或只更换图算子。
