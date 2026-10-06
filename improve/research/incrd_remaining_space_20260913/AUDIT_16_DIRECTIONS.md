# iNcRD-HG 的 16 个方向：原文、代码与项目实验对照

核查日期：2026-09-13。本轮为研究与代码核查，没有启动训练、读取新的测试预测或替换当前模型。后续跨领域文献建议见 [RECOMMENDATIONS.md](RECOMMENDATIONS.md)。

## 原文中需要澄清的三点

原论文为 [Wei 等，Bioinformatics 2026](https://doi.org/10.1093/bioinformatics/btag029)，公开全文 [PMC12866671](https://pmc.ncbi.nlm.nih.gov/articles/PMC12866671/)。本目录保存官方全文 XML、文本和 Crossref 元数据。

1. 用户引述的三个未来方向确实出现在原文，位置是 **第 4 节 Conclusion**。其中多任务部分写的是 “A sophisticated multi-task learning module can be introduced to extend iNcRD-HG”，随后列出跨任务一致性、任务感知对比目标和自适应损失权重。它是未来扩展建议。原发布 [predict_association.py](../../../predict_association.py) 为 LncDrug、MiDrug 分别初始化模型、优化器并训练，不能把当前实现描述成已经联合优化两个任务的简单多任务模型。
2. **原文第 3.6 节已做留一 ncRNA 和留一药物实验。** 药物实验挑选了 9 种已有超过 30 条关联的药物；ncRNA 实验选择超过 10 条关联的实体。剩余空间在长尾、家族/骨架分组、跨来源、跨癌种以及双端同时未见等更严格设置。[predict_simulated_drug.py](../../../predict_simulated_drug.py) 按目标关系删除该药物的边；如果研究所有 RNA 任务均无关联支持的药物，须另行统一屏蔽其他 RNA 关系。
3. 原文第 3.2 节已有 **HG_SeqSim**：用 Word2Vec 序列表示构造相似图，表现低于表达相似图。序列作为独立模态、局部靶位点和结构仍可研究，但“原文完全没有做过序列对照”不准确。2026 年 8 月的 [勘误](https://doi.org/10.1093/bioinformatics/btag497) 只补充共同通讯作者，不涉及这些方法或结果。

## 逐项对照

| 用户编号与方向 | 实际覆盖到哪里 | 尚可检验的增量与优先级 |
|---|---|---|
| ① 多模态 ncRNA/circRNA | 本地已有 RNA-FM、circRNA 扩展输入及相关实验；不是空白起点 | 有来源的功能调控证据与模态缺失建模，比继续拼接完整向量更值得研究；新增数据与新增算子须分开评价 |
| ② 精细多任务 | 原实现分任务训练；当前已试共享 CARE、RELI、输入适配与约束更新 | 支持集条件化的共享生成器仍可检验。不能把梯度保护写成实际 AUC/AP 保证 |
| ③ 细粒度关联 | 已有方向标签、上下文模型和记录审计 | 记录级实验类型、剂量、时间、来源及实际功能读数；需要新增数据整理，不能从聚合标签推导真实干预效果 |
| ④ 加权、signed、top-k | [weighted_edges.py](../../weighted_edges.py)、[graph_prep.py](../../graph_prep.py) 已实现并试验。历史结果中拓扑、绝对边权和符号效应不同，部分早期正结果在扩大种子后减弱 | 常规阈值/top-k 搜索优先级低。表达相关符号与功能调控方向必须区别；不能预先赋予“高收益” |
| ⑤ mRNA/protein/pathway | [target_net.py](../../target_net.py) 只把药物–靶点压成药物间重叠矩阵，再经已知 RNA–drug 关联生成 7 列统计；没有直接的 RNA–gene–drug 机制编码 | 保留具体 RNA 调控基因、药物作用方向与通路证据，而不是再投影成 drug similarity。优先做覆盖与独立增量审计 |
| ⑥ 癌种/细胞系 | [CELL_CONTEXT_MIL.md](../../CELL_CONTEXT_MIL.md) 已做 3 个来源诊断组加剩余组、多实例均值/soft-OR/随机分组对照 | 主 soft-OR 方案未通过；MiDrug 相对随机分组差异很小。真实记录对应的细胞状态与药物条件尚不是该实验的输入 |
| ⑦ 表达降噪 | [FIRST_ROUND_FEATURES.md](../../FIRST_ROUND_FEATURES.md) 已试 PCA、表达统计、AE；PCA 在原 GNN 的部分设置有益 | 属于已覆盖的输入工程。不能将弱基线上的增益预先转写为当前 TabICLv2 的剩余空间 |
| ⑧ RNA 序列/结构 | 早期 RNA-FM 特征探针；后来的 [RELI](../../TABICL_RELI.md) 已修复成熟 miRNA 并比较局部/池化 | 固定局部匹配可提高 MiDrug，但没有共同合格候选。靶位点的功能语义、已测调控方向与可学习匹配仍是未完成的假设 |
| ⑨ drug 多视图/pooling | 原代码对 padding 也取均值；首轮已比较同检查点 masked/unmasked pooling，并试过靶点重叠特征 | 药物处理后的表达变化、作用机制与 RNA 调控程序的匹配，提供比另一种 pooling 更不同的信息 |
| ⑩ HAN/HGT/关系融合 | [GRAPH_TRANSFORMER_BACKBONE.md](../../GRAPH_TRANSFORMER_BACKBONE.md) 已完成结构增强 HGT 及内部调参，没有超过后续强表格基线 | 不宜再以“换 HGT”为主贡献。可检验交互对象与计算位置的变化，例如基因程序与药物子结构先交互，再进入 ICL |
| ⑪ 自监督/对比 | 本地有表达一致性、集合去噪等尝试；同领域已有 GSLRDA、MIFNDRA 等 | 需定义具体保留什么机制、什么样本可对齐。不可默认相互作用的两类 RNA 应有相同表示或相同效应方向 |
| ⑫ 过平滑 | 原发布模型只有两层 GeneralConv；项目没有证据确立“深层过平滑”为当前 TabICLv2 的误差来源 | 低优先级。先诊断表征退化，不能把通用残差/归一化当作已定位的解决方案 |
| ⑬ 配对解码 | [STRUCTURAL_MODEL_FINDINGS.md](../../STRUCTURAL_MODEL_FINDINGS.md) 已有节点–配对联合、双流等完整对照；部分弱基线增益被相同训练设置解释 | 普通 concat/product/bilinear 可作必要对照。重点应是配对获得了哪些新的、可迁移的证据 |
| ⑭ 负样本 | 已试额外负样本与可靠负例路线；当前 0 标签主要是未观测配对 | 分层的标注选择机制与 PU 假设值得研究。高分未知配对可能是真阳性，直接 hard mining 会强化错误监督 |
| ⑮ 泛化 | 原文与本地均已有 cold-start 相关工作 | 家族/骨架/来源分组、双端未见及跨任务共同药物屏蔽；这些是验证贡献，不自动成为架构创新 |
| ⑯ 定量解释 | 原文有贡献分析、候选富集；[plot_case_drug.py](../../../plot_case_drug.py) 有 ENCORI→gene set 的后处理 | 解释稳定性、来源独立的功能边恢复、靶标扰动/救援验证；不能用模型输入的同一数据库证明解释正确 |

本表区分“已有实现/有限实验”与“整个方向被否定”。历史 [RESULTS.md](../../RESULTS.md) 中一些较早的强结论，后来已有追加种子或更公平对照修正；不据其个别段落推断所有新模态都无用。

## 两个仍有实质空间的数据接口

**直接 RNA–gene 证据。** 本地 ChEMBL 缓存有 103/216 种药物具备机制靶点、116 个靶点和 200 条边，见既有靶点实验记录。`target_net.py` 使用 `B @ B.T` 构造重叠图，丢掉了具体基因/靶标身份及 action；这次核查缓存确实包含 `action` 和 `moa` 字段。因此，旧靶点重叠探针不能否定直接机制证据的价值。但 ChEMBL target ID 与具体蛋白/复合物成员仍需映射，RNA 侧功能证据覆盖也未完成，不能称为已就绪输入。

**相互作用的方向与证据类型。** 原 case-study 代码用 `TDMDScore > 0.8` 筛选 miRNA 相关基因；代码注释明确称其为 TDMD 潜力。这不能直接视为经典“miRNA 抑制靶 mRNA”的验证分数。本轮新阅读的 [TDMD 实验论文](https://doi.org/10.1038/s44318-026-00889-8) 展示转录本诱导 miRNA 降解的反向作用。这是值得区分的机制，但具体数据库字段的版本、分数计算及覆盖仍待核验；当前没有据此修改图或标签。

## 现有硬约束

后续候选以当前图上下文 TabICLv2 为参考：MiDrug 争取改善，LncDrug **AUC、AP 均不下降**，按药物宏 AP 单独报告。两个任务使用同一组新增参数，不设置 LncDrug 专用恒等分支。

[共享输入适配的实际结果](../../TABICL_INPUT_ADAPTER_FINDINGS.md) 已说明：保护训练上的平滑 AUC/AP 损失，不能保证验证 AUC/AP 不降。投影 64 步候选的 LncDrug 内部平均差值仍为 AUC −0.000033783、AP −0.000128387。后续应保留真实指标的联合验收，而非只检查训练代理。

已有外层和内部划分被反复用于开发；新增推理种子不构成独立数据重复。新方法若通过开发筛选，正式泛化结论仍需要未用于方案选择的来源、时间批次或外部数据。
