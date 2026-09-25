**FORGE 启发的联合功能回归未通过外部留出细胞门槛，没有进入 RNA–药物关联训练。**

2026-09-14。完成四组、五折共 20 次拟合。每个细胞在每组中恰好作为留出对象一次；两类功能实验、表达标准化和输出可用性筛选都只使用训练细胞。没有读取 RNA–药物关联标签，没有拟合计划中的四个全数据模型，也没有增加一项验证有效的关联技术。

|预测终点|训练均值基线的标准化 MSE|联合模型|仅药物模型|联合模型胜过均值的折数|
|---|---:|---:|---:|---:|
|CRISPR gene effect|0.894308448|0.980462048|0.894308448（该头不训练）|0/5|
|全部 PRISM 药物活性|1.010799413|1.106422551|1.092847391|0/5|
|精确匹配项目药物的 PRISM 活性|1.011225049|1.055742007|1.044873215|0/5|

MSE 越低越好，表中是各折指标的均值。联合模型的药物 MSE 在五折中均高于仅药物训练组，均值差为 +0.013575159；该辅助任务没有支持预设的药物预测增量。联合模型的药物宏 Pearson 为 0.063712，匹配药物为 0.182682，CRISPR 为 0.104872。正相关不能替代误差与机制门槛。

同谱系内打乱细胞的对照已完成；联合模型的药物 MSE 胜出 5/5，基因 MSE 胜出 4/5。但随机编码器两折存在明显外推异常：训练中恒定的 LINC01256、HAO2-IT1 在留出细胞中非零，按冻结的 SD 下限 1e-5 标准化后，极值分别约 7,039 和 23,879。其他低变异列也存在大幅外推。随机组基因 MSE 均值 34.272558、药物 MSE 3.636298 受此影响，不能据此宣称联合训练机制优越。这是拟合完成后发现的诊断，未据它调整当前协议或选择模型。即便不看随机组，训练均值和仅药物对照已分别给出 0/5 的失败证据。

模型是 rank 32 的共享线性 RNA 表达编码器、独立基因效应与药物活性头。固定 200 轮、AdamW 学习率 0.01、维度归一化权重惩罚 0.1；缺失实验无损失、无梯度。它是对 [FORGE，Nature Communications 2026](https://doi.org/10.1038/s41467-026-73977-2) 共享功能监督思路的独立适配。原文使用靶点相关的稀疏因子、dependency probabilities 和 IC50，本实验使用全局连续 gene-effect 与 PRISM 活性头；不能称为 FORGE 复现，更不能由此次失败否定原论文或所有功能迁移方法。

数据来自 Bioconductor ExperimentHub 的公开登记归档：EH7554（22Q2 CRISPR）、EH7558（22Q2 元数据）、EH7530（PRISM 19Q3 测量及 21Q2 化合物元数据更新）。记录与合法下载来源见 [获取记录](research/dependency_transfer_20260914/experimenthub_selected_records.json) 和 [数据审计](results/dependency_data_20260914/DATA_AUDIT.json)。未绕过 DepMap 门户的人机验证；原本希望取得的较新 dependency-probability 数据仍未取得。

对齐结果为 725 个细胞、1,927 个 RNA 表达特征、17,386 个基因及固定 HTS、2.5 ± 0.001 µM 条件的 3,939 个化合物。625 个细胞有 CRISPR，543 个有 PRISM，443 个两者都有。两个 RNA 表达文件的行顺序不同，按 ModelID 重排；没有新增对数变换。216 个项目药物中，34 个精确匹配 canonical isomeric SMILES；保留电荷、立体化学和片段，解析失败或盐形式不一致的结构没有模糊归并。另 92 个药物有映射后的靶基因注释，当前外部回归未使用该注释训练。gene effect 不是 0–1 依赖概率，PRISM logfold viability 不是 IC50，二者均不是 ncRNA 干预效应。

五项针对性测试通过。全部留出预测在重新加载模型、逆序和改变批量后复现，最大绝对误差 2.3841858e-6；从保存预测重新计算全部指标，源文件、输入、模型和预测哈希通过。五折来自同一归档队列，不能称为五个独立生物队列。

可复核入口：[冻结协议](results/functional_response_transfer_20260914/protocol.json)、[完整评估](results/functional_response_transfer_20260914/ASSESSMENT.json)、[逐折指标](results/functional_response_transfer_20260914/metrics.csv)、[重放与完整性检查](results/functional_response_transfer_20260914/CV_COMPLETION.json)、[事后尺度诊断](results/functional_response_transfer_20260914/POSTHOC_SCALE_DIAGNOSTIC.json)。实现为 [functional_response_transfer.py](functional_response_transfer.py)，输入构建为 [functional_response_data.py](functional_response_data.py)；二者已封存，不应直接修改。
