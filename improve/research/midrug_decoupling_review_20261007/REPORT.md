# FlexBIP近期论文核查：不直接以新名称启动旧机制

上一目标轮为progress：9次药物token拟合、重放、审计及独立诊断已完成；本轮先重查了
归档状态和运行进程，未发现仍在运行的训练。原双指标+.02目标未完成。

Wu Q等，**Decoupling topological and molecular features for interpretable biomolecular
interaction prediction**，Briefings in Bioinformatics，2026-09-01。
DOI https://doi.org/10.1093/bib/bbag471 ，PMCID PMC13544631。
从上一轮Europe PMC已存检索结果定位，获取OA全文XML；部分精读Modular architecture、
Multi-modal fusion、Sequence attention、Adaptive graph processing、Unified loss。
数学抽取保留在methods_extracted.txt；不宣称全文及补充材料已全部核验。

方法正文宣称条件融合，但式7实际写ReLU标量加权拼接，后续式24–25为绝对差解码。
固定作者代码提交58c7c0b0155d9fbf35a50ca136c15682a1a83289，仅读model.py、
AttentionModel.py和trian_model.py。UniBIP.forward返回各分支torch.cat；Similarity
用clamp到[1,4]的幂次作用于绝对端点差，有序列表示则再拼接后线性输出。
在本次检查范围内，未定位“拓扑条件动态调制分子”的明确模块，不能自行补出后称复现。
这只是论文/所查代码范围的差异，不能推断其它版本或补充材料不存在该机制。
README获取connection reset已记录；三份源码下载成功，均未导入或执行。

式27在m=1且预测为sigmoid概率时，正项(1-p)^2、负项p^2，与Brier形式一致，另含
每批等量负采样；负采样不意味着干净标签恢复或动态优化器梯度平衡已被证明。
本地proper_loss已检验Brier，既有条件表达和药物token分支已覆盖多种独立属性交互。
因而不把FlexBIP标题当作一个已验证、必然有效的新机制来重新训练。

## 尚未充分覆盖的具体信息边界

旧midrug_internal/probe.py有12个XGBoost探针，但最大输入仅195维：M3统计、手工
相似/正负证据、每侧PCA32；其结果来自较早原验证探索。不能迁移那些混合图基线，
不能导入其旧加载器或lnc辅助脚本。本轮只静态读取这些文件查重。

新问题不是“树从未尝试”，而是“原938维表达/768维ChemBERTa逐坐标阈值信息，
在独立加性修正中是否优于PCA32瓶颈”。这只是经典梯度提升的定向内部检验，
不是FlexBIP复现，也不声称新方法家族。冻结M123作为base margin保留全部原路径，
只在当前训练折学习树修正；对照为PCA32修正和仅M3修正，预算相同。源M123检查点
已在同一内部折选过，明确属于自适应开发，不能当独立确认。

该决定无新外部数据/权重，无lncRNA；新研究目录midrug_raw_residual_20261007。
原验证/测试保持封存。若未过内部双指标门槛，不调树深/融合权重或事后提升对照。
