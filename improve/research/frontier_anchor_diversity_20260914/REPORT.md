**表格集成与校准文献支持进一步检验互补性，不能直接证明本项目的增益机制或三项创新。**

2026-09-14 更正。以下两个 arXiv 标识、题名和公开摘要已重新核验，原始页面及状态、哈希见 `metadata_recheck_20260914.json`。本次核查深度为元数据与摘要，没有重新复现论文实验。

|文献|摘要直接支持的结论|对本项目的边界|
|---|---|---|
|*Ensembling Tabular Foundation Models — A Diversity Ceiling And A Calibration Trap*，[arXiv:2605.18696](https://arxiv.org/abs/2605.18696)，2026 预印本|六个 TFM、153 个分类任务；池内平均 Q 为 0.961。摘要称最佳 two-level cascade stacking 相对最强单模型增加 0.18% accuracy，计算量为 253 倍。部分集成与最强单模型无显著区别，logistic stacking 的 log-loss 表现差。|撤回旧记录中“skip-connection 集成提高约 3 个点”的未准确限定表述。该基准不能证明本项目 TFM 的错误近乎相同，也不能替代对经典学习器组合的评价。|
|*High Performance, Low Reliability: Uncertainty Benchmarking for Tabular Foundation Models*，[arXiv:2605.28554](https://arxiv.org/abs/2605.28554)，2026 预印本|在 TALENT 的 112 个数据集上，TFM 的 AUC 高，但 conformal prediction 的条件覆盖指标 SSCS 较弱。|条件覆盖较弱不等同于已经测得本项目过度自信，也不能直接解释 AP 的变化。需要本项目自己的校准与不确定性检查。|

本项目保存的 `anchor_logit`、`anchor_rank`、`anchor_mixed` 都使用同一表格学习器池，属于同一集成方向的变体。三个开发重划分相互重叠，且已用于反复开发；不能称为三个独立验证集。`anchor_mixed` 对 MiDrug 增加 TabDPT 分支，对 LncDrug 不加，必须公开这一任务特定配置。

标签打乱对照的平均 AP 较差，支持监督标签影响结果；不能据此排除方差减少或校准变化。当前结果也没有确立“首次”的广泛创新性结论。成员与权重的后续搜索属于开发探索，不是独立确认。

数值和更正范围见 [ANCHOR_SHRINKAGE_FINDINGS.md](../../ANCHOR_SHRINKAGE_FINDINGS.md)。此方向可以保留工程价值，但不能用三个组合公式满足“三种经过验证的不同非表格创新技术”的目标。
