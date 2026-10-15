# 固定编码器的反馈分支与分类头适配

目标仍为原测试 seeds42/43/44 均值 AUC≥.9733929274189016、AP≥.933683373732677，
PLMF MDR/MDS 两项不下降。只使用原 miRNA/drug 属性、相似边及当折训练正关联；
无 lncRNA、外部数据、注释、序列或新预训练权重。只在原训练内部固定三折开发。

## 依据和要检验的问题

上一轮完整二阶矩反馈没有超过均值反馈。full 的新增 lin1 激活 RMS 在三折差别很大，
但非零/较小激活不证明分类头是瓶颈。已有 retrieval joint_label 和 reconstruction
finetune_v2 均退回 stage0；这些同时改变编码器、BN 与头，不能单独回答分类头适配问题。
本轮检验：保持原图编码器、M2/M3 投影及全部缓冲不变，只让原 lin1/lin2/lin3
适配新增分支，是否让完整二阶矩提供超出头微调和均值反馈的留出收益。

机制复用上一轮核读的 [GeoF, arXiv2609.34711v1](https://arxiv.org/abs/2609.34711v1)，
详见 ../midrug_feedback_geometry_20261007/README.md、DERIVATION.md 和数学核验。
本轮未新增文献搜索，未把局部适配称为作者完整实现。modules.py 与上一轮逐字相同：
4×4 块、两轮、图无关初值、完整/均值残差矩、温度1及全部常数均不变。
不追加温度、权重、深度或宽度搜索，不把旧均值对照事后提为候选。

## 预先固定的九次拟合

full_head（唯一候选）、mean_head（去掉完整二阶矩对照）、head_only（无新分支）
各三折 seed42。从相同 unmasked10 原所选检查点恢复，不从上一轮反馈所选权重继续。
full_head/mean_head 新分支相同初始化、零投影；三臂 stage0 恢复原模型预测。
只训练 lin1/lin2/lin3 及存在时的 feedback/feedback_projection。
原头 RMSprop lr1e-5、wd0；新增分支 lr1e-4、wd1e-4；alpha.99、空优化器、
set_to_none=True。原模型始终 eval，反馈分支训练时开启 activation checkpointing。
这不是恢复原优化器后继续训练；head_only 比带分支臂容量更小，是组成对照。

BCE 原目标，冻结重构为无关常数，不更新也不重复计算。max300/min100/patience100，
最早最大内部 (AUC+AP)/2 选轮，包含 stage0；此 stage0 已在同一内折选过，
不将本轮称为独立验证。保存所选/最终权重、两份优化器、曲线和所选预测。
旧 control/unmasked10/frozen_full/frozen_mean 重用既有完整重放并核对哈希。
full_head 只有相对上述四个旧对照及 mean_head/head_only 的三折平均 AUC/AP
均≥+.002 才推进原验证。不能事后选择 mean_head/head_only 或改变门槛。

## 验证和冻结后诊断

新预检：三臂初始源预测/原状态，两个真实优化步，所有固定参数和 BN 缓冲逐位不变，
头/分支有效梯度、正负查询物理删图及查询批次不变、留出标签翻转不变。
复用相同模块已归档独立数学与删图梯度检查（不声称重新运行该套数学核验）。
训练后九个所选模型完整重放，九个最终状态检查及所选/最终逐参数优化器步数检查。
选型冻结后可测无反馈推理干预、头参数偏移和训练/留出表现；这些仅解释已选模型，
不允许继续按干预系数或单折收益选型。
PLMF 十个检查点、原图/划分/发布源码只作哈希保护，不称新增 GPU 重放。
