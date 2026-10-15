# 原始坐标树修正的折外/训练内offset匹配实验

原目标不变：MiDrug原固定测试seed42/43/44均值AUC≥.9733929274189016、AP≥.933683373732677，
PLMF MDR/MDS不下降。仅原miRNA/drug属性与原训练配对；无lncRNA、外部数据或新权重。

## 已有工作和本轮差异

上一轮raw树修正有内部+.000386/+.000492收益，但未达门槛；额外训练降低训练BCE、
增加内部留出BCE。该现象本身不证明样本内offset是原因，也不构成改变树超参的依据。
旧midrug_crossfit有子图OOF残差，midrug_retrieval_crossfit有案例OOF/训练内匹配对照，
均未达推进门槛。不能说OOF方法首次使用或默认有效。本轮对同一原始坐标树进行匹配
实验，保持树输入和推理基模型完全相同，只改变训练offset；另设PCA32输入对照。

## 基模型复用和边界

复用midrug_retrieval_crossfit的18个纯M123+unmasked10基模型，seed42固定300轮：
每内部折有5个4/5子模型和1个完整训练模型。分区为原同一5折StratifiedKFold，
seed20261006+fold；被预测折未用于选轮/早停。全量当前哈希核对、18个基模型重新
重放train/预测logit、索引重建、OOF恰好一次覆盖检查；没有新训练这些基模型。

OOF offset来自不含该配对及标签的4/5模型；训练内offset来自完整300轮模型。
所有臂推理均用相同完整300轮模型。OOF的训练数据/图较稀疏，这是方案固有差异，
不能把结果全归因为消除单样本标签影响。树的8维精确拓扑计数来自完整当前内部训练
图并删除自身查询；它们不因OOF子模型而变化。原逐坐标/PCA和整数特征复用上一轮
已核验输入，不重新提取混合RNA数据或使用旧子图缓存。

## 三个冻结配置

- oof_raw（唯一可推进候选）：OOF logit作为训练base_margin，原8+938+768维树输入。
- insample_raw：完整300轮训练logit作为训练base_margin，树输入相同。
- oof_pca32：OOF训练base_margin，8维计数+每侧PCA32输入。

推理始终sigmoid(full300_logit+tree_delta)，不使用训练OOF offset代替推理基模型。
完整M1/M2/M3均保留、参数和缓冲冻结。stage0树为0，三臂推理精确退回full300。
树参数不变：XGBoost2.1.4 CPU hist，depth2，min_child_weight10，reg_lambda10，eta.03，
subsample.85，colsample_bytree.85，seed42，nthread2。最多1600轮、至少300、patience150，
包含stage0，按最早最大内部mean(AUC,AP)选轮。3臂×3折=9次新树拟合。

候选对control、已选unmasked10、full300、insample_raw、oof_pca32及前轮raw_selected
各三折平均AUC/AP均≥+.002才推进原验证。前轮raw_selected用较强已选主干，本轮须
超过它，不能借较弱full300夸大改善。不调树深/学习率/融合权重，不事后提升对照。
原测试双+.02才是最终验收，内部小门槛不代替它。

## 验证

只允许原训练内部配对及对应当前训练阳性图。源模型索引/OOF覆盖重新核验，18个
保存模型全部重放。stage0推理相同、实际DMatrix训练offset等于指定数组、匹配臂输入
逐位相同；五次预检更新后独立规范JSON数组逐树遍历及float32阈值边界检查。
真实正负查询完整删除，树特征逐位相同，full300模型训练mu/sd保持原值。
留出标签翻转不变。选后9个树重放、选轮/最终轮数、源hash、全预测删除和树oracle。
所有指标属原训练内部开发，不能当作新的原验证/测试成绩。PLMF仅保护性哈希复核。
