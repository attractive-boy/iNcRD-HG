# 固定M123的原始特征坐标树修正

原测试MiDrug seed42/43/44均值AUC≥.9733929274189016、AP≥.933683373732677，PLMF
MDR/MDS不下降。仅原miRNA/drug、原属性、训练阳性图；无lncRNA、外部数据或新权重。

## 精确问题及来源

这是经典梯度提升的定向检验，不是新模型家族或FlexBIP完整复现。
近期FlexBIP的解耦表述与所查代码差异及旧实验查重见../midrug_decoupling_review_20261007/REPORT.md。
旧midrug_internal XGBoost探针只有手工统计/PCA32，不含938+768原逐坐标输入，且用
更早原验证探索。本轮不加载其图或模型；用独立固定内部三折的纯miRNA源检查点。

从同折原unmasked10选择模型得到不可训练的logit f0。新预测为
sigmoid(f0(r,d)+Σ_t eta h_t(x(r,d)))。直接把f0传给XGBoost base_margin；不把它
当成可分裂的特征，不使用概率截断/变换反推logit，也不再按留出集选混合系数。
树以binary:logistic目标拟合该整体的训练梯度和Hessian，主干/BN/解码器保持固定。
保留完整M1/M2/M3，只添加一个输出logit修正。基模型已在同一内部折选过，明确为
自适应开发，不能声称新独立确认或跨实体冷启动。

## 三个固定臂

共有原M3原始qis及ease统计，不再做额外手工内容核；输入没有任何直接查询标签。
raw（唯一候选）：追加938维原miRNA表达、768维原ChemBERTa逐坐标。
pca32：追加每侧32维PCA，full SVD，仅原节点属性无监督拟合，不用关联标签。
structure：仅M3统计。三者深度/轮数/正则/采样预算相同，但输入维数及表达容量不同，
不称等参数或等计算量。树坐标划分属于归纳偏置对照，不把PCA低方差方向说成必然有用。

xgboost2.1.4，CPU hist；深度2、min_child_weight10、reg_lambda10、eta.03、subsample.85、
colsample_bytree.85、seed42、nthread2，其他包默认。沿用旧树探针固定数值以避免扫描。
最多1600轮，至少300轮，距最佳150轮早停。包含stage0（零棵树=原M123），选择最早
最大mean(inner AUC,AP)。raw/pca32/structure×3折共9次新拟合，不训练第二随机种子。
保存选择/最终树、每轮两指标、原margin、全部输入和概率。

raw须对control、unmasked10、pca32、structure各自三折配对平均AUC/AP均≥+.002，
才进入冻结原验证确认。不能事后提升对照、扫描深度/阈值/学习率或调融合权重。
最终仍须在原固定测试三种子达到双+.02，内部小门槛不代替最终验收。

## 边界与验证

原图仅当前训练阳性；M1/M2/M3保持完整查询删除，PCA和逐坐标输入不依赖目标标签。
重建删除查询训练记录时源M123训练mu/sd保持原值，不能以重新拟合预处理改变模型。
预检stage0精确base_margin、独立Python逐树遍历、改变margin的加性恒等、logistic
梯度/Hessian；真实三折特征复算、训练后非零树的完整正负查询删除及留出标签翻转。
选择后9个模型重放概率与指标/选轮，独立逐树参考、最终轮数、完整删查询、全源哈希；
PLMF10个既有检查点及冻结数据/发布源码做哈希保护，不计为新GPU重放。
