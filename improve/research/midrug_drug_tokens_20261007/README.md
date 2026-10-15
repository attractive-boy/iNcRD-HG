# 原 ChemBERTa token 的 miRNA 条件汇总

原目标未完成：原 MiDrug 固定测试 seed42/43/44 均值 AUC≥0.9733929274189016，
AP≥0.933683373732677；PLMF MDR/MDS 保持已验证结果。仅原 miRNA–drug 数据，
无 lncRNA 节点、特征、关系、监督，无外部数据或新预训练权重。

## 本轮问题与设计（看内部成绩前冻结）

既有 ChemBERTa 最后层 LoRA、池化适配器仍先做与候选 miRNA 无关的汇总。
表达列注意力方向为 drug→miRNA 表达列。本轮检验相反侧的具体信息边界：
原605个miRNA表达向量能否选择原216种药物的局部 ChemBERTa token。
这不是新方法家族或完整论文复现；参考及与旧实验区别见 READING_NOTES.md。

严格复用三折原训练拆分；目标图仅该折训练阳性，原M1/M2/M3完整查询删除不变。
只使用原 ChemBERTa 版本761d6a18...；从已核验的倒数第二层缓存及原最后层重新生成
216×512×768最终状态。冻结全部ChemBERTa参数。复核原SMILES重分词和CID顺序。
保留原512截断：CID145994598原1146 token被截断，其余215种未截断。
原BOS/EOS等参与上下文编码，但所有tokenizer特殊ID和padding不参与新分支汇总。
不是原子/化学子结构标注；注意力权重不是因果机制证据。

记 h[d,j]为固定状态，k=Wk LN(h)，v=Wv LN(h)（768→32、无bias）。
q[r]=Linear32(LeakyReLU(Linear938→32(LN(x[r]))))；第二Linear无bias。
共同全局交互 g=q[r]⊙mean_valid(k[d])。三组均计算：

- mirna：c=Σ softmax_valid(q[r]·k[d,j]/sqrt32) v[d,j]。
- population：仅注意力query替换为全部605原miRNA的q均值，g仍使用当前q[r]。
- pooled：c=mean_valid(v[d])；g仍使用当前q[r]。Q/K通过g保持有效参数。

共同输出readout=LeakyReLU(Linear64→32(LN(concat(c,g))))，经零初始化32→512
投影加入原lin1激活前。先初始化原M123和重建解码器，再初始化分支。三组全部可训练
参数数目和初值一致。pooled只使用逐token标准化后的一阶均值，并非原padding均值；
该对照用于排除非线性全局配对交互带来的增益。population用于排除共享局部汇总增益。
计算量并不严格一致，参数量一致。query256分块，所有有效token均保留。

## 训练和推进

三配置×三折，共9次，seed42。BCE+10×原特征重建，权重50轮线性升至10；每轮私有
mask RNG20261006+fold抽一次，unmasked重建。RMSprop alpha.99 eps1e-8，momentum0，
centeredFalse；主干lr1e-4/wd0，分支lr3e-4/wd1e-4，解码器lr1e-4/wd1e-4。
最多800，最少300，距最好轮200早停，选择最早最大mean(inner AUC,AP)。
仅mirna可推进：对control、unmasked10、population、pooled四个对照，三折配对平均
AUC和AP增益必须均≥.002。绝不事后提升对照为候选，不追加宽度/头数/温度扫描。
通过后才冻结原验证确认方案；原测试双+.02仍是最终验收。

## 预检查与审计

float64独立逐token/逐配对输出及全部参数梯度参考；同步token置换、padding污染、
分块不变，mask概率归零且归一。真实图初始主干/解码器相等、原始预测相等，非零投影
完整反传各组有限非零梯度，BN一次，hook清理，留出标签翻转不变，完整M1/M2/M3
物理删查询记录。保存选择/最终状态、优化器及mask RNG；重放9个选择模型，核查
全部18个训练状态、选轮、损失、概率/配对/索引、mask RNG步数和优化器步数。
PLMF仅本轮哈希保护，不把既有GPU重放计作新实验。
