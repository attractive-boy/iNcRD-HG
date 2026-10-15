# 原特征分组与双向组间频率

这是 CGLE（2025，arXiv:2511.06982v1）启发的独立二部图适配，不是原样复现。
阅读依据与差异见 ../midrug_literature_20261006/COMPLETION_READING_NOTES.md。
只使用原 miRNA/drug 输入与原训练内部三折。没有生物学类别注释、序列或新增权重。

分别将原表达谱/ChemBERTa 按行 L2 归一化，使用固定 KMeans16/8、n_init10、
seed20261006 分组。此处组数在看候选性能前固定，不使用目标关联图聚合后再聚类。
分组是数值特征簇，不是已知生物学类别。匹配对照在类型内固定置换成员，保留组大小。
先前 PLMF crossed random-effects probe 使用 miRNA family/sequence 和 drug clusters；
本轮不读取那些数据，也不同于在已知测试分数上拟合随机效应。

令 E_ab 为当前训练正关联在两个组之间的计数，n_b/N_D 为药物组大小比例。
对查询 (r,d)，将该查询存在时的计数 t=A_rd 从 E_ab 和对应行/列总数精确扣除。
两个方向的特征为：

log(((E_ab-t)+10*n_b/N_D)/((E_a.-t)+10)/(n_b/N_D))，及对称反方向。

它们是经 Dirichlet 平滑的组间连接富集度，不是具体配对的校准存在概率。
全局无关联时两个特征均为零；无三边路径时仍可能保留组间信息。
两个输入进入 2→64→1 的 LeakyReLU 残差头，末层零初始化，初始恢复原 M123。
计数属于每次输入图的预处理，不在 checkpoint 参数中，显式删图时必须重建。

固定四配置：feature/shuffled × 重建0/10，每项原训练内部三折、seed42，共12次拟合。
RMSprop lr1e-4、alpha .99，主干wd0、新分支/重建wd1e-4；最多800轮，至少300轮，
patience200，按内部(AUC+AP)/2选轮。feature 的每种重建版本须同时超过匹配 shuffled、
纯 M123、unmasked10，两个指标的平均增益均≥.002才推进原验证。
该门槛只是内部开发条件，最终原测试三种子双指标绝对+.02目标不变，PLMF不得下降。

运行顺序：prepare.py → check.py → launch.py → audit.py → summarize.py。
当前路径矩实验完成并归档前不启动本轮正式拟合。
