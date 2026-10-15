# 从图补全转向训练扰动：文献与适用范围

2026-10-06，nature-academic-search multi-source-search。学术MCP未挂载，
预检PubMed/Crossref/arXiv公开API均连通，实际使用arXiv与Crossref检索。
optimization_discovery.json保存4个完整查询、原始响应位置与结果；
optimization_methods_fetch.json保存下面两篇PDF及2026论文Crossref记录的URL/哈希。
搜索命中不等于相关，也不等于全文阅读。当前核读以下两篇的方法，不导入数据或权重。

| 文献 | 阅读范围和核对 | 对本任务的判断 |
|---|---|---|
| [Rethinking Graph Generalization through the Lens of Sharpness-Aware Minimization](https://arxiv.org/abs/2602.08855v1)，2026，[DOI](https://doi.org/10.1145/3774904.3792707) | arXiv正文§2–4，尤其§4.2–4.4及算法1；Crossref题名、作者和会议元数据 | E2A研究OOD图分类；先训练条件VAE，再在VAE潜空间做能量上升，最后用生成表示校准分类器。不是标准参数SAM，也没有证明当前MiDrug随机配对划分存在同样的Minimal Shift Flip。 |
| [Efficient Sharpness-Aware Minimization for Molecular Graph Transformer Models](https://arxiv.org/abs/2406.13137v1)，ICLR 2024（PDF首页标注） | §3.3式1–2与§4.1–4.2式3、逐轮re-anchor说明 | GraphSAM用上一步更新梯度平滑近似扰动梯度，目标是减少SAM两次反传开销。论文主要是分子图Transformer属性预测，当前是miRNA–drug配对GeneralConv，实验结论不能直接外推。 |

SAM每步先计算g=∇L(theta)，扰动epsilon=rho*g/||g||₂，再在扰动参数处计算
更新梯度，恢复原参数后更新。GraphSAM的梯度近似在每个epoch首步重置；
本项目一轮只有一个全批次步，照此重置便没有减少扰动梯度计算的收益。
因此先做完整SAM与等半径随机方向对照，不为采用较新论文而引入近似算法或cVAE。

E2A能量E(S)=−logsumexp(S)依赖logit共同偏移；预测类别和softmax概率却不变。
所以其能量—稳健半径关系需要结合论文假设理解，不能把未固定logit约定的能量
直接作为本项目单输出二分类器的校准置信度。本轮也不引入任何能量伪标签。

本地证据：纯M123和unmasked10的3折均在已选轮后仍降低训练损失，但内部AUC/AP
回落。它提示需要检查泛化，不能单凭此现象宣称已测得尖锐极小值。
已尝试过的BPR、全局AUC/AP surrogate均登记为前驱，不能把同类损失称为全新方法；
全局surrogate旧实验使用完整历史图，不作为当前纯miRNA候选证据，也不运行其加载器。

预先固定四配置sam/random×重建0/10，rho=.05，三内部折；
复用纯M123和unmasked10，检验梯度方向是否比随机方向及原训练更好。
只有SAM对上述匹配对照平均AUC/AP均至少+.002才推进原验证。
最终原测试AUC/AP绝对+.02和PLMF保护要求不变，论文的其他数据集增益不计入验收。
