# DC-MetaMG 同名公开仓库的方法核对

论文：Tang et al., Dual-granularity feature learning reveals static and dynamic mechanisms of ncRNA-drug interactions. IEEE JBHI,2026，DOI [10.1109/JBHI.2026.3717803](https://doi.org/10.1109/JBHI.2026.3717803)。本轮核对了Europe PMC摘要及Crossref元数据，未取得论文全文。摘要描述静态结合/动态调节的分离；这不是已验证的因果结论。

GitHub搜索得到唯一同名仓库：[xiaozhouyouhuayaojiang/DC-MetaMG](https://github.com/xiaozhouyouhuayaojiang/DC-MetaMG)。固定commit `5dc602c89278d7a941467d4b183c0f70baaf372d`。**尚未从出版社页面或作者声明核实该仓库与论文正式版本的对应关系**；以下只描述这个commit的代码，不能直接归为论文正式实验。

读取了README和5个Python源文件：预测阶段model/main/cbam，特征阶段model/model与trainer。只静态阅读，未导入或执行。没有下载任何miRNA/lncRNA数据、标签、序列、预计算特征或模型权重；Git树响应仅为文件清单。README为“Research Report … Facts for README.md”的工作笔记形式，部分描述与当前源代码不一致，不能替代方法核对。

| 核查位置 | 当前commit中的事实 | 对迁移决策的影响 |
|---|---|---|
| DC_Prediction/model.py, MLP.forward | 两通道特征先经CBAM，再分别MLP，最终取两个sigmoid概率的算术平均。可学习通道权重和另一融合公式在三引号注释中。 | 文档所说的“sigmoid of summed logits”及README中的可学习权重并非当前活动代码。不能据此宣传新条件融合。 |
| DC_Prediction/cbam.py, CBAM.forward | 活动分支为通道注意力乘输入再加残差；空间注意力调用被注释。 | 这是通道重标定，不能自动等同于静态/动态因果解耦。 |
| DC_Prediction/main.py, loss_fn与训练循环 | 活动损失为BCEWithLogitsLoss，却接收model.forward已输出的概率；FocalLoss未启用。 | 不能直接照搬这个损失接口。这里是代码语义核对，未运行其训练。 |
| DC_Prediction/main.py, evaluate及early_stopping | evaluate读取固定test_data/test_label；warmup后用test AUC驱动early_stopping。kfold_split返回的留出项被丢弃。 | 与本任务严格封存原测试集的协议不兼容。未复制此选择规则，也不由此推断论文全文协议。 |
| GetFeature/model/model.py | 两通道GCN+MLP+残差，初始输入包括GIP/拓扑、RNA序列特征和分子特征。活动训练损失为4个初末层InfoNCE与0.1倍正边相似性损失。 | 当前禁止导入序列、额外网络及外部预计算表示。现有自监督、通道/分支建模已有覆盖，未发现足以直接开启新实验的已核实差异。 |
| GetFeature/trainer.py | 先训练特征模型，再导出四组嵌入和划分；预测阶段另行训练。 | 尚未核对其全部图构建实现，不声称已经完成该方法复现或全面泄漏审计。 |

网络记录：首个README原始地址请求连接重置，后续README/main原始地址超时；改用GitHub内容API在同一commit成功取得文本。原始失败记录保留。源文件均有SHA256；未用失败的请求内容替代证据。

结论：这次阅读提供了明确的不可直接照搬之处，但没有得到新的、已核实且满足当前输入限制的候选机制。本轮仍未达到原测试AUC/AP双+.02。下一步应继续检查原始Drug_Name等尚未使用的合法协变量和更直接适配miRNA的证据，避免把既有注意力/自监督方向换名重试。没有新训练、原验证或测试评估。
