# 微调失败后的参数平均文献核查

检索于2026-10-07（北京时间），nature-academic-search / multi-source-search。
学术MCP未挂载，使用公开Europe PMC、Crossref、arXiv、GitHub API。
检索截至2026-10-06，四个查询返回100条发现记录，按DOI及标题/首作者核对后
仍为100条；不等于100篇相关或已读全文论文。原始查询、响应及去重结果保留。

## 当前实验依据

[修正后的辅助任务微调](../midrug_reconstruction_finetune_v2_20261007/REPORT.md)
三折九次拟合均回选源unmasked10；固定关闭/衰减方案未获得双指标增益。
前一版本decay第100步保留零梯度，仍更新解码器，已完整保留并明确判为协议失败；
修正set_to_none并加入边界检查后重新训练，不能以原失败实现代替修正实验。

## 两项主要阅读

| 文献 | 核实范围 | 对当前任务的意义 |
|---|---|---|
| Sapkota、Demir、Sharma、Ngonga Ngomo. [Parameter Averaging in Link Prediction](https://doi.org/10.1145/3731443.3771365)，K-CAP2025；[arXiv2510.25361v1](https://arxiv.org/abs/2510.25361v1) | Crossref出版信息（2025-12-10）、arXiv元数据/开放HTML§3–4、§5设置、讨论；公开作者库当前固定commit的SWA/ASWA实现 | 固定参数平均可在同一轨迹上获得另一预测器，不添加输入或推理分支。论文的KGE/MRR实验不保证MiDrug二分类AUC/AP收益。 |
| Milanesio、Ingrosso、Osella. [An Analytical Theory of Auxiliary Learning](https://arxiv.org/abs/2609.29774v1)，2026-09-24预印本，元数据注明在审 | arXiv元数据、开放HTML§2–3模型/分析和§4讨论；未逐行审核附录证明 | 任务关联与梯度噪声共同影响辅助学习。高斯独立输入、共享teacher可实现最优点、固定二层权重、小学习率在线SGD等假设不适用于当前RMSprop全配对BCE+节点重构；不能照搬最优权重sqrt(2)-1。 |

## 参数平均的具体边界

论文§3式(2)给出普通累积均值 `(n * mean + current) / (n + 1)`。
本地只采用这一算术平均，固定从第100更新起纳入每一步参数，维持原学习率和
原raw停止规则。它是论文启发的本地适配，不是完整复制作者SWA训练日程。
参数进入平均不依赖验证分数；平均模型仍需像raw一样使用内部留出选轮，
因此没有声称完全无选型或独立验证。测试集不参与任何步骤。

论文§4的ASWA会依据验证成绩硬替换、尝试合并或拒绝快照。其内部得分单调
不等于独立测试有增益，也不能仅凭拒绝规则保证超过另一条固定SWA轨迹。
不把自适应选模造成的内部提升自动归因于权重平均。

作者公开develop分支固定至`d0fc565d530d5dbe425fc32ccf0b607634401724`；这是本次
访问时的实现，未证明与论文提交版本相同。只读取源码，不导入其数据/模型入口。
- `SWA.moving_average`按1/(n+1)更新参数，另有学习率日程；本地保持原学习率，
  以便在同一训练轨迹上比较raw与平均本身。
- `ASWA.decide`硬更新会清空alphas，随后临时合并使用sum(alphas)，使硬更新后
  第一次临时合并中旧均值权重为0；论文Algorithm1初始化/硬更新的计数表述也
  与简单“已接收模型数”不同。本地不复制这一自适应状态机。
- 当前M123的BN输入是固定训练EASE值，与模型权重无关。因此复制当轮BN统计量，
  不平均整数计数，也不通过额外前向改变训练BN。累积和使用float64保存。

历史`improve/coldstart.py`已经实现按验证AUC选top10检查点后做参数平均；
故参数平均不是项目首次出现的方法家族。该旧训练还会随机采样负例，且不属于
当前严格纯miRNA流程。本轮不会调用该数据入口，也不将旧分数作为当前证据。

## 其他命中与不采用项

此次再次检出Nie等2026年谱编码/稀疏交叉注意力miRNA–drug文章
（DOI10.1109/TCBBIO.2026.3729865），此前已核查含gene介导视图和序列输入，
没有把它当作新的可直接复现方案。mDADGAN标题及摘要明确引入lncRNA，排除。
大量临床/组学或LLM辅助任务条目与本任务不匹配，仅保留发现记录。

原始HTML/元数据/脚本与作者代码按哈希归档。raw.githubusercontent.com三次读取
失败记录保留；相同commit的GitHub contents API成功。HTML解析先因环境缺少bs4
失败，后改用标准库HTMLParser；API解码先遇Python3.8不支持removesuffix，后从
已保存base64回复解码。这些访问问题不解释为论文或作者代码缺失。

下一实验为[固定参数平均](../midrug_weight_average_20261007/README.md)：三条训练
轨迹、六个选中检查点，唯一候选mean须同时超过纯M123、既有unmasked10和匹配raw
的平均AUC/AP各.002。最终双+.02验收目标仍未达成。
