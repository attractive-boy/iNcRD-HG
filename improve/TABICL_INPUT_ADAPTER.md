# TabICLv2 共享输入适配与受约束更新

本轮实现并检验文献方案中的第一步：冻结完整 TabICLv2，只训练两个 RNA–药物任务共用的输入适配器。研究依据及其他候选路线见 [文献与架构方案](research/tabicl_next_architecture_20260913/RECOMMENDATIONS.md)。

正式实验已完成：MiDrug 有提升，但 4 个候选均未守住 LncDrug AUC/AP，未进入外层。完整比较见 [实验结果与复核记录](TABICL_INPUT_ADAPTER_FINDINGS.md)。

## 结构与训练

两任务均为 36 维，但只有前 31 列的名称和角色对应。对这 31 列使用

`z = x + U tanh(V x)`，其中 `V: 8 × 31`、`U: 31 × 8`，共 496 参数；`U` 从零开始。后 5 列沿用原输入。

新增权重在两个任务、支持和查询、所有图分支与集成成员之间共享。模块不接收任务名称或 RNA 类型。执行位置在支持集拟合的标准预处理之后、列编码之前：显式还原被过滤列和特征置换，按规范列语义适配，再恢复模型要求的列顺序。被过滤的常数列用零占位，适配后仍被移除。全部列编码、行编码和 ICL 表示重新计算，不复用旧 KV 或表示缓存。

第一轮固定两条轨迹，各保存第 16、64 步，共 4 个性能候选：

- `plain`：仅以 MiDrug 的 BCE、平滑 AUC 和平滑 AP 之和训练。
- `projected`：采用相同 MiDrug episode、初始参数和 Adam 配置，将 Adam 实际参数改变量投影到两个 LncDrug 排序损失的一阶不增加半空间；随后检查有限步长的实际代理损失，必要时缩步或跳过。

Adam 学习率为 `0.0003`，无权重衰减。动量保存 MiDrug 目标的提案历史，包括被缩步或跳过的提案。保护检查的 `1e-7` 仅为代理损失浮点误差容差，**性能验收的 AUC/AP 没有下降容差**。

MiDrug 使用 20 个拟合 episode，每个支持 192 行、伪查询 64 行；LncDrug 使用五个固定保护 episode，每个遮边折一个，支持 96 行、伪查询 32 行。分层抽样保留各折的标签比例，支持与查询互斥，整个目标折均从图中遮蔽。每个 episode 使用官方 `none/power` 预处理和两个成员。LncDrug 保护损失在五个 episode 拼接后的查询排名上计算。验证标签不进入梯度、抽样或缩步判断。

## 比较与验收

复用历史图上下文模型的支持集与拟合预处理状态，每种子 32 个成员，种子为 `666/0/1`。LncDrug 沿用 `query_bag_050`，MiDrug 沿用 `fold_match_050`。每个候选均作用于两任务全部相关分支。

内部要求 LncDrug 平均 AUC 和 AP 分别不低于历史图上下文与原始 TabICLv2 参考，MiDrug 平均 AUC/AP 均提升且每项至少 2/3 推理种子为正。按药物宏 AP 单独报告。合格者优先选择受约束轨迹，再优先选择较早检查点。内部不合格则不读取本轮外层特征；外层只能评估已固定的同一适配器及零对照，不重新训练或选择。

按药物宏 AP 沿用基线定义：先计算各药物的 AP，再对验证中同时包含正、负标签的药物取算术平均。

比较使用历史参考的原始预测，另外报告本轮零适配重算，避免将浮点重算差异解释为收益。所有保存状态均完整重放，并检查查询重排、子集和重复行。原始测试数组不读取。

## 文件与复现

- [适配器和投影实现](tabicl_input_adapter_modules.py)
- [训练、评估与验收入口](tabicl_input_adapter.py)
- [结构及优化不变量测试](test_tabicl_input_adapter.py)
- 正式实验：`improve/results/tabicl_input_adapter_v2_20260913/`
- 工程检查：`improve/results/tabicl_input_adapter_smoke_v2_20260913/`

使用现有 `.venv-tabicl` 环境，在仓库根目录运行，设置 `PYTHONHASHSEED=0`。不需要安装测试依赖。

```bash
python -m unittest improve.test_tabicl_input_adapter -v
python -m improve.tabicl_input_adapter prepare
python -m improve.tabicl_input_adapter bank
python -m improve.tabicl_input_adapter train --mode plain --device cuda:0
python -m improve.tabicl_input_adapter train --mode projected --device cuda:1
```

然后对两个任务分别执行 `run`、`verify`、`audit`，例如 `python -m improve.tabicl_input_adapter run --task MiDrug --device cuda:1`；可用 `--seed` 分开执行已登记的三个种子。完成两个任务后执行 `select` 和 `report`。仅当 `SELECTION.json` 的 `eligible` 为真时，才允许上述推理和验证命令加 `--phase outer`。代码、协议、训练银行、参数和结果均记录哈希；若修改已冻结的实现，必须使用新的输出目录。

工程检查的初版在建立训练银行时发现极小查询样本可能缺少类别，尚未训练或评估即停止；`smoke_v2` 使用确定性的分层抽样修复。该修复在任何正式性能评估之前完成。

第一份正式运行在读取历史预测文件时停止：历史格式的字段为 `views`，新运行自身保存的是 `scores`。兼容性修复只改变读取接口，并增加独立格式测试；失败运行的协议和训练产物保留，尚未产生任何候选性能指标。正式 `v2` 使用相同超参数重新冻结和训练。

输入适配与梯度投影均有既有方法依据。本轮是项目内的新结构尝试，不据组合或命名宣称方法首创。历史开发分区已多次查看，训练记忆的代理约束不能保证新数据上的 AUC/AP 不下降。
