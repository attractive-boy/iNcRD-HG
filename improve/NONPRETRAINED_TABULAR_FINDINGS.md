**表格模型有大量不依赖预训练的选择。本轮从头训练的 TabM 和 CatBoost，在扩展证据输入下都有高于当前 HGT 的平均全局 AUC/AP；训练预算会影响这个结论。**

CatBoost、LightGBM、XGBoost 从当前训练数据拟合树；MLP、TabM、FT-Transformer 等神经表格模型可以随机初始化后训练。本轮实际补跑的是 CatBoost 1.2.8 和项目现有的 TabM 实现，比较对象为上一轮已验证的 HGT 与配对 MLP。没有使用 TabICL 或 TabPFN 预测器。

2026-09-13 完成 36 个新模型的训练和 90 个检查点的恢复验证。三档输入与上一轮完全一致：36 个图配对特征、增加旧功能证据后的 62 维、扩充药物靶点证据后的 62 维。各方案保留共有的 ChemBERTa 药物输入；本轮排除的是预测器预训练，不是删除所有上游预训练特征。

**最新扩展 62 维输入下的主要比较。** 均为同一历史内部开发集的 3 个种子均值；以下预算事先指定，未按本轮表现挑选。

|任务|模型|预算|AUC|AP|
|---|---|---|---:|---:|
|MiDrug|HGT|74 次更新|0.959536|0.919795|
|MiDrug|配对 MLP|74 次更新|0.957304|0.918293|
|MiDrug|TabM|441 次更新|0.962057|0.925282|
|MiDrug|CatBoost|300 棵树|0.960549|0.920759|
|LncDrug|HGT|24 次更新|0.991471|0.980800|
|LncDrug|配对 MLP|24 次更新|0.981477|0.962799|
|LncDrug|TabM|141 次更新|0.992194|0.983619|
|LncDrug|CatBoost|300 棵树|0.993332|0.986034|

MiDrug 的 TabM 在这组主要预算中有更高的全局 AUC/AP，LncDrug 则是 CatBoost 更高。这表明非预训练表格预测器在当前数据上有竞争力，不能用普通配对 MLP 的结果代表整个表格模型家族。

**相同更新次数下，HGT 优于这版 TabM。**

将 TabM 的检查点固定为 HGT 的 74/24 次更新，扩展输入下 TabM 的 AUC/AP 为 MiDrug 0.941810/0.901154、LncDrug 0.989561/0.972644，均低于同次数 HGT。TabM 每次累积全部训练配对的梯度，HGT 每次抽取一半配对；TabM 还有 16 个内部成员，因此相同更新次数也不代表相同算力。

TabM 的主要预算和学习率来自历史内部选择，继续沿用其完整支持集梯度累积训练方式；CatBoost 使用固定的标准配置，没有为它重新搜索超参数。这里比较的是预先固定的现有方案，不能声称严格隔离了架构或计算量的影响。完整报告同时列出所有检查点，没有跨预算挑最佳数值拼接。

**增加证据的效果仍取决于任务。**

- LncDrug 从 36 维增加到旧 62 维时，TabM 的 AP 从 0.978588 升至 0.983243，CatBoost 从 0.982647 升至 0.986051；两种模型的 AUC/AP 均在 3/3 个种子上改善。
- MiDrug 中，TabM 的 AP 按三档输入依次为 0.927778、0.927435、0.925282，CatBoost 为 0.923213、0.922754、0.920759。继续扩充靶点证据后的 AP 下降，在两种模型上均为 3/3 个种子。
- 在维度不变的“旧 62 → 扩展 62”比较中，LncDrug TabM 的平均 AP 略升而 AUC 略降；CatBoost 的 AP 接近不变、AUC 略降。证据覆盖更多并不保证两项指标同步改善。

CatBoost 的 1000 棵树作为预定敏感性分析另行报告：扩展输入下 MiDrug AP 为 0.918829，低于 300 棵树的 0.920759；LncDrug 为 0.986443，略高于 0.986034。本轮没有据此改变主要预算。

**分组诊断没有给出所有模型一致的排名。**

扩展输入下，MiDrug 逐药物平均 AP 为配对 MLP 0.745322、TabM 0.742645、HGT 0.732874、CatBoost 0.708935。平衡药物频率后的子集 AP 则为 CatBoost 0.740887、MLP 0.710997、TabM 0.708476、HGT 0.704625。前者对药物等权，后者采用匹配后的配对权重，优化目标不同。加权子集 AP 不能直接与全量 AP 比较。

**无预训练预测器约束下，建议把 TabM 和 CatBoost 作为主要候选继续验证。** MiDrug 的全局指标更支持当前 TabM，LncDrug 更支持当前 CatBoost；HGT 在较少更新次数下有优势。下一步应围绕实际目标明确预算和分组指标，而不是仅以“图模型／表格模型”判断优劣。

这轮没有替换当前已接受模型，没有进行外层或原始测试集评价。历史内部开发集已被反复使用，三个种子不是独立划分；本轮结果不构成独立泛化或统计显著性证据。

**复现与产物。**

- [完整报告与所有预算](results/nonpretrained_tabular_compare_20260913/REPORT.md)
- [逐种子结果](results/nonpretrained_tabular_compare_20260913/metrics.csv)、[配对增量](results/nonpretrained_tabular_compare_20260913/paired_differences.csv)、[药物分组诊断](results/nonpretrained_tabular_compare_20260913/diagnostics.csv)
- [实验入口](nonpretrained_tabular_compare.py)、[模型与训练逻辑](nonpretrained_tabular_modules.py)、[测试](test_nonpretrained_tabular.py)

10 项单元检查通过；其中 8 项另在历史 PyTorch 环境验证兼容性。全数据短训练的 30 个检查点、正式训练的 90 个新检查点均通过恢复验证，正式恢复预测最大误差为 0。三档输入中 TabM 的初始权重和参数量一致；CatBoost 从未拟合状态开始，不传入验证集或初始模型。

TabM 沿用成员平均概率，通过稳定的 log-odds 计算排序指标，避免高置信概率饱和。CatBoost 使用原始二分类分数。CatBoost 安装在独立的依赖目录，已有 Python 环境和冻结模型源码均未修改。

已有结果可以用以下命令重新汇总：

```bash
/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.nonpretrained_tabular_compare report
```

重新训练应以新目录运行 `prepare --out-dir ...`；随后分别指定 `run --learner tabm` 或 `run --learner catboost`、`--task MiDrug/LncDrug` 和同一输出目录。TabM 使用历史 `ncRNAdrug` Python 环境及 GPU，CatBoost 使用 `.venv-tabicl` Python 环境及 CPU。配置、依赖、输入、模型状态和参考结果的哈希均记录在协议与验证文件中。
