# 冻结选型后的分量诊断

仅用于解释原训练内部结果，不改变selection.json或据此改选配置/轮次。
backbone_without_residual仅去掉预测残差，不能撤销已经发生的辅助损失训练。
scorer_only是第一层图预测器的直接查询输出；full是已审计模型。

| 配置 | 分量 | 内部AUC | 内部AP |
|---|---|---:|---:|
| direct | full | 0.952674330 | 0.912080527 |
| direct | scorer_only | 0.948432574 | 0.900261149 |
| direct | backbone_without_residual | 0.952202407 | 0.911495465 |
| direct_rec10 | full | 0.953604163 | 0.914122272 |
| direct_rec10 | scorer_only | 0.948209232 | 0.900796361 |
| direct_rec10 | backbone_without_residual | 0.952626878 | 0.912578144 |
| uniform | full | 0.952724112 | 0.912306372 |
| uniform | scorer_only | 0.948742959 | 0.900775513 |
| uniform | backbone_without_residual | 0.952287968 | 0.911693134 |
| uniform_rec10 | full | 0.953894772 | 0.914118105 |
| uniform_rec10 | scorer_only | 0.949075345 | 0.901915236 |
| uniform_rec10 | backbone_without_residual | 0.953617568 | 0.913642887 |
| soft | full | 0.952602213 | 0.911674335 |
| soft | scorer_only | 0.949117886 | 0.901905913 |
| soft | backbone_without_residual | 0.952312504 | 0.911269694 |
| soft_rec10 | full | 0.953584282 | 0.914198606 |
| soft_rec10 | scorer_only | 0.949156930 | 0.903712497 |
| soft_rec10 | backbone_without_residual | 0.953032468 | 0.913357525 |
