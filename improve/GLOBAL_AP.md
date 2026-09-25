# 全局 AP 软排序训练

2026-09-12。作为 PLE 调参失败后的备用机制，保持原标量 TabM 的图特征输入，在原 BCE 之外增加全局 AP 的可微排序代理。正式 `prepare` 会核对前序 PLE 调参的失败判定和核验文件；短流程仅检查实现，不作性能判断。

正式记录：[全局 AP 实验报告](results/global_ap_20260912/REPORT.md)；两轮汇总：[PLE 调参与机制切换最终比较](results/global_ap_20260912/RETRY_REPORT.md)。

## 已完成结果

**新机制未通过双任务提升条件。** 76 次正式训练及检查点重放全部完成。以下是同一历史外层验证划分、seed 666/0/1 的均值：

| 任务 | 组 | AUC | AP | 按药物宏 AP |
|---|---|---:|---:|---:|
| LncDrug | 原/调参 TabM | 0.982052 | 0.966260 | 0.825590 |
| LncDrug | 全局 AP 软排序 | 0.981484 | 0.965234 | 0.817402 |
| MiDrug | 原/调参 TabM | 0.958717 | 0.921461 | 0.718469 |
| MiDrug | 全局 AP 软排序 | 0.958586 | 0.921142 | 0.717940 |

LncDrug 的 AUC/AP 差值分别为 -0.000568/-0.001026，MiDrug 为 -0.000131/-0.000318；四项指标的正向种子均为 0/3。本轮保留原 TabM 基线。

内部验证选中 LncDrug 的 lambda=1.0、temperature=0.05、lr=0.003，重训 103 轮；MiDrug 为 lambda=0.1、temperature=0.01、lr=0.001，重训 450 轮。内部选择分数的轻微改善没有转化为外层验证收益。部分候选的选中轮数接近训练上限，结论限于本轮搜索范围和训练预算。

合并核验通过：两轮共 190 个正式检查点全部重放，38 个重复 TabM 对照的配置、所选轮数和预测一致；所有候选的评价配对与标签相同。重复对照不计为新增独立种子。

## 机制

原模型逐成员计算 BCE，推理时平均成员概率。新增项直接作用于这个平均概率，并对一个拟合正例与**全部拟合样本**的分数差使用 sigmoid 近似排序关系。

对正例 i，令 `h_ij = sigmoid((score_j - score_i) / temperature)`：

- 软总排名：`R_i = 1 + sum_{j != i} h_ij`。
- 软正例排名：`R_i+ = 1 + sum_{j != i, y_j=1} h_ij`。
- 排序损失：`1 - mean_i(R_i+ / R_i)`。

每轮均匀抽取至多 256 个拟合正例作为锚点，所有拟合正负例仍参与分母和正例计数。抽样只近似最外层的正例平均，不按药物分组，也不删掉未被抽为锚点的负例。私有随机数生成器保证该抽样不会改变 BCE 的 dropout 随机数序列。

总目标为 `BCE + lambda * AP_loss`。BCE 保留原 dropout；AP 项在关闭 dropout 的模型上计算，仍对全部模型参数求梯度。两个梯度在同一次 AdamW 更新前累积，lambda 在前十轮线性增加到目标值。自身比较被准确移除；并列分数采用软排名含义，与 sklearn 的精确 AP 并列处理并不相同。

## 对照与选参

AP 组固定 TabM 宽度 128、深度 3、16 成员、dropout 0.1、weight decay 0.0001，搜索：

- lambda：0.1、0.5、1.0；
- temperature：0.01、0.05；
- learning rate：0.001、0.003。

共 12 个候选。标量 TabM 对照重复前序调参的 12 个网络/正则/学习率配置。两组都采用内部前两名的三种子确认，再固定轮数外层重训；正式实验共 76 次训练。AP 组的网络结构没有额外参数。

主比较是相对原 TabM 和相对同预算调参 TabM 的双任务 AUC/AP，宏 AP 单列。所有结果都属于反复使用的历史验证划分；没有测试集选参或测试集评价。未观测标签不能解释成实验确认的药物敏感性。

11 项单元检查覆盖软 AP 的低温极限、并列分数、全部参考样本的梯度、成员梯度、独立锚点抽样、置换/平移不变性，以及关闭新增项时精确恢复旧训练。短流程重放全部 20 个检查点。

## 复现

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
OUT=improve/results/global_ap_repeat

$PY -m unittest improve.test_global_ap improve.test_global_ap_training -v
$PY -m improve.global_ap prepare --out-dir "$OUT" \
    --tuning-dir improve/results/numeric_embedding_tuning_20260912
$PY -m improve.global_ap pipeline --out-dir "$OUT" --devices cuda:0,cuda:1
```

`--smoke` 必须使用新目录，它可在前序调参结束前核验代码。正式流程只在前序调参已完成核验且未通过预设条件后执行。两轮的源码、配置、特征引用、检查点和结果保持独立。

已完成的两轮实验可用以下命令重新核对并生成合并报告：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.analyze_mechanism_retry
```
