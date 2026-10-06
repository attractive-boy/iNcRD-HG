# 三数据集 miRNA 基线实验（2026-09-28）

正式运行完成 **169/169**；完整组合 **39/39**。

每个数据集使用同一套划分、负例和验证集选轮规则。表中的 adapted/reimpl./supervised-only 标记是方法定义的一部分。

| 模型 | 类别 | MiDrug AUC | MiDrug AUPR | MDR AUC | MDR AUPR | MDS AUC | MDS AUPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PLMF-MDA | miRNA 专项 | 0.9463 ± 0.0003 | 0.8935 ± 0.0025 | 0.9215 ± 0.0059 | 0.9095 ± 0.0103 | 0.9225 ± 0.0064 | 0.9105 ± 0.0045 |
| DLMVF (k-mer adapted) | miRNA 专项 | 0.9491 ± 0.0020 | 0.9048 ± 0.0023 | 0.9215 ± 0.0050 | 0.9068 ± 0.0117 | 0.9286 ± 0.0057 | 0.9198 ± 0.0076 |
| DLST-MDA | miRNA 专项 | 0.9472 ± 0.0014 | 0.8943 ± 0.0013 | 0.9246 ± 0.0029 | 0.9140 ± 0.0073 | 0.9295 ± 0.0047 | 0.9223 ± 0.0039 |
| SubKNet (adapted) | 生物关联网络（含 miRNA） | 0.9430 ± 0.0018 | 0.8997 ± 0.0021 | 0.9160 ± 0.0059 | 0.9059 ± 0.0090 | 0.9196 ± 0.0060 | 0.9105 ± 0.0034 |
| JMSSMMA (adapted) | miRNA 专项 | 0.9542 ± 0.0003 | 0.9112 ± 0.0013 | 0.9036 ± 0.0094 | 0.8971 ± 0.0143 | 0.9256 ± 0.0115 | 0.9214 ± 0.0132 |
| GCNNMMA | miRNA 专项 | 0.9389 ± 0.0058 | 0.8812 ± 0.0067 | 0.9095 ± 0.0111 | 0.8947 ± 0.0135 | 0.9110 ± 0.0069 | 0.8997 ± 0.0077 |
| AMMGC (reimpl.) | miRNA 专项 | 0.9513 ± 0.0017 | 0.9095 ± 0.0025 | 0.9267 ± 0.0070 | 0.9157 ± 0.0126 | 0.9278 ± 0.0074 | 0.9197 ± 0.0081 |
| GCMDR (reimpl.) | miRNA 专项 | 0.9382 ± 0.0085 | 0.8891 ± 0.0108 | 0.8898 ± 0.0052 | 0.8910 ± 0.0069 | 0.9136 ± 0.0075 | 0.9109 ± 0.0059 |
| GraphDTA (adapted) | 药物–靶标迁移 | 0.9327 ± 0.0022 | 0.8765 ± 0.0042 | 0.9027 ± 0.0057 | 0.8859 ± 0.0136 | 0.9064 ± 0.0071 | 0.8978 ± 0.0087 |
| ML-DTI (adapted) | 药物–靶标迁移 | 0.9417 ± 0.0008 | 0.8821 ± 0.0027 | 0.9215 ± 0.0031 | 0.9098 ± 0.0081 | 0.9261 ± 0.0053 | 0.9180 ± 0.0048 |
| SubMDTA (supervised-only) | 药物–靶标迁移 | 0.9291 ± 0.0011 | 0.8680 ± 0.0065 | 0.8976 ± 0.0062 | 0.8825 ± 0.0098 | 0.8995 ± 0.0048 | 0.8872 ± 0.0101 |
| LAGCN (adapted reimpl.) | 药物–疾病迁移 | 0.9525 ± 0.0000 | 0.9092 ± 0.0001 | 0.9294 ± 0.0051 | 0.9172 ± 0.0104 | 0.9325 ± 0.0054 | 0.9263 ± 0.0044 |
| iNcRD-HG (original; PLMF adapted) | 本项目原始模型 | 0.9534 ± 0.0005 | 0.9137 ± 0.0011 | 0.9243 ± 0.0073 | 0.9171 ± 0.0122 | 0.9317 ± 0.0073 | 0.9286 ± 0.0061 |

## 协议与解释

- **完成状态**：169/169 次正式运行；39/39 个完整模型×数据集组合。短跑 smoke 不计入。
- **MiDrug**：保留原有 train/val/test 成员；只删除训练中 2 条同标签重复行。605 miRNA × 216 药物；正负 1:2。3 个模型种子 42/43/44，标准差反映初始化波动。
- **MDR/MDS**：作者公开代码为单次 80/20；本项目按论文描述构建 seed=42 的外层 5 折，与前续基线逐行相同，并非作者发布的折索引。每个训练折中分层取 10% 验证。正负 1:1。只做一套外层划分，标准差为折间波动，不是 5 次独立重复。
- **选模型**：仅以验证 AUROC 选 epoch；固定 0.5 分类阈值；测试标签只在选定模型后用于一次评估。未做嵌套超参搜索。
- **图与特征**：目标 miRNA–drug 关联图/GIP 只使用训练正例；验证正例与测试正例均不进入消息图。冻结内容特征可使用已知实体全集。原始 iNcRD-HG 在 MiDrug 上额外保留发布的 lncRNA 辅助关系与表达谱，方法间输入信息并非完全相同。
- **成熟序列**：MiDrug 605 条成熟序列已按名称补全；604 条不同序列。以 PLMF 元数据表为历史名称来源，miRBase 为补充，miR-302f 的 arm 更名单独记录。未把前体序列头部误作成熟序列。
- **内容重合**：作者实体 ID 未合并；按相同成熟序列+原始 SMILES 字符串检查，MiDrug 有 4 个、MDR 每折 2–4 个、MDS 每折 6–9 个 train-test 内容相同对。ID 对无重叠。本轮是常规转导式关联预测，不是内容去重或冷启动。
- **AP 定义**：AUPR 使用 sklearn average_precision_score（AP），不是梯形积分 PR-AUC；不同正负比例下不直接比较三个数据集的 AP 高低。
- **负例语义**：使用公开 benchmark 未关联/负例池；这些不是全部经过生物实验验证的真阴性，属于闭世界采样基准。
- **与旧结果比较**：旧脚本多按 test-best 选轮次，部分还在测试图加入验证正例。本次严格 train-only/validation-select，不能把旧分数直接拼入主表。
- **模型来源差异**：PLMF/DLST/DLMVF/SubKNet 等保留作者主要网络并适配数据；GCMDR/AMMGC/LAGCN 是本地结构重实现。AMMGC 本地版联合训练四视图，并非原代码分阶段训练。
- **不完整方法**：SubMDTA 只有监督预测结构，缺原文子结构预训练；DLMVF 缺原始 miRNA–基因图，使用 k-mer 替代视图；JMSSMMA 使用本任务训练负例替代原作者任意节点类型负例。必须随分数保留这些标记。
- **PLMF 适配修复**：原函数截节点到 100 却保留全部键；MiDrug 的一个 551 原子药物因此越界/跨图。现保留大分子的全部原子与键，小分子保持原 100-node 填充；MDR/MDS 最大 96 原子，处理不变。已检查全部药物 batch 的边不跨分子。
- **原始 iNcRD-HG**：锁定最初代码提交 eb8de624a731a4c076143bf2f9c91e539affee43 的 Model/GNNEncoder/EdgeDecoder，GeneralConv×2、hidden=256、关系 mean、三层 MLP、全批量 RMSprop(lr=1e-4, alpha=.99)，训练 300 epochs。MiDrug 恢复原始 lncRNA/miRNA/drug 三类节点及辅助关系，去除后加的 circRNA；使用发布表达谱/ChemBERTa 特征。
- **原始版跨数据集适配**：MDR/MDS 缺原始表达谱与 lncRNA 辅助图：使用 80 维 2+3-mer 和原版 ChemBERTa-zinc-base-v1 768 维特征，行 min-max、双向余弦 kNN(10)，保留原网络及训练配方。原始版一行的 PLMF adapted 标记必须保留。目标边用共享训练正例，验证集选轮；不是原脚本逐轮看测试/测试图加入验证边的旧分数。
- **统计范围**：表中均值±样本标准差；没有把 3 个种子和 5 折混作相同重复数，也没有跨数据集平均排名；没有据这些重实现分数推翻原文结论。
- **复核**：verify.py 核对每条结果的预测、指标、checkpoint hash、数据 hash 和验证集最优 epoch。literature/verified_references.json 与 registry.json 保留论文出处。

## 候选模型与论文来源

| 模型 | 类别 | 年份 | 论文 DOI | 实现 |
| --- | --- | --- | --- | --- |
| PLMF-MDA | miRNA 专项 | 2026 | [10.1371/journal.pcbi.1013968](https://doi.org/10.1371/journal.pcbi.1013968) | 作者网络适配；冻结语言编码器 eval 缓存 |
| DLMVF (k-mer adapted) | miRNA 专项 | 2026 | [10.1021/acs.jcim.5c02839](https://doi.org/10.1021/acs.jcim.5c02839) | 作者网络适配；基因网络视图替换为 k-mer；训练关联图 |
| DLST-MDA | miRNA 专项 | 2025 | [10.1021/acs.jcim.5c00038](https://doi.org/10.1021/acs.jcim.5c00038) | 作者 GCNNetmuti 网络与数据接口适配 |
| SubKNet (adapted) | 生物关联网络（含 miRNA） | 2024 | [10.1109/JBHI.2024.3390092](https://doi.org/10.1109/JBHI.2024.3390092) | 作者子图提取、RW 核和全局 GCN；相似度换为序列/结构余弦 PCA |
| JMSSMMA (adapted) | miRNA 专项 | 2024（在线 2023） | [10.1016/j.omtn.2023.102103](https://doi.org/10.1016/j.omtn.2023.102103) | 作者 GIN 与双解码器；固定二部图训练负例替代任意类型负例 |
| GCNNMMA | miRNA 专项 | 2023 | [10.3389/fgene.2023.1201934](https://doi.org/10.3389/fgene.2023.1201934) | 作者结构批量移植；修复无键原子/不连通分子 WL 索引 |
| AMMGC (reimpl.) | miRNA 专项 | 2022（卷年 2021） | [10.3389/fphar.2021.799108](https://doi.org/10.3389/fphar.2021.799108) | 本地 PyTorch 结构重实现；四视图内容/GIP 注意力融合 |
| GCMDR (reimpl.) | miRNA 专项 | 2020 | [10.1093/bioinformatics/btz621](https://doi.org/10.1093/bioinformatics/btz621) | 本地 PyTorch 重实现；图卷积、节点偏置、双线性解码 |
| GraphDTA (adapted) | 药物–靶标迁移 | 2021 | [10.1093/bioinformatics/btaa921](https://doi.org/10.1093/bioinformatics/btaa921) | 蛋白分支改为 24-nt miRNA，回归改为二分类 |
| ML-DTI (adapted) | 药物–靶标迁移 | 2021 | [10.1021/acs.jpclett.1c00867](https://doi.org/10.1021/acs.jpclett.1c00867) | 作者互学习注意力结构；miRNA 二分类适配 |
| SubMDTA (supervised-only) | 药物–靶标迁移 | 2023 | [10.1186/s12859-023-05460-4](https://doi.org/10.1186/s12859-023-05460-4) | GIN + 多尺度 BiLSTM 预测结构；未使用原文子结构自监督预训练 |
| LAGCN (adapted reimpl.) | 药物–疾病迁移 | 2021 | [10.1093/bib/bbaa243](https://doi.org/10.1093/bib/bbaa243) | 本地层注意力 GCN 重实现，迁移到 miRNA–药物；非 miRNA 原生模型 |
| iNcRD-HG (original; PLMF adapted) | 本项目原始模型 | 原始发布版 | [原始源码](https://github.com/attractive-boy/iNcRD-HG/tree/eb8de624a731a4c076143bf2f9c91e539affee43) | 原始 GeneralConv×2 + MLP；MiDrug 保留原始 lncRNA 辅助图；MDR/MDS 用序列/ChemBERTa 与双类型图适配；RMSprop 300 epochs |

## 产物

- Excel：`/data1/licheng/incrd-hg/三数据集_miRNA基线实验_20260928.xlsx`。
- `summary.csv`：数值均值和标准差。
- `results/*_predictions.npz`：逐样本 ID、标签、概率；`*.pt`：验证集选定的模型。
- `results/*_curves.json`：训练损失和验证指标；没有逐轮测试分数。
- `data/manifest.json`、`registry.json`、`literature/verified_references.json`：数据、方法和引用溯源。
- `verification.json`：完整性和结果重算核查。
- 运行方式见 `README.md`。
