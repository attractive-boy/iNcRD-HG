# 三数据集 miRNA 基线实验

13 个基线（新增最初发布的 iNcRD-HG），在 MiDrug（iNcRD-HG）、MDR 和 MDS（PLMF-MDA）上使用共享划分运行，共 169 次正式实验。候选与引用见 `registry.json`；最终数字由 `summarize.py` 从逐次预测产生，见 `REPORT.md`。此前 12 模型的完整表与版本记录保留在 `archive_12_models/`。

使用作者 miRNA 专项模型、原始 iNcRD-HG 与常用迁移对照。原有 8 个方法属于 miRNA/生物关联预测方向，另加入本项目原始模型；GraphDTA、ML-DTI、SubMDTA 为药物–靶标迁移，LAGCN 为药物–疾病迁移。不能把全部方法都称为“miRNA 原生 SOTA”。AMMGC/GCMDR/LAGCN 的本地结构重实现、DLMVF 替代特征、SubMDTA 缺失预训练等差异必须和数值一起报告。

## 固定协议

- MiDrug：现有独立 train/val/test；只去除训练集 2 条同标签重复样本，未改测试集；模型种子 42/43/44。
- MDR/MDS：作者公开代码只实现一次 80/20 划分。本项目按论文描述构建 seed=42 外层 5 折，与此前 `plmf_baselines_20260920` 逐行相同；从外层训练集分层划 10% 验证。这不是作者发布的折索引文件。
- 目标 miRNA–drug 图/GIP 只用训练正例；验证 AUROC 选轮次；阈值固定 0.5。测试仅在模型选定后打分。原始 iNcRD-HG 的 MiDrug 额外使用发布的 lncRNA 辅助关系，保留其原始输入信息。
- AUROC、AP（AUPR）、ACC、Precision、Recall、Specificity、F1、MCC；均值 ± 样本标准差。MiDrug 为种子间 SD，MDR/MDS 为折间 SD。
- MiDrug 正负比 1:2，MDR/MDS 1:1。含作者实体 ID 的极少量内容别名重复，详见 manifest，属于转导式常规关联预测。
- MiDrug 成熟序列由 PLMF 实体元数据及 miRBase 补齐；来源是无关联标签的序列表，逐条出处存于 `data/MiDrug_sequence_provenance.json`。全套数据使用原始 ID，不以字符串去重合并实体。

## 运行

本机环境为 `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`，Python 3.8，PyTorch 1.13.1+cu117、PyG 2.6.1、NumPy 1.24.4、SciPy 1.10.1、scikit-learn 1.3.2、pandas 2.0.3、RDKit 2024.03.5、openpyxl 3.1.5。PLMF 使用既有独立 `plmf_site` 和本地预训练权重。

```bash
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/mirna_baselines_20260928
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
# 已生成的数据在 data/ 下；重新生成时需保留下载的 miRBase FASTA 和所有原始数据。
$PY bench_data.py
$PY plmf_adapter.py  # 仅首次缓存冻结语言模型
$PY launch.py --models gcmdr,ammgc,lagcn,jmssmma --devices cuda:0,cuda:1
$PY launch.py --models graphdta,mldti,dlstmda,gcnnmma,submdta,dlmvf,plmf --devices cuda:0,cuda:1
$PY launch.py --models subknet --devices cuda:0,cuda:1
$PY incrd_adapter.py  # 冻结原始 iNcRD-HG 的三数据集输入，仅首次需要
$PY launch.py --models incrd_original --devices cuda:0,cuda:1
$PY verify.py --require-complete
$PY provenance.py
$PY summarize.py --require-complete
```

上述 launch 可对不重叠的模型集合并发运行。单次失败会写日志，其他组合继续；再次运行会跳过已完成组合。同一组合有文件锁避免重复写入。

## 最初发布版 iNcRD-HG

`original_source/` 锁定最初上传代码的 Git 提交 `eb8de624a731a4c076143bf2f9c91e539affee43`，其中 `Model`、`GNNEncoder`、`EdgeDecoder` 三个类直接通过 AST 加载，不运行旧脚本的测试循环。网络保留两层 `GeneralConv`、256 维隐藏层、关系 mean 聚合、三层 MLP、LeakyReLU(0.1)；保留 RMSprop(lr=1e-4, alpha=0.99)、BCE、全批量训练 300 epochs。没有使用后续的查询掩蔽、共同邻居、EASE、融合或随机效应模块。

- MiDrug：从现有发布训练图提取原始 lncRNA/miRNA/drug 三类节点、表达谱/ChemBERTa 特征和 7 类辅助关系，移除后续加入的 circRNA。目标 MiDrug 及反向边替换为本轮共享训练正例，共 9 类关系；保留原有同类边和双自环。保留原始的 lncRNA–drug 侧信息，因此它与只用序列/分子结构的模型并非完全相同输入条件。
- MDR/MDS：数据缺少原始表达谱及 lncRNA 辅助图，保留网络结构，miRNA 使用 80 维 2+3-mer，drug 使用原版 `ChemBERTa-zinc-base-v1` 768 维缓存，按 DrugBank ID 和 SMILES 同时核对。行 min-max 后构建对称余弦 kNN(10)；使用 miRNA/drug 两类节点和 4 类关系。结果名称保留 `PLMF adapted`。
- 评估：与其他 12 方法相同，训练/验证/测试均使用训练目标正例图，以验证 AUROC 选轮，最后评估一次测试。原始脚本逐轮观察测试及测试图加入验证边的行为不用于本轮比较。
- 输入与来源：`data/incrd_original_manifest.json`、`data/*_incrd_original_inputs.pt`；原代码提交及文件哈希见 `original_source/SOURCE.json`。`verify.py` 检查侧信息逐项相等、目标训练边、留出标签隔离和 RMSprop 配方。

SubKNet 可加 `--compact-subknet`：将独热输入的第一层线性映射改为等价索引运算，并把评估时随机游走核的求和改为小特征 Gram 矩阵的收缩，避免大尺寸中间张量。训练时保留作者的逐元素 dropout；参数与 checkpoint 格式一致。前向/梯度检查见 `verify.py`。部分已完成运行保留原始密集计算，其余使用该等价计算选项，逐次配置中有记录。

PLMF 原始分子图预处理把节点截到 100，却保留所有键。MiDrug 的 CID 145994598 有 551 个原子，旧函数会产生越界或连接到下一分子的错误边。`safe_drug_graph` 保留这个药物的全部原子/键；小于 100 原子的药物仍沿用作者的填充行为。MDR/MDS 最大为 96 个原子，其处理数值不变。回归检查覆盖三个数据集全部药物，验证 batch 内每条边的两端始终属于同一分子。失败的 PLMF 运行没有进入主表。

JMSSMMA 的扩展包只装在本目录 `vendor/`，未改共享 conda 环境：

```bash
$PY -m pip install --target vendor --no-deps --no-index \
  --find-links https://data.pyg.org/whl/torch-1.13.0+cu117.html torch-sparse torch-scatter
```

单次运行示例：

```bash
$PY run.py --model dlstmda --dataset MDR --fold 0 --seed 42 --device cuda:1
```

`run.py` 固定各方法最大 epoch、学习率和早停参数；完整数值在每条 `results/*.json` 的 `config` 中。`--epochs` 等参数可显式覆盖。短跑必须写到独立目录，例如 `--epochs 2 --output smoke`；短跑结果不进入汇总。

## 可复核文件

- `data/manifest.json`：源文件 SHA256、样本数、划分 SHA256、ID/内容重合检查。
- `data/{dataset}_fold{fold}.npz`：每个模型完全相同的 train/val/test 配对数组。
- `results/{model}_{dataset}_f{fold}_s{seed}.json`：参数、所选轮次、指标、checkpoint/hash。
- 对应 `_predictions.npz`：验证/测试标签及概率，顺序与划分一致。
- 对应 `_curves.json`：训练损失、验证 AUROC/AP；对应 `.pt`：所选权重。
- `literature/verified_references.json` 和 `references.bib`：已按 DOI 匹配的 Europe PMC/OpenAlex 记录；搜索失败日志也保留。
- `verify.py`：检查外层折与旧版一致、ID 无交叉、成熟序列覆盖；用故意反转的测试最优轮构造检查验证选模；扰动留出标签检查图/GIP 不变；重算全部结果。
- `source_manifest.json` / `source_snapshot/`：本地适配器及被调用作者/历史实现的源文件快照和哈希，预训练权重、缓存与实体特征的哈希，运行环境版本。
- 根目录 Excel 的“全部指标汇总”保留 8 项指标的数值均值与标准差，“逐次运行”保留所有折/种子的原始分数。主对照表同时显示方法适配标记；未跑满的组合不计算部分折均值。

文献检索的 MCP 在当前环境未挂载，因此使用 OpenAlex 与 Europe PMC 公共 API。OpenAlex 部分请求返回 429 或超时，这些失败保留在检索记录中；最终 12 条方法引用均已从成功返回的元数据按 DOI 匹配，未用猜测的题名/DOI 填表。

## 不应混用的旧口径

本项目此前大量结果使用测试最优轮次（test-best），部分测试消息图还加入验证正例。它们不能拼接到本轮主表。作者原文分数与本地重新训练分数也不能混填。不把迁移实现的性能当作原论文完整模型性能，也不据此断言原论文结论错误。

本轮未做冷启动、重复外层 CV 或嵌套超参优化。若用于投稿，应按论文任务需求补充这些评估，并保留当前主表完整的适配说明。
