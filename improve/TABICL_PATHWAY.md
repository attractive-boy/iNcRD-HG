# RNA–gene–pathway 配对特征与 TabICLv2

本轮补充公开 RNA–基因、基因–通路和药物–基因注释，检验它们能否提高 MiDrug，同时保持 LncDrug 平均 AUC、AP 不下降。通路特征代表注释支持的关联，不表示细胞系中的实测通路活性。

已完成的结果见[实验结论](TABICL_PATHWAY_FINDINGS.md)：首轮未通过，后续共享权重校准选出 alpha=0.001，在历史外层满足约束，增量极小。校准与首轮分开登记，复用预测不视为独立实验重复。

## 已补数据

数据目录：[tabicl_pathway_inputs_20260913](results/tabicl_pathway_inputs_20260913/)。

| 实体 | 总数 | 有直接基因证据 | 有直接通路证据 | 加入跨 RNA 邻域后有通路证据 |
|---|---:|---:|---:|---:|
| miRNA | 605 | 372 | 372 | 404 |
| lncRNA | 1,322 | 64 | 52 | 141 |
| Drug | 216 | 93 | 92 | 不适用 |

保留 540,491 条 miRNA–基因边、2,886 条 lncRNA–基因边和 645 条去重药物–基因边。共同字典含 17,322 个基因、1,663 个 Reactome 通路；并非每个基因都有通路注释。Reactome 版本 97，采用人类 Ensembl gene 注释、直接通路映射，通路大小限定 5–500 个基因。

身份复用原 605 个成熟链 MIMAT 编号、GENCODE gene ID，并经 HGNC 额外确认 15 条 lncRNA 身份。最终 1,307/1,322 条 LncDrug 实体有稳定身份；身份覆盖不等于靶基因覆盖。所有实体行序与原网络 CSV 逐项一致。

- [逐 RNA 覆盖](results/tabicl_pathway_inputs_20260913/PER_RNA_COVERAGE.tsv)
- [RNA–基因证据表](results/tabicl_pathway_inputs_20260913/rna_to_gene.tsv)
- [基因–通路表](results/tabicl_pathway_inputs_20260913/gene_to_pathway.tsv)
- [药物–基因表](results/tabicl_pathway_inputs_20260913/drug_to_gene.tsv)
- [完整覆盖审计](results/tabicl_pathway_inputs_20260913/COVERAGE.json)

ENCORI 提供结合/相互作用候选，不自动赋促进或抑制方向。miRNA 保留至少 1 个 CLIP 实验数据集支持的 protein-coding 靶点；lncRNA 统一要求 `interactionNum >= 3` 且 `totalExpNum >= 3`，并限定配对端为 protein-coding。直接基因边按 RNA–gene 去重，位点重复及实验数量不当成独立文献置信概率。

miRTarBase、LncTarD 的公开入口本次访问失败，未填入功能调控记录。药物沿用本地 ChEMBL 机制靶点，将 UniProt 映射到人类 Ensembl gene；9 个仅靠模糊搜索匹配的药物被排除，复合物/家族目标保留原类别及成员记录。

## 下载完整性

ENCORI 整表接口曾在 HTTP 200 正文中返回数据库错误和 PHP 执行超时。失败响应保存在 `sources/failed_bulk/`，未用于完整数据计算。改为按本项目实体查询，检查表头、行字段数、结束处错误、稳定 ID 与响应哈希；当前 [查询清单](results/tabicl_pathway_inputs_20260913/ENTITY_QUERIES.json) 的未完成请求数为 0。

名称不可用/没有注释、身份不匹配和请求失败分别记录。采用历史名称补查 miRNA 时仍以相同 MIMAT 编号为必要条件。无法确定的身份不强行连接。

NEAT1、MALAT1 的大响应采用较高筛选阈值取得完整结果，随后对全部 lncRNA 统一应用至少 3 个实验数据集及对应相互作用数量条件。这个阈值在任何模型指标产生前确定，依据下载完整性；宽松响应也按同一阈值过滤。该选择使直接 lncRNA 覆盖很低，是本轮实质限制，不能称为完整功能靶点库。

[构建源码快照](results/tabicl_pathway_inputs_20260913/BUILD_SOURCE.py) 与 INPUTS.json 内构建源码哈希一致。后续源码增加了防止刷新已冻结目录的保护，不改变已构建矩阵。

## 检验的结构

保留强图上下文基线的原 36 列及其图视图组合。两任务共用同一个冻结 TabICLv2 检查点，没有更新神经参数。

- `direct`：追加 14 列直接靶基因/通路配对特征，包括存在标记、目标数量、交集、覆盖率和通路程序余弦。
- `cross_rna`：在 direct 上追加 8 列跨 RNA 邻域程序特征。通过已有 lncRNA–miRNA 相互作用，把另一端的归一化基因程序传播过来，两个方向使用同一规则。这类联系单独编码，不能视为已验证 ceRNA 或新直接靶点。

通路投影按基因参与的通路数、通路包含的基因数分别做平方根降权，然后对实体程序 L2 归一化。输入中保留缺失标记。新特征定义、归一化规则和融合系数跨任务共享，没有新增 LncDrug 专用恒等分支。

每个新模型沿用原任务已冻结的图上下文组合，再与历史强基线做概率融合。预设 alpha 为 0.05、0.10、0.25、0.50、1.00；选择时同一 variant 和 alpha 同时作用于两个任务。原基线作为零增量对照。

## 评估边界

使用历史内部划分、推理种子 666/0/1，每种子 32 个成员。LncDrug 平均 AUC、AP 逐项不下降，无负容差；MiDrug 两项平均值均提高，且每项至少 2/3 个配对种子为正。按药物宏 AP 另报，沿用仅统计验证中同时有正、负样本药物的定义。

仅在内部满足约束并冻结共同选择后加载本轮外层数组，外层只评估选定方案和基线。不读取原始测试集。历史划分已多次参与开发，推理种子不是独立数据重复，不能据此声称独立泛化或统计显著提升。

全部新上下文状态均保存并重载复算；工程小样本另检查查询顺序和分批不变性。8 项检查覆盖字典重排、缺失证据、两任务同规则、查询标签隔离、外层门控以及低于零容差的指标退化。

新增列也会改变 TabICL 的特征置换。因此首轮结果回答“这组输入增量是否有用”，不能单独识别通路知识相对置换变化的因果贡献。非 1.0 的融合权重在实际部署时需要同时计算原模型和新模型。

## 复现

使用新目录下载或构建新版本，已冻结输入目录禁止刷新。当前保存的原始响应、证据表及矩阵可以直接复核。

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /data1/licheng/incrd-hg/.venv-tabicl/bin/python -m unittest improve.test_tabicl_pathway -v

/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.tabicl_pathway prepare \
  --out-dir improve/results/tabicl_pathway_repeat

/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.tabicl_pathway run \
  --out-dir improve/results/tabicl_pathway_repeat --task MiDrug --device cuda:0

/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.tabicl_pathway run \
  --out-dir improve/results/tabicl_pathway_repeat --task LncDrug --device cuda:1

/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.tabicl_pathway report \
  --out-dir improve/results/tabicl_pathway_repeat --phase inner
```

仅当 SELECTION.json 的 eligible 为 true，才继续以 `--phase outer` 运行两个任务并生成外层报告。工程小样本在新的 prepare 目录加 `--smoke`，其预测不得用作性能结论。
