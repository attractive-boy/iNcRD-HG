本轮验证 Q1 位点证据分层、Q2 药物特定池化和 Q3 冻结药物表示。完成后的汇总入口为 [FRONTIER_VALIDATION_FINDINGS.md](FRONTIER_VALIDATION_FINDINGS.md)，详细报告为 [REPORT.md](results/frontier_validation_20260914/REPORT.md)。

关联任务使用既有接受的图上下文＋分层证据 TabICLv2，保持同一支持样本、查询样本、32 个成员、666/0/1 三个推理种子和固定等权合成。只使用 inner 开发验证；来源任务只加载 train/validation，复用已有 PMID 排除元数据。所有输出目录需要为空；已完成结果不能覆盖。

本轮程序与职责：

|程序|职责|
|---|---|
|[site_evidence_features.py](site_evidence_features.py)|从缓存构建四类证据池化、原机制／新增结合靶点通道及保留拓扑的错配对照|
|[site_evidence_context.py](site_evidence_context.py)|MiDir 来源分组、固定线性交互模型、500 次来源连通组重采样|
|[frozen_feature_validation.py](frozen_feature_validation.py)|冻结协议、支持侧投影、关联推理、重载核验及预设条件判断|
|[drug_multimodal_inputs.py](drug_multimodal_inputs.py)|编码固定药物字典，保留未覆盖条目和同覆盖的分子对照|
|[m2umol_frozen_encoder.py](m2umol_frozen_encoder.py)|提取作者预训练版的完整 2D 分支；严格拒绝缺失参数|
|[drug_multimodal_features.py](drug_multimodal_features.py)|药物支持侧 PCA 和保持覆盖／支持成员／分子统计分层的打乱|
|[audit_frontier_boundaries.py](audit_frontier_boundaries.py)|逐个检查实际保存的拟合集合与支持实体一致，验证产物哈希|
|[summarize_frontier_validation.py](summarize_frontier_validation.py)|生成完整报告、逐药物结果和完成清单|

关联与来源程序使用 `/data1/licheng/incrd-hg/.venv-tabicl/bin/python`。M2UMol 编码与兼容性测试使用独立的 `/data1/licheng/incrd-hg/.venv-m2umol/bin/python`，继承已有 ncRNAdrug 的 PyTorch 1.13.1、PyG 2.6.1、RDKit 2024.03.5，并单独安装 torch-scatter 2.1.1+pt113cu117、OGB 1.3.6、SymPy 1.13.3、mpmath 1.3.0；没有改动 TabICL 环境。

关联实验的调用顺序如下。`--family site` 只用于 MiDrug；`--family drug` 覆盖 MiDrug/LncDrug。新输出路径和输入路径需要显式传入，先用独立 smoke 目录检查，再运行正式目录。

```bash
python -m improve.frozen_feature_validation prepare --family site --inputs-dir INPUT_DIR --out-dir NEW_SMOKE_DIR --smoke
python -m improve.frozen_feature_validation run --out-dir NEW_SMOKE_DIR --task MiDrug --device cuda:0
python -m improve.frozen_feature_validation prepare --family site --inputs-dir INPUT_DIR --out-dir NEW_FORMAL_DIR
python -m improve.frozen_feature_validation run --out-dir NEW_FORMAL_DIR --task MiDrug --device cuda:0
python -m improve.frozen_feature_validation report --out-dir NEW_FORMAL_DIR
```

`--variant` 支持逗号分隔的已登记分支，便于在两个 GPU 上分配互不重叠的工作。`--seed` 只允许协议内种子。现有完成配置可按哈希校验后跳过，但不能修改冻结实现或参数继续运行。

M2UMol 的公开示例与所链接检查点结构不同，首次缺依赖及随后参数不匹配的尝试保存在输入目录 v1/v2 的 FAILURE.json，均未产生药物向量或任务成绩。正式 v3 在评分前记录修订：使用作者预训练代码的 OGB 原子／键编码和 GraphGPS `feats2d`，103 个所需状态全部匹配；不使用缺少预训练权重的示例融合头。原始来源、提交、LFS 校验及版本见 v3 的 ENCODING_PROTOCOL.json 和 INPUTS.json。

十二项针对性测试覆盖证据类别、错配约束、缺失处理、真实支持侧投影边界、单类别指标边界，以及公开检查点完整加载和初始化独立性：

```bash
/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m unittest improve.test_site_evidence_features improve.test_drug_multimodal_features improve.test_frontier_report_guard
/data1/licheng/incrd-hg/.venv-m2umol/bin/python -m unittest improve.test_m2umol_frozen_encoder
```

后续复验应使用新的冻结输出目录。原始外层、原始关联测试和保留来源结果标签不属于本轮运行范围；现有接受模型不由这些脚本自动替换。

药物任务的未覆盖诊断子集只有一种标签，原报告函数的匹配辅助程序因此拒绝计算。本轮由 [frontier_report_guard.py](frontier_report_guard.py) 把这种匹配指标记为 NA，对双类别子集完全沿用原函数；已存档 REPORTING_AMENDMENT.json 和原失败日志。模型、预测、维度及接受条件没有改动。遇到相同情况可用 `python -m improve.frontier_report_guard --out-dir NEW_FORMAL_DIR` 生成报告。
