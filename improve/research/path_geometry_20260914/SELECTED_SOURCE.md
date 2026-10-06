**BioPathNet：全文与代码核读，以及本项目适配边界。**

Hu, E. Y. et al. *Enhancing link prediction in biomedical knowledge graphs with BioPathNet*. Nature Biomedical Engineering, published 20 January 2026. [DOI:10.1038/s41551-025-01598-z](https://doi.org/10.1038/s41551-025-01598-z). Crossref 与 Europe PMC 核验，PMCID PMC13472943；出版商全文按 CC BY 4.0 公开提供。没有将检索命中数量当作已阅读全文数量。

核读了 Methods 中的源节点边界条件、DistMult 消息、PNA 聚合、广义 Bellman–Ford 递推、背景调控图与监督图的区分、NTA/SANS 采样，以及 Discussion 对数据偏差、背景图依赖性、训练代价的限制。论文在基因功能、药物–疾病、合成致死、lncRNA–靶标等任务上验证；这些结果不是本项目 MiDrug/LncDrug 的预期涨幅。路径归因可以提出解释线索，不能据此确认分子因果机制。

作者仓库为 [emyyue/BioPathNet](https://github.com/emyyue/BioPathNet)，核读提交 b051a9003a1363b5655cdb7dfdad60609c0e4fe3（2026-02-19）的 `biopathnet/model.py`、`biopathnet/layer.py`、lncTarD 配置及许可证。该配置使用六层 width 32 的 DistMult/PNA、shortcut、LayerNorm、query-dependent relations，Adam lr 0.005，10 × 150 步。作者代码默认 ReLU；正文概述的一处 sigmoid 描述不能覆盖实际配置。MIT 许可证保留于 `../../vendor/biopathnet_20260914/`。

本项目编写普通 PyTorch 的传播实现，免除 TorchDrug 定制 C++ 扩展；小图数值与缓存的作者 `message` / `aggregate` / `combine` 源码相符。模型固定六层、width 32，但作为已通过开发验证的 REIFM＋属性核的残差模块。输出层零初始化，学习目标加入残差能量惩罚。背景仅使用现成、冻结的 RNA 功能证据、无符号药物靶点与基因通路注释；不额外采样“阴性”，不将未知配对认定为生物阴性，不采用原文的双向预测或 SANS。这是独立适配，不是官方实验复现。

主比较要求 MiDrug 在已验证父模型之上继续提高，并胜过不加背景图和精确保度的 RNA–gene 打乱对照；同时实际检查 LncDrug AUC/AP 的损失范围。整个训练不依验证分数早停。三条原 GNN 种子预测不会被计成三个路径网络拟合。若结果失败，保留原冻结模型与全部对照。

检索与核验原始响应、URL、时间和哈希见 `manifest.json`、`biopathnet_verification.json`、`author_source_manifest.json`。arXiv API 与一条 OpenAlex 检索返回 429，未将其视为无相关论文；随后通过 Crossref、Europe PMC、出版商和作者仓库完成本篇核验。研究过程没有访问本项目原始测试标签。
