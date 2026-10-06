**非表格技术：所选原文、核验深度与实验对应。**

核查截至 2026-09-14。沿用 `nature-academic-search` 的多源发现与引文核验流程、`nature-downloader` 的公开授权全文流程。当前会话没有挂载学术 MCP，使用公开 Crossref、OpenAlex、arXiv HTTP 接口；检索原始响应和失败记录保存在本目录。检索命中不等于相关文献，更不等于全部读过全文。

|文献|已核验/核读内容|本项目使用方式与限制|
|---|---|---|
|Felix L. Opolka, Pietro Liò. *Graph Convolutional Gaussian Processes for Link Prediction*. [arXiv:2002.04337v1](https://arxiv.org/abs/2002.04337)，2020-02-11|arXiv 题名、作者和版本；公开 HTML 的核/链路预测方法相关段落，缓存 `gp_paper.html/.txt`|作为图核与链路预测的历史依据。当前实现为 MAP 乘积核逻辑模型，没有复现论文的变分图 GP，也没有 GP 后验不确定性。|
|Rémi Bourgerie, Šarūnas Girdzijauskas, Viktoria Fodor. *Deep Neural Sheaf Diffusion*. [arXiv:2605.19021v2](https://arxiv.org/abs/2605.19021)，2026-05-22|公开 HTML；邻接传播、奇激活、归一化、门控的相关方法，缓存 `sheaf_paper.html/.txt`|实现异构图上的 SO(2) 边传输及标量/恒等对照。是独立适配；此处按预印本引用，不称为 ICML 主会论文。实验没有建立矩阵传输的独立优势。|
|Yunqin Zhu, Chao Wang, Qi Zhang, Hui Xiong. *Graph Signal Diffusion Model for Collaborative Filtering*. SIGIR 2024. [DOI:10.1145/3626772.3657759](https://doi.org/10.1145/3626772.3657759)；[arXiv:2311.08744v3](https://arxiv.org/abs/2311.08744)，2024-04-24|DOI/元数据；公开 arXiv HTML 的前向滤波、去噪器、确定性逆过程公式；官方 `VinciZhu/GiffCF` TensorFlow 核心源码，缓存 `giffcf_paper.*` 和 `giffcf_source.py/giffcf_denoiser.py`|独立 PyTorch 双轴实现，128 模态谱近似、等权注释锚定；不是原推荐数据集上的官方复现。原始逆扩散未满足两任务条件；训练内校准后续另行封存。|
|Camille Pradel. *Reification as a Transferable Vocabulary: Zero-Shot Link Prediction with Vanilla GNNs*. [arXiv:2609.11347v1](https://arxiv.org/abs/2609.11347)，2026-09-10|公开 HTML 的题名/作者/版本、第 3 节表示与查询屏蔽方法及限制；缓存 `reification_paper.html/.txt`；随后核读官方模型/图重写/评价源码与模型卡|已使用官方 GAT 检查点，比较冻结预训练与随机权重，经过五折训练内校准后，组合未通过两任务门槛。权重、许可证、源码版本和运行适配见 `../graph_foundation_20260914/` 及 `../../vendor/reifm_20260914/PROVENANCE.json`。后续属性核＋冻结图表示属于同一方向。|

GiffCF 的 ACM PDF 请求返回 403，随后读取公开 arXiv 全文；没有绕过访问限制。官方 GitHub 原始文件部分请求超时，后用公开 contents API 成功获取。具体 URL、状态、时间及哈希见 [source_verification.json](source_verification.json)；发现阶段的广泛命中与错误见 [manifest.json](manifest.json)。

选择原则是：保留原图/矩阵或关系词表的机制，与反复压缩成配对表格后更换预测器区别开；必须同时检查原 GNN、强表格参考及适当机制对照。经典核正结果只计工程改进，前沿论文标题与复杂结构不能替代本项目的增益证据。
