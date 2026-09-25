**RNA–基因–通路补充数据下载清单（2026-09-13 核对）**

更新：用户已完成三组主数据下载，并确认 DepMap 三文件为 25Q2。原文件已收到并接入新的冻结输入，见[本轮数据与模型说明](TABICL_PATHWAY_CONTEXT.md)。下文保留下载前的核对记录；其中服务器访问失败不再表示本地缺少这些文件。

数据覆盖不足是限制之一，但现在还不能认定它是模型提升小的唯一原因。当前需要补的是匹配细胞系的基因表达和功能调控证据。RNA 身份映射已经完成大部分：miRNA 为 605/605，lncRNA 为 1,307/1,322。

本次重新检查了已下载的 ENCORI 原始响应，保持 RNA 身份和蛋白编码靶基因筛选不变，只审计两个支持阈值：

|条件：interactionNum、totalExpNum 均至少为|有 RNA–基因相互作用的项目 lncRNA|去重 RNA–基因对|
|---|---:|---:|
|1|至少 858 / 1,322|至少 43,102|
|2|至少 163 / 1,322|至少 7,651|
|3，上一轮实际模型输入|64 / 1,322|2,886|

NEAT1 只有支持数至少为 2 的完整响应，MALAT1 只有至少为 3 的完整响应，所以较低阈值的结果按下界报告。它们都是结合/相互作用证据，不能等同于已验证的功能调控。该审计说明大量低支持证据已经在本地，**不需要重新下载 ENCORI 才能研究证据分层**，但没有检验放宽阈值能否提高预测性能。[审计记录](research/pathway_data_downloads_20260913/LOCAL_COVERAGE_AUDIT.json)

**1. 最先下载：CCLE / DepMap 基因表达，共三个文件**

入口：[DepMap All Data](https://depmap.org/portal/data_page/?tab=allData)。也可从 [Broad CCLE 官方数据页](https://sites.broadinstitute.org/ccle/datasets) 点击 **Broad DepMap Portal** 进入。

操作：打开 **All Data** → 文件集选 **DepMap Public** → 固定一个 release → 搜索下表文件名并逐个下载。原始 LncExp 的 release 目前尚未确认；如果能找回原 release，优先使用与其匹配的基因表达。否则可以先用 **DepMap Public 25Q2** 作为可复现的起点，这一版的文件名和含义已经从官方发布说明核实；它不是已确认与当前 RNA 表同一版本，也不表示它是最新版。

|文件|用途|本轮需要保留的内容|
|---|---|---|
|`OmicsExpressionProteinCodingGenesTPMLogp1.csv`|为已有 RNA 表配上同细胞系的蛋白编码基因表达，计算通路分数或表达关联|下载完整 CSV，保留全部基因列和原始细胞 ID；读取该 release 的单位说明|
|`Model.csv`|对应细胞系名称、癌种/组织、模型 ID 及历史别名|完整表，尤其 ModelID、ModelIDAlias、细胞系和癌种字段|
|`OmicsProfiles.csv`|对应测量 Profile 与模型；辨别默认表达样本和不同处理条件|完整表，保留 Profile、Model、数据类型及默认样本标记|

25Q2 的官方说明明确：表达矩阵每个 model 一行；`OmicsProfiles.csv` 增加 `is_default_entry`；`Model.csv` 增加 `ModelIDAlias`。若选其他 release，必须按该版实际列名解析。[官方发布说明](https://forum.depmap.org/t/announcing-the-25q2-release/4257)

当前 `LncExp.csv` 和 `MiExp.csv` **都包含相同的 938 个 ACH 细胞系编号，但行顺序不同**。我已经导出 [ACH938.txt](research/pathway_data_downloads_20260913/ACH938.txt)，顺序与 LncExp 一致。无需逐个在网页上下载细胞系；下载完整矩阵后按 ACH/ModelID 取交集。若文件使用 PR 开头的 ProfileID，需要利用同版 OmicsProfiles 映射到 ACH，并依照该版默认表达样本规则选择；历史 ACH 别名应通过 Model 表审计，不能按相似名称猜测或按行位置拼接。

这组数据能建立“同细胞系 RNA 表达—基因/通路表达关联”，得到的相关性会单独标记，不能当作实验验证的直接调控。新 release 能匹配多少个原始细胞系、测量和处理流程有何差异，要在实际取得文件后核对。

**2. 优先补强 LncDrug：LncTarD 2.0 的 lncRNA–target 功能调控表**

入口：[论文提供的 LncTarD 数据入口](https://bio-bigdata.hrbmu.edu.cn/LncTarD/)，[论文中的 2.0 域名](https://lnctard.bio-database.com/)。

操作：进入 **Download** → 下载完整的 **lncRNA–target regulation / association 数据表**。如果页面把功能调控和表达关联分别提供，分别下载并保留原文件名；如果提供合并总表，完整下载后保留区分这两类记录的字段。

需要保留：lncRNA 名称/ID、target 名称/ID及类型、调控方向或机制、实验方法、疾病/细胞系、PMID。网页字段名可能不同，直接保存原表即可。后续对齐到项目 RNA，并区分蛋白编码靶基因、其他 RNA 靶点、实验功能证据和表达关联。

论文报告 5,531 条功能调控和 2,829 条表达关联，但这个总量不等于能覆盖本项目所有 lncRNA。论文明确全量数据通过 Download 页面提供；本次未能进入下载页，因此**没有核实当前按钮文字、压缩包文件名或直链**。[论文原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9825480/)（DOI: 10.1093/nar/gkac984）

**3. 补强 MiDrug：miRTarBase 的 human miRNA–target interactions 表**

入口：[miRTarBase 2025 / 10.0](https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/)。

操作：进入下载区 → 物种选择 **Homo sapiens / human** → 下载完整的 **miRNA–target interactions，MTI** 表。保留强功能证据和高通量证据的原始分类，后续分别编码。

需要保留：miRNA 名称/稳定 ID、物种、target gene 名称和 Gene ID、实验方法、support/evidence type、PMID。完整 CSV/TSV/Excel 均可，保留原始列和文件名。这里要取得的是 RNA–靶基因证据；该版本另外提供的 miRNA–drug resistance 模块整合了 ncRNADrug，与当前预测标签来源有关，不能作为独立外部特征直接回填。[论文原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701613/)（DOI: 10.1093/nar/gkae1072）

本次服务器无法完成官网 TLS 握手，因此未核实当前下载文件名和直链。

**下载位置与命令**

已经创建接收目录，文件下载后保留原名，放入对应目录：

```text
/data1/licheng/incrd-hg/data_external/pathway_20260913/
  depmap/       # 表达矩阵、Model、OmicsProfiles；来自同一 release
  lnctard/      # lncRNA–target 原始调控/关联表
  mirtarbase/   # human MTI 原始表
```

把所选 release、下载页网址和下载日期一起保存到各目录的 `SOURCE.txt`。如果多个 release 都要保留，先建立不同的 release 子目录，避免覆盖同名 CSV。

浏览器能打开官方文件按钮时，可右键复制实际下载链接，然后在服务器运行。例如只下载 DepMap 表达矩阵：

```bash
cd /data1/licheng/incrd-hg/data_external/pathway_20260913/depmap
read -r -p '粘贴该 release 官方表达矩阵下载按钮的链接: ' pathway_expression_url
curl --fail --location --retry 2 --connect-timeout 20 \
  --output OmicsExpressionProteinCodingGenesTPMLogp1.csv \
  "$pathway_expression_url"
```

这个命令需要你复制的真实文件链接，网页首页 URL 不是数据文件链接。若服务器仍访问失败，可以直接在浏览器下载文件后放入上述目录。我没有提供未经核实的文件编号或猜测的直链。

**本次服务器访问结果与已有数据**

|来源|2026-09-13 实际观察|
|---|---|
|Broad CCLE 数据页|HTTP 200，明确链接到 DepMap 下载门户|
|DepMap 下载页|HTTP 403；完整数据文件未下载|
|DepMap 官方发布公告/官方代码|HTTP 200；核实上述三文件名及 ID 对齐方式|
|LncTarD `bio-bigdata.hrbmu.edu.cn`|HTTP 403|
|LncTarD `lnctard.bio-database.com`|TLS 证书主机名不匹配|
|miRTarBase|TLS 握手失败|
|LncRNA2Target `bio-annotation.cn`|连接被拒绝，因此暂未列为本轮必需下载项|

这些是当前服务器的访问结果，不能据此判断你的浏览器也无法访问。**先取得第 1 组即可支持下一轮表达/通路实验设计**；其余两组补充功能证据，不必等所有失效站点恢复后才能推进。

ENCORI 原始相互作用、Reactome 97、HGNC、UniProt 映射及部分 ChEMBL 机制靶点已经在本地，无需重复下载。药物端目前也仅有 93/216 能对应到基因，后续仍要报告 RNA 与药物两端的共同覆盖率；不会把补足 RNA 名称映射说成整条机制链已经完整。

本轮完成的是数据清单、文件名核查和覆盖审计，没有重跑模型。取得数据后应先校验真实文件格式、版本、哈希、RNA/细胞匹配和证据类型，再冻结新的输入做对照。已有冻结输入和实验结果保留；后续仍按 **LncDrug AUC、AP 均不下降，按药物宏 AP 单独报告** 验收，当前没有保证下载这些数据就一定带来提升。
