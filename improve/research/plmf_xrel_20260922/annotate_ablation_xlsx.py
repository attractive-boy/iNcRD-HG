"""Add detailed notes to results/ablation_PLMF-MDA_seed42.xlsx.
Sheet 消融: keep A-C, add D/E (Δ AUC/AUPR vs ctx+diff×LR) and F (per-row 说明).
Sheet 说明: protocol, metrics, feature groups, leakage safeguards, data sources, reproduction, reading notes.
"""
import openpyxl, os
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, 'results', 'ablation_PLMF-MDA_seed42.xlsx')
wb = openpyxl.load_workbook(F); ws = wb['消融']

NOTES = {
 'PLMF-MDA（官方模型，本机复现）':
  '主对比基线。PLMF-MDA（PLoS CB 2026）官方代码在本机按其论文 5 折协议重跑，同一 seed 42 的折划分与 1:1 负采样（../plmf_baselines_20260920）。此处为 5 个测试折的均值。',
 'MPHGNN（无泄漏口径）':
  '第二基线。MPHGNN（BiB 2026）移植到 PLMF-MDA 数据（../mphgnn_plmf_20260921）。其官方"全局分支"在评分时会读到测试批次的度信息，带来 +0.01~0.02 AUC 的泄漏；此处用逐对隔离评分口径，元路径通道置零/打乱结果不变。',
 'iNcRD-HG base（代码分支 GeneralConv 异质图，k-mer + ChemBERTa + kNN 边）':
  '本仓库原始端到端 GNN：miRNA 侧 k-mer 序列特征、药物侧 ChemBERTa SMILES 嵌入，两侧各加 kNN 相似边组成异质图，GeneralConv 消息传递 + 双线性解码。数值来自 ../plmf_innov_20260921（plain 设置）。',
 'iNcRD-HG base + 图上下文 ctx33 + kNN20':
  '同一 GNN，节点/边输入再拼上 33 列图上下文特征（见说明页 ctx），kNN 邻居数 20。是 GNN 路线的最好结果；再往上换架构/损失/对比学习均无提升（../plmf_innov_20260921 §3）。',
 '+ 图上下文 + 扩散谱特征 × 线性读头（ctx+diff × LR）':
  '改为"确定性特征 + 线性读头"的单模型：33 列图上下文 ctx + 52 列扩散谱 diff（个性化 PageRank / Katz / path-7 及其逐 miRNA 行 z、逐药物列 z）共 85 列，L2 逻辑回归 C=0.3。所有依赖标签的特征在训练折内做 K=10 内折交叉拟合，测试对只看训练正例。以下各行均在此 85 列上叠加列，读头不变；D/E 列即相对本行的增量。',
 '+ 溯源超边（iNcRD-HG 记录，覆盖 60%）':
  '溯源超图启发式（prov，24 列 = pmid 6 + costudy 6 + ctx 4 + rec 8）：把每条训练正例的策展记录（报道它的 PMID、细胞系/表型/疾病语境、通量、年份）建成"研究超边"和"语境超边"，计算 m 与 d 经其它对被同一研究报道的桥接数、研究共现图 S_r=M_rM_rᵀ 上的共研究关联数、语境共现、记录统计等。记录取自 iNcRD-HG 自带表，只覆盖 58~62% 的正例（605 个 miRNA）。ncol=109。',
 '+ 溯源超边（ncRNADrug 完整记录，覆盖 95%）':
  '同样 24 列，记录换成 ncRNADrug DR_Curated 完整策展表（29.5k 条；miRNA 名 + DrugBank id 精确匹配 89.6/89.7%，去 -3p/-5p 后缀后 95.4/96.1%，build_full_records.py）。数据事实：PLMF-MDA 的正例就是这张表的 resistant/sensitive 对，故只用训练正例的记录 + 交叉拟合，query 对永不出现在自己的 support 里。本轮第一个正向改进（provf）。',
 '　└ 对照：溯源记录置换':
  '打乱对照（provfs）：完整记录在正例对之间随机置换后重算同样 24 列。结果回到 ctx+diff×LR 水平，说明上一行的增益来自真实溯源结构，而不是列数增加或实体流行度。',
 '+ miRNA 旁信息关系（仅旁信息）':
  '不加溯源，只加 19 列 miRNA 侧标签无关旁信息（mprior）：名称家族 5（去 -3p/-5p 与字母后缀的成员数、成员对 d 的关联数/均值/占比、家族总度）、seed 7-mer 家族 5、HMDD v4 疾病谱 7（疾病数、疾病桥、HMDD 相似 kNN 标签迁移、邻居流行度）、另一表型（MDS↔MDR）的度 2。ncol=104。作用对象是训练图中度数 0~3 的稀疏 miRNA。',
 '+ 完整溯源 + 旁信息':
  '第一 + 第二改进叠加（phlf，128 列）。相对 provf 再 +0.002 左右且 4/4 种子为正；研究分组划分下旁信息的增益放大到 +0.004（REPORT_XREL §5）。',
 '　└ 对照：溯源置换 + 真实旁信息':
  '打乱对照（phlfs）：记录置换、旁信息真实。落回"仅旁信息"一行的水平，证明溯源与旁信息两块增益相互独立、可加。',
 '+ 完整溯源 + 旁信息 + 基因组簇':
  '在 phlf 上加 miRBase v23.1 基因组簇关系 5 列（hsa.gff3，前体同链 10 kb 内成簇；茎名映射 97% miRNA，51~53% 在多成员簇）：簇成员对 d 的支持统计。标签无关证据，置换对照 → 0。ncol=133。',
 '+ 完整溯源 + 旁信息 + GEO 组学证据（研究无关）':
  '在 phlf 上加 ncRNADrug DR_GEO（耐药/敏感样本 miRNA 差异表达）与 DT_GEO（给药前后差异表达）证据 7 列：同向/反向 GSE 数、最大 −log10 p、给药差异 GSE 数、显著数、覆盖标志。每个 GSE 经 eutils 映射到 PMID，剔除被任何策展记录引用过的系列（73 个里 32 个），只保留研究无关的实验证据。ncol=135。随机划分下增益小，研究分组划分下放大到 +0.0016/+0.0026。',
 '+ 完整溯源 + 旁信息 + 无标签池训练 ×10':
  '第三改进（协议侧）。特征同 phlf；训练集从数据集发布的未标注对池（MDR 132,874 / MDS 170,226 对，排除协议中的全部训练与测试对）再抽 10×n_pos 作为额外负例，按类别先验重加权（负类权重 = n_pos/n_neg）训练同一线性读头；测试集与其它行完全相同。机制是降低负类分布的估计方差，×1→×3→×10 单调、×40 饱和。',
 '★ iNcRD-HG（完整溯源 + 旁信息 + 簇 + GEO + 无标签池）':
  '最终单模型：135 列（ctx 33 + diff 52 + prov 24 + mprior 19 + cluster 5 + geo 7）+ 无标签池 ×10，L2 逻辑回归，可训练参数约 140。命令：probe_extraneg.py --dataset {MDR,MDS} --seed 42 --mult 10 --prov full --xrel cluster,geo。相对 ctx+diff×LR：MDR +0.0127/+0.0162，MDS +0.0100/+0.0122；相对 PLMF-MDA 官方复现约 +0.04 AUC / +0.05 AUPR。去噪评估（剔除别处已记录为正的测试负例）：MDR 0.9694/0.9698，MDS 0.9690/0.9707。',
}
HEAD_NOTE = {
 'MDR': 'MDR = miRNA–drug resistance（耐药）表型矩阵。PLMF-MDA 论文 5 折协议：正例随机分 5 折，负例按 1:1 从未标注对随机采样，seed 42 同时控制折划分与负采样；表中 AUC/AUPR 为 5 个测试折的均值。4 个种子（42,0,1,2）的均值见同目录 ablation_PLMF-MDA.xlsx。',
 'MDS': 'MDS = miRNA–drug sensitivity（敏感）表型矩阵。协议、seed 与折划分同 MDR 块；两块共用同一套代码与超参，没有按数据集单独调参。',
}
thin = Side(style='thin', color='999999'); border = Border(top=thin, bottom=thin, left=thin, right=thin)
hfont = Font(bold=True); title_font = Font(bold=True, size=12); fill = PatternFill('solid', fgColor='EFEFEF'); star = PatternFill('solid', fgColor='FFF2CC')
ws['D2'] = None  # will set below
base = None
for r in range(1, ws.max_row + 1):
    a = ws.cell(r, 1).value
    if a is None: base = None; continue
    if ws.cell(r, 2).value is None:   # block title
        ws.cell(r, 1).font = title_font
        key = 'MDR' if a.startswith('MDR') else 'MDS'
        ws.cell(r, 6, HEAD_NOTE[key]); ws.cell(r, 6).alignment = Alignment(wrap_text=True, vertical='top'); base = None; continue
    if a == '方法':
        for c, v in enumerate(['方法', 'AUC', 'AUPR', 'Δ AUC vs ctx+diff×LR', 'Δ AUPR vs ctx+diff×LR', '说明'], 1):
            cell = ws.cell(r, c, v); cell.font = hfont; cell.fill = fill; cell.border = border; cell.alignment = Alignment(wrap_text=True, vertical='center')
        continue
    auc, aupr = ws.cell(r, 2).value, ws.cell(r, 3).value
    if a.startswith('+ 图上下文 + 扩散谱'): base = (auc, aupr)
    if base is not None:
        ws.cell(r, 4, round(auc - base[0], 4)); ws.cell(r, 5, round(aupr - base[1], 4))
        for c in (4, 5): ws.cell(r, c).number_format = '+0.0000;-0.0000;0.0000'
    note = NOTES.get(a)
    if note is None: raise SystemExit(f'no note for row {r}: {a!r}')
    ws.cell(r, 6, note)
    for c in range(1, 7):
        ws.cell(r, c).border = border; ws.cell(r, c).alignment = Alignment(wrap_text=True, vertical='top')
        if a.startswith('★'): ws.cell(r, c).fill = star; ws.cell(r, c).font = hfont
    for c in (2, 3): ws.cell(r, c).number_format = '0.0000'
for col, w in zip('ABCDEF', [52, 9, 9, 12, 12, 110]): ws.column_dimensions[col].width = w
ws.freeze_panes = 'B3'

# ---------- 说明 sheet ----------
if '说明' in wb.sheetnames: del wb['说明']
S = wb.create_sheet('说明')
rows = []
def sec(t): rows.append(('SEC', t))
def kv(k, v): rows.append((k, v))
sec('1. 表格口径')
kv('基准', 'PLMF-MDA（PLoS Comput Biol 2026）发布的 miRNA–药物关联数据：MDR（耐药）与 MDS（敏感）两个二值矩阵，实体为 miRNA 与 DrugBank 药物。')
kv('划分协议', '论文 5 折：正例随机分 5 折；每折从未标注对中按 1:1 随机采样负例；seed 42 同时决定折划分和负采样。训练/测试对集合在本表所有行之间完全一致（GNN 两行来自同一 seed 的独立跑，折划分相同）。')
kv('指标', 'AUC = ROC 曲线下面积；AUPR = 精确率–召回率曲线下面积（average precision）。均为 5 个测试折上分别计算后取均值，四位小数。')
kv('单种子 vs 多种子', '本文件为 seed 42 单种子（与 base 及早期 GNN 行同口径，便于逐行配对）。种子间标准差约 0.002~0.004，单种子 <0.001 的差异不应过度解读；4 种子（42,0,1,2）配对均值与逐种子正负计数见 ablation_PLMF-MDA.xlsx 与 REPORT_XREL.md §0、§4。')
kv('Δ 列', 'D/E 列 = 该行减去"+ 图上下文 + 扩散谱特征 × 线性读头（ctx+diff × LR）"一行，只对该行之后的行计算。GNN 两行与两条外部基线不参与差值。')
kv('读头与超参', '自 ctx+diff×LR 行起所有行使用同一个 L2 逻辑回归（C=0.3，max_iter=5000），内折 K=10 交叉拟合；没有按行或按数据集调参。')
sec('2. 特征组（列数 / 来源 / 含义）')
kv('ctx（33 列）', '图上下文：miRNA/药物在训练支持图中的度、共邻数、3/5 步路径数、SVD-16 低秩打分、属性支撑（k-mer / ChemBERTa 相似邻居的关联数）、相似度支撑。复用 dlstmda_strongest_20260920/features.py。')
kv('diff（52 列）', '扩散谱：支持图上的个性化 PageRank、Katz（两组 α、β）、7 步路径数，各自再做逐 miRNA 行 z 分数、行百分位、逐药物列 z 分数。来自 plmf_innov_20260921。')
kv('prov（24 列）', '溯源超图启发式：pmid 6（研究超边 PMID→其报道的对；m、d 经其它对被同一 PMID 报道的桥接数、按研究规模加权、Jaccard、实体研究数），costudy 6（研究共现图 S_r=M_rM_rᵀ、S_d=M_dM_dᵀ 上的共研究关联数及归一化），ctx 4（细胞系/表型/疾病语境 token 的共现），rec 8（实体记录数、低通量比例、首末报道年份）。"iNcRD-HG 记录"版用仓库自带表（覆盖 58~62%），"完整记录"版用 ncRNADrug DR_Curated（覆盖 95.4/96.1%，data/full_{ds}/data/pair_records.pkl）。')
kv('mprior（19 列）', 'miRNA 旁信息：fam 5（名称家族：去 -3p/-5p 与字母后缀）、seed 5（seed 7-mer 家族）、hmdd 7（HMDD v4 miRNA–疾病谱，茎名匹配覆盖 83~85%）、xdeg 2（另一表型矩阵中的度）。均为标签无关或来自另一基准的旁信息，目标是给训练图中度数 0~3 的稀疏 miRNA 补信息。')
kv('cluster（5 列）', 'miRBase v23.1 hsa.gff3 基因组簇：前体同链 10 kb 内成簇；簇成员对目标药物的支持统计。97% miRNA 可映射，51~53% 处于多成员簇。')
kv('geo（7 列）', 'ncRNADrug DR_GEO / DT_GEO 差异表达证据：同向/反向 GSE 数、最大 −log10 p、给药差异 GSE 数、显著数、覆盖标志。研究无关过滤：GSE→PMID（eutils，gse_pmid.json），剔除被任何策展记录引用过的 GSE（73 个中 32 个）。不过滤版（geoall）增益并不更大。')
kv('无标签池 ×10', '不是特征而是训练集扩充：从数据集发布的未标注对池（MDR 132,874 / MDS 170,226）中排除协议内全部训练与测试对后再抽 10×n_pos 作为额外负例；样本权重按类别先验（负类 = n_pos/n_neg）。测试集不变。')
sec('3. 泄漏防护与对照')
kv('交叉拟合', '所有依赖训练标签的列（ctx/diff/prov/mprior/cluster/geo 中用到支持图或记录的部分）在训练折内做 K=10 内折交叉拟合：某训练对的特征只由其余 9/10 训练正例计算；测试对由全部训练正例计算。测试对永不进入任何支持图。')
kv('溯源同源性', 'PLMF-MDA 正例即 ncRNADrug 策展对（4849 个 resistant 对里 4848 个是 PLMF 正例）。因此溯源特征只用训练正例的记录，query 对不在自己的 support 里；同时给出"记录置换"对照（回到 base）和研究分组划分结果（REPORT_XREL §5：完整溯源仍 +0.006~0.007 AUC 且置换归零）。')
kv('置换对照', '两条"└ 对照"行分别把溯源记录在正例对之间随机置换（保留列数、分布、实体流行度）。置换后回到未加该特征的水平，是增益来自结构信息的直接证据。cluster/geo 的置换对照 → 0（见 REPORT_XREL §3、§4.4）。')
kv('隐藏正例', '采样负例中约 2%（MDR 2.1%，MDS 1.7%）在另一表型矩阵或 ncRNADrug 中已记录为正，是基准的标签噪声天花板。去噪评估（剔除这些测试负例）：最终模型 MDR 0.9694/0.9698，MDS 0.9690/0.9707；base 0.9573/0.9595。')
kv('MPHGNN 口径', 'MPHGNN 官方全局分支在评分时读取测试批次的度信息（+0.01~0.02 AUC）。表中为逐对隔离评分口径；元路径通道置零或打乱不改变结果。')
sec('4. 数据来源')
kv('ncRNADrug', 'DR_Curated / DR_GEO / DT_GEO / DT_Curated，2026-09-22 下载，data_external/ncrnadrug_20260922/。药物名 → DrugBank id 经 PubChem 同义词（data/drug_synonyms.json）。')
kv('miRBase', 'v23.1 hsa.gff3，data_external/mirbase_20260922/。')
kv('HMDD', 'v4 miRNA–疾病关联（上一轮 plmf_warm_20260922 已缓存）。')
kv('外部基线', 'PLMF-MDA 官方代码：../plmf_baselines_20260920；MPHGNN 移植：../mphgnn_plmf_20260921；GNN 两行：../plmf_innov_20260921。')
sec('5. 复现')
kv('环境', 'PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python；OMP_NUM_THREADS=4；特征缓存来自 ../plmf_warm_20260922/cache。')
kv('单行命令', 'final2.py --dataset MDR --split 5cv --seed 42 --variant {base|prov|provf|provfs|mpr|phlf|phlfs|phlf_cluster|phlf_geo}；probe_extraneg.py --dataset MDR --seed 42 --mult 10 --prov full [--xrel cluster,geo]。')
kv('全流程', './run_final.sh（外部资源抓取、溯源补全、外部关系缓存、4 种子 × 2 数据集 × 全部变体、80/20 协议、研究分组划分、诊断）。')
kv('产物', 'results/summary.txt、results/extraneg.txt、results/paired_summary_vs_{base,phlf}.csv、results/scores_*.csv（每对得分）、logs/*.log（每行的原始输出，含每折标准差）。')
sec('6. 解读要点')
kv('增益落点', '增益几乎全部来自训练图中度数 0~3 的稀疏 miRNA（seed 42 正例平均损失：度 1 从 0.094 降到 0.073，度 2~3 从 0.056 降到 0.041；密集 miRNA 持平）。药物侧 95% 正例落在度 16+ 的药物上，不是瓶颈。')
kv('五类外部关系无效', 'CTD 药物–疾病 × HMDD 桥、miRTarBase 靶基因 × 药物基因集（LINCS/GEO/DrugMatrix/CMap/ChEMBL）、lncRNA ceRNA 桥、GDSC2 IC50 谱药物 kNN、"药物改变 miRNA 表达"策展记录在 warm 面全部 ≤ +0.0008，置换对照同水平；单变量诊断显示它们只含药物流行度，不含"这个稀疏 miRNA 与这个药物"的配对信息。故未进入本表。')
kv('边界', '相对 ctx+diff×LR 的提升为 +0.010~0.013 AUC / +0.012~0.016 AUPR，相对 PLMF-MDA 官方复现 +0.04 / +0.05；随机划分重度"同研究内"，研究分组划分下 base 掉 0.05~0.08，溯源仍正、置换归零。论文应同时报告随机划分、研究分组划分与去噪评估。')
kv('叙述顺序', '① 溯源补全 + 研究/语境超边（信息侧主贡献）；② miRNA 旁信息关系；③ 无标签池训练（协议侧修正）；基因组簇与研究无关 GEO 证据作为"标签无关证据超边"补充。')
r = 1
S.cell(r, 1, 'ablation_PLMF-MDA_seed42.xlsx 说明').font = Font(bold=True, size=13); r += 2
for k, v in rows:
    if k == 'SEC':
        S.cell(r, 1, v).font = title_font; r += 1; continue
    S.cell(r, 1, k).font = hfont; S.cell(r, 2, v)
    for c in (1, 2): S.cell(r, c).alignment = Alignment(wrap_text=True, vertical='top'); S.cell(r, c).border = border
    r += 1
S.column_dimensions['A'].width = 22; S.column_dimensions['B'].width = 130
wb.save(F); print('saved', F)
