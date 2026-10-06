"""Paper-style baseline comparison workbook.

Turns the flat "高于我的模型" summary into a table shaped like a paper's baseline
comparison: one sheet per dataset (methods as rows, AUC/AUPR columns, ours
highlighted, winners in red) plus AUC/AUPR matrices across datasets.

Output: improve/results/external_specialized_20260916/外部专用模型对比_论文格式.xlsx
"""
import os

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = ("/data1/licheng/incrd-hg/iNcRD-HG/improve/results/"
       "external_specialized_20260916/外部专用模型对比_论文格式.xlsx")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# (sheet, dataset, task, our_variant, our_auc, our_aupr,
#  [(model, type, auc, aupr, note), ...], reliability)
DATA = [
    ("MPHGNN", "MPHGNN 独立测试集", "miRNA-drug 抗性",
     "iNcRD-HG (GNN 配对嵌入 + frozen TabICLv2)", 0.9605, 0.9024, [
        ("MPHGNN", "miRNA-only", 0.9780, 0.9370, "reported (BIB 2026); 本地复现 0.9743/0.9382"),
        ("SubKNet", "miRNA-only", 0.9717, 0.9341, "reported (MPHGNN Table 1)"),
        ("GAM-MDR", "miRNA-only", 0.9557, 0.9150, "reported (MPHGNN Table 1)"),
        ("DeepLDA", "lncRNA-only, re-tested", 0.9311, 0.8804, "reported (MPHGNN Table 1)"),
        ("AGCLNDA", "pan-ncRNA", 0.9170, 0.8670, "reported (MPHGNN Table 1)"),
        ("NDSGCL", "pan-ncRNA", 0.8070, 0.6732, "reported (MPHGNN Table 1)"),
     ], "ok"),
    ("ncDR", "ncDR (GAM-MDR 701x101)", "miRNA-drug 抗性",
     "iNcRD-HG (图特征 + frozen TabICLv2)", 0.8322, 0.8135, [
        ("GAM-MDR (本地复现)", "miRNA-only", 0.9179, 0.8977, "本地同协议复现(5 seeds, 官方代码移植); AUC 0.9179±0.0105 AUPR 0.8977±0.0124"),
        ("SGCLDGA", "BIB 图对比(迁移到 ncDR)", 0.8878, 0.9089, "本地同协议复现(5 seeds); AUC 0.8878±0.0075 AUPR 0.9089±0.0070"),
        ("JMSSMMA", "BIB 图掩码自编码(迁移到 ncDR)", 0.6145, 0.6305, "本地同协议复现(5 seeds); AUC 0.6145±0.0164 AUPR 0.6305±0.0145"),
        ("SMCL-MDR", "miRNA-only", 0.9896, 0.9881, "reported (BIBM 2024); own ncDR protocol"),
        ("SVMMDR", "miRNA-only", 0.9780, None, "reported (Front Oncol 2022); ncDR 626x85"),
        ("BNEMDI", "miRNA-only", 0.9568, 0.9565, "reported (Front Genet 2022)"),
        ("NASMDR", "miRNA-only", 0.9468, None, "reported (BIB 2022); AUPR 未公开"),
        ("AMMGC", "miRNA-only", 0.9467, 0.2399, "reported (Front Pharmacol 2022); 1:1 CV"),
        ("AMMGC (本地复现)", "miRNA-only", 0.9024, 0.9064, "本地同协议复现(PyTorch 忠实重实现, 适配 ncDR 视图); AUC 0.9024±0.0252 AUPR 0.9064±0.0171"),
        ("GCMDR", "miRNA-only", 0.9369, None, "reported (Bioinformatics 2019); AUPR 未公开"),
        ("GCMDR (本地复现)", "miRNA-only", 0.7875, 0.6887, "本地同协议复现(PyG重实现); AUC 0.7875±0.0155 AUPR 0.6887±0.0197"),
        ("DLP (本地复现)", "drug-miRNA (Python移植)", 0.7622, 0.8219, "MATLAB→Python 移植后同协议复现; AUC 0.7622±0.0073 AUPR 0.8219±0.0083"),
        ("DeepLDA (本地复现)", "lncRNA-only(迁移)", 0.6364, 0.7207, "本地同协议复现(本地移植); AUC 0.6364±0.0656 AUPR 0.7207±0.0482"),
        ("NDSGCL (本地复现)", "pan-ncRNA(迁移)", 0.9004, 0.9070, "本地同协议复现(重实现); AUC 0.9004±0.0058 AUPR 0.9070±0.0039"),
        ("AGCLNDA (本地复现)", "pan-ncRNA(迁移)", 0.6892, 0.7485, "官方代码同协议复现; AUC 0.6892±0.0156 AUPR 0.7485±0.0111"),
     ], "ok"),
    ("41lncRNA", "NoncoRNA+ncDR 的 41 lncRNA 子集", "lncRNA-drug 抗性",
     "iNcRD-HG (图特征 + frozen TabICLv2)", 0.7234, 0.7347, [
        ("KBMF2K", "baseline", 0.8410, 0.5210, "reported (DMGAT Table 2)"),
        ("DMGAT (lncRNA)", "pan-ncRNA", 0.8397, 0.8431, "reported (BIB 2025)"),
        ("CMF", "baseline", 0.8340, 0.5380, "reported (DMGAT Table 2)"),
        ("NetLapRLS", "baseline", 0.8110, 0.5470, "reported (DMGAT Table 2)"),
        ("BLM-NI", "baseline", 0.7840, 0.5600, "reported (DMGAT Table 2)"),
     ], "同数据同折数; lncRNA 子集仅 41 个, 指标波动大"),
    ("DeepLDA-NoncoRNA", "DeepLDA NoncoRNA", "lncRNA-drug 抗性",
     "iNcRD-HG (图特征 + frozen TabICLv2)", 0.9997, 0.9856, [
        ("DeepLDA", "lncRNA-only", 0.9583, 0.8601, "reported (Front Microbiol 2023)"),
     ], "低: 全矩阵相似度特征含标签泄漏; 该协议下图特征+TabICLv2≈1.0, 跨模型数值不可靠"),
    ("DeepLDA-ncDR", "DeepLDA ncDR", "lncRNA-drug 抗性",
     "iNcRD-HG (图特征 + frozen TabICLv2)", 0.9948, 0.9303, [
        ("DeepLDA", "lncRNA-only", 0.9819, 0.8687, "reported (Front Microbiol 2023); 仅 184 正例"),
     ], "低: 同协议含泄漏; 仅 184 正例; 图特征+TabICLv2≈0.99, 不可靠"),
    ("GCFMCL", "GCFMCL D2R", "miRNA-drug 敏感性",
     "iNcRD-HG (图特征 + frozen TabICLv2)", 0.7482, 0.7492, [
        ("GCFMCL", "miRNA-only", 0.9528, 0.9566, "reported (BIB 2023); balanced 负例"),
     ], "graph-features + frozen TabICLv2 (balanced 5 折)"),
    ("NoncoRNA2693", "NoncoRNA+ncDR 2693", "ncRNA-drug 抗性(混合)",
     "iNcRD-HG (图特征 + frozen TabICLv2)", 0.7617, 0.7782, [
        ("RDRGSE", "pan-ncRNA", 0.9768, 0.9736, "reported (ACS Omega 2023)"),
        ("B-NDRA", "pan-ncRNA", 0.9207, 0.9173, "reported (JCIM 2024)"),
        ("LRGCPND", "pan-ncRNA", 0.8987, 0.9094, "reported (IJMS 2021)"),
        ("GSLRDA", "pan-ncRNA", 0.9101, None, "reported (JCIM 2022); AUPR 未公开"),
        ("DMGAT", "pan-ncRNA", 0.8964, 0.8984, "reported (BIB 2025)"),
     ], "ok"),
    ("RNAactDrug", "RNAactDrug (HGNNLDA 数据)", "lncRNA-drug 敏感性",
     "iNcRD-HG (图特征 + frozen TabICLv2)", 0.9016, 0.9144, [
        ("HGNNLDA", "lncRNA-only", None, None, "指标付费墙, 未取到"),
     ], "ok"),
    ("SMD", "自有 S_MD 测试集", "miRNA-drug 抗性",
     "iNcRD-HG (anchor-shrinkage 集成)", 0.9638, 0.9294, [], "自有基准"),
    ("SLD", "自有 S_LD 测试集", "lncRNA-drug 抗性",
     "iNcRD-HG (anchor-shrinkage 集成)", 0.9948, 0.9896, [], "自有基准"),
]

# ---------------------------------------------------------------------------
# 新增候选基线：检索到但尚未复现的论文（重点 BIB / B刊(JBHI/TCBB/TBD/BIBM)）
# 字段: 方法, 任务, 数据集, 期刊/会议, 分区, 年份, 报告指标, DOI, 代码/数据, 备注
# ---------------------------------------------------------------------------
CANDIDATES = [
    ("MSMP-GAE", "miRNA-drug 关联", "miRNA-drug 异质网络", "IEEE JBHI", "CCF-B / 中科院1区", 2026,
     "AUC 0.9861 / AUPR 0.9823", "10.1109/JBHI.2025.3558303", "待确认",
     "多源信息+元路径增强图自编码器；指标高，最值得优先复现对比"),
    ("DLP", "drug-miRNA 关联", "drug-miRNA 基准", "IEEE/ACM TCBB", "CCF-B / 中科院2区", 2026,
     "未取到", "10.1109/TCBB.2024.3402215", "github.com/kaizheng-academic/DLP",
     "子空间分割+对偶链路预测；MATLAB 代码公开，B刊可复现"),
    ("Dual-Stream Heterogeneous GNN (Zero-Shot)", "miRNA-drug 敏感性", "miRNA-drug", "IEEE BIBM", "CCF-B 会议", 2024,
     "未取到", "待查", "待确认", "零样本嵌入的双流异质图；同为 miRNA-drug 敏感性"),
    ("DNMDA", "miRNA-drug 关联", "miRNA-drug", "IEEE BIBM", "CCF-B 会议", 2024,
     "未取到", "待查", "待确认", "深度非负矩阵分解多级融合"),
    ("LGCMDS", "miRNA-drug 敏感性", "miRNA-drug", "IEEE BIBM", "CCF-B 会议", 2021,
     "未取到", "10.1109/BIBM52615.2021.9669566", "待确认", "轻量图卷积；常被 ncDR 系方法引用作基线"),
    ("LAGCN-MDR (Layer Attention GCN)", "drug-miRNA 抗性", "drug-miRNA 抗性", "IEEE BIBM", "CCF-B 会议", 2021,
     "未取到", "待查", "待确认", "层注意力图卷积+多通道特征，直击抗性"),
    ("Integrating Molecular LLM + Multi-View", "miRNA-drug 关联", "miRNA-drug", "IEEE BIBM", "CCF-B 会议", 2025,
     "未取到", "待查", "待确认", "分子大模型+多视图表示，新方法候选"),
    ("MHTMG", "miRNA-drug 关联", "miRNA-drug", "IEEE HPCC", "CCF-C 会议", 2025,
     "未取到", "10.1109/HPCC67675.2025.00029", "待确认", "多头注意力+参数化变换矩阵+GraphSAGE"),
    ("Dual-Channel Signed Bipartite GNN", "miRNA-drug 交互", "miRNA-drug", "IEEE Trans. Big Data", "CCF-B", 2025,
     "未取到", "待查", "待确认", "结构特征+符号二部图双通道"),
    ("IRPCA", "miRNA-drug 关联", "miRNA-drug", "J. Chem. Inf. Model.", "中科院1区", 2025,
     "未取到", "10.1021/acs.jcim.4c02385", "待确认", "可解释鲁棒 PCA；与 LRGCPND/GSLRDA 同实验室系列"),
    ("ALRTGCL", "miRNA-drug 关联", "miRNA-drug", "IEEE Trans. Consumer Electronics", "中科院2区", 2025,
     "未取到", "10.1109/TCE.2025.3593065", "待确认", "自适应低秩 Transformer+图对比学习"),
    ("VGAE + CMF", "miRNA-drug 敏感性", "miRNA-drug", "Eng. Appl. Artif. Intell.", "中科院1区", 2026,
     "未取到", "10.1016/j.engappai.2026.113901", "待确认", "变分图自编码器+协同矩阵分解"),
    ("Metapath-aware Graph Fusion Encoder", "miRNA-drug 关联", "miRNA-drug", "ACM (会议)", "—", 2025,
     "未取到", "10.1145/3766671.3766753", "待确认", "元路径感知注意力图融合编码器"),
    ("BCheM", "疾病-药物抗性", "disease-drug 抗性", "Front. Bioinformatics", "—", 2026,
     "AUC 0.8890 / AUPR 0.8295", "10.3389/fbinf.2026.1838904", "待确认",
     "Chebyshev 图卷积；基线含 NASMDR/GCFMCL，可直接对照"),
    ("SGCLDGA", "drug-gene 关联", "drug-gene 二部图", "Briefings in Bioinformatics", "CCF-B / 中科院1区", 2024,
     "AUC 0.8860 / AUPR 0.9072", "10.1093/bib/bbae231", "官方代码(见论文)",
     "BIB；简单图对比学习，基线含 LRGCPND(ncRNA 抗性)，方法可迁移"),
    ("LR-GNN", "分子关联(通用)", "lncRNA-disease/miRNA-disease/PPI/DDI", "Briefings in Bioinformatics", "CCF-B / 中科院1区", 2022,
     "未取到", "10.1093/bib/bbab513", "待确认", "链路表示 GNN；BIB 通用图关联方法，可作结构基线"),
    ("SAGCN", "circRNA-drug 敏感性", "circRNA-drug", "IEEE/ACM TCBB", "CCF-B / 中科院2区", 2024,
     "未取到", "待查", "待确认", "子图感知图卷积；circRNA 版，可作 ncRNA 扩展对比"),
    ("Entropy-Based Drug-Resistant miRNA Ranking", "drug-resistant miRNA 识别", "drug-miRNA 抗性", "IEEE/ACM TCBB", "CCF-B / 中科院2区", "—",
     "未取到", "待查", "待确认", "早期抗性 miRNA 排序方法，TCBB"),
]
CANDS_COLS = ["方法", "任务", "数据集", "期刊/会议", "分区", "年份",
              "报告指标", "DOI", "代码/数据", "备注"]

# ---------------------------------------------------------------------------
# 同协议复现台账 (baseline_repro_20260918)
# 字段: 方法, 期刊, 任务, 代码仓库, 语言/环境, 复现状态, ncDR同协议结果, 备注
# ---------------------------------------------------------------------------
LEDGER = [
    ("SGCLDGA", "BIB 2024", "drug-gene", "github.com/one-melon/SGCLDGA", "torch1.13 (ncRNAdrug)",
     "已完成", "AUC 0.8878±0.0075 / AUPR 0.9089±0.0070",
     "官方代码两处笔误(xavier_gniform_/__dnit__)已修；迁移到 ncDR 701x101 同协议"),
    ("AMMGC", "Front Pharmacol 2022", "miRNA-drug 抗性", "github.com/scz760904126/AMMGC", "TF2 (gapTrick)",
     "代码已获取", "-", "官方 106x754 自有数据；固定维度需改为 101x701；缺 sklearn 需装"),
    ("JMSSMMA", "BIB 2022", "small molecule-miRNA", "github.com/ZZCrazy00/JMSSMMA",
     "torch2.1+pyg2.3 (molcl-sp)", "已完成", "AUC 0.6145±0.0164 / AUPR 0.6305±0.0145",
     "官方代码原样运行；ncDR 无 SM/miRNA 富相似度特征，仅用训练集 GIP，故偏低(公平)"),
    ("DLP", "TCBB 2026", "drug-miRNA", "github.com/kaizheng-academic/DLP", "MATLAB",
     "阻塞", "-", "无 MATLAB/Octave，需移植到 Python 才能跑"),
    ("MGCNA", "PLoS Comput Biol 2025", "miRNA-drug", "github.com/sheng-n/MGCNA", "torch2.1+pyg",
     "克隆受阻", "-", "仓库较大，clone 反复超时；需重试"),
    ("SAGCN", "TCBB 2024", "circRNA-drug 敏感性", "github.com/lizhiqi49/SAGCN", "torch+pyg",
     "克隆受阻", "-", "仓库较大，clone 反复超时；需重试"),
    ("GCNLASMMA", "Front Genet/PMC", "small molecule-miRNA", "github.com/1054366388/GCNLASMMA",
     "python", "代码已获取", "-", "与 LAGCN-MDR 同族(层注意力 GCN)，可作迁移基线"),
    ("GCLMTP", "Brief Bioinform 2023", "lncRNA/miRNA-disease", "github.com/sheng-n/GCLMTP",
     "torch1.13+xgb", "代码已获取", "-", "非 drug 任务，仅作结构参考"),
    ("MSMP-GAE", "IEEE JBHI 2026", "miRNA-drug", "未找到公开代码", "-", "待人工实现", "-",
     "AUC 0.9861/AUPR 0.9823 (论文值)，实现成本高"),
    ("Dual-Stream Heterogeneous GNN (Zero-Shot)", "BIBM 2024", "miRNA-drug 敏感性",
     "未找到公开代码", "-", "待人工实现", "-", "会议论文，无代码"),
    ("DNMDA", "BIBM 2024", "miRNA-drug", "未找到公开代码", "-", "待人工实现", "-", "会议论文，无代码"),
    ("LGCMDS", "BIBM 2021", "miRNA-drug 敏感性", "未找到公开代码", "-", "待人工实现", "-", "会议论文，无代码"),
    ("LAGCN-MDR", "BIBM 2021", "drug-miRNA 抗性", "未找到公开代码", "-", "待人工实现", "-",
     "可参考同族 GCNLASMMA 代码"),
    ("Integrating Molecular LLM + Multi-View", "BIBM 2025", "miRNA-drug", "未找到公开代码", "-",
     "待人工实现", "-", "会议论文，无代码"),
    ("MHTMG", "HPCC 2025", "miRNA-drug", "未找到公开代码", "-", "待人工实现", "-", "会议论文，无代码"),
    ("Dual-Channel Signed Bipartite GNN", "IEEE TBD 2025", "miRNA-drug", "未找到公开代码", "-",
     "待人工实现", "-", "期刊论文，未找到代码"),
    ("IRPCA", "JCIM 2025", "miRNA-drug", "未找到公开代码", "-", "待人工实现", "-", "未找到代码"),
    ("ALRTGCL", "IEEE TCE 2025", "miRNA-drug", "未找到公开代码", "-", "待人工实现", "-", "未找到代码"),
    ("VGAE + CMF", "EAAI 2026", "miRNA-drug 敏感性", "未找到公开代码", "-", "待人工实现", "-", "未找到代码"),
    ("Metapath-aware Graph Fusion Encoder", "ACM 2025", "miRNA-drug", "未找到公开代码", "-",
     "待人工实现", "-", "未找到代码"),
    ("BCheM", "Front Bioinform 2026", "disease-drug 抗性", "未找到公开代码", "-", "待人工实现", "-",
     "AUC 0.8890/AUPR 0.8295 (论文值)"),
    ("LR-GNN", "BIB 2022", "分子关联(通用)", "需检索", "-", "待检索", "-", "BIB 通用图关联"),
    ("Entropy-Based Drug-Resistant miRNA Ranking", "TCBB", "drug-miRNA 抗性", "未找到公开代码", "-",
     "待人工实现", "-", "早期方法"),
]
LEDGER_COLS = ["方法", "期刊", "任务", "代码仓库", "语言/环境", "复现状态", "ncDR同协议结果", "备注"]

# ---------------------------------------------------------------------------
# 各基线论文自身报告的其他指标（只填对应论文确实报告的指标；缺省留空）
# key = (dataset_sheet, 方法) ; value = 字符串，如 "ACC 0.912 / SN 0.905 / SP 0.918 / F1 0.905 / MCC 0.82"
# ---------------------------------------------------------------------------
EXTRA = {
    # MPHGNN 论文 Table 1 (同数据同协议): ACC / Precision / F1 / MCC / BS
    ("MPHGNN", "MPHGNN"): "ACC 0.9394 / Precision 0.8780 / F1 0.9131 / MCC 0.8633 / BS 0.0511",
    ("MPHGNN", "SubKNet"): "ACC 0.8964 / Precision 0.9007 / F1 0.8329 / MCC 0.7630 / BS 0.0905",
    ("MPHGNN", "GAM-MDR"): "ACC 0.8780 / Precision 0.7612 / F1 0.8350 / MCC 0.7489 / BS 0.2339",
    ("MPHGNN", "DeepLDA"): "ACC 0.8813 / Precision 0.8640 / F1 0.8681 / MCC 0.7368 / BS 0.1083",
    ("MPHGNN", "AGCLNDA"): "ACC 0.8398 / Precision 0.6944 / F1 0.7943 / MCC 0.6863 / BS 0.1471",
    ("MPHGNN", "NDSGCL"): "ACC 0.7059 / Precision 0.5373 / F1 0.6580 / MCC 0.4563 / BS 0.2556",
    # ncDR: SVMMDR 论文 Table 2 (ncDR 626x85, 10 折)
    ("ncDR", "SVMMDR"): "ACC 0.9393 / Precision 0.8705 / Recall 0.8905 / F1 0.8800 / MCC 0.8399",
    # ncDR: AMMGC 论文 Table 1 (ncDR 106x754)
    ("ncDR", "AMMGC"): "ACC 0.9867 / Recall 0.3587 / F1 0.3184",
}

# ---------------------------------------------------------------------------
# 我们自己跑出来的其他指标(阈值=best-F1, 同协议)；与论文报告值分开
# key = (dataset_sheet, 方法)
# ---------------------------------------------------------------------------
SELF_RUN = {
    ("MPHGNN", "MPHGNN"):
        "官方代码自跑(5折,seed42): AUC 0.9755 / AUPR 0.9415 / ACC 0.9246 / Precision 0.8883 / Recall 0.8850 / Specificity 0.9443 / F1 0.8866 / MCC 0.8301",
    ("MPHGNN", "iNcRD-HG (GNN 配对嵌入 + frozen TabICLv2)"): "未计算(仅 AUC/AUPR)",
    ("MPHGNN", "SubKNet"):
        "官方代码自跑(5折,seed42): AUC 0.9493 / AUPR 0.9070 (阈值指标因输出接口不稳定, 仅报 AUC/AUPR)",
    ("ncDR", "iNcRD-HG (图特征 + frozen TabICLv2)"):
        "ACC 0.7718 / Precision 0.7404 / Recall 0.8427 / Specificity 0.7009 / F1 0.7869 / MCC 0.5515",
    ("ncDR", "SGCLDGA"):
        "ACC 0.8378 / Precision 0.8366 / Recall 0.8408 / Specificity 0.8348 / F1 0.8384 / MCC 0.6761",
    ("ncDR", "JMSSMMA"): "阈值退化(MCC=0), 以 AUC/AUPR 为准",
    ("ncDR", "GAM-MDR (本地复现)"):
        "ACC 0.8540 / Precision 0.8299 / Recall 0.8908 / Specificity 0.8171 / F1 0.8590 / MCC 0.7104",
    ("ncDR", "GCMDR (本地复现)"):
        "ACC 0.7763 / Precision 0.7649 / Recall 0.8079 / Specificity 0.7446 / F1 0.7838 / MCC 0.5567",
    ("ncDR", "AMMGC (本地复现)"):
        "ACC 0.8525 / Precision 0.8374 / Recall 0.8782 / Specificity 0.8269 / F1 0.8564 / MCC 0.7077",
    ("ncDR", "DLP (本地复现)"):
        "ACC 0.7419 / F1 0.7414 / MCC 0.4858",
    ("ncDR", "DeepLDA (本地复现)"):
        "ACC 0.6422 / Precision 0.6615 / Recall 0.8066 / Specificity 0.4778 / F1 0.6982 / MCC 0.2947",
    ("ncDR", "NDSGCL (本地复现)"):
        "ACC 0.8282 / Precision 0.8254 / Recall 0.8335 / Specificity 0.8228 / F1 0.8291 / MCC 0.6569",
    ("ncDR", "AGCLNDA (本地复现)"): "阈值退化(MCC≈0.07), 以 AUC/AUPR 为准",
    ("GCFMCL", "iNcRD-HG (图特征 + frozen TabICLv2)"):
        "ACC 0.6759 / Precision 0.6341 / Recall 0.8472 / Specificity 0.5046 / F1 0.7236 / MCC 0.3767",
    ("41lncRNA", "iNcRD-HG (图特征 + frozen TabICLv2)"):
        "ACC 0.6677 / Precision 0.6114 / Recall 0.9308 / Specificity 0.4047 / F1 0.7376 / MCC 0.3928",
    ("NoncoRNA2693", "iNcRD-HG (图特征 + frozen TabICLv2)"):
        "ACC 0.6951 / Precision 0.6610 / Recall 0.8058 / Specificity 0.5845 / F1 0.7255 / MCC 0.4012",
    ("RNAactDrug", "iNcRD-HG (图特征 + frozen TabICLv2)"):
        "ACC 0.8323 / Precision 0.8332 / Recall 0.8310 / Specificity 0.8336 / F1 0.8321 / MCC 0.6647",
}

# ---------------------------------------------------------------- styling
DARK = PatternFill("solid", fgColor="305496")
OURS = PatternFill("solid", fgColor="FFF2CC")
HEAD = Font(bold=True, color="FFFFFF", size=11)
TITLE = Font(bold=True, size=12, color="1F3864")
RED = Font(bold=True, color="C00000")
GREEN = Font(bold=True, color="107C10")
BOLD = Font(bold=True)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
GRAY = PatternFill("solid", fgColor="D9E1F2")


def style_header(ws, row, ncol):
    for c in range(1, ncol + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = DARK
        cell.font = HEAD
        cell.alignment = CENTER
        cell.border = BORDER


def set_widths(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def is_winner(base, ours):
    return base is not None and ours is not None and base > ours


# ---------------------------------------------------------------- build
wb = openpyxl.Workbook()

# ---- 0. 说明
ws = wb.active
ws.title = "说明"
notes = [
    ("表结构", "每个数据集一个 sheet，方法为行、AUC/AUPR 为列，与论文基线表一致；★/黄底＝本文方法，红色＝高于本文的数值。"),
    ("Δ定义", "ΔAUC = 基线 AUC − 我的 AUC；Δ 为正(红)表示基线更高，为负(绿)表示我更高。"),
    ("汇总", "所有数据集一览，含高于我的模型数量。"),
    ("总表-AUC / 总表-AUPR", "方法 × 数据集 的宽表，便于横向比较。"),
    ("same/cross", "部分对比为跨论文报告值(数据集子集/负采样/划分不同)，AUPR 尤其不可跨行直接比较。"),
    ("数据来源", "improve/research/external_specialized_20260916/REPORT.md 与 build_comparison_excel.py。"),
    ("新增候选基线", "sheet《新增候选基线(BIB-B刊)》为新检索到的 miRNA/lncRNA-drug 论文(BIB / JBHI / TCBB / TBD / BIBM 等), 尚未复现, 指标多为论文报告值。"),
    ("同协议复现台账", "sheet《同协议复现台账》记录各候选的代码可获取性与同协议复现进度；已完成者(如 SGCLDGA)已并入对应 dataset sheet。"),
    ("本文方法口径", "所有外部数据集的“本文方法”统一采用 iNcRD-HG 可迁移核 + 冻结 TabICLv2(不再是 GNN decoder/HistGB)；DeepLDA 两个数据集因协议(1.5% 正例/标签泄漏)TabICLv2 失效，保留 GNN 迁移值并注明。"),
    ("重要警告", "DeepLDA 数据集的评估协议存在标签泄漏(全矩阵相似度), 其跨模型数值不可靠, 仅供记录。"),
]
ws.append(["项", "说明"])
for a, b in notes:
    ws.append([a, b])
style_header(ws, 1, 2)
for r in range(2, ws.max_row + 1):
    ws.cell(row=r, column=1).font = BOLD
    ws.cell(row=r, column=1).alignment = LEFT
    ws.cell(row=r, column=2).alignment = LEFT
set_widths(ws, [18, 110])

# ---- 1. 汇总
ws = wb.create_sheet("汇总")
sum_head = ["数据集", "任务", "我的AUC", "我的AUPR",
            "高于我的模型数(AUC)", "高于我的模型数(AUPR)",
            "高于我的模型(AUC)", "可靠性/备注"]
ws.append(sum_head)
for _, ds, task, _, oa, op, comps, rel in DATA:
    aa = sorted([(m, a, a - oa) for m, _, a, _, _ in comps
                 if a is not None and a > oa], key=lambda x: -x[2])
    ap = sorted([(m, p, p - op) for m, _, _, p, _ in comps
                 if p is not None and p > op], key=lambda x: -x[2])
    ws.append([ds, task, oa, op, len(aa), len(ap),
               "、".join(f"{m} {v:.4f}(+{d:.4f})" for m, v, d in aa) or "无",
               rel])
style_header(ws, 1, len(sum_head))
set_widths(ws, [30, 22, 10, 10, 16, 16, 62, 34])
for r in range(2, ws.max_row + 1):
    for c in (3, 4):
        ws.cell(row=r, column=c).number_format = "0.0000"
    for c in range(1, len(sum_head) + 1):
        ws.cell(row=r, column=c).border = BORDER
        ws.cell(row=r, column=c).alignment = LEFT if c in (1, 2, 7, 8) else CENTER
    if ws.cell(row=r, column=5).value >= 3 or ws.cell(row=r, column=6).value >= 3:
        for c in (5, 6):
            ws.cell(row=r, column=c).font = RED

# ---- 2/3. matrices (method x dataset)
ALL_DS = [d[1] for d in DATA]
method_order, method_meta = [], {}
for _, _, _, our_label, _, _, comps, _ in DATA:
    for m in [our_label] + [c[0] for c in comps]:
        if m not in method_meta:
            method_meta[m] = "本文方法" if m == our_label else ""
            method_order.append(m)
for _, _, _, our_label, _, _, comps, _ in DATA:
    for m, t, _, _, _ in comps:
        method_meta[m] = t

for metric, mi in (("AUC", 2), ("AUPR", 3)):
    ws = wb.create_sheet(f"总表-{metric}")
    ws.append(["方法", "类型"] + ALL_DS)
    for m in method_order:
        row = [m, method_meta[m]]
        for _, _, _, our_label, oa, op, comps, _ in DATA:
            if m == our_label:
                row.append(oa if metric == "AUC" else op)
            else:
                v = next((c[mi] for c in comps if c[0] == m), None)
                row.append(v)
        ws.append(row)
    style_header(ws, 1, len(ALL_DS) + 2)
    set_widths(ws, [34, 20] + [14] * len(ALL_DS))
    ours_rows = set()
    for r in range(2, ws.max_row + 1):
        if ws.cell(row=r, column=2).value == "本文方法":
            ours_rows.add(r)
        for c in range(3, len(ALL_DS) + 3):
            cell = ws.cell(row=r, column=c)
            cell.number_format = "0.0000"
            cell.border = BORDER
            cell.alignment = CENTER
        ws.cell(row=r, column=1).font = BOLD
        ws.cell(row=r, column=1).border = BORDER
        ws.cell(row=r, column=2).border = BORDER
    for c in range(1, len(ALL_DS) + 3):
        ws.cell(row=1, column=c).border = BORDER
        ws.cell(row=1, column=c).alignment = Alignment(horizontal="center",
                                                       vertical="center", wrap_text=True)
    for r in ours_rows:
        for c in range(1, len(ALL_DS) + 3):
            ws.cell(row=r, column=c).fill = OURS
    # mark per-dataset best in each column (red bold)
    for ci, (_, _, _, _, _, _, _, _) in enumerate(DATA):
        col = ci + 3
        vals = [(r, ws.cell(row=r, column=col).value)
                for r in range(2, ws.max_row + 1)
                if isinstance(ws.cell(row=r, column=col).value, (int, float))]
        if vals:
            bi = max(vals, key=lambda x: x[1])[0]
            ws.cell(row=bi, column=col).font = RED
    ws.freeze_panes = "C2"

# ---- 4.. per-dataset sheets
for sheet_name, ds, task, our_label, oa, op, comps, rel in DATA:
    ws = wb.create_sheet(sheet_name)
    title = f"数据集：{ds}    任务：{task}    本文方法：{our_label}"
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=11)
    tc = ws.cell(row=1, column=1, value=title)
    tc.font = TITLE
    tc.alignment = LEFT
    ws.append([])
    head = ["排名", "方法", "类型", "AUC", "AUPR",
            "ΔAUC(基线−我)", "ΔAUPR(基线−我)", "是否高于我", "来源/备注",
            "论文其他指标(照抄)", "自跑其他指标(@bestF1)"]
    ws.append(head)
    hrow = ws.max_row

    entries = [(our_label, "本文方法", oa, op, "", EXTRA.get((sheet_name, our_label), ""),
                SELF_RUN.get((sheet_name, our_label), ""))]
    for m, t, a, p, note in comps:
        entries.append((m, t, a, p, note, EXTRA.get((sheet_name, m), ""),
                        SELF_RUN.get((sheet_name, m), "")))
    ranked = sorted(entries, key=lambda e: (e[2] is None, -(e[2] or 0)))
    for rank, (m, t, a, p, note, extra, selfrun) in enumerate(ranked, start=1):
        ours_row = (t == "本文方法")
        da = None if (a is None or ours_row) else round(a - oa, 4)
        dp = None if (p is None or ours_row) else round(p - op, 4)
        flags = []
        if is_winner(a, oa):
            flags.append("AUC")
        if is_winner(p, op):
            flags.append("AUPR")
        ws.append([rank, ("★ " if ours_row else "") + m, t, a, p, da, dp,
                   "、".join(flags) if not ours_row else "—", note, extra, selfrun])
        r = ws.max_row
        for c in range(1, 12):
            ws.cell(row=r, column=c).border = BORDER
            ws.cell(row=r, column=c).alignment = LEFT if c in (2, 3, 9, 10, 11) else CENTER
            if ours_row:
                ws.cell(row=r, column=c).fill = OURS
                ws.cell(row=r, column=c).font = BOLD
        ws.cell(row=r, column=2).font = BOLD if ours_row else Font(bold=True)
        for c in (4, 5):
            ws.cell(row=r, column=c).number_format = "0.0000"
        for c in (6, 7):
            ws.cell(row=r, column=c).number_format = "+0.0000;-0.0000;0.0000"
        if is_winner(a, oa):
            ws.cell(row=r, column=4).font = RED
        if is_winner(p, op):
            ws.cell(row=r, column=5).font = RED
        if da is not None and da > 0:
            ws.cell(row=r, column=6).font = RED
        elif da is not None and da < 0:
            ws.cell(row=r, column=6).font = GREEN
        if dp is not None and dp > 0:
            ws.cell(row=r, column=7).font = RED
        elif dp is not None and dp < 0:
            ws.cell(row=r, column=7).font = GREEN
        if flags:
            ws.cell(row=r, column=8).font = RED
    style_header(ws, hrow, 11)
    for c in range(1, 12):
        ws.cell(row=hrow, column=c).border = BORDER
    set_widths(ws, [6, 26, 20, 10, 10, 15, 15, 12, 40, 40, 40])
    ws.freeze_panes = ws.cell(row=hrow + 1, column=1)
    rr = ws.max_row + 2
    ws.cell(row=rr, column=1, value="备注").font = BOLD
    ws.merge_cells(start_row=rr, start_column=2, end_row=rr, end_column=11)
    ws.cell(row=rr, column=2, value=rel).alignment = LEFT

# ---- 新增候选基线（检索到、待复现的 BIB / B刊论文）
ws = wb.create_sheet("新增候选基线(BIB-B刊)")
ws.append(CANDS_COLS)
for row in CANDIDATES:
    ws.append(list(row))
style_header(ws, 1, len(CANDS_COLS))
set_widths(ws, [30, 18, 26, 26, 18, 8, 24, 34, 30, 44])
for r in range(2, ws.max_row + 1):
    for c in range(1, len(CANDS_COLS) + 1):
        cell = ws.cell(row=r, column=c)
        cell.border = BORDER
        cell.alignment = LEFT if c in (1, 2, 3, 4, 5, 7, 9, 10) else CENTER
    ws.cell(row=r, column=1).font = BOLD
ws.freeze_panes = "A2"

# ---- 同协议复现台账
ws = wb.create_sheet("同协议复现台账")
ws.append(LEDGER_COLS)
for row in LEDGER:
    ws.append(list(row))
style_header(ws, 1, len(LEDGER_COLS))
set_widths(ws, [34, 20, 20, 38, 22, 12, 40, 46])
for r in range(2, ws.max_row + 1):
    for c in range(1, len(LEDGER_COLS) + 1):
        cell = ws.cell(row=r, column=c)
        cell.border = BORDER
        cell.alignment = LEFT if c in (1, 2, 3, 4, 5, 7, 8) else CENTER
    ws.cell(row=r, column=1).font = BOLD
    if ws.cell(row=r, column=6).value == "已完成":
        ws.cell(row=r, column=6).font = Font(bold=True, color="107C10")
    elif ws.cell(row=r, column=6).value in ("阻塞", "克隆受阻"):
        ws.cell(row=r, column=6).font = RED
ws.freeze_panes = "A2"

wb.save(OUT)
print("wrote", OUT)
print("sheets:", wb.sheetnames)
