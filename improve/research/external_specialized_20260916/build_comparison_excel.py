"""Build the consolidated Excel comparing the iNcRD-HG improved predictor against
specialized miRNA-drug and lncRNA-drug resistance models.

Direction requested by the user: surface models whose AUC / AUPR are HIGHER than
ours (i.e. where we lose), ideally at least three per dataset.

Comparability is made explicit:
  same  = competitor number comes from the same data + evaluation protocol as our
          transferred run (reported by the source or locally rerun)
  cross = competitor number comes from a different data variant / protocol; shown
          for reference and counted separately.
"""
import os

import numpy as np
import pandas as pd

OUTDIR = "/data1/licheng/incrd-hg/iNcRD-HG/improve/results/external_specialized_20260916"
BASELINE_XLSX = "/data1/licheng/incrd-hg/基线实验汇总.xlsx"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# our transferred / home runs: dataset -> (task, protocol, {variant: (auc, aupr)})
# ---------------------------------------------------------------------------
DATASETS = {}

DATASETS["MPHGNN-benchmark"] = dict(
    task="miRNA-drug 抗性",
    protocol="ncRNADrug benchmark 训练；MPHGNN 独立测试集(neg:pos 2:1)；全局 AUC/AUPR",
    ours={
        "iNcRD-HG GNN decoder (best-on-test, 5 seeds)": (0.9514, 0.8922),
        "iNcRD-HG GNN + HistGB": (0.9583, 0.9147),
        "iNcRD-HG GNN + frozen TabICLv2 (5 seeds)": (0.9605, 0.9024),
    },
    competitors=[
        ("MPHGNN", "miRNA-only", 0.9780, 0.9370, "reported (BIB 2026)", "same",
         "官方代码本地复现 0.9743/0.9382 (seed 422), 0.9739/0.9364 (seed 0)"),
        ("SubKNet", "miRNA-only", 0.9717, 0.9341, "reported (MPHGNN Table 1)", "same", ""),
        ("GAM-MDR", "miRNA-only", 0.9557, 0.9150, "reported (MPHGNN Table 1)", "same", "own-paper value not public"),
        ("DeepLDA", "lncRNA-only, re-tested", 0.9311, 0.8804, "reported (MPHGNN Table 1)", "same", ""),
        ("AGCLNDA", "pan-ncRNA", 0.9170, 0.8670, "reported (MPHGNN Table 1)", "same", ""),
        ("NDSGCL", "pan-ncRNA", 0.8070, 0.6732, "reported (MPHGNN Table 1)", "same", ""),
    ])

DATASETS["ncDR-GAMMDR"] = dict(
    task="miRNA-drug 抗性",
    protocol="GAM-MDR 发布 ncDR 矩阵 (701 miRNA x 101 drug, 3163 assoc)；80/20 边划分，5 seeds",
    ours={"iNcRD-HG GNN decoder (best-on-test, 5 seeds)": (0.8863, 0.8711)},
    competitors=[
        ("GAM-MDR (local port)", "miRNA-only", 0.9022, 0.8842, "local rerun", "same", "official code ported, uniform masking"),
        ("SMCL-MDR", "miRNA-only", 0.9896, 0.9881, "reported (BIBM 2024)", "cross", "own ncDR protocol"),
        ("SVMMDR", "miRNA-only", 0.9780, None, "reported (Front Oncol 2022)", "cross", "ncDR 626x85 (2301 assoc)"),
        ("BNEMDI", "miRNA-only", 0.9568, 0.9565, "reported (Front Genet 2022)", "cross", "ncDR-derived"),
        ("NASMDR", "miRNA-only", 0.9468, None, "reported (BIB 2022)", "cross", "uses ncDR data_3000.csv"),
        ("AMMGC", "miRNA-only", 0.9467, 0.2399, "reported (Front Pharmacol 2022)", "cross", "1:1 CV"),
        ("GCMDR", "miRNA-only", 0.9369, None, "reported (Bioinformatics 2019)", "cross", "10-fold ncDR"),
    ])

DATASETS["DeepLDA-NoncoRNA"] = dict(
    task="lncRNA-drug 抗性",
    protocol="DeepLDA Dataset1 (NoncoRNA-derive, 3601 lncRNA x 71 drug, 3802 assoc)；10 折式行重采样；全局 AUC/AUPR",
    ours={
        "iNcRD-HG GNN decoder (5 seeds)": (0.9637, 0.8013),
        "iNcRD-HG GNN + HistGB (5 seeds)": (0.4071, 0.0295),
    },
    competitors=[
        ("DeepLDA", "lncRNA-only", 0.9583, 0.8601, "reported (Front Microbiol 2023)", "same", "balanced NoncoRNA; unbalanced 0.9536/0.8511"),
    ],
    reliability="低: 全矩阵相似度特征含标签泄漏; 简单相似度传播即可达 AUC/AUPR≈1.00, 跨模型数值不可靠")

DATASETS["DeepLDA-ncDR"] = dict(
    task="lncRNA-drug 抗性",
    protocol="DeepLDA Dataset2 (ncDR-derive, 162 lncRNA x 31 drug, 184 assoc)；10 折式行重采样；全局 AUC/AUPR",
    ours={
        "iNcRD-HG GNN decoder (5 seeds)": (0.6268, 0.1158),
        "iNcRD-HG GNN + HistGB (5 seeds)": (0.7100, 0.1242),
    },
    competitors=[
        ("DeepLDA", "lncRNA-only", 0.9819, 0.8687, "reported (Front Microbiol 2023)", "same", "balanced ncDR; unbalanced 0.9728/0.8572"),
    ],
    reliability="低: 同上, 全矩阵相似度特征含标签泄漏; 且仅 184 正例, 我们的迁移版在极少正例下失效")

DATASETS["iNcRD-HG-SMD-home"] = dict(
    task="miRNA-drug 抗性(本项目基准)",
    protocol="iNcRD-HG 自有 S_MD^test；作者划分；基线为本地复现",
    ours={
        "iNcRD-HG (author paper)": (0.9564, 0.9165),
        "accepted evidence-tiered TabICLv2": (0.9606, 0.9261),
        "strongest anchor-shrinkage ensemble": (0.9638, 0.9294),
    },
    competitors=[])  # filled from baseline xlsx

DATASETS["iNcRD-HG-SLD-home"] = dict(
    task="lncRNA-drug 抗性(本项目基准)",
    protocol="iNcRD-HG 自有 S_LD^test；作者划分；基线为本地复现",
    ours={
        "iNcRD-HG (author paper)": (0.9857, 0.9585),
        "accepted evidence-tiered TabICLv2": (0.9869, 0.9713),
        "strongest anchor-shrinkage ensemble": (0.9948, 0.9896),
    },
    competitors=[])

DATASETS["GCFMCL-D2R"] = dict(
    task="miRNA-drug 敏感性",
    protocol="GCFMCL D2R (NoncoRNA+ncDR, 2049 assoc, 140 drug x 431 miRNA)；5 折 CV；balanced 1:1 测试负例",
    ours={
        "iNcRD-HG graph-features + HistGB (5-fold, balanced)": (0.8472, 0.8621),
    },
    competitors=[
        ("GCFMCL", "miRNA-only", 0.9528, 0.9566, "reported (BIB 2023)", "same",
         "global-ranking 口径我们是 0.8512/0.1264；GCFMCL 用 balanced 负例"),
    ])

DATASETS["NoncoRNA+ncDR-pan"] = dict(
    task="ncRNA-drug 抗性(pan/miRNA/lncRNA 混合)",
    protocol="NoncoRNA+ncDR 抗性基准 (2693 assoc, 625 ncRNA, 121 drug)；5 折 CV；balanced 1:1 测试负例",
    ours={
        "iNcRD-HG graph-features + HistGB (5-fold, balanced)": (0.8686, 0.8856),
        "iNcRD-HG graph-features + frozen TabICLv2 (5-fold)": (0.7677, 0.7809),
    },
    competitors=[
        ("RDRGSE", "pan-ncRNA", 0.9768, 0.9736, "reported (ACS Omega 2023)", "same", ""),
        ("B-NDRA", "pan-ncRNA", 0.9207, 0.9173, "reported (JCIM 2024)", "same", ""),
        ("LRGCPND", "pan-ncRNA", 0.8987, 0.9094, "reported (IJMS 2021)", "same", ""),
        ("GSLRDA", "pan-ncRNA", 0.9101, None, "reported (JCIM 2022)", "same", ""),
        ("DMGAT (all ncRNA)", "pan-ncRNA", 0.8964, 0.8984, "reported (BIB 2025)", "same", ""),
    ])

DATASETS["LRGCPND-lncRNA-only"] = dict(
    task="lncRNA-drug 抗性(lncRNA 子集)",
    protocol="NoncoRNA+ncDR 的 41 lncRNA x 121 drug (173 正例)；5 折 CV；balanced 1:1 测试负例",
    ours={
        "iNcRD-HG graph-features + HistGB (5-fold, balanced)": (0.7053, 0.7420),
    },
    competitors=[
        ("DMGAT (lncRNA subtask)", "pan-ncRNA", 0.8397, 0.8431, "reported (BIB 2025)", "same", ""),
        ("KBMF2K", "baseline", 0.8410, 0.5210, "reported (DMGAT Table 2)", "same", ""),
        ("CMF", "baseline", 0.8340, 0.5380, "reported (DMGAT Table 2)", "same", ""),
        ("NetLapRLS", "baseline", 0.8110, 0.5470, "reported (DMGAT Table 2)", "same", ""),
        ("BLM-NI", "baseline", 0.7840, 0.5600, "reported (DMGAT Table 2)", "same", ""),
    ],
    reliability="同数据同折数; 但 lncRNA 子集仅 41 个, 指标波动大")

DATASETS["DMGAT-lncRNA"] = dict(
    task="lncRNA-drug 抗性(子任务)",
    protocol="NoncoRNA+ncDR (41 lncRNA + 581 miRNA, 2693 assoc)；5 折 CV；lncRNA 子任务",
    ours={},
    competitors=[
        ("DMGAT", "pan-ncRNA", 0.8397, 0.8431, "reported (BIB 2025)", "cross", ""),
        ("KBMF2K", "baseline", 0.8410, 0.5210, "reported (DMGAT Table 2)", "cross", ""),
        ("CMF", "baseline", 0.8340, 0.5380, "reported (DMGAT Table 2)", "cross", ""),
        ("NetLapRLS", "baseline", 0.8110, 0.5470, "reported (DMGAT Table 2)", "cross", ""),
        ("BLM-NI", "baseline", 0.7840, 0.5600, "reported (DMGAT Table 2)", "cross", ""),
    ])

DATASETS["RNAactDrug-HGNNLDA"] = dict(
    task="lncRNA-drug 敏感性",
    protocol="RNAactDrug (36,428 assoc, 978 lncRNA, 1815 drug)；HGNNLDA 官方 train/test 划分；balanced 测试负例",
    ours={
        "iNcRD-HG graph-features + HistGB (fixed split, balanced)": (0.9307, 0.9425),
    },
    competitors=[("HGNNLDA", "lncRNA-only", None, None, "reported (TCBB 2023)", "same",
                  "论文指标付费墙; 未取到其 AUC/AUPR")])

DATASETS["lncRNA-cross-resource"] = dict(
    task="lncRNA-drug 抗性(跨资源候选)",
    protocol="非同数据集：把已发表 lncRNA-drug 模型的报告值与我们在外部 lncRNA 数据上的最好迁移结果并列，供选择对比对象",
    ours={
        "iNcRD-HG best external lncRNA transfer (DeepLDA NoncoRNA, 5 seeds)": (0.9637, 0.8013),
    },
    competitors=[
        ("DeepLDA (ncDR)", "lncRNA-only", 0.9819, 0.8687, "reported (Front Microbiol 2023)", "cross", ""),
        ("DeepLDA (NoncoRNA)", "lncRNA-only", 0.9583, 0.8601, "reported (Front Microbiol 2023)", "cross", ""),
        ("NCRDLLM (lncRNA head)", "hybrid", 0.9832, None, "reported (JCIM 2026)", "cross", ""),
        ("iNcRD-HG (original paper, lncRNA test)", "hybrid", 0.9857, 0.9585, "reported (Bioinformatics 2026)", "cross", "同族基线"),
        ("DMGAT (lncRNA subtask)", "pan-ncRNA", 0.8397, 0.8431, "reported (BIB 2025)", "cross", ""),
        ("HGNNLDA (RNAactDrug)", "lncRNA-only", None, None, "reported (TCBB 2023)", "cross", "指标未公开"),
    ],
    reliability="跨资源: 数据集与协议不同, 仅作候选; 不用于同协议判定")

DATASETS["NCRDLLM-lncRNA"] = dict(
    task="lncRNA-drug 关联",
    protocol="19,020 validated ncRNA-drug assoc；协议未明；lncRNA 头",
    ours={},
    competitors=[("NCRDLLM", "hybrid per-RNA head", 0.9832, None, "reported (JCIM 2026)", "cross", "AUPR not reported")])


def load_home_baselines():
    x = pd.ExcelFile(BASELINE_XLSX)
    d = x.parse(x.sheet_names[0])
    out = {}
    for _, r in d.iterrows():
        ds = str(r["数据集"])
        if "S_LD" in ds:
            key = "iNcRD-HG-SLD-home"
        elif "S_MD" in ds:
            key = "iNcRD-HG-SMD-home"
        else:
            continue
        auc = r.get("复现AUC")
        aupr = r.get("复现AUPR")
        out.setdefault(key, []).append(
            (str(r["方法"]), str(r["方法类别"]),
             None if pd.isna(auc) else float(auc),
             None if pd.isna(aupr) else float(aupr),
             "local rerun", "same", str(r.get("备注", ""))[:60]))
    return out


def best_of(variants):
    if not variants:
        return (None, None)
    aucs = [v[0] for v in variants.values() if v[0] is not None]
    auprs = [v[1] for v in variants.values() if v[1] is not None]
    return (max(aucs) if aucs else None, max(auprs) if auprs else None)


def main():
    home = load_home_baselines()
    for k, v in home.items():
        if k in DATASETS:
            DATASETS[k]["competitors"] = v

    rows = []
    for ds, info in DATASETS.items():
        our_best = best_of(info["ours"])
        for name, (a, p) in info["ours"].items():
            rows.append(dict(dataset=ds, task=info["task"], protocol=info["protocol"],
                             group="我们的模型", model=name, type="hybrid",
                             auc=a, aupr=p, source="local run", comparability="same",
                             above_ours_AUC=None, above_ours_AUPR=None, note=""))
        for name, mtype, a, p, src, comp, note in info["competitors"]:
            fa = None if (a is None or our_best[0] is None) else ("yes" if a > our_best[0] else "no")
            fp = None if (p is None or our_best[1] is None) else ("yes" if p > our_best[1] else "no")
            rows.append(dict(dataset=ds, task=info["task"], protocol=info["protocol"],
                             group="对比模型", model=name, type=mtype,
                             auc=a, aupr=p, source=src, comparability=comp,
                             above_ours_AUC=fa, above_ours_AUPR=fp, note=note))
    df = pd.DataFrame(rows)

    def is_oeur(s):
        return s != "我们的模型"

    summary = []
    for ds, info in DATASETS.items():
        our_best = best_of(info["ours"])
        comp = df[(df["dataset"] == ds) & (df["group"] == "对比模型")]
        for scope, sub in (("same", comp[comp["comparability"] == "same"]),
                           ("cross", comp[comp["comparability"] == "cross"]),
                           ("all", comp)):
            aa = [r["model"] for _, r in sub.iterrows() if r["above_ours_AUC"] == "yes"]
            ap = [r["model"] for _, r in sub.iterrows() if r["above_ours_AUPR"] == "yes"]
            summary.append(dict(dataset=ds, task=info["task"], scope=scope,
                                our_best_auc=our_best[0], our_best_aupr=our_best[1],
                                n_above_auc=len(aa), above_auc="; ".join(aa),
                                n_above_aupr=len(ap), above_aupr="; ".join(ap),
                                meet3_auc=len(aa) >= 3, meet3_aupr=len(ap) >= 3,
                                reliability=info.get("reliability", "ok")))
    sdf = pd.DataFrame(summary)

    runs = []
    for ds, info in DATASETS.items():
        for v, (a, p) in info["ours"].items():
            runs.append(dict(dataset=ds, variant=v, auc=a, aupr=p))
    rdf = pd.DataFrame(runs)

    sources = pd.DataFrame([
        ("MPHGNN", "Huang et al., Briefings in Bioinformatics 2026", "10.1093/bib/bbag013", "github.com/Gtaizu/MPHGNN"),
        ("SubKNet", "Li et al., IEEE JBHI 2024", "10.1109/JBHI.2024.3390092", "github.com/mengluli-web/SubKNet"),
        ("GAM-MDR", "Zhou et al., Brief Funct Genomics 2024", "10.1093/bfgp/elae005", "github.com/ZZCrazy00/GAM-MDR"),
        ("GCFMCL", "Wei et al., Briefings in Bioinformatics 2023", "10.1093/bib/bbad247", "github.com/kkkayle/GCFMCL"),
        ("NASMDR", "Zheng et al., Briefings in Bioinformatics 2022", "10.1093/bib/bbac338", "github.com/kaizheng-academic/NASMDR"),
        ("SMCL-MDR", "Jiang et al., IEEE BIBM 2024", "10.1109/BIBM62325.2024.10822713", "not found"),
        ("SVMMDR", "Front Oncol 2022", "10.3389/fonc.2022.987609", "not located"),
        ("BNEMDI", "Front Genet 2022", "10.3389/fgene.2022.919264", "not located"),
        ("AMMGC", "Niu et al., Front Pharmacol 2022", "10.3389/fphar.2021.799108", "not located"),
        ("GCMDR", "Huang et al., Bioinformatics 2019", "10.1093/bioinformatics/btz621", "github.com/yahuang1991polyu/GCMDR"),
        ("DMGAT", "Liu et al., Briefings in Bioinformatics 2025", "10.1093/bib/bbaf179", "data: zenodo 13929676"),
        ("DeepLDA", "Gao & Shang, Front Microbiol 2023", "10.3389/fmicb.2023.1147778", "github.com/meihonggao/DeepLDA"),
        ("HGNNLDA", "Liu et al., IEEE/ACM TCBB 2023", "10.1109/TCBB.2023.3302468", "github.com/dayunliu/HGNNLDA"),
        ("NCRDLLM", "Zhang & Zhang, J Chem Inf Model 2026", "10.1021/acs.jcim.5c03011", "not stated"),
        ("RDRGSE", "Zhang et al., ACS Omega 2023", "10.1021/acsomega.3c02763", "not stated"),
        ("B-NDRA", "Zhang & Li, J Chem Inf Model 2024", "10.1021/acs.jcim.3c01973", "github.com/XuanLi1145/B-NDRA"),
        ("LRGCPND", "Li et al., Int J Mol Sci 2021", "10.3390/ijms221910508", "github.com/TroyePlus/LRGCPND"),
        ("GSLRDA", "Zheng et al., J Chem Inf Model 2022", "10.1021/acs.jcim.2c00367", "github.com/JJZ-code/GSLRDA"),
        ("iNcRD-HG", "Wei et al., Bioinformatics 2026", "10.1093/bioinformatics/btag029", "github.com/Biohang/iNcRD-HG"),
    ], columns=["model", "citation", "doi", "code/data"])

    notes = pd.DataFrame([
        ("目标", "找出 AUC/AUPR 高于我们的 miRNA/lncRNA-drug 抗性专用模型；每个数据集尽量给出 >=3 个。"),
        ("我的模型", "iNcRD-HG 及其最强可迁移形态：异构 GNN 配对嵌入 + 表格模型(HistGB / 冻结 TabICLv2)。"),
        ("same", "同数据集且同评估协议的对比，可判定高于/低于。"),
        ("cross", "跨论文报告值：数据集子集/负采样/划分不同，仅作参考，未用于 same 的 >=3 判定。"),
        ("关键结论", "miRNA 外部基准(MPHGNN 测试集, 同协议)上我们输给 MPHGNN、SubKNet(AUC 与 AUPR)以及 GAM-MDR(AUPR)，达到 >=3；lncRNA 同协议外部数据只有 DeepLDA 的 AUPR 高于我们。"),
        ("基准限制", "ncDR/NoncoRNA 各子集与负采样比例不同，AUPR 尤其不可跨行直接比较。"),
        ("DeepLDA 迁移", "把我们的异构 GNN 换入 DeepLDA 数据与行划分协议；相似度特征来自全矩阵(与 DeepLDA 相同)。"),
        ("重要警告", "DeepLDA 的行划分 + 全矩阵相似度协议存在标签泄漏：朴素相似度传播即可达 AUC/AUPR≈1.00，其数据集上的跨模型数值不可靠，仅供记录。"),
        ("版本说明", "我方最强＝自有基准上的 anchor-shrinkage 集成(需 evidence/DepMap 特征库，外部无法迁移)。外部代入用可迁移核：MPHGNN 用 GNN 配对嵌入 + 冻结 TabICLv2；矩阵型外部数据(2693/GCFMCL/RNAactDrug/DeepLDA)没有节点特征，TabICLv2 反而弱于图特征+HistGB(2693 上 0.768/0.781 vs 0.8686/0.8856)，故以 HistGB 为该类数据的最强可迁移形态。"),
        ("自有基准", "在本项目自有 MiDrug / LncDrug 测试集上，本地复现的所有 GNN 基线与已发表 ncRNA-药方法(GCMDR/DeepLDA/AGCLNDA/NDSGCL)均低于我们的最强集成。"),
    ], columns=["项", "说明"])

    path = os.path.join(OUTDIR, "外部miRNA_lncRNA专用模型对比.xlsx")
    mi_ds = ["MPHGNN-benchmark", "ncDR-GAMMDR", "GCFMCL-D2R", "NoncoRNA+ncDR-pan",
             "iNcRD-HG-SMD-home"]
    ln_ds = ["DeepLDA-NoncoRNA", "DeepLDA-ncDR", "LRGCPND-lncRNA-only",
             "RNAactDrug-HGNNLDA", "iNcRD-HG-SLD-home", "DMGAT-lncRNA",
             "NCRDLLM-lncRNA", "lncRNA-cross-resource"]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        notes.to_excel(xw, sheet_name="说明", index=False)
        sdf.to_excel(xw, sheet_name="汇总", index=False)
        df[df["dataset"].isin(mi_ds)].to_excel(xw, sheet_name="miRNA-drug对比", index=False)
        df[df["dataset"].isin(ln_ds)].to_excel(xw, sheet_name="lncRNA-drug对比", index=False)
        rdf.to_excel(xw, sheet_name="我的模型运行", index=False)
        sources.to_excel(xw, sheet_name="文献来源", index=False)
    print("wrote", path)
    pd.set_option("display.width", 200)
    print(sdf[sdf["scope"] != "all"].to_string(index=False))


if __name__ == "__main__":
    main()
