"""Minimal two-sheet comparison workbook: which external models beat ours (by AUC / AUPR)."""
import os

import pandas as pd

OUT = "/data1/licheng/incrd-hg/iNcRD-HG/improve/results/external_specialized_20260916/外部专用模型对比_简版.xlsx"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# dataset, task, our_best_auc, our_best_aupr, [(model, auc, aupr), ...]
DATA = [
    ("MPHGNN 独立测试集", "miRNA-drug 抗性", 0.9605, 0.9147, [
        ("MPHGNN", 0.9780, 0.9370), ("SubKNet", 0.9717, 0.9341), ("GAM-MDR", 0.9557, 0.9150),
        ("DeepLDA", 0.9311, 0.8804), ("AGCLNDA", 0.9170, 0.8670), ("NDSGCL", 0.8070, 0.6732)]),
    ("ncDR (GAM-MDR 701x101)", "miRNA-drug 抗性", 0.8863, 0.8711, [
        ("GAM-MDR (本地复现)", 0.9022, 0.8842), ("SMCL-MDR", 0.9896, 0.9881), ("SVMMDR", 0.9780, None),
        ("BNEMDI", 0.9568, 0.9565), ("NASMDR", 0.9468, None), ("AMMGC", 0.9467, 0.2399),
        ("GCMDR", 0.9369, None)]),
    ("GCFMCL D2R", "miRNA-drug 敏感性", 0.8472, 0.8621, [
        ("GCFMCL", 0.9528, 0.9566)]),
    ("NoncoRNA+ncDR 2693", "ncRNA-drug 抗性(混合)", 0.8686, 0.8856, [
        ("RDRGSE", 0.9768, 0.9736), ("B-NDRA", 0.9207, 0.9173), ("LRGCPND", 0.8987, 0.9094),
        ("GSLRDA", 0.9101, None), ("DMGAT", 0.8964, 0.8984)]),
    ("NoncoRNA+ncDR 的 41 lncRNA 子集", "lncRNA-drug 抗性", 0.7053, 0.7420, [
        ("DMGAT (lncRNA)", 0.8397, 0.8431), ("KBMF2K", 0.8410, 0.5210), ("CMF", 0.8340, 0.5380),
        ("NetLapRLS", 0.8110, 0.5470), ("BLM-NI", 0.7840, 0.5600)]),
    ("RNAactDrug (HGNNLDA 数据)", "lncRNA-drug 敏感性", 0.9307, 0.9425, [
        ("HGNNLDA", None, None)]),
    ("DeepLDA NoncoRNA", "lncRNA-drug 抗性", 0.9637, 0.8013, [
        ("DeepLDA", 0.9583, 0.8601)]),
    ("DeepLDA ncDR", "lncRNA-drug 抗性", 0.7100, 0.1242, [
        ("DeepLDA", 0.9819, 0.8687)]),
    ("自有 S_MD 测试集", "miRNA-drug 抗性", 0.9638, 0.9294, []),
    ("自有 S_LD 测试集", "lncRNA-drug 抗性", 0.9948, 0.9896, []),
]

summary, detail = [], []
for ds, task, oa, op, models in DATA:
    above_a = [(m, a, a - oa) for m, a, p in models if a is not None and a > oa]
    above_p = [(m, p, p - op) for m, a, p in models if p is not None and p > op]
    above_a.sort(key=lambda x: -x[2]); above_p.sort(key=lambda x: -x[2])
    summary.append(dict(数据集=ds, 任务=task, 我的AUC=oa, 我的AUPR=op,
                        高于我的模型_AUC_值_差距="、".join(f"{m} {v:.4f}(+{d:.4f})" for m, v, d in above_a) or "无",
                        高于我的模型_AUPR_值_差距="、".join(f"{m} {v:.4f}(+{d:.4f})" for m, v, d in above_p) or "无",
                        达到3个_AUC=len(above_a) >= 3, 达到3个_AUPR=len(above_p) >= 3))
    detail.append(dict(数据集=ds, 模型="★ 我的模型", AUC=oa, AUPR=op, 高于我="",
                       高出AUC=None, 高出AUPR=None))
    for m, a, p in models:
        if a is None and p is None:
            detail.append(dict(数据集=ds, 模型=m, AUC=None, AUPR=None, 高于我="指标未公开",
                               高出AUC=None, 高出AUPR=None))
            continue
        flag = "、".join([x for x, c in (("AUC", a is not None and a > oa),
                                        ("AUPR", p is not None and p > op)) if c])
        if flag:
            detail.append(dict(数据集=ds, 模型=m, AUC=a, AUPR=p, 高于我=flag,
                               高出AUC=(round(a - oa, 4) if a is not None else None),
                               高出AUPR=(round(p - op, 4) if p is not None else None)))

with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    pd.DataFrame(summary).to_excel(xw, sheet_name="汇总", index=False)
    pd.DataFrame(detail).to_excel(xw, sheet_name="明细", index=False)
    for name, widths in (("汇总", [26, 18, 10, 10, 60, 60, 10, 10]),
                         ("明细", [26, 20, 10, 10, 12, 11, 11])):
        ws = xw.sheets[name]
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
print("wrote", OUT)
print(pd.DataFrame(summary).to_string(index=False))
