# -*- coding: utf-8 -*-
"""Markdown tables for REPORT_NOLQO.md from results/F_*.csv (run_nl.py outputs).
Every number is the valsel test metric (epoch picked on the 10% training hold-out), averaged over folds x seeds;
deltas are paired over identical (split, seed, fold) and reported with the number of folds that improve."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
PROT = {'5cv': [('5cv', s) for s in (42, 43, 44)], '80/20': [('single', s) for s in range(5)]}
METRIC = sys.argv[1] if len(sys.argv) > 1 else 'valsel'


def load(tag, ds, prot):
    out = []
    for sp, sd in PROT[prot]:
        f = os.path.join(R, f'{tag}_{ds}_{sp}_seed{sd}.csv')
        if os.path.exists(f):
            d = pd.read_csv(f); d['key'] = f'{sp}_{sd}_' + d['fold'].astype(str); out.append(d)
    return pd.concat(out) if out else None


def cell(tag, ref, ds, prot):
    d = load(tag, ds, prot)
    if d is None:
        return 'n/a', ''
    m = METRIC
    s = f'{d[m + "_auc"].mean():.4f} / {d[m + "_aupr"].mean():.4f}'
    if ref is None:
        return s, ''
    r = load(ref, ds, prot)
    if r is None:
        return s, ''
    j = d.merge(r, on='key', suffixes=('', '_r'))
    da = j[m + '_auc'] - j[m + '_auc_r']; dp = j[m + '_aupr'] - j[m + '_aupr_r']
    se = da.std(ddof=1) / np.sqrt(len(j))
    return s, f'{da.mean():+.4f} / {dp.mean():+.4f} ({int((da > 0).sum())}/{len(j)} · {int((dp > 0).sum())}/{len(j)}; se {se:.4f})'


def table(title, rows, ref_mode):
    print(f'\n#### {title}\n')
    print('| 配置 | MDR 5cv×3 | Δ | MDS 5cv×3 | Δ | MDR 80/20×5 | Δ | MDS 80/20×5 | Δ |')
    print('|---|---|---|---|---|---|---|---|---|')
    prev = None
    for name, tag, ref in rows:
        rr = prev if ref_mode == 'ladder' else ref
        cells = []
        for prot in ('5cv', '80/20'):
            for ds in ('MDR', 'MDS'):
                pass
        line = [name]
        for ds, prot in (('MDR', '5cv'), ('MDS', '5cv'), ('MDR', '80/20'), ('MDS', '80/20')):
            s, dl = cell(tag, rr, ds, prot)
            line += [s, dl or '–']
        print('| ' + ' | '.join(line) + ' |')
        prev = tag


table('编码器阶梯（每级对上一级配对 Δ；不含 M2–M4 pair 通道）', [
    ('L0 plain iNcRD-HG 编码器', 'P_plain', None),
    ('L1 + M1 查询掩蔽消息传递（上一轮）', 'P_mask', None),
    ('L2 + E1 CKA 内容键控消息权重', 'P_cka', None),
    ('L3 + E2 H-NBF 空心源条件消息传递（药物源）', 'P_cka_nbf30d128', None),
    ('L4 + E3 反向条件（miRNA 源）', 'P_E3_bi32', None),
    ('L5 + E4 条件状态逐层读出', 'P_L5', None),
], 'ladder')
table('逐项移除（对 L5 配对 Δ）', [
    ('L5 完整编码器', 'P_L5', None),
    ('− E1 CKA', 'P_L5_noCKA', 'P_L5'),
    ('− E2 药物源 H-NBF（保留 miRNA 源）', 'P_L5_noNBFd', 'P_L5'),
    ('− E3 miRNA 源', 'P_E3_njk', 'P_L5'),
    ('− E4 逐层读出', 'P_E3_bi32', 'P_L5'),
    ('对照：不掩蔽查询边（M1 + 条件传播都关掩蔽）', 'P_L5_nomask', 'P_L5'),
], 'ref')
table('第一轮编码器候选（对 L2 或 L1 配对 Δ；均为零或弱正，未进入最终编码器）', [
    ('L2 CKA', 'P_cka', None),
    ('+ HJK 跳跃知识读出', 'P_enc2', 'P_cka'),
    ('+ HJK + HRP 空心预解传播', 'P_enc3', 'P_cka'),
], 'ref')
table('full 栈（M1–M4）+ 编码器模块（对 F_full 配对 Δ）', [
    ('F_full（上一轮最终模型）', 'F_full', None),
    ('F_full + E1–E4', 'F_L5', 'F_full'),
], 'ref')
