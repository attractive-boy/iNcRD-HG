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


table('阶梯（每级对上一级配对 Δ）', [
    ('L0 plain iNcRD-HG', 'F_plain', None),
    ('L1 + M1 QMP 查询掩蔽消息传递', 'F_qmp', None),
    ('L2 + M2 SP-RNCN 简单路径关系型共同邻居', 'F_sp', None),
    ('L3 + M3 MC-EASE 掩蔽单元闭式协同回归', 'F_ease', None),
    ('L4 + M4 SeqSide 序列来源旁信息（= full）', 'F_full', None)], 'ladder')
table('逐个移除（对 full 配对 Δ）', [
    ('full', 'F_full', None),
    ('full − M1（编码器不掩蔽）', 'F_noM1', 'F_full'),
    ('full − M2（无配对结构通道）', 'F_noM2', 'F_full'),
    ('full − M3（无 EASE 通道）', 'F_noM3', 'F_full'),
    ('full − M4（无序列旁信息）', 'F_ease', 'F_full')], 'ref')
table('机制对照（对 full 配对 Δ）', [
    ('full', 'F_full', None),
    ('全部通道不做查询掩蔽（编码器 / 结构 / EASE 都含查询边）', 'F_nomask', 'F_full'),
    ('M4 序列关系与 k-mer 核在 miRNA 间置换', 'F_seqS', 'F_full')], 'ref')
table('严格版阶梯：编码器保持原样（不用 M1），只加配对通道', [
    ('plain', 'F_plain', None),
    ('+ M2', 'F_sp_p', None),
    ('+ M3', 'F_ease_p', None),
    ('+ M4（= full − M1）', 'F_noM1', None)], 'ladder')
