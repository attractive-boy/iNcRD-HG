# -*- coding: utf-8 -*-
"""Rank the user's strongest model against the reproduced DLST-MDA baselines (1:1)."""
import csv
import os
import statistics as st

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

HERE = os.path.dirname(os.path.abspath(__file__))
BL = os.path.join(HERE, '..', 'dlstmda_baselines_20260919', 'results')
OUT = os.path.join(HERE, 'DLST_MDA_strongest_ranking.xlsx')


def mean(fp, keys):
    rows = list(csv.DictReader(open(fp)))
    return {k: st.mean(float(r[k]) for r in rows) for k in keys}, len(rows)


def main():
    rows = []

    # our strongest model: pool 3 inference seeds x 5 folds = 15 runs
    per_seed = {}
    pooled = []
    for seed in ('', '_seed666', '_seed0', '_seed1'):
        fp = os.path.join(HERE, 'results', f'strongest{seed or "_per_fold"}.csv')
        with open(fp) as fh:
            rr = list(csv.DictReader(fh))
        per_seed[seed or '_primary'] = rr
        pooled += rr
    auc = st.mean(float(r['fused_auc']) for r in pooled)
    aupr = st.mean(float(r['fused_aupr']) for r in pooled)
    acc = st.mean(float(r['acc']) for r in pooled)
    rows.append(dict(method='iNcRD-HG-strongest(本方法)', auc=auc, aupr=aupr, acc=acc,
                     source='本轮(本文方法, 3seed x 5fold)'))
    seed_detail = {k: (st.mean(float(r['fused_auc']) for r in v),
                       st.mean(float(r['fused_aupr']) for r in v),
                       st.mean(float(r['ref_auc']) for r in v),
                       st.mean(float(r['ref_aupr']) for r in v)) for k, v in per_seed.items()}

    # reproduced baselines (best-epoch 5-fold means)
    base = {
        'GraphDTA': ('graphdta_per_fold.csv', False),
        'GCNNMMA': ('gcnnma_per_fold.csv', False),
        'SubMDTA': ('submdta_per_fold.csv', False),
        'ML-DTI': ('mldti_per_fold.csv', False),
        'DAESTB(leak-free)': ('daestb_foldwise_per_fold.csv', False),
        'DAESTB(transductive,leaky)': ('daestb_per_fold.csv', False),
    }
    for name, (fp, _) in base.items():
        d, n = mean(os.path.join(BL, fp), ['auc', 'aupr', 'acc'])
        rows.append(dict(method=name, auc=d['auc'], aupr=d['aupr'], acc=d['acc'], source='基线复现'))

    # paper DLST-MDA + official-code reproduction
    rows.append(dict(method='DLST-MDA(官方代码复现)', auc=0.9437, aupr=0.9356, acc=0.8753, source='论文方法'))
    rows.append(dict(method='DLST-MDA(论文报告)', auc=0.9411, aupr=0.9326, acc=0.8736, source='论文方法'))

    # rank: fair board excludes the leaky transductive DAESTB
    fair = [r for r in rows if 'leaky' not in r['method']]
    fair.sort(key=lambda r: (-r['auc'], -r['aupr']))
    for i, r in enumerate(fair, 1):
        r['rank'] = i

    allr = sorted(rows, key=lambda r: (-r['auc'], -r['aupr']))
    for i, r in enumerate(allr, 1):
        r['rank_all'] = i

    wb = Workbook(); ws = wb.active; ws.title = 'Strongest ranking'
    hdr = ['公平排名', '含泄漏排名', '方法', '类别', 'AUC', 'AUPR', 'ACC', '备注']
    ws.append(hdr)
    for c in ws[1]:
        c.font = Font(bold=True, color='FFFFFF')
        c.fill = PatternFill('solid', fgColor='305496')
        c.alignment = Alignment(horizontal='center', wrap_text=True)
    OUR = PatternFill('solid', fgColor='FFF2CC')
    for r in allr:
        leak = 'leaky' in r['method']
        ws.append([r.get('rank', '—'), r['rank_all'], r['method'], r['source'],
                   round(r['auc'], 4), round(r['aupr'], 4), round(r['acc'], 4) if r.get('acc') else '—',
                   '转导 GIP（含测试关联，仅参考）' if leak else ''])
        if r['method'].startswith('iNcRD-HG-strongest'):
            for c in range(1, 9):
                ws.cell(ws.max_row, c).fill = OUR
    for i, w in enumerate([9, 10, 30, 14, 9, 9, 9, 30], 1):
        ws.column_dimensions[chr(64 + i)].width = w
    ws2 = wb.create_sheet('per-seed detail')
    ws2.append(['inference seed', 'graphctx-ref AUC', 'graphctx-ref AUPR', 'fused AUC', 'fused AUPR'])
    for k, (a, u, ra, ru) in seed_detail.items():
        ws2.append([k, round(ra, 4), round(ru, 4), round(a, 4), round(u, 4)])
    for c in ws2[1]:
        c.font = Font(bold=True)
    wb.save(OUT)

    print(f"{'rank':>4s} {'method':32s} {'AUC':>7s} {'AUPR':>7s}")
    for r in fair:
        print(f"{r['rank']:>4d} {r['method']:32s} {r['auc']:7.4f} {r['aupr']:7.4f}")
    print('\n(with leaky transductive DAESTB included):')
    for r in allr:
        print(f"{r['rank_all']:>4d} {r['method']:32s} {r['auc']:7.4f} {r['aupr']:7.4f}")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
