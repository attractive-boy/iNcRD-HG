# -*- coding: utf-8 -*-
"""Full ranking of the user's strongest model vs baselines & DLST-MDA on DLST-MDA data:
1:1, 1:5, 1:10 and cold-start (miRNA / drug)."""
import csv
import os
import statistics as st

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

HERE = os.path.dirname(os.path.abspath(__file__))
BL = os.path.join(HERE, '..', 'dlstmda_baselines_20260919', 'results')
OUT = os.path.join(HERE, 'DLST_MDA_strongest_ranking.xlsx')

BASE_M = ['graphdta', 'gcnnma', 'submdta', 'daestb', 'mldti']
PRETTY = {'graphdta': 'GraphDTA', 'gcnnma': 'GCNNMMA', 'submdta': 'SubMDTA',
          'daestb': 'DAESTB(leak-free)', 'mldti': 'ML-DTI'}
# paper DLST-MDA (AUC, AUPR) for ratios and cold start
PAPER_R = {5: (0.9510, 0.8026), 10: (0.9514, 0.7049)}
PAPER_C = {'mirna': (0.9394, 0.9277), 'drug': (0.7327, 0.5533)}


def means(fp, ak='auc', pk='aupr', ck='acc'):
    rows = list(csv.DictReader(open(fp)))
    return (st.mean(float(r[ak]) for r in rows),
            st.mean(float(r[pk]) for r in rows),
            st.mean(float(r[ck]) for r in rows) if ck in rows[0] else None, rows)


def board(title, entries):
    """entries: list of (label, auc, aupr, acc, note). Returns rows sorted with ranks."""
    entries.sort(key=lambda e: (-e[1], -e[2]))
    out = []
    for i, (lab, a, u, c, note) in enumerate(entries, 1):
        out.append([i, lab, a, u, c, note])
    return out


def add_sheet(wb, name, title, rows):
    ws = wb.create_sheet(name)
    ws.append([title, '方法', 'AUC', 'AUPR', 'ACC', '备注'])
    for c in ws[1]:
        c.font = Font(bold=True, color='FFFFFF')
        c.fill = PatternFill('solid', fgColor='305496')
        c.alignment = Alignment(horizontal='center', wrap_text=True)
    OUR = PatternFill('solid', fgColor='FFF2CC')
    LEAK = PatternFill('solid', fgColor='F2F2F2')
    for r in rows:
        ws.append([r[0], r[1], round(r[2], 4), round(r[3], 4),
                   round(r[4], 4) if r[4] else '—', r[5]])
        if r[1].startswith('iNcRD-HG'):
            for c in range(1, 7):
                ws.cell(ws.max_row, c).fill = OUR
        elif 'leaky' in r[1]:
            for c in range(1, 7):
                ws.cell(ws.max_row, c).fill = LEAK
    for i, w in enumerate([6, 30, 9, 9, 9, 34], 1):
        ws.column_dimensions[chr(64 + i)].width = w
    return ws


def main():
    wb = Workbook(); wb.remove(wb.active)

    # ---------- 1:1 ----------
    e1 = []
    pooled = []
    for seed in ('_per_fold', '_seed666', '_seed0', '_seed1'):
        pooled += list(csv.DictReader(open(os.path.join(HERE, 'results', f'strongest{seed}.csv'))))
    e1.append(('iNcRD-HG-strongest(本方法)',
               st.mean(float(r['fused_auc']) for r in pooled),
               st.mean(float(r['fused_aupr']) for r in pooled),
               st.mean(float(r['acc']) for r in pooled), '3seed x 5fold'))
    for m in BASE_M:
        fp = 'daestb_foldwise_per_fold.csv' if m == 'daestb' else f'{m}_per_fold.csv'
        a, u, c, _ = means(os.path.join(BL, fp))
        e1.append((PRETTY[m], a, u, c, '基线复现'))
    e1.append(('DLST-MDA(官方代码复现)', 0.9437, 0.9356, 0.8753, '论文方法'))
    e1.append(('DLST-MDA(论文报告)', 0.9411, 0.9326, 0.8736, '论文方法'))
    a, u, c, _ = means(os.path.join(BL, 'daestb_per_fold.csv'))
    e1.append(('DAESTB(transductive,leaky)', a, u, c, '转导GIP,仅参考'))
    add_sheet(wb, '1to1', '1:1 排名', board('1:1', e1))

    # ---------- ratios ----------
    for ratio in (5, 10):
        e = []
        fp = os.path.join(HERE, 'results', f'strongest_1to{ratio}.csv')
        a, u, c, _ = means(fp, 'fused_auc', 'fused_aupr', 'acc')
        e.append(('iNcRD-HG-strongest(本方法)', a, u, c, '图上下文+anchor'))
        for m in BASE_M:
            fp = os.path.join(BL, f'ratio_1to{ratio}_{m}.csv')
            a, u, c, _ = means(fp)
            e.append((PRETTY[m], a, u, c, '基线复现'))
        e.append((f'DLST-MDA(论文报告)', PAPER_R[ratio][0], PAPER_R[ratio][1], None, '论文方法'))
        add_sheet(wb, f'1to{ratio}', f'1:{ratio} 排名', board(f'1:{ratio}', e))

    # ---------- cold start ----------
    for s in ('mirna', 'drug'):
        e = []
        a, u, c, _ = means(os.path.join(HERE, 'results', f'strongest_cold_{s}.csv'),
                           'fused_auc', 'fused_aupr', 'acc')
        e.append(('iNcRD-HG-strongest(本方法)', a, u, c, '图上下文+anchor'))
        for m in BASE_M:
            fp = os.path.join(BL, f'coldstart_{s}_{m}.csv')
            a, u, c, _ = means(fp)
            e.append((PRETTY[m], a, u, c, '基线复现'))
        e.append(('DLST-MDA(论文报告)', PAPER_C[s][0], PAPER_C[s][1], None, '论文方法'))
        add_sheet(wb, f'cold_{s}', f'冷启动 {s} 排名', board(f'cold_{s}', e))

    wb.save(OUT)
    print('wrote', OUT)
    # console summary
    for sn in wb.sheetnames:
        ws = wb[sn]
        print(f'\n--- {ws.title} ---')
        for row in ws.iter_rows(min_row=2, values_only=True):
            print(f"{row[0]:>3} {str(row[1]):30s} AUC {row[2]:.4f}  AUPR {row[3]:.4f}")


if __name__ == '__main__':
    main()
