# -*- coding: utf-8 -*-
"""Summarise the reproduced DLST-MDA 1:1 baselines and compare with the paper.

Paper values: Supplementary Table 1, ratio 1:1 (DLST-MDA, JCIM 2025).
"""
import csv
import os
import statistics as st

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, 'DLST_MDA_1to1_baselines.xlsx')

# method -> (repro csv, paper Acc, MCC, Sen, Spe, AUC, AUPR)
PAPER = {
    'DAESTB':   ('daestb_per_fold.csv',   0.8447, 0.6944, 0.7954, 0.8939, 0.9279, 0.9169),
    'GCNNMMA':  ('gcnnma_per_fold.csv',   0.7615, 0.6052, 0.9842, 0.6249, 0.8630, 0.7222),
    'SubMDTA':  ('submdta_per_fold.csv',  0.8554, 0.7109, 0.8573, 0.8536, 0.9032, 0.8659),
    'GraphDTA': ('graphdta_per_fold.csv', 0.8478, 0.7119, 0.9537, 0.7419, 0.8692, 0.8692),
    'ML-DTI':   ('mldti_per_fold.csv',    0.6719, 0.4284, 0.7155, 0.6802, 0.7858, 0.7058),
    'DAESTB-leakfree': ('daestb_foldwise_per_fold.csv', 0.8447, 0.6944, 0.7954, 0.8939, 0.9279, 0.9169),
}
# DLST-MDA itself (official-code reproduction from the sibling task)
DLST = (0.8753, 0.8827, 0.9437, 0.9356, 0.8736, 0.7532, 0.9411, 0.9326)


def read(fp):
    rows = list(csv.DictReader(open(os.path.join(RES, fp))))
    return rows


def mean(rows, key):
    return st.mean(float(r[key]) for r in rows)


def main():
    order = ['GraphDTA', 'GCNNMMA', 'SubMDTA', 'DAESTB', 'DAESTB-leakfree', 'ML-DTI']
    table = []
    for m in order:
        fp, pacc, pmcc, psen, pspe, pauc, paupr = PAPER[m]
        if not os.path.exists(os.path.join(RES, fp)):
            table.append((m, None))
            continue
        rows = read(fp)
        table.append((m, dict(
            acc=mean(rows, 'acc'), mcc=mean(rows, 'mcc'), sen=mean(rows, 'recall'),
            spe=mean(rows, 'specificity'), auc=mean(rows, 'auc'), aupr=mean(rows, 'aupr'),
            n=len(rows), paper=(pacc, pmcc, psen, pspe, pauc, paupr))))

    hdr = ['方法', '数据/协议', '复现 AUC', '论文 AUC', 'ΔAUC', '复现 AUPR', '论文 AUPR',
           'ΔAUPR', '复现 ACC', '论文 ACC', '复现 F1', '论文 F1', '复现 MCC', '论文 MCC',
           '复现 Sen', '论文 Sen', '复现 Spe', '论文 Spe', '备注']
    wb = Workbook()
    ws = wb.active
    ws.title = 'DLST-MDA 1to1 基线'
    ws.append(hdr)
    HEAD = Font(bold=True, color='FFFFFF', size=10)
    HFILL = PatternFill('solid', fgColor='305496')
    OUR = PatternFill('solid', fgColor='FFF2CC')
    POS = PatternFill('solid', fgColor='C6EFCE')
    NEG = PatternFill('solid', fgColor='FFC7CE')
    thin = Side(style='thin', color='BFBFBF')
    bord = Border(left=thin, right=thin, top=thin, bottom=thin)
    cen = Alignment(horizontal='center', vertical='center')
    for c in ws[1]:
        c.font = HEAD
        c.fill = HFILL
        c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.freeze_panes = 'A2'

    def cell(r, c, v, fill=None, fmt='0.0000'):
        x = ws.cell(row=r, column=c)
        x.value = round(v, 4) if isinstance(v, float) else v
        x.number_format = fmt
        x.alignment = cen
        x.border = bord
        if fill:
            x.fill = fill
        return x

    r = 2
    for m, d in table:
        ws.cell(row=r, column=1, value=m).font = Font(bold=True)
        ws.cell(row=r, column=1).alignment = cen
        ws.cell(row=r, column=1).border = bord
        ws.cell(row=r, column=2, value='DLST-MDA benchmark, 5-fold, 正负1:1').alignment = cen
        ws.cell(row=r, column=2).border = bord
        if d is None:
            for c in range(3, 19):
                ws.cell(row=r, column=c, value='-').alignment = cen
                ws.cell(row=r, column=c).border = bord
        else:
            pacc, pmcc, psen, pspe, pauc, paupr = d['paper']
            cell(r, 3, d['auc'], OUR)
            cell(r, 4, pauc)
            cell(r, 5, d['auc'] - pauc, POS if d['auc'] - pauc >= 0 else NEG)
            cell(r, 6, d['aupr'], OUR)
            cell(r, 7, paupr)
            cell(r, 8, d['aupr'] - paupr, POS if d['aupr'] - paupr >= 0 else NEG)
            cell(r, 9, d['acc'], OUR)
            cell(r, 10, pacc)
            cell(r, 11, '-')
            cell(r, 12, '-')
            cell(r, 13, d['mcc'], OUR)
            cell(r, 14, pmcc)
            cell(r, 15, d['sen'], OUR)
            cell(r, 16, psen)
            cell(r, 17, d['spe'], OUR)
            cell(r, 18, pspe)
            note = ''
            if abs(d['auc'] - pauc) < 0.02 and abs(d['aupr'] - paupr) < 0.02:
                note = '复现命中(<0.02)'
            else:
                note = '有偏差'
            ws.cell(row=r, column=19, value=note).alignment = cen
            ws.cell(row=r, column=19).border = bord
        r += 1
    # DLST-MDA row
    ws.cell(row=r, column=1, value='DLST-MDA(参考)').font = Font(bold=True)
    ws.cell(row=r, column=1).alignment = cen
    ws.cell(row=r, column=1).border = bord
    ws.cell(row=r, column=2, value='官方代码 5 折(前序任务)').alignment = cen
    ws.cell(row=r, column=2).border = bord
    ref = [DLST[2], DLST[6], DLST[2] - DLST[6], DLST[3], DLST[7], DLST[3] - DLST[7],
           DLST[0], DLST[4], '-', '-', DLST[1], '-', DLST[5], DLST[7] * 0 + DLST[5], '-', '-', '-']
    for i, v in enumerate([DLST[2], DLST[6], DLST[2] - DLST[6], DLST[3], DLST[7], DLST[3] - DLST[7],
                           DLST[0], DLST[4]], start=3):
        cell(r, i, v, OUR if i in (3, 6, 9, 13, 15, 17) else None)
    for c in range(11, 19):
        ws.cell(row=r, column=c, value='-').alignment = cen
        ws.cell(row=r, column=c).border = bord
    ws.cell(row=r, column=19, value='官方代码复现 +0.0026/+0.0030').alignment = cen
    ws.cell(row=r, column=19).border = bord

    widths = [14, 32, 10, 10, 9, 10, 10, 9, 10, 10, 9, 9, 10, 10, 9, 9, 9, 9, 26]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + i) if i <= 26 else 'A'].width = w
    wb.save(OUT)
    print('wrote', OUT)
    print()
    print(f"{'method':10s} {'reproAUC':>9s} {'paperAUC':>9s} {'dAUC':>7s} "
          f"{'reproAUPR':>10s} {'paperAUPR':>10s} {'dAUPR':>7s}")
    for m, d in table:
        if d is None:
            print(f'{m:10s}  (missing)')
            continue
        pacc, pmcc, psen, pspe, pauc, paupr = d['paper']
        print(f'{m:10s} {d["auc"]:9.4f} {pauc:9.4f} {d["auc"]-pauc:7.4f} '
              f'{d["aupr"]:10.4f} {paupr:10.4f} {d["aupr"]-paupr:7.4f}')


if __name__ == '__main__':
    main()
