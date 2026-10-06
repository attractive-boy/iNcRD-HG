# -*- coding: utf-8 -*-
"""Summarise the cold-start reproductions and compare with Supp. Table 2.

NOTE: the paper does not release its cold-start split, so our splits are a documented
surrogate (see make_coldstart_splits.py); absolute values are therefore only indicative.
"""
import csv
import os
import statistics as st

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, 'DLST_MDA_coldstart_baselines.xlsx')

# Supp. Table 2: setting -> method -> (Acc, MCC, Sen, Spe, AUC, AUPR)
PAPER = {
    'mirna': {
        'DAESTB':   (0.8329, 0.7668, 0.8544, 0.8121, 0.8795, 0.8745),
        'GCNNMMA':  (0.8585, 0.7169, 0.8687, 0.8498, 0.9208, 0.8857),
        'SubMDTA':  (0.8437, 0.6880, 0.8177, 0.8640, 0.8960, 0.8624),
        'GraphDTA': (0.8671, 0.7364, 0.9025, 0.8328, 0.9223, 0.8945),
        'ML-DTI':   (0.8484, 0.6967, 0.8429, 0.8537, 0.9044, 0.8687),
    },
    'drug': {
        'DAESTB':   (0.5828, 0.3515, 0.3012, 0.8348, 0.6962, 0.5270),
        'GCNNMMA':  (0.6281, 0.2166, 0.3730, 0.8454, 0.7210, 0.5455),
        'SubMDTA':  (0.6620, 0.2598, 0.4331, 0.8077, 0.6879, 0.5363),
        'GraphDTA': (0.6357, 0.1736, 0.2691, 0.8691, 0.6992, 0.5307),
        'ML-DTI':   (0.6891, 0.3166, 0.4253, 0.8569, 0.6668, 0.5494),
    },
}
FILE = {
    'GraphDTA': 'coldstart_{s}_graphdta.csv',
    'GCNNMMA': 'coldstart_{s}_gcnnma.csv',
    'SubMDTA': 'coldstart_{s}_submdta.csv',
    'DAESTB': 'coldstart_{s}_daestb.csv',
    'ML-DTI': 'coldstart_{s}_mldti.csv',
}


def mean(rows, k):
    return st.mean(float(x[k]) for x in rows)


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = 'Coldstart'
    hdr = ['冷启动', '方法', '复现 AUC', '论文 AUC', 'ΔAUC', '复现 AUPR', '论文 AUPR',
           'ΔAUPR', '复现 ACC', '论文 ACC', '复现 Sen', '论文 Sen', '复现 Spe', '论文 Spe',
           '复现 MCC', '论文 MCC']
    ws.append(hdr)
    HF = Font(bold=True, color='FFFFFF')
    HFILL = PatternFill('solid', fgColor='305496')
    OUR = PatternFill('solid', fgColor='FFF2CC')
    for c in ws[1]:
        c.font = HF
        c.fill = HFILL
        c.alignment = Alignment(horizontal='center', wrap_text=True)
    r = 2
    print(f"{'setting':7s} {'method':10s} {'repAUC':>7s} {'papAUC':>7s} {'repAUPR':>8s} {'papAUPR':>8s}")
    for s in ('mirna', 'drug'):
        for m in ['GraphDTA', 'GCNNMMA', 'SubMDTA', 'DAESTB', 'ML-DTI']:
            fp = os.path.join(RES, FILE[m].format(s=s))
            if not os.path.exists(fp):
                ws.append([s, m] + ['-'] * 14)
                r += 1
                continue
            rows = list(csv.DictReader(open(fp)))
            pacc, pmcc, psen, pspe, pauc, paupr = PAPER[s][m]
            ra, ru = mean(rows, 'auc'), mean(rows, 'aupr')
            ws.append([s, m, round(ra, 4), pauc, round(ra - pauc, 4), round(ru, 4), paupr,
                       round(ru - paupr, 4), round(mean(rows, 'acc'), 4), pacc,
                       round(mean(rows, 'recall'), 4), psen,
                       round(mean(rows, 'specificity'), 4), pspe,
                       round(mean(rows, 'mcc'), 4), pmcc])
            for c in range(3, 17):
                ws.cell(r, c).alignment = Alignment(horizontal='center')
            ws.cell(r, 3).fill = OUR
            ws.cell(r, 6).fill = OUR
            print(f'{s:7s} {m:10s} {ra:7.4f} {pauc:7.4f} {ru:8.4f} {paupr:8.4f}')
            r += 1
    for i, w in enumerate([9, 12, 10, 10, 8, 11, 11, 8, 10, 10, 10, 10, 10, 10, 10, 10], 1):
        ws.column_dimensions[chr(64 + i)].width = w
    wb.save(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
