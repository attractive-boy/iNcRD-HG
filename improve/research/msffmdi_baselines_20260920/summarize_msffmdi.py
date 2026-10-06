# -*- coding: utf-8 -*-
"""Summarise MSFFMDI baseline reproduction vs the paper's Tables 3/4."""
import glob
import os
import statistics as st

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, 'MSFFMDI_baselines.xlsx')

# MSFFMDI paper Tables 3 (ncDR) and 4 (RNAInter): AUC, AUPR, ACC, Sen, Spec
PAPER = {
    'ncDR': {
        'LAGCN': (0.8911, 0.8489, 0.9194, 0.8039, 0.8332),
        'MFIDMA': (0.9454, 0.9276, 0.8840, 0.8845, 0.8835),
        'GCNNMMA': (0.9123, 0.8608, 0.8477, 0.8081, 0.8607),
        'LDAMAN': (0.9389, 0.9199, 0.8583, 0.7848, 0.8317),
        'DLST-MDA': (0.9446, 0.9255, 0.8712, 0.9144, 0.8303),
        'MSFFMDI': (0.9483, 0.9461, 0.8834, 0.8890, 0.8863),
    },
    'RNAInter': {
        'LAGCN': (0.8337, 0.7360, 0.7831, 0.7238, 0.7425),
        'MFIDMA': (0.9168, 0.9074, 0.8414, 0.9059, 0.7830),
        'GCNNMMA': (0.9116, 0.8988, 0.8424, 0.8184, 0.8500),
        'LDAMAN': (0.9081, 0.8989, 0.8219, 0.9043, 0.8294),
        'DLST-MDA': (0.9143, 0.9036, 0.8778, 0.9109, 0.8449),
        'MSFFMDI': (0.9216, 0.9163, 0.8524, 0.8754, 0.8516),
    },
}
ORDER = ['LAGCN', 'GCNNMMA', 'MFIDMA', 'LDAMAN', 'DLST-MDA']

HEAD = Font(bold=True, color='FFFFFF', size=10)
HEADF = PatternFill('solid', fgColor='305496')
OURF = PatternFill('solid', fgColor='FFF2CC')
POS = PatternFill('solid', fgColor='C6EFCE')
NEG = PatternFill('solid', fgColor='FFC7CE')
thin = Side(style='thin', color='BFBFBF')
BORD = Border(left=thin, right=thin, top=thin, bottom=thin)
CEN = Alignment(horizontal='center', vertical='center', wrap_text=True)
LEFT = Alignment(horizontal='left', vertical='center')


def load(dataset, method):
    key = {'LAGCN': 'lagcn', 'GCNNMMA': 'gcnnma', 'MFIDMA': 'mfidma',
           'LDAMAN': 'ldaman', 'DLST-MDA': 'dlstmda'}[method]
    fp = os.path.join(RESULTS, f'{key}_{dataset}_seed42.csv')
    if not os.path.exists(fp):
        return None
    return pd.read_csv(fp)


def main():
    wb = Workbook()
    ws0 = wb.active; ws0.title = '说明'
    notes = [
        ('MSFFMDI 论文对比基线复现', ''),
        ('论文', 'Lei et al., Interdiscip. Sci. Comput. Life Sci. 18(3):986-1009, 2026 (DOI 10.1007/s12539-025-00775-7)'),
        ('论文对比基线', 'LAGCN / GCNNMMA / MFIDMA / LDAMAN / DLST-MDA（论文 Sect.3.4, Tables 3/4）'),
        ('数据集', 'ncDR (624 miRNA x 95 drug, 4457 关联) / RNAInter (1009 x 281, 5740)，来自 MFIDMA 官方仓库'),
        ('协议', '正负 1:1，5 折 CV（clean 划分；见报告协议说明），seed 42，指标 AUC/AUPR/ACC/Sen/Spec'),
        ('模型选择', '测试集最优 epoch（本项目统一口径），另存最后一轮'),
        ('Δ 定义', 'Δ = 复现 − 论文；绿色为正、红色为负'),
        ('重要说明', 'MFIDMA 仓库的每折 test 集合完全相同且含重复/训练-测试重叠的正例，无法支持真正 5-CV；本复现改用去重后的正例 + 等量随机负例做 clean 5-CV，故绝对数值不与论文严格可比'),
    ]
    for i, (a, b) in enumerate(notes, 1):
        ws0.cell(i, 1, a).font = Font(bold=True, size=10)
        ws0.cell(i, 1).alignment = LEFT
        ws0.cell(i, 2, b).alignment = LEFT
    ws0.column_dimensions['A'].width = 14
    ws0.column_dimensions['B'].width = 100

    for ds in ('ncDR', 'RNAInter'):
        ws = wb.create_sheet(ds)
        hdr = ['方法', '复现AUC', '论文AUC', 'ΔAUC', '复现AUPR', '论文AUPR', 'ΔAUPR',
               '复现ACC', '论文ACC', '复现Sen', '论文Sen', '复现Spec', '论文Spec']
        ws.append([f'{ds}dataset — MSFFMDI 论文对比基线复现（5 折均值, seed42）'])
        ws.cell(1, 1).font = Font(bold=True, size=11)
        ws.append(hdr)
        for c in range(1, len(hdr) + 1):
            cell = ws.cell(2, c); cell.font = HEAD; cell.fill = HEADF
            cell.alignment = CEN; cell.border = BORD
        ws.freeze_panes = 'A3'
        rows = []
        for m in ORDER:
            df = load(ds, m)
            if df is None:
                continue
            pa, pp, pacc, psen, pspec = PAPER[ds][m]
            rows.append((m, df.auc.mean(), pa, df.aupr.mean(), pp, df.acc.mean(), pacc,
                         df.sen.mean(), psen, df.spec.mean(), pspec))
        rows.sort(key=lambda r: -r[1])
        r = 3
        for (m, a, pa, u, pp, acc, pacc, sen, psen, spec, pspec) in rows:
            vals = [m, a, pa, a - pa, u, pp, u - pp, acc, pacc, sen, psen, spec, pspec]
            for c, v in enumerate(vals, 1):
                cell = ws.cell(r, c)
                if isinstance(v, float):
                    cell.value = round(v, 4); cell.number_format = '0.0000'
                else:
                    cell.value = v
                cell.alignment = LEFT if c == 1 else CEN
                cell.border = BORD
                if c in (4, 7) and isinstance(v, float):
                    cell.fill = POS if v >= 0 else NEG
            r += 1
        # paper MSFFMDI row for reference
        pa, pp, pacc, psen, pspec = PAPER[ds]['MSFFMDI']
        vals = ['MSFFMDI (论文方法)', None, pa, None, None, pp, None, None, pacc, None, psen, None, pspec]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(r, c)
            if isinstance(v, float):
                cell.value = round(v, 4); cell.number_format = '0.0000'
            else:
                cell.value = v
            cell.alignment = LEFT if c == 1 else CEN
            cell.border = BORD
            cell.fill = OURF
        for i, w in enumerate([24, 9, 9, 8, 9, 9, 8, 9, 9, 9, 9, 9, 9], 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        print(f'\n=== {ds} ===')
        for (m, a, pa, u, pp, acc, pacc, sen, psen, spec, pspec) in rows:
            print(f'{m:10s} AUC {a:.4f} (paper {pa:.4f}, Δ{a-pa:+.4f})  AUPR {u:.4f} (paper {pp:.4f}, Δ{u-pp:+.4f})')
    wb.save(OUT)
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
