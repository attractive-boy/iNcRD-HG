# -*- coding: utf-8 -*-
"""Aggregate HGNNLDA reproduction results and build the Excel + markdown table.

Paper values transcribed from the PDF (TCBB 2023, Fig. 2 and Fig. 3).
"""
import glob
import os
import statistics as st

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, 'HGNNLDA_reproduction.xlsx')

HEAD = Font(bold=True, color='FFFFFF', size=10)
HEADF = PatternFill('solid', fgColor='305496')
OURF = PatternFill('solid', fgColor='FFF2CC')
PAPF = PatternFill('solid', fgColor='E2EFDA')
thin = Side(style='thin', color='BFBFBF')
BORD = Border(left=thin, right=thin, top=thin, bottom=thin)
CEN = Alignment(horizontal='center', vertical='center', wrap_text=True)
LEFT = Alignment(horizontal='left', vertical='center')

# ---- paper-reported values (Fig. 2 / Fig. 3) ----
PAPER_BENCH = [('HGNNLDA (论文)', 0.9325, 0.9348),
               ('MMGCN', 0.9104, 0.9051), ('LightGCN', 0.9132, 0.9134),
               ('LAGCN', 0.8984, 0.8966), ('NGCF', 0.9033, 0.8982),
               ('GANLDA', 0.8580, 0.8566), ('SDLDA', 0.8323, 0.8210),
               ('DMFCDA', 0.8273, 0.8014), ('NTSHMDA', 0.7810, 0.7312)]
PAPER_IND = [('HGNNLDA (论文)', 0.9095, 0.8897),
             ('MMGCN', 0.8938, 0.8792), ('LAGCN', 0.8821, 0.8658),
             ('GANLDA', 0.8429, 0.8351), ('SDLDA', 0.8385, 0.8221),
             ('DMFCDA', 0.8091, 0.7743), ('NTSHMDA', 0.7748, 0.7775)]
PAPER_HGNN = {'bench': (0.9325, 0.9348), 'indep': (0.9095, 0.8897)}

# project's earlier transfer runs on the SAME data (official fold 0, balanced)
OURS_FOLD0 = {'iNcRD-HG 图特征 + HistGB': (0.9307, 0.9425, 'transfer_matrix_benchmark.py'),
              'iNcRD-HG TabICLv2': (0.9017, 0.9144, 'metrics_ours_rnaactdrug.json')}

METRICS = ['bal_auc', 'bal_aupr', 'glob_auc', 'glob_aupr', 'ACC@bestF1', 'F1@bestF1', 'MCC@bestF1']


def load(pattern):
    fs = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def agg(df, cols):
    return {c: (st.mean(df[c]), st.stdev(df[c]) if len(df) > 1 else 0.0) for c in cols}


def style_header(ws, row, ncol):
    for c in range(1, ncol + 1):
        cell = ws.cell(row, c); cell.font = HEAD; cell.fill = HEADF
        cell.alignment = CEN; cell.border = BORD


def put(ws, r, vals, floats=(), left=()):
    for c, v in enumerate(vals, 1):
        cell = ws.cell(r, c)
        if isinstance(v, float):
            cell.value = round(v, 4); cell.number_format = '0.0000'
        else:
            cell.value = v
        cell.alignment = LEFT if c in left else CEN
        cell.border = BORD
    return r + 1


def main():
    folds = load('hgnnlda_seed*.csv')          # e0 (official code) variant
    iterf = load('hgnnlda_iter_seed*.csv')     # iterated-propagation variant
    ind = load('hgnnlda_independent_seed*.csv')
    fa = agg(folds, METRICS)
    ia = agg(ind, ['bal_auc', 'bal_aupr']) if ind is not None else None

    print('=== HGNNLDA RNAactDrug 5-fold CV (official-code variant "e0", 3 seeds pooled) ===')
    for c in METRICS:
        print(f'  {c:12s} {fa[c][0]:.4f} +/- {fa[c][1]:.4f}')
    print(f'  paper       bal_auc  {PAPER_HGNN["bench"][0]:.4f}   bal_aupr {PAPER_HGNN["bench"][1]:.4f}')
    print(f'  ΔAUC {fa["bal_auc"][0]-PAPER_HGNN["bench"][0]:+.4f}  ΔAUPR {fa["bal_aupr"][0]-PAPER_HGNN["bench"][1]:+.4f}')
    if iterf is not None:
        ia2 = agg(iterf, ['bal_auc', 'bal_aupr'])
        print(f'  iter variant: {ia2["bal_auc"][0]:.4f}/{ia2["bal_aupr"][0]:.4f}')
    if ia is not None:
        print(f'\n=== independent ===\n  {ia["bal_auc"][0]:.4f}/{ia["bal_aupr"][0]:.4f}  '
              f'paper {PAPER_HGNN["indep"][0]:.4f}/{PAPER_HGNN["indep"][1]:.4f}  '
              f'ΔAUC {ia["bal_auc"][0]-PAPER_HGNN["indep"][0]:+.4f}  ΔAUPR {ia["bal_aupr"][0]-PAPER_HGNN["indep"][1]:+.4f}')

    wb = Workbook(); ws0 = wb.active; ws0.title = '说明'
    notes = [
        ('HGNNLDA 基线复现 (TCBB 2023)', ''),
        ('论文', 'Dayun Liu et al. "HGNNLDA: Predicting lncRNA-Drug Sensitivity Associations via a Dual Channel Hypergraph Neural Network", IEEE/ACM TCBB 2023, 3547-3555. DOI 10.1109/TCBB.2023.3302468'),
        ('论文报告值', 'benchmark 5 折 AUC 0.9325 / AUPR 0.9348 (Fig.3a/b); 独立集 AUC 0.9095 / AUPR 0.8897 (Fig.3c/d)'),
        ('官方代码', 'github.com/dayunliu/HGNNLDA (TF1.15, 只含训练无评测)'),
        ('复现实现', 'PyTorch 移植, 照抄 main.py + HGNNLDA.conf: emb=64, 3 层超图卷积, BPR pairwise, lr=0.005, iter=100, batch=2048, reg_u=1e-4'),
        ('复现要点', 'e0 变体: 逐层 message 恒为 A@E0 (官方写法, 主结果); iter 变体: message 用上一层嵌入 (论文 Eq.10-11)'),
        ('数据', 'RNAactDrug lncRNA-drug 敏感性, 978 lncRNA x 1815 drug, 36428 正例; 官方 5 折 + 独立集 2093'),
        ('评测协议', '论文: 全关联随机分 5 折, 1 折测试 4 折训练; 负例 = 等量随机未知对 (1:1, 论文取 36248 个)'),
        ('复现协议', 'balanced: 每折测试正例 + 等量随机未知负例; global: 测试正例 vs 全部未知对'),
        ('结论', 'e0 变体 benchmark 0.9241/0.9319 (ΔAUC -0.0084, ΔAUPR -0.0029), 独立集 0.9056/0.8898 (ΔAUPR +0.0001); 与论文高度吻合'),
        ('同数据本文模型', 'iNcRD-HG 图特征+HistGB (官方 fold0, balanced) 0.9307/0.9425; TabICLv2 0.9017/0.9144'),
    ]
    for i, (a, b) in enumerate(notes, 1):
        ws0.cell(i, 1, a).font = Font(bold=True, size=10); ws0.cell(i, 1).alignment = LEFT
        ws0.cell(i, 2, b).alignment = LEFT
    ws0.column_dimensions['A'].width = 16; ws0.column_dimensions['B'].width = 115

    # ---- 5-fold sheet ----
    ws = wb.create_sheet('5fold_CV')
    ws.append(['HGNNLDA 复现 — RNAactDrug 官方 5 折交叉验证'])
    ws.cell(1, 1).font = Font(bold=True, size=11)
    hdr = ['variant', 'seed', 'fold', 'balanced AUC', 'balanced AUPR', 'global AUC', 'global AUPR',
           'ACC@bestF1', 'F1@bestF1', 'MCC@bestF1', 'n_train', 'n_test']
    ws.append(hdr); style_header(ws, 2, len(hdr))
    r = 3
    for _, row in folds.sort_values(['seed', 'fold']).iterrows():
        r = put(ws, r, ['e0(官方代码)', int(row.seed), int(row.fold), row.bal_auc, row.bal_aupr,
                        row.glob_auc, row.glob_aupr, row['ACC@bestF1'], row['F1@bestF1'],
                        row['MCC@bestF1'], int(row.n_train), int(row.n_test)])
    r = put(ws, r, ['e0  mean', '', '', fa['bal_auc'][0], fa['bal_aupr'][0], fa['glob_auc'][0],
                    fa['glob_aupr'][0], fa['ACC@bestF1'][0], fa['F1@bestF1'][0], fa['MCC@bestF1'][0], '', ''])
    r = put(ws, r, ['e0  std', '', '', fa['bal_auc'][1], fa['bal_aupr'][1], fa['glob_auc'][1],
                    fa['glob_aupr'][1], fa['ACC@bestF1'][1], fa['F1@bestF1'][1], fa['MCC@bestF1'][1], '', ''])
    for rr in (r - 2, r - 1):
        for c in range(1, len(hdr) + 1):
            ws.cell(rr, c).fill = OURF; ws.cell(rr, c).font = Font(bold=True, size=10)
    if iterf is not None:
        for _, row in iterf.sort_values(['seed', 'fold']).iterrows():
            r = put(ws, r, ['iter(论文式)', int(row.seed), int(row.fold), row.bal_auc, row.bal_aupr,
                            row.glob_auc, row.glob_aupr, row['ACC@bestF1'], row['F1@bestF1'],
                            row['MCC@bestF1'], int(row.n_train), int(row.n_test)])
    r = put(ws, r, ['论文报告', '', '', PAPER_HGNN['bench'][0], PAPER_HGNN['bench'][1], '', '', '', '', '', '', ''])
    for c in range(1, len(hdr) + 1):
        ws.cell(r - 1, c).fill = PAPF; ws.cell(r - 1, c).font = Font(bold=True, size=10)
    for i, w in enumerate([13, 7, 7, 13, 13, 12, 12, 12, 11, 11, 9, 8], 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    # ---- independent ----
    if ind is not None:
        wsi = wb.create_sheet('independent')
        hdr = ['seed', 'n_test', 'balanced AUC', 'balanced AUPR', 'ACC@bestF1', 'F1@bestF1', 'MCC@bestF1']
        wsi.append(hdr); style_header(wsi, 1, len(hdr))
        rr = 2
        for _, row in ind.iterrows():
            rr = put(wsi, rr, [int(row.seed), int(row.n_test), row.bal_auc, row.bal_aupr,
                               row['ACC@bestF1'], row['F1@bestF1'], row['MCC@bestF1']])
        rr = put(wsi, rr, ['论文报告', 2093, PAPER_HGNN['indep'][0], PAPER_HGNN['indep'][1], '', '', ''])
        for c in range(1, len(hdr) + 1):
            wsi.cell(rr - 1, c).fill = PAPF; wsi.cell(rr - 1, c).font = Font(bold=True, size=10)
        for i, w in enumerate([10, 8, 13, 13, 12, 11, 11], 1):
            wsi.column_dimensions[get_column_letter(i)].width = w

    # ---- comparison ----
    wsc = wb.create_sheet('对比')
    wsc.append(['RNAactDrug — HGNNLDA 复现 vs 论文 & 本文模型 (balanced)'])
    wsc.cell(1, 1).font = Font(bold=True, size=11)
    r = 3
    wsc.cell(r, 1, 'A. benchmark 5 折 CV'); wsc.cell(r, 1).font = Font(bold=True, size=10); r += 1
    hdr = ['方法', 'AUC', 'AUPR', '来源']
    for c, v in enumerate(hdr, 1):
        wsc.cell(r, c, v)
    style_header(wsc, r, len(hdr)); r += 1
    r = put(wsc, r, ['HGNNLDA (本文复现, e0, 5 折均值)', fa['bal_auc'][0], fa['bal_aupr'][0], 'run_hgnnlda.py'])
    for rr in (r - 1,):
        for c in range(1, 5):
            wsc.cell(rr, c).fill = OURF; wsc.cell(rr, c).font = Font(bold=True, size=10)
    if iterf is not None:
        ia2 = agg(iterf, ['bal_auc', 'bal_aupr'])
        r = put(wsc, r, ['HGNNLDA (本文复现, iter 变体)', ia2['bal_auc'][0], ia2['bal_aupr'][0], 'run_hgnnlda.py'])
    for m, a, u in PAPER_BENCH:
        r = put(wsc, r, [m, a, u, '论文 Fig.3a/b' if m.startswith('HGNNLDA') else '论文 Fig.2/3a'])
        if m.startswith('HGNNLDA'):
            for c in range(1, 5):
                wsc.cell(r - 1, c).fill = PAPF; wsc.cell(r - 1, c).font = Font(bold=True, size=10)
    r += 1
    wsc.cell(r, 1, 'B. 独立数据集'); wsc.cell(r, 1).font = Font(bold=True, size=10); r += 1
    for c, v in enumerate(hdr, 1):
        wsc.cell(r, c, v)
    style_header(wsc, r, len(hdr)); r += 1
    if ia is not None:
        r = put(wsc, r, ['HGNNLDA (本文复现)', ia['bal_auc'][0], ia['bal_aupr'][0], 'run_hgnnlda_independent.py'])
        for c in range(1, 5):
            wsc.cell(r - 1, c).fill = OURF; wsc.cell(r - 1, c).font = Font(bold=True, size=10)
    for m, a, u in PAPER_IND:
        r = put(wsc, r, [m, a, u, '论文 Fig.3c/d'])
        if m.startswith('HGNNLDA'):
            for c in range(1, 5):
                wsc.cell(r - 1, c).fill = PAPF; wsc.cell(r - 1, c).font = Font(bold=True, size=10)
    r += 1
    wsc.cell(r, 1, 'C. 同数据本文模型 (官方 fold0, balanced)'); wsc.cell(r, 1).font = Font(bold=True, size=10); r += 1
    for c, v in enumerate(hdr, 1):
        wsc.cell(r, c, v)
    style_header(wsc, r, len(hdr)); r += 1
    for k, (a, u, src) in OURS_FOLD0.items():
        r = put(wsc, r, [k, a, u, src])
    for i, w in enumerate([36, 9, 9, 20], 1):
        wsc.column_dimensions[get_column_letter(i)].width = w

    wb.save(OUT)
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
