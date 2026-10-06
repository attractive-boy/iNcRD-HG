# -*- coding: utf-8 -*-
"""Compile the HGNNLDA paper's experiment tables (TCBB 2023) into a clean workbook.

All numbers are transcribed from the paper (Fig.2 / Fig.3 / Table 1 / Table 2).
Our own reproduction is added as a reference row, clearly marked.
"""
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'HGNNLDA_paper_tables.xlsx')

HEAD = Font(bold=True, color='FFFFFF', size=10)
HEADF = PatternFill('solid', fgColor='305496')
PAPF = PatternFill('solid', fgColor='E2EFDA')
OURF = PatternFill('solid', fgColor='FFF2CC')
thin = Side(style='thin', color='BFBFBF')
BORD = Border(left=thin, right=thin, top=thin, bottom=thin)
CEN = Alignment(horizontal='center', vertical='center', wrap_text=True)
LEFT = Alignment(horizontal='left', vertical='center')

# ---- Fig.2 (recommender GNNs) + Fig.3a/b (six related models) : benchmark 5-fold ----
BENCH = [  # method, AUC, AUPR, source
    ('HGNNLDA', 0.9325, 0.9348, 'Fig.3a/b'),
    ('LightGCN', 0.9132, 0.9134, 'Fig.2'),
    ('MMGCN', 0.9104, 0.9051, 'Fig.3a/b'),
    ('NGCF', 0.9033, 0.8982, 'Fig.2'),
    ('LAGCN', 0.8984, 0.8966, 'Fig.3a/b'),
    ('GANLDA', 0.8580, 0.8566, 'Fig.3a/b'),
    ('SDLDA', 0.8323, 0.8210, 'Fig.3a/b'),
    ('DMFCDA', 0.8273, 0.8014, 'Fig.3a/b'),
    ('NTSHMDA', 0.7810, 0.7312, 'Fig.3a/b'),
]
# ---- Fig.3c/d : independent test set ----
INDEP = [
    ('HGNNLDA', 0.9095, 0.8897, 'Fig.3c/d'),
    ('MMGCN', 0.8938, 0.8792, 'Fig.3c/d'),
    ('LAGCN', 0.8821, 0.8658, 'Fig.3c/d'),
    ('GANLDA', 0.8429, 0.8351, 'Fig.3c/d'),
    ('SDLDA', 0.8385, 0.8221, 'Fig.3c/d'),
    ('DMFCDA', 0.8091, 0.7743, 'Fig.3c/d'),
    ('NTSHMDA', 0.7748, 0.7775, 'Fig.3c/d'),
]
# ---- Table 1 / Table 2 : case studies ----
KALAMYCIN = [('PDCD4-AS1', 'Unconfirmed'), ('GS1-24F4.2', 'Unconfirmed'), ('LOXL1-AS1', '32449981'),
             ('LINC00477', 'Unconfirmed'), ('RHPN1-AS1', '34917132'), ('MCM3AP-AS1', '32678686'),
             ('LINC00937', '33376751'), ('EMX2OS', 'Unconfirmed'), ('MIR210HG', '34897892'),
             ('EXTL3-AS1', 'Unconfirmed'), ('GATA3-AS1', '34358678'), ('ZBED3-AS1', '29749482'),
             ('TOB1-AS1', '31482275'), ('MIR31HG', '30125988'), ('CACNA1G-AS1', '34238292')]
BULLETYANIN = [('LINC00710', 'Unconfirmed'), ('LINC00592', 'Unconfirmed'), ('LY86-AS1', '32854616'),
               ('THRB-AS1', 'Unconfirmed'), ('LINC01016', '31856144'), ('GS1-24F4.2', 'Unconfirmed'),
               ('ERVH48-1', 'Unconfirmed'), ('WT1-AS', '30468780'), ('NR2F1-AS1', '34128858'),
               ('HOXA-AS2', '33174119'), ('LOXL1-AS1', '32449981'), ('LINC00312', '34336850'),
               ('IGSF11-AS1', 'Unconfirmed'), ('UBL7-AS1', 'Unconfirmed'), ('PRKCQ-AS1', 'Unconfirmed')]
PARAMS = [
    ('高阶连通深度 (Fig.4a)', '1, 2, 3, 4', '2 阶最优（原文：the optimal value when using the second-order connectivity）'),
    ('嵌入维度 (Fig.4b)', '16, 32, 64, 128, 256', '64 最优'),
    ('网络层数 (Fig.4c)', '1, 2, 3, 4', '3 层最优'),
]
REPRO = [('HGNNLDA (本文复现, e0 官方代码写法, 5 折均值)', 0.9241, 0.9319),
         ('HGNNLDA (本文复现, iter 论文公式写法, 5 折均值)', 0.9222, 0.9298)]
REPRO_IND = [('HGNNLDA (本文复现, 独立集)', 0.9056, 0.8898)]


def style_header(ws, row, ncol):
    for c in range(1, ncol + 1):
        cell = ws.cell(row, c); cell.font = HEAD; cell.fill = HEADF
        cell.alignment = CEN; cell.border = BORD


def put(ws, r, vals, left=()):
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
    wb = Workbook(); ws0 = wb.active; ws0.title = '说明'
    notes = [
        ('HGNNLDA 论文实验数据整理', ''),
        ('论文', 'Dayun Liu et al. "HGNNLDA: Predicting lncRNA-Drug Sensitivity Associations via a Dual Channel Hypergraph Neural Network", IEEE/ACM TCBB 2023, 3547-3555. DOI 10.1109/TCBB.2023.3302468'),
        ('数据', 'RNAactDrug; benchmark 36,248 关联 / 978 lncRNA / 1,815 drug; 负例 1:1 随机未知对; 独立集 2,093 关联 (273 lncRNA / 480 drug)'),
        ('协议', '全部关联随机分 5 折, 1 折测试 / 4 折训练; 指标 AUC / AUPR; 独立集用全部 benchmark 训练'),
        ('表说明', 'benchmark=Fig.2(推荐 GNN)+Fig.3a/b(6 个相关方法); 独立集=Fig.3c/d; 案例研究=Table 1/2'),
        ('数值出处', 'Fig.3b 图例 HGNNLDA AUPR = 0.9348（正文 3.3 处写作 0.9328，以图例为准）'),
        ('本文复现', 'PyTorch 忠实移植官方 main.py（TF1.15 无评测脚本）; benchmark 0.9241/0.9319, 独立集 0.9056/0.8898'),
    ]
    for i, (a, b) in enumerate(notes, 1):
        ws0.cell(i, 1, a).font = Font(bold=True, size=10); ws0.cell(i, 1).alignment = LEFT
        ws0.cell(i, 2, b).alignment = LEFT
    ws0.column_dimensions['A'].width = 14; ws0.column_dimensions['B'].width = 120

    for title, data, repro, sheet in [
            ('benchmark 5 折交叉验证（论文 Fig.2 / Fig.3a,b）', BENCH, REPRO, 'benchmark_5fold'),
            ('独立测试集（论文 Fig.3c,d）', INDEP, REPRO_IND, 'independent')]:
        ws = wb.create_sheet(sheet)
        ws.append([title]); ws.cell(1, 1).font = Font(bold=True, size=11)
        hdr = ['排名', '方法', 'AUC', 'AUPR', '来源']
        ws.append(hdr); style_header(ws, 2, len(hdr))
        r = 3
        for i, (m, a, u, src) in enumerate(data, 1):
            r = put(ws, r, [i, m, a, u, src], left=(2, 5))
            if m == 'HGNNLDA':
                for c in range(1, len(hdr) + 1):
                    ws.cell(r - 1, c).fill = PAPF; ws.cell(r - 1, c).font = Font(bold=True, size=10)
        r += 1
        ws.cell(r, 1, '本文复现（参考）').font = Font(bold=True, size=10); r += 1
        for m, a, u in repro:
            r = put(ws, r, ['', m, a, u, 'run_hgnnlda.py'], left=(2, 5))
            for c in range(1, len(hdr) + 1):
                ws.cell(r - 1, c).fill = OURF
        for i, w in enumerate([6, 44, 9, 9, 12], 1):
            ws.column_dimensions[get_column_letter(i)].width = w

    # ---- our strongest model transferred onto this dataset ----
    ws = wb.create_sheet('本文模型代入')
    ws.append(['本文最强模型代入 HGNNLDA 数据集（RNAactDrug，官方 5 折，balanced，AUC/AUPR）'])
    ws.cell(1, 1).font = Font(bold=True, size=11)
    hdr = ['方法', 'AUC', 'AUPR', '说明']
    ws.append(hdr); style_header(ws, 2, len(hdr))
    rows = [
        ('HGNNLDA (论文)', 0.9325, 0.9348, '论文 Fig.3a/b'),
        ('本文最强模型：图上下文特征 + HistGB', 0.9310, 0.9422, '本文最好；5 折均值（±0.002）'),
        ('本文：TabICLv2 参考（单模型）', 0.8877, 0.9030, '5 折均值'),
        ('本文：TabICLv2 + anchor 融合', 0.8806, 0.8981, 'anchor 成员在此数据上更强，融合反而拉低'),
    ]
    r = 3
    for m, a, u, note in rows:
        r = put(ws, r, [m, a, u, note], left=(1, 4))
        if m.startswith('HGNNLDA'):
            for c in range(1, len(hdr) + 1):
                ws.cell(r - 1, c).fill = PAPF; ws.cell(r - 1, c).font = Font(bold=True, size=10)
        elif 'HistGB' in m:
            for c in range(1, len(hdr) + 1):
                ws.cell(r - 1, c).fill = OURF; ws.cell(r - 1, c).font = Font(bold=True, size=10)
    for i, w in enumerate([38, 9, 9, 48], 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws = wb.create_sheet('案例研究')
    ws.append(['论文 Table 1/2：两个药物的 top-15 预测 lncRNA 及文献验证']); ws.cell(1, 1).font = Font(bold=True, size=11)
    ws.append(['Table 1 — kalamycin（去掉 26 个已知关联，952 候选；top15 中 10 个被文献验证）'])
    ws.cell(2, 1).font = Font(bold=True, size=10)
    hdr = ['#', 'kalamycin lncRNA', 'PubMed', '', '#', 'bulletyanin lncRNA', 'PubMed']
    ws.append(hdr); style_header(ws, 3, len(hdr))
    r = 4
    for i in range(15):
        r = put(ws, r, [i + 1, KALAMYCIN[i][0], KALAMYCIN[i][1], '',
                        i + 1, BULLETYANIN[i][0], BULLETYANIN[i][1]], left=(2, 6))
    ws.append([])
    ws.append(['Table 2 — bulletyanin（去掉 43 个已知关联，1772 候选；top15 中 7 个被文献验证）'])
    for i, w in enumerate([5, 16, 12, 3, 5, 16, 12], 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws = wb.create_sheet('参数敏感性')
    ws.append(['论文 Fig.4：参数敏感性（原文结论）']); ws.cell(1, 1).font = Font(bold=True, size=11)
    ws.append(['参数', '取值范围', '论文结论']); style_header(ws, 2, 3)
    r = 3
    for p, rng, concl in PARAMS:
        r = put(ws, r, [p, rng, concl], left=(1, 2, 3))
    for i, w in enumerate([22, 22, 70], 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    wb.save(OUT)
    print('wrote', OUT)
    print('\nbenchmark:', [(m, a, u) for m, a, u, _ in BENCH])
    print('independent:', [(m, a, u) for m, a, u, _ in INDEP])


if __name__ == '__main__':
    main()
