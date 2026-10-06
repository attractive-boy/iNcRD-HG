# -*- coding: utf-8 -*-
"""Summarise the PLMF-MDA Fig-3 baseline reproduction into an Excel workbook."""
import glob
import os
import statistics as st

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, 'PLMF_MDA_baselines.xlsx')

# Paper Fig 3 (legend values printed in each panel) -------------------------------
PAPER = {
    ('MDR', 'PLMF-MDA'): (0.9227, 0.9076),
    ('MDR', 'GCNNMMA'): (0.9070, 0.8736),
    ('MDR', 'SubMDTA'): (0.8936, 0.8514),
    ('MDR', 'GraphDTA'): (0.8086, 0.7979),
    ('MDR', 'ML-DTI'): (0.7660, 0.7870),
    ('MDS', 'PLMF-MDA'): (0.9301, 0.9207),
    ('MDS', 'GCNNMMA'): (0.9143, 0.8889),
    ('MDS', 'SubMDTA'): (0.8858, 0.8587),
    ('MDS', 'GraphDTA'): (0.7864, 0.7911),
    ('MDS', 'ML-DTI'): (0.7820, 0.7881),
}
# paper main-text PLMF-MDA values (slightly different from the Fig 3 legend)
PAPER_TEXT = {('MDR', 'PLMF-MDA'): (0.9222, 0.9062), ('MDS', 'PLMF-MDA'): (0.9301, 0.9207)}
METHOD_ORDER = ['PLMF-MDA', 'GCNNMMA', 'SubMDTA', 'GraphDTA', 'ML-DTI']


def load_rows():
    frames = []
    for fp in glob.glob(os.path.join(RESULTS, '*.csv')):
        if os.path.basename(fp).startswith('_'):
            continue
        frames.append(pd.read_csv(fp))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def agg(rows, dataset, split, method):
    g = rows[(rows.dataset == dataset) & (rows.split == split) & (rows.method == method)]
    if len(g) == 0:
        return None
    return dict(n=len(g), auc=st.mean(g.auc), auc_sd=st.stdev(g.auc) if len(g) > 1 else 0.0,
                aupr=st.mean(g.aupr), aupr_sd=st.stdev(g.aupr) if len(g) > 1 else 0.0,
                acc=st.mean(g.acc), f1=st.mean(g.f1), mcc=st.mean(g.mcc),
                last_auc=st.mean(g.last_auc), last_aupr=st.mean(g.last_aupr))


def main():
    rows = load_rows()
    if rows.empty:
        print('no result csv found'); return
    print(rows.groupby(['dataset', 'split', 'method']).size())

    tables = {}
    for split in ['5cv', 'single']:
        recs = []
        for ds in ['MDR', 'MDS']:
            for m in METHOD_ORDER:
                a = agg(rows, ds, split, m)
                if a is None:
                    continue
                pa, pp = PAPER[(ds, m)]
                recs.append(dict(dataset=ds, split=split, method=m,
                                 auc=a['auc'], auc_sd=a['auc_sd'], aupr=a['aupr'], aupr_sd=a['aupr_sd'],
                                 acc=a['acc'], f1=a['f1'], mcc=a['mcc'],
                                 paper_auc=pa, paper_aupr=pp,
                                 d_auc=a['auc'] - pa, d_aupr=a['aupr'] - pp, n=a['n']))
        tables[split] = pd.DataFrame(recs)

    with pd.ExcelWriter(OUT, engine='openpyxl') as xl:
        for split, df in tables.items():
            df.to_excel(xl, sheet_name=f'baselines_{split}', index=False)
        # protocol comparison for PLMF-MDA itself
        ref = []
        for ds in ['MDR', 'MDS']:
            for split in ['5cv', 'single']:
                a = agg(rows, ds, split, 'PLMF-MDA')
                if a is None:
                    continue
                ref.append(dict(dataset=ds, split=split, auc=a['auc'], auc_sd=a['auc_sd'],
                                aupr=a['aupr'], aupr_sd=a['aupr_sd'], n=a['n'],
                                paper_fig3_auc=PAPER[(ds, 'PLMF-MDA')][0],
                                paper_fig3_aupr=PAPER[(ds, 'PLMF-MDA')][1],
                                paper_text_auc=PAPER_TEXT[(ds, 'PLMF-MDA')][0],
                                paper_text_aupr=PAPER_TEXT[(ds, 'PLMF-MDA')][1]))
        pd.DataFrame(ref).to_excel(xl, sheet_name='PLMF-MDA_reference', index=False)

    for split, df in tables.items():
        print(f'\n===== split={split} =====')
        cols = ['dataset', 'method', 'auc', 'auc_sd', 'paper_auc', 'd_auc',
                'aupr', 'aupr_sd', 'paper_aupr', 'd_aupr']
        print(df[cols].to_string(index=False))
    print('\nsaved', OUT)


if __name__ == '__main__':
    main()
