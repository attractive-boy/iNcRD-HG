# -*- coding: utf-8 -*-
"""Build markdown tables from results/*.csv for the frontier report."""
import os, glob, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results')
out = []
import csv
_rows = list(csv.reader(open(os.path.join(R, 'cold_probe.csv'))))
_hdr = ['tag', 'dataset', 'protocol', 'fit', 'model', 'families', 'score_col', 'kshot', 'auc', 'auc_sd', 'aupr', 'aupr_sd', 'macro_auc', 'pooled_auc', 'pooled_aupr', 'n_runs']
_fixed = []
for r in _rows:
    if r[0] == 'tag':
        continue
    if len(r) == 15:
        r = r[:7] + ['0'] + r[7:]
    _fixed.append(r)
cp = pd.DataFrame(_fixed, columns=_hdr)
for c in ['kshot', 'n_runs']:
    cp[c] = pd.to_numeric(cp[c], errors='coerce').fillna(0).astype(int)
for c in ['auc', 'auc_sd', 'aupr', 'aupr_sd', 'macro_auc', 'pooled_auc', 'pooled_aupr']:
    cp[c] = pd.to_numeric(cp[c], errors='coerce')
cp['score_col'] = cp['score_col'].replace('', np.nan)
tab = pd.read_csv(os.path.join(R, 'tabicl_cold.csv')) if os.path.exists(os.path.join(R, 'tabicl_cold.csv')) else pd.DataFrame()
plmf = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, 'plmf_cold_*.csv'))]) if glob.glob(os.path.join(R, 'plmf_cold_*.csv')) else pd.DataFrame()

def fmt(m, s):
    return f'{m:.4f} ± {s:.4f}' if not np.isnan(s) else f'{m:.4f}'

def block(title, df, cols, key='tag'):
    out.append(f'\n#### {title}\n'); out.append('| 方法 | ' + ' | '.join(cols) + ' |'); out.append('|---|' + '---:|' * len(cols))
    for _, r in df.iterrows():
        out.append(f'| {r[key]} | ' + ' | '.join(r[c] for c in cols) + ' |')

for ds in ['MDR', 'MDS']:
    out.append(f'\n### {ds}\n')
    # PLMF official
    if len(plmf):
        p = plmf[(plmf.dataset == ds) & (plmf.protocol == 'plmf')]
        if len(p):
            out.append(f'PLMF-MDA 官方模型（同一冷启动划分，{len(p)} seeds）：test-best AUC **{p.best_auc.mean():.4f} ± {p.best_auc.std():.4f}** / AUPR {p.best_aupr.mean():.4f}；末轮 AUC {p.last_auc.mean():.4f} ± {p.last_auc.std():.4f} / AUPR {p.last_aupr.mean():.4f}（论文 Table 2 报告：MDR 0.6642/0.6534，MDS 0.6579/0.6352）\n')
    d = cp[(cp.dataset == ds) & (cp.protocol == 'plmf') & (cp.kshot == 0)].copy()
    d['AUC'] = [fmt(a, b) for a, b in zip(d.auc, d.auc_sd)]; d['AUPR'] = [fmt(a, b) for a, b in zip(d.aupr, d.aupr_sd)]; d['macroAUC'] = d.macro_auc.map('{:.4f}'.format)
    d['配置'] = d.families + ' / ' + d.fit.where(d.score_col.isna() | (d.score_col == ''), 'scorer=' + d.score_col.astype(str)) + ' / ' + d.model
    d = d.sort_values('auc', ascending=False)
    block(f'{ds} · PLMF Table-2 冷启动药物协议（11–20 条关联的药物，5 个负采样 seed）', d, ['配置', 'AUC', 'AUPR', 'macroAUC'])
    if len(tab):
        t = tab[(tab.dataset == ds) & (tab.protocol == 'plmf') & (tab.kshot == 0)].copy()
        if len(t):
            t['AUC'] = [fmt(a, b) for a, b in zip(t.auc, t.auc_sd)]; t['AUPR'] = [fmt(a, b) for a, b in zip(t.aupr, t.aupr_sd)]; t['macroAUC'] = t.macro_auc.map('{:.4f}'.format)
            t['配置'] = t.families + ' / ' + t.fit
            block(f'{ds} · frozen TabICLv2 in-context（同划分）', t, ['配置', 'AUC', 'AUPR', 'macroAUC'])
    k = cp[(cp.dataset == ds) & (cp.protocol == 'plmf') & (cp.kshot > 0)].copy()
    if len(k):
        k['AUC'] = [fmt(a, b) for a, b in zip(k.auc, k.auc_sd)]; k['AUPR'] = [fmt(a, b) for a, b in zip(k.aupr, k.aupr_sd)]; k['macroAUC'] = k.macro_auc.map('{:.4f}'.format)
        k['配置'] = k.families + ' / ' + k.fit + ' / k=' + k.kshot.astype(str)
        k = k.sort_values(['families', 'kshot'])
        block(f'{ds} · few-shot 冷启动药物（每个测试药物先给 k 条已知正例）', k, ['配置', 'AUC', 'AUPR', 'macroAUC'])
        if len(tab):
            t = tab[(tab.dataset == ds) & (tab.protocol == 'plmf') & (tab.kshot > 0)].copy()
            if len(t):
                t['AUC'] = [fmt(a, b) for a, b in zip(t.auc, t.auc_sd)]; t['AUPR'] = [fmt(a, b) for a, b in zip(t.aupr, t.aupr_sd)]; t['macroAUC'] = t.macro_auc.map('{:.4f}'.format)
                t['配置'] = 'TabICLv2 ' + t.families + ' / k=' + t.kshot.astype(str)
                block(f'{ds} · few-shot（TabICLv2）', t, ['配置', 'AUC', 'AUPR', 'macroAUC'])
    m = cp[(cp.dataset == ds) & (cp.protocol == 'plmf_mirna')].copy()
    if len(m):
        m['AUC'] = [fmt(a, b) for a, b in zip(m.auc, m.auc_sd)]; m['AUPR'] = [fmt(a, b) for a, b in zip(m.aupr, m.aupr_sd)]; m['macroAUC'] = m.macro_auc.map('{:.4f}'.format)
        m['配置'] = m.families + ' / ' + m.fit.where(m.score_col.isna() | (m.score_col == ''), 'scorer=' + m.score_col.astype(str))
        block(f'{ds} · PLMF Table-2 冷启动 miRNA 协议（5–6 条关联的 miRNA；论文 PLMF-MDA：MDR 0.9115/0.8977，MDS 0.9190/0.9037）', m.sort_values('auc', ascending=False), ['配置', 'AUC', 'AUPR', 'macroAUC'])
    l = cp[(cp.dataset == ds) & (cp.protocol == 'ldo')].copy()
    if len(l):
        l['AUC(折均值)'] = [fmt(a, b) for a, b in zip(l.auc, l.auc_sd)]; l['pooled AUC'] = l.pooled_auc.map('{:.4f}'.format); l['pooled AUPR'] = l.pooled_aupr.map('{:.4f}'.format); l['macroAUC'] = l.macro_auc.map('{:.4f}'.format)
        l['配置'] = l.families + ' / ' + l.fit.where(l.score_col.isna() | (l.score_col == ''), 'scorer=' + l.score_col.astype(str))
        block(f'{ds} · 全药物 leave-drugs-out 5 折（按度分层，3 seeds）', l.sort_values('macro_auc', ascending=False), ['配置', 'AUC(折均值)', 'pooled AUC', 'pooled AUPR', 'macroAUC'])
open(os.path.join(HERE, 'RESULT_TABLES.md'), 'w').write('\n'.join(out)); print('\n'.join(out))
