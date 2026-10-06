# -*- coding: utf-8 -*-
"""Aggregate incrd_* runs (4 boards) and compare with PLMF-MDA + baselines
(plmf_baselines_20260920), MPHGNN (mphgnn_plmf_20260921) and iNcRD-HG-strongest
(plmf_strongest_20260920).  Prints a markdown table per board + a compact
'delta vs plain' table across variants; writes summary.csv."""
import glob, json, os, statistics as st, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
BL = os.path.join(BASE, 'plmf_baselines_20260920', 'results')
MPH = os.path.join(BASE, 'mphgnn_plmf_20260921', 'results')
STR = os.path.join(BASE, 'plmf_strongest_20260920', 'results')
BOARDS = [('MDR', '5cv'), ('MDR', 'single'), ('MDS', '5cv'), ('MDS', 'single')]


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def agg(files, auc='auc', aupr='aupr', extra=()):
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if auc not in d:
        return None
    r = dict(auc=d[auc].mean(), auc_sd=sd(d[auc].tolist()), aupr=d[aupr].mean(), aupr_sd=sd(d[aupr].tolist()), n=len(d))
    for e in extra:
        if e in d:
            r[e] = d[e].mean()
    return r


def ours(tag, ds, split):
    fs = sorted(glob.glob(os.path.join(HERE, 'results', f'incrd_{tag}_{ds}_{split}_seed*.csv')))
    return agg(fs, extra=('last_auc', 'last_aupr', 'valsel_auc', 'valsel_aupr', 'best_epoch', 'valsel_epoch'))


def tags():
    t = set()
    for f in glob.glob(os.path.join(HERE, 'results', 'incrd_*_seed*.csv')):
        b = os.path.basename(f)[len('incrd_'):]
        for ds, split in BOARDS:
            suf = f'_{ds}_{split}_seed'
            if suf in b:
                t.add(b.split(suf)[0])
    return sorted(t)


def refs(ds, split):
    out = {}
    for m, pat in [('PLMF-MDA', 'plmf_'), ('ML-DTI', 'mldti_'), ('GCNNMMA', 'gcnnma_'), ('GraphDTA', 'graphdta_'), ('SubMDTA', 'submdta_')]:
        fs = sorted(glob.glob(os.path.join(BL, f'{pat}{ds}_{split}_seed*.csv')))
        r = agg(fs)
        if r:
            out[m] = r
    fs = sorted(glob.glob(os.path.join(MPH, f'mphgnn_kmer_{ds}_{split}_seed*.csv')))
    r = agg(fs)
    if r:
        out['MPHGNN (test-best, batch scoring)'] = r
    fs = sorted(glob.glob(os.path.join(MPH, f'mphgnn_kmer_val_iso_{ds}_{split}_seed*.csv')))
    r = agg(fs, auc='iso_valsel_auc', aupr='iso_valsel_aupr')
    if r:
        out['MPHGNN (val-selected, isolated-pair)'] = r
    fs = sorted(glob.glob(os.path.join(STR, f'strongest_{ds}_{split}_seed*.csv')))
    if fs:
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        for c in ('fused_auc', 'ref_auc'):
            if c in d:
                out['iNcRD-HG-strongest (TabICLv2 ctx + anchor)'] = dict(auc=d[c].mean(), auc_sd=sd(d[c].tolist()),
                                                                          aupr=d[c.replace('auc', 'aupr')].mean(), aupr_sd=0, n=len(d))
                break
    return out


def main():
    T = tags()
    rows = []
    for ds, split in BOARDS:
        print(f'\n### {ds} · {split}')
        print('| method | AUC | ± | AUPR | ± | last AUC/AUPR | valsel AUC/AUPR | n |')
        print('|---|---|---|---|---|---|---|---|')
        entries = []
        for t in T:
            r = ours(t, ds, split)
            if r:
                entries.append((f'iNcRD-HG[{t}]', r)); rows.append(dict(tag=t, dataset=ds, split=split, **r))
        for m, r in refs(ds, split).items():
            entries.append((m, r))
        for m, r in sorted(entries, key=lambda x: -x[1]['aupr']):
            last = f"{r['last_auc']:.4f}/{r['last_aupr']:.4f}" if 'last_auc' in r else '-'
            vs = f"{r['valsel_auc']:.4f}/{r['valsel_aupr']:.4f}" if 'valsel_auc' in r and r['valsel_auc'] == r['valsel_auc'] else '-'
            print(f"| {m} | {r['auc']:.4f} | {r['auc_sd']:.4f} | {r['aupr']:.4f} | {r['aupr_sd']:.4f} | {last} | {vs} | {r['n']} |")
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'summary.csv'), index=False)
    # delta-vs-plain table
    base = {b: ours('plain', *b) for b in BOARDS}
    print('\n### variants: test-best AUPR (delta vs plain) per board, mean delta over 4 boards')
    print('| tag | ' + ' | '.join(f'{d} {s}' for d, s in BOARDS) + ' | mean dAUC | mean dAUPR |')
    print('|---|' + '---|' * (len(BOARDS) + 2))
    for t in T:
        cells, da, dp = [], [], []
        for b in BOARDS:
            r = ours(t, *b); p = base[b]
            if r and p:
                cells.append(f"{r['aupr']:.4f} ({r['aupr'] - p['aupr']:+.4f})"); da.append(r['auc'] - p['auc']); dp.append(r['aupr'] - p['aupr'])
            else:
                cells.append('-')
        ma = f'{st.mean(da):+.4f}' if da else '-'; mp = f'{st.mean(dp):+.4f}' if dp else '-'
        print(f'| {t} | ' + ' | '.join(cells) + f' | {ma} | {mp} |')


if __name__ == '__main__':
    main()
