# -*- coding: utf-8 -*-
"""Aggregate MPHGNN-on-PLMF runs and rank against PLMF-MDA + its 4 reproduced
baselines (plmf_baselines_20260920) and iNcRD-HG-strongest (plmf_strongest_20260920)."""
import glob, json, os, statistics as st
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
BL = os.path.join(BASE, 'plmf_baselines_20260920', 'results')
STR = os.path.join(BASE, 'plmf_strongest_20260920', 'results')
PAPER = {('MDR', 'PLMF-MDA'): (0.9227, 0.9076), ('MDR', 'GCNNMMA'): (0.9070, 0.8736), ('MDR', 'SubMDTA'): (0.8936, 0.8514),
         ('MDR', 'GraphDTA'): (0.8086, 0.7979), ('MDR', 'ML-DTI'): (0.7660, 0.7870),
         ('MDS', 'PLMF-MDA'): (0.9301, 0.9207), ('MDS', 'GCNNMMA'): (0.9143, 0.8889), ('MDS', 'SubMDTA'): (0.8858, 0.8587),
         ('MDS', 'GraphDTA'): (0.7864, 0.7911), ('MDS', 'ML-DTI'): (0.7820, 0.7881)}
VARIANTS = [('kmer', 'real', 'MPHGNN (k-mer miRNA feat, ENCORI/ChEMBL/HumanNet gene graph)'),
            ('kmer', 'none', 'MPHGNN w/o gene graph (no metapath, bipartite only)'),
            ('kmer', 'shuffled', 'MPHGNN, gene ids permuted (shuffle control)'),
            ('expr', 'real', 'MPHGNN (CCLE expr miRNA feat; 42-45% miRNAs covered)'),
            ('kmer_expr', 'real', 'MPHGNN (k-mer + CCLE expr)'),
            ('kmer_gg-train', 'real', 'MPHGNN, global branch on fixed training graph (official uses batch-pair graph)'),
            ('kmer_val', 'real', 'MPHGNN, 10% of training pairs held out for early stopping (test-best column still shown)'),
            ('kmer_val_iso', 'real', 'MPHGNN, 10% val hold-out + isolated-pair test scoring (test-best column = batch scoring)')]


def honest(ds, split):
    """Leakage-free numbers for the main variant: epoch chosen on a 10% validation
    hold-out; test pairs scored (a) in 1024-pair batches (query-pair graph visible to
    the global branch, as in the official evaluation) and (b) one pair at a time."""
    fs = sorted(glob.glob(os.path.join(HERE, 'results', f'mphgnn_kmer_val_iso_{ds}_{split}_seed*.csv')))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return dict(n=len(d), valsel_epoch=d.valsel_epoch.mean(),
                batch_valsel_auc=d.valsel_auc.mean(), batch_valsel_auc_sd=sd(d.valsel_auc.tolist()), batch_valsel_aupr=d.valsel_aupr.mean(),
                iso_valsel_auc=d.iso_valsel_auc.mean(), iso_valsel_auc_sd=sd(d.iso_valsel_auc.tolist()), iso_valsel_aupr=d.iso_valsel_aupr.mean(),
                batch_best_auc=d.auc.mean(), iso_best_auc=d.iso_best_auc.mean(),
                batch_last_auc=d.last_auc.mean(), iso_last_auc=d.iso_last_auc.mean(), iso_last_aupr=d.iso_last_aupr.mean())


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def ours(ds, split, feat, gene):
    tag = f"mphgnn_{feat}{'' if gene == 'real' else '_gene-' + gene}_{ds}_{split}_seed"
    fs = sorted(glob.glob(os.path.join(HERE, 'results', tag + '*.csv')))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    n_exp = 5
    # epoch-30 metrics (official fixed training length) from the per-epoch curves
    e30a, e30p = [], []
    for f in fs:
        cj = f.replace('.csv', '_curves.json')
        if os.path.exists(cj):
            for hist in json.load(open(cj)).values():
                h = [x for x in hist if x['epoch'] == 30]
                if h:
                    e30a.append(h[0]['auc']); e30p.append(h[0]['aupr'])
    return dict(auc=d.auc.mean(), auc_sd=sd(d.auc.tolist()), aupr=d.aupr.mean(), acc=d.acc.mean(),
                last_auc=d.last_auc.mean(), last_aupr=d.last_aupr.mean(),
                ep30_auc=float(np.mean(e30a)) if e30a else float('nan'), ep30_aupr=float(np.mean(e30p)) if e30p else float('nan'),
                minloss_auc=d.minloss_auc.mean(), minloss_aupr=d.minloss_aupr.mean(),
                best_epoch=d.best_epoch.mean(), n=len(d), complete=len(d) == n_exp)


def baseline(ds, split, m):
    rows = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(BL, '*.csv'))], ignore_index=True)
    g = rows[(rows.dataset == ds) & (rows.split == split) & (rows.method == m)]
    if not len(g):
        return None
    return dict(auc=g.auc.mean(), auc_sd=sd(g.auc.tolist()), aupr=g.aupr.mean(), acc=g.acc.mean(),
                last_auc=g.last_auc.mean(), last_aupr=g.last_aupr.mean(), n=len(g))


def strongest(ds, split):
    fs = glob.glob(os.path.join(STR, f'strongest_{ds}_{split}_seed*.csv'))
    if not fs:
        return None
    r = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return dict(auc=r.fused_auc.mean(), auc_sd=sd(r.fused_auc.tolist()), aupr=r.fused_aupr.mean(), acc=r.acc.mean(), n=len(r))


def board(ds, split):
    ent = []
    for feat, gene, label in VARIANTS:
        r = ours(ds, split, feat, gene)
        if r:
            ent.append(dict(method=label, kind='mphgnn', **r))
    s = strongest(ds, split)
    if s:
        ent.append(dict(method='iNcRD-HG-strongest (+anchor)', kind='ours', **s))
    for m in ['PLMF-MDA', 'GCNNMMA', 'SubMDTA', 'GraphDTA', 'ML-DTI']:
        r = baseline(ds, split, m)
        if r:
            ent.append(dict(method=m, kind='paper' if m == 'PLMF-MDA' else 'baseline', paper=PAPER.get((ds, m)), **r))
    ent.sort(key=lambda e: (-e['auc'], -e['aupr']))
    return ent


def main():
    out = []
    for ds in ('MDR', 'MDS'):
        for split in ('5cv', 'single'):
            print(f"\n=== {ds} / {split} ({'paper 5-fold seed42' if split == '5cv' else 'released 80/20 seeds 0-4'}) ===")
            print(f"{'#':>2} {'method':62s} {'AUC':>7} {'±':>6} {'AUPR':>7} {'ACC':>6} {'lastAUC':>7} {'lastAUPR':>8} {'ep30AUC':>7} {'n':>2}")
            for i, e in enumerate(board(ds, split), 1):
                def f4(v):
                    return f'{v:.4f}' if isinstance(v, float) and v == v else '   -  '
                print(f"{i:>2} {e['method']:62s} {e['auc']:.4f} {e['auc_sd']:.4f} {e['aupr']:.4f} {e['acc']:.4f} "
                      f"{f4(e.get('last_auc')):>7} {f4(e.get('last_aupr')):>8} {f4(e.get('ep30_auc')):>7} {e['n']:>2}"
                      + ('' if e.get('complete', True) else '  (incomplete)'))
                out.append(dict(dataset=ds, split=split, rank=i, **{k: v for k, v in e.items() if k != 'paper'}))
    pd.DataFrame(out).to_csv(os.path.join(HERE, 'results', '_summary.csv'), index=False)
    print('\n=== leakage-free protocol (main variant: 10% val early-stop; batch vs isolated-pair test scoring) ===')
    for ds in ('MDR', 'MDS'):
        for split in ('5cv', 'single'):
            h = honest(ds, split)
            if h:
                print(f"{ds} {split:6s} n={h['n']}  val-selected ep {h['valsel_epoch']:.1f}: batch AUC {h['batch_valsel_auc']:.4f} AUPR {h['batch_valsel_aupr']:.4f} | "
                      f"isolated AUC {h['iso_valsel_auc']:.4f} AUPR {h['iso_valsel_aupr']:.4f} || test-best batch {h['batch_best_auc']:.4f} iso {h['iso_best_auc']:.4f} || last batch {h['batch_last_auc']:.4f} iso {h['iso_last_auc']:.4f}")


if __name__ == '__main__':
    main()
