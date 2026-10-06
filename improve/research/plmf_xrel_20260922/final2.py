# -*- coding: utf-8 -*-
"""PHL + external relations (XRel) on PLMF-MDA warm protocols — single linear model, cross-fitted features.
  python final2.py --dataset MDR --split 5cv --seed 42 --variant phl_x
Variants: phl (reference = plmf_warm final), phl_<group>, phl_x (all groups), *_shuf (external profiles permuted)."""
import os, sys, argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
WARM = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_warm_20260922'; sys.path.insert(0, WARM)
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from mprior import MPrior, build_extra
from prov import Prov, crossfit
from xrel import XRel
ALL = ['dz', 'gene', 'cerna', 'cluster', 'pgx']   # geo, geoall are opt-in (not part of phl_x)
PHL = dict(prov='pmid,costudy,ctx,rec', mprior='fam,seed,hmdd,xdeg')


def parse_variant(v):
    """phl | phl_x | phl_dz | phl_x_shuf | phl_dz_shuf | phl_x-dz (all but dz) | base_x (ctx+diff + xrel, no PHL) | phlf_* (full ncRNADrug provenance) | phlfs_* (full provenance, records shuffled)"""
    shuf = v.endswith('_shuf'); v = v[:-5] if shuf else v
    head, _, rest = v.partition('_')
    if rest == '': groups = []
    elif rest == 'x': groups = list(ALL)
    elif rest.startswith('x-'): groups = [g for g in ALL if g not in rest[2:].split('-')]
    else: groups = rest.split('-')
    return head, groups, shuf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--split', default='5cv'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--inner', type=int, default=10); ap.add_argument('--variant', default='phl_x'); ap.add_argument('--C', type=float, default=0.3)
    ap.add_argument('--k', type=int, default=10); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    head, groups, shuf = parse_variant(a.variant)
    prefix = f'{a.dataset}_{"single" if a.split == "single" else ""}s{a.seed}'
    xr = XRel(a.dataset, groups=groups, shuffle=groups if shuf else None, rng=np.random.RandomState(a.seed + 7), k=a.k) if groups else None
    pv = mp = None
    # heads: base (ctx+diff) | phl (prov + mprior, iNcRD-HG records) | phlf (full ncRNADrug provenance) | phlfs (full provenance, records shuffled)
    #        provf (full provenance only) | provfs (full provenance shuffled, no mprior) | mpr (mprior only) | prov (iNcRD-HG provenance only)
    if head in ('phl', 'phlf', 'phlfs', 'provf', 'provfs', 'prov', 'phlfg'):
        import prov as provmod
        if head in ('phlf', 'phlfs', 'provf', 'provfs'): provmod.HERE = os.path.join(HERE, 'data', f'full_{a.dataset}')     # provenance completion
        if head == 'phlfg': provmod.HERE = os.path.join(HERE, 'data', f'fullg_{a.dataset}')   # + mechanism tokens (curated target genes / pathways) as context hyperedges
        pv = Prov(a.dataset, head in ('phlfs', 'provfs'), np.random.RandomState(a.seed), groups=PHL['prov'].split(','))
        provmod.HERE = WARM
    if head in ('phl', 'phlf', 'phlfs', 'mpr', 'phlfg'):
        mp = MPrior(a.dataset)
    res, rows = [], []
    for f in (range(5) if a.split == '5cv' else [0]):
        z = np.load(os.path.join(WARM, 'cache', f'{prefix}_f{f}_ctx+diffK{a.inner}_kernel10.npz'))
        Xtr, Xte, tr, ytr, te, yte = z['Xtr'], z['Xte'], z['tr'], z['ytr'], z['te'], z['yte']
        if pv is not None or mp is not None:
            pc = os.path.join(HERE, 'cache', f'{head}_{prefix}_f{f}_K{a.inner}.npz')
            if os.path.exists(pc):
                zz = np.load(pc); parts_tr, parts_te = [zz['Ptr']], [zz['Pte']]
                if 'Etr' in zz.files: parts_tr.append(zz['Etr']); parts_te.append(zz['Ete'])     # older cache layout (prov, mprior stored separately)
            else:
                parts_tr, parts_te = [], []
                if pv is not None:
                    Ptr, Pte = crossfit(pv, tr, ytr, te, a.seed * 100 + f, lambda o, s, q: o.transform(s, q), K=a.inner); parts_tr.append(Ptr); parts_te.append(Pte)
                if mp is not None:
                    Etr, Ete, _ = build_extra(mp, tr, ytr, te, set(PHL['mprior'].split(',')), a.seed * 100 + f, K=a.inner); parts_tr.append(Etr); parts_te.append(Ete)
                np.savez_compressed(pc, Ptr=np.concatenate(parts_tr, 1), Pte=np.concatenate(parts_te, 1))
            Xtr = np.concatenate([Xtr] + parts_tr, 1); Xte = np.concatenate([Xte] + parts_te, 1)
        if xr is not None:
            Rtr, Rte = crossfit(xr, tr, ytr, te, a.seed * 100 + f, lambda o, s, q: o.transform(s, q), K=a.inner)
            Xtr = np.concatenate([Xtr, Rtr], 1); Xte = np.concatenate([Xte, Rte], 1)
        m = LogisticRegression(max_iter=5000, C=a.C).fit(Xtr, ytr.astype(int))
        s = m.predict_proba(Xte)[:, 1]
        res.append((roc_auc_score(yte, s), average_precision_score(yte, s)))
        sup = tr[ytr == 1]; dr = np.bincount(sup[:, 0], minlength=xr.nr if xr else mp.nr if mp else 100000)
        rows.append(pd.DataFrame({'fold': f, 'r': te[:, 0], 'd': te[:, 1], 'y': yte.astype(int), 'score': s, 'deg_r': dr[te[:, 0]]}))
    res = np.array(res)
    name = a.variant + (('_' + a.tag) if a.tag else '')
    pd.concat(rows).to_csv(os.path.join(HERE, 'results', f'scores_{prefix}_K{a.inner}_{name}.csv'), index=False)
    line = f'## {prefix} K{a.inner} {name} ncol={Xtr.shape[1]}: AUC {res[:,0].mean():.4f}±{res[:,0].std():.4f} AUPR {res[:,1].mean():.4f}±{res[:,1].std():.4f}'
    print(line, flush=True); open(os.path.join(HERE, 'results', 'summary.txt'), 'a').write(line + '\n')


if __name__ == '__main__':
    main()
