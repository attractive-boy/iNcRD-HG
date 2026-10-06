# -*- coding: utf-8 -*-
"""Leakage audit of the provenance pair channel of ../plmf_gnnarch_20260923/run_gnn.py (--prov).
No training: replicates each fold's split + 10% val hold-out exactly as run_fold, rebuilds the
study / ctxtok memberships from the message-graph (train) positives only, and reads the saved
per-pair test scores of the ladder tags.

Questions
  Q1 own-record leak  : does any test pair's own record reach its features?  (asserted, not assumed)
  Q2 what fires       : shared-study count c = |S_train(m) & S_train(d)| on test positives vs negatives,
                        split into  sib  = a shared study is one of the test pair's OWN PMIDs
                                           (the same paper reported sibling pairs that sit in train)
                                    xs   = only other studies are shared
  Q3 where the gain is: per-subgroup AUC (positive subgroup vs all negatives) for pair -> pair_prov -> full
                        and the record-shuffle control full_provS
  Q4 strict protocol  : how much 'sib' survives the primary-study grouped split
  Q5 hidden positives : negatives documented as same-effect positives in DR_Curated, and their bridge rate
Output: stdout + results/*.csv"""
import os, re, sys, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
from plmf_splits import get_split, load_dataset  # noqa: E402

RES = os.path.join(BASE, 'plmf_gnnarch_20260923', 'results')
RECORDS = os.path.join(BASE, 'plmf_xrel_20260922', 'data', 'full_{}', 'data', 'pair_records.pkl')
DR = '/data1/licheng/incrd-hg/data_external/ncrnadrug_20260922/DR_Curated.txt'
TAGS = ['plain', 'lqo', 'pair', 'pair_prov', 'full', 'full_provS']
RUNS = [('5cv', s) for s in (42, 43, 44)] + [('single', s) for s in range(5)] + [('study', s) for s in (42, 43)]
EFFECT = {'MDR': 'resistant', 'MDS': 'sensitive'}


def stem(n):
    return re.sub(r'-(3P|5P)$', '', n)


def split_with_val(ds, split, fold, seed, val_frac=0.1):
    """== run_gnn.run_fold: split, then a stratified 10% hold-out drawn with RandomState(seed*100+fold)."""
    tr, ytr, te, yte = get_split(ds, split, fold, seed)
    rng = np.random.RandomState(seed * 100 + fold)
    vidx = []
    for c in (0, 1):
        ids = np.where(ytr == c)[0]; rng.shuffle(ids); vidx.append(ids[:int(round(val_frac * len(ids)))])
    mask = np.ones(len(ytr), bool); mask[np.concatenate(vidx)] = False
    return tr[mask], ytr[mask], te, yte


def documented(ds, names, dbs):
    """(i, j) -> True if DR_Curated has a same-effect miRNA record for the pair (exact or -3p/-5p stem)."""
    d = pd.read_csv(DR, sep='\t', dtype=str, encoding='latin1', on_bad_lines='skip', quoting=3)
    d = d[d['ncRNA_Type'].str.lower().str.contains('mirna', na=False) & (d['Effect'].str.lower() == EFFECT[ds])]
    ex = set(zip(d['ncRNA_Name'].str.upper().str.strip(), d['DrugBank_ID'].str.strip()))
    st = {(stem(n), b) for n, b in ex}
    return lambda i, j: (names[i], dbs[j]) in ex or (stem(names[i]), dbs[j]) in st


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def main():
    rows, subs = [], []
    for ds in ('MDR', 'MDS'):
        mir, dru, pos, neg = load_dataset(ds)
        names = mir['miRNA_Name'].astype(str).str.upper().values; dbs = dru['DrugBank_ID'].astype(str).values
        recs = pickle.load(open(RECORDS.format(ds), 'rb'))
        rec = {(i, j): recs[(names[i], dbs[j])] for i, j in map(tuple, pos) if (names[i], dbs[j]) in recs}
        is_doc = documented(ds, names, dbs)
        for split, seed in RUNS:
            folds = [0] if split == 'single' else range(5)
            scores = {t: pd.read_csv(os.path.join(RES, f'{t}_{ds}_{split}_seed{seed}_scores.csv')) for t in TAGS
                      if os.path.exists(os.path.join(RES, f'{t}_{ds}_{split}_seed{seed}_scores.csv'))}
            for fold in folds:
                tr, ytr, te, yte = split_with_val(ds, split, fold, seed)
                train_pos = set(map(tuple, tr[ytr == 1]))
                test_keys = set(map(tuple, te))
                # Q1: the message graph (and so every Mr / Md row) is built from train_pos only
                assert not (train_pos & test_keys), 'test pair inside the message graph'
                Sm, Sd, Tm, Td = {}, {}, {}, {}
                for (i, j) in train_pos:
                    v = rec.get((i, j))
                    if v is None:
                        continue
                    Sm.setdefault(i, set()).update(v['pmids']); Sd.setdefault(j, set()).update(v['pmids'])
                    Tm.setdefault(i, set()).update(v['toks']); Td.setdefault(j, set()).update(v['toks'])
                ann = []
                for (i, j), y in zip(map(tuple, te), yte):
                    sh = Sm.get(i, set()) & Sd.get(j, set())
                    own = set(rec[(i, j)]['pmids']) if (y == 1 and (i, j) in rec) else set()
                    ann.append(dict(fold=fold, m=i, d=j, y=int(y), c_study=len(sh),
                                    c_tok=len(Tm.get(i, set()) & Td.get(j, set())),
                                    has_rec=(i, j) in rec, sib=len(sh & own) > 0, cross=len(sh - own) > 0,
                                    doc=bool(y == 0 and is_doc(i, j))))
                A = pd.DataFrame(ann)
                P, N = A[A.y == 1], A[A.y == 0]
                r = dict(ds=ds, split=split, seed=seed, fold=fold, n_pos=len(P), n_neg=len(N),
                         pos_has_rec=P.has_rec.mean(),
                         pos_fire=(P.c_study > 0).mean(), neg_fire=(N.c_study > 0).mean(),
                         pos_sib=P.sib.mean(), pos_xs_only=(P.cross & ~P.sib).mean(),
                         neg_doc=N.doc.mean(), doc_fire=(N[N.doc].c_study > 0).mean() if N.doc.any() else np.nan,
                         undoc_fire=(N[~N.doc].c_study > 0).mean(),
                         auc_c_study=auc(A.y.values, A.c_study.values), auc_c_tok=auc(A.y.values, A.c_tok.values))
                rows.append(r)
                # Q3: subgroup AUCs from the saved scores
                groups = {'all': np.ones(len(A), bool),
                          'pos_sib': (A.y == 0) | A.sib, 'pos_nosib': (A.y == 0) | ~A.sib,
                          'nobridge': A.c_study == 0}
                for t, sc in scores.items():
                    s = sc[sc.fold == fold]
                    M = A.merge(s[['m', 'd', 'y', 'score']].rename(columns={'y': 'y_s'}), on=['m', 'd'], how='left', validate='one_to_one')
                    assert M.score.notna().all() and (M.y == M.y_s.astype(int)).all(), f'{t} fold {fold}: saved test pairs differ'
                    for g, mask in groups.items():
                        mm = M[np.asarray(mask)]
                        subs.append(dict(ds=ds, split=split, seed=seed, fold=fold, tag=t, group=g,
                                         n_pos=int(mm.y.sum()), auc=auc(mm.y.values, mm.score.values)))
    R = pd.DataFrame(rows); S = pd.DataFrame(subs)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    R.to_csv(os.path.join(HERE, 'results', 'fire_rates.csv'), index=False)
    S.to_csv(os.path.join(HERE, 'results', 'subgroup_auc.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    R['proto'] = R.split.map({'5cv': 'random', 'single': 'random', 'study': 'study'})
    print('\n== Q2/Q4/Q5 bridge firing on test pairs (mean over folds x seeds)')
    print(R.groupby(['ds', 'split']).mean(numeric_only=True)[
        ['n_pos', 'pos_has_rec', 'pos_fire', 'pos_sib', 'pos_xs_only', 'neg_fire', 'neg_doc', 'doc_fire', 'undoc_fire', 'auc_c_study', 'auc_c_tok']].round(4))
    print('\n== Q3 subgroup AUC (positive subgroup vs all negatives; nobridge = pairs with c_study == 0)')
    T = S.groupby(['ds', 'split', 'group', 'tag']).agg(auc=('auc', 'mean'), n_pos=('n_pos', 'mean')).reset_index()
    W = T.pivot_table(index=['ds', 'split', 'group'], columns='tag', values='auc')
    W['n_pos'] = T.groupby(['ds', 'split', 'group']).n_pos.first()
    W['d_prov'] = W['pair_prov'] - W['pair']; W['d_full_vs_provS'] = W['full'] - W['full_provS']
    print(W[['n_pos', 'pair', 'pair_prov', 'full', 'full_provS', 'd_prov', 'd_full_vs_provS']].round(4))


if __name__ == '__main__':
    main()
