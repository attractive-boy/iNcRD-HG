# -*- coding: utf-8 -*-
"""Cold-drug (inductive) probe on PLMF-MDA MDR/MDS.

Protocols
  plmf : PLMF-MDA Table-2 style. Test drugs = drugs with 11..20 positives; test pairs =
         all their positives + equal sampled negatives among (miRNA, test-drug) zeros;
         training pairs = positives of remaining drugs + equal negatives among their zeros.
         Repeated over negative-sampling seeds.
  ldo  : leave-drugs-out 5-fold over ALL drugs, stratified by degree bin. Same pair
         construction per fold. Reports pooled + per-fold metrics.

Training-feature construction (--fit)
  warm   : StratifiedKFold(5) over training pairs, support = other folds' positives
           (same as the warm benchmark; the query drug keeps its other edges -> mismatch)
  drugcv : GroupKFold(5) over training DRUGS, support = positives of the other drug groups
           (the query drug has zero support degree -> matches the cold test condition)

Feature families: ctx, diff, cdiff, scf, align (plmf_innov bank) + knn (dense content
label-transfer, Morgan-Tanimoto and ChemBERTa-cosine) + mech (miRNA-target x drug-target
bridge from ENCORI/ChEMBL/HumanNet cache).
"""
import argparse, os, sys, time, csv, hashlib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
for p in ('plmf_innov_20260921', 'mphgnn_plmf_20260921', 'dlstmda_strongest_20260920'):
    sys.path.insert(0, os.path.join(BASE, p))
from plmf_splits import load_dataset  # noqa
from bank import Bank  # noqa
from run_probe import build_adj  # noqa

MPH = os.path.join(BASE, 'mphgnn_plmf_20260921', 'cache')


def load_all(ds):
    mir, dru, pos, neg = load_dataset(ds)
    g = np.load(os.path.join(MPH, f'graph_{ds}.npz'), allow_pickle=True)
    xr = g['mirna_kmer'].astype(np.float32)
    star = np.load(os.path.join(BASE, 'plmf_strongest_20260920', f'graph_cache_{ds}.npz'), allow_pickle=True)
    xd = star['xd'].astype(np.float32)                     # Morgan 2048
    # raw ChemBERTa mean-pool by DrugBank id
    cb = np.load(os.path.join(MPH, 'drug_chemberta_zinc.npz'), allow_pickle=True)
    m = {k: i for i, k in enumerate(cb['drugbank_id'])}
    xcb = np.stack([cb['x'][m[k]] for k in g['drug_ids']]).astype(np.float32)
    nr, nd = xr.shape[0], xd.shape[0]
    MG = np.zeros((nr, g['genes'].shape[0]), np.float32); MG[g['mirna_gene'][:, 0], g['mirna_gene'][:, 1]] = 1
    GD = np.zeros((g['genes'].shape[0], nd), np.float32); GD[g['gene_drug'][:, 0], g['gene_drug'][:, 1]] = 1
    GG = np.zeros((g['genes'].shape[0],) * 2, np.float32); GG[g['gene_gene'][:, 0], g['gene_gene'][:, 1]] = 1
    GG = np.maximum(GG, GG.T)
    return dict(xr=xr, xd=xd, xcb=xcb, pos=pos, nr=nr, nd=nd, MG=MG, GD=GD, GG=GG)


def tanimoto(x):
    xs = (x > 0).astype(np.float32); inter = xs @ xs.T; cnt = xs.sum(1, keepdims=True)
    s = inter / (cnt + cnt.T - inter + 1e-8); np.fill_diagonal(s, 0); return s


def cosine(x):
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8); s = xn @ xn.T
    np.fill_diagonal(s, 0); return np.clip(s, 0, None)


class ColdBank:
    """Wraps plmf_innov Bank and adds knn / mech families."""
    def __init__(self, D, graph='kernel', knn=10):
        sr = build_adj(D['xr'], knn, 'cosine', graph); sd = build_adj(D['xd'], knn, 'tanimoto', graph)
        self.bank = Bank(D['xr'], D['xd'], sr, sd)
        self.D = D
        self.S_tan = tanimoto(D['xd']); self.S_cb = cosine(D['xcb'])
        MG, GD, GG = D['MG'], D['GD'], D['GG']
        self.B0 = MG @ GD                                      # shared targets
        self.B1 = MG @ GG @ GD                                  # one-hop gene neighbour bridge
        self.mg_cnt = MG.sum(1); self.gd_cnt = GD.sum(0)
        self.cache = {}

    def _A(self, sup):
        A = np.zeros((self.D['nr'], self.D['nd']), np.float32); A[sup[:, 0], sup[:, 1]] = 1; return A

    def _knn(self, sup, q):
        A = self._A(sup); r, d = q[:, 0], q[:, 1]; cols = {}
        for name, S in (('tan', self.S_tan), ('cb', self.S_cb)):
            for k in (5, 20, 0):                                # 0 = dense
                Sk = S.copy()
                if k:
                    thr = np.sort(Sk, axis=1)[:, -k][:, None]; Sk = np.where(Sk >= thr, Sk, 0)
                num = A @ Sk.T                                   # [nr, nd]: sum_j A[r,j] S[d,j]
                den = np.maximum(Sk.sum(1), 1e-8)                # per drug d
                cols[f'knn_{name}{k}_sum'] = num[r, d]
                cols[f'knn_{name}{k}_mean'] = (num / den[None, :])[r, d]
                # miRNA-side normalisation: fraction of miRNA r's positives that sit on similar drugs
                cols[f'knn_{name}{k}_rfrac'] = (num[r, d] / np.maximum(A.sum(1)[r], 1))
                # similarity-weighted degree of the drug's neighbours
                cols[f'knn_{name}{k}_nbdeg'] = (Sk @ A.sum(0))[d] / den[d]
                # rank of d among all drugs for miRNA r (percentile) -> drug-popularity free
                cols[f'knn_{name}{k}_rpc'] = (num[r] < num[r, d][:, None]).mean(1)
        return cols

    def _mech(self, q):
        r, d = q[:, 0], q[:, 1]
        return {
            'mech_b0': self.B0[r, d], 'mech_b1': self.B1[r, d],
            'mech_b0_norm': self.B0[r, d] / np.maximum(np.sqrt(self.mg_cnt[r] * self.gd_cnt[d]), 1),
            'mech_b1_norm': self.B1[r, d] / np.maximum(np.sqrt(self.mg_cnt[r] * self.gd_cnt[d]), 1),
            'mech_b0_rpc': (self.B0[r] < self.B0[r, d][:, None]).mean(1),
            'mech_b1_rpc': (self.B1[r] < self.B1[r, d][:, None]).mean(1),
            'mech_has_r': (self.mg_cnt[r] > 0).astype(np.float32), 'mech_has_d': (self.gd_cnt[d] > 0).astype(np.float32),
            'mech_cnt_r': self.mg_cnt[r], 'mech_cnt_d': self.gd_cnt[d],
        }

    def transform(self, sup, q, fams):
        sup = np.asarray(sup, np.int64); q = np.asarray(q, np.int64)
        parts = []
        bf = fams & {'ctx', 'diff', 'cdiff', 'scf', 'align'}
        if bf:
            parts.append(self.bank.transform(sup, q, bf))
        if 'knn' in fams:
            parts.append(pd.DataFrame(self._knn(sup, q)))
        if 'mech' in fams:
            parts.append(pd.DataFrame(self._mech(q)))
        return pd.concat(parts, axis=1).astype(np.float64)


def cold_split(D, test_drugs, seed, side='drug', kshot=0):
    """Balanced pairs: test = positives of test entities + equal negatives among their zeros;
    train = positives of other entities + equal negatives among their zeros.
    kshot>0: for every test entity move k random positives + k random negatives into the training set."""
    rng = np.random.RandomState(seed)
    A = np.zeros((D['nr'], D['nd']), np.int8); A[D['pos'][:, 0], D['pos'][:, 1]] = 1
    n_ent = D['nd'] if side == 'drug' else D['nr']
    is_te = np.zeros(n_ent, bool); is_te[test_drugs] = True

    def ent_mask(mask):
        return mask[None, :] if side == 'drug' else mask[:, None]

    def make(mask):
        P = np.argwhere((A == 1) & ent_mask(mask)); Z = np.argwhere((A == 0) & ent_mask(mask))
        N = Z[rng.permutation(len(Z))[:len(P)]]
        idx = np.concatenate([P, N]); y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
        o = rng.permutation(len(idx)); return idx[o], y[o]
    tr, ytr = make(~is_te); te, yte = make(is_te)
    if kshot > 0:
        col = 1 if side == 'drug' else 0
        move = np.zeros(len(yte), bool)
        for e in test_drugs:
            for lab in (1, 0):
                cand = np.where((te[:, col] == e) & (yte == lab))[0]
                move[rng.permutation(cand)[:kshot]] = True
        tr = np.concatenate([tr, te[move]]); ytr = np.concatenate([ytr, yte[move]])
        te, yte = te[~move], yte[~move]
    return tr, ytr, te, yte


def train_features(cb, tr, ytr, fams, fit, seed):
    Xtr = None
    if fit == 'warm':
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, ytr.astype(int))
    elif fit == 'lodo':
        # leave-one-drug-out at train time: query drug d sees a support with every other training drug intact
        drugs = tr[:, 1]
        splitter = ((np.where(drugs != d)[0], np.where(drugs == d)[0]) for d in np.unique(drugs))
    else:
        drugs = tr[:, 1]; ud = np.unique(drugs); rng = np.random.RandomState(seed)
        grp_of = dict(zip(ud, rng.permutation(len(ud)) % 5)); groups = np.array([grp_of[x] for x in drugs])
        splitter = GroupKFold(5).split(tr, ytr, groups)
    for fit_idx, q_idx in splitter:
        sup = tr[fit_idx[ytr[fit_idx] == 1]]
        f = cb.transform(sup, tr[q_idx], fams)
        if Xtr is None:
            Xtr = np.zeros((len(ytr), f.shape[1]), np.float64); names = list(f.columns)
        Xtr[q_idx] = f.values
    return Xtr, names


def prep(Xtr, Xte):
    both = np.concatenate([Xtr, Xte]); both = np.log1p(np.abs(both)) * np.sign(both)
    Xtr, Xte = both[:len(Xtr)], both[len(Xtr):]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_predict(model, Xtr, ytr, Xte, seed):
    if model.startswith('lr'):
        C = float(model.split(':')[1]) if ':' in model else 1.0
        m = LogisticRegression(max_iter=5000, C=C)
    elif model == 'hgb':
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=seed)
    else:
        raise ValueError(model)
    m.fit(Xtr, ytr.astype(int)); return m.predict_proba(Xte)[:, 1], m


def macro_auc(te, yte, s, col=1):
    out = []
    for d in np.unique(te[:, col]):
        mk = te[:, col] == d
        if len(np.unique(yte[mk])) == 2:
            out.append(roc_auc_score(yte[mk], s[mk]))
    return float(np.mean(out)) if out else float('nan'), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--protocol', default='plmf', choices=['plmf', 'ldo', 'plmf_mirna'])
    ap.add_argument('--kshot', type=int, default=0, help='reveal k positives (and k negatives) of every test drug into the training set')
    ap.add_argument('--families', default='ctx'); ap.add_argument('--fit', default='drugcv', choices=['warm', 'drugcv', 'lodo'])
    ap.add_argument('--model', default='lr:0.3'); ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--graph', default='kernel'); ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--score_col', default='', help='rank directly by this single column (no training)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--scorer', default='', help="'transfer[:k]' = warm LR on --families, kernel-transfer predicted profiles to the cold drug; 'rankavg:c1+c2' = rank-average of columns")
    a = ap.parse_args(); fams = set(a.families.split(','))
    D = load_all(a.dataset); cb = ColdBank(D, a.graph, a.knn)
    A = np.zeros((D['nr'], D['nd']), np.int8); A[D['pos'][:, 0], D['pos'][:, 1]] = 1; ddeg = A.sum(0)
    rows = []
    for seed in [int(s) for s in a.seeds.split(',')]:
        side = 'mirna' if a.protocol == 'plmf_mirna' else 'drug'
        if a.protocol == 'plmf':
            folds = [np.where((ddeg >= 11) & (ddeg <= 20))[0]]
        elif a.protocol == 'plmf_mirna':
            mdeg = A.sum(1); folds = [np.where((mdeg >= 5) & (mdeg <= 6))[0]]
        else:
            bins = np.digitize(ddeg, [2, 6, 21, 101])
            folds = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(D['nd']), bins)]
        pooled_s, pooled_y, pooled_te = [], [], []
        for fi, test_drugs in enumerate(folds):
            t0 = time.time()
            tr, ytr, te, yte = cold_split(D, test_drugs, seed * 10 + fi, side, a.kshot)
            if a.score_col:
                f = cb.transform(tr[ytr == 1], te, fams); s = f[a.score_col].values
                ncol = 1
            elif a.scorer.startswith('kron'):
                lam = float(a.scorer.split(':')[1]) if ':' in a.scorer else 1.0
                sup = tr[ytr == 1]; A_ = np.zeros((D['nr'], D['nd'])); A_[sup[:, 0], sup[:, 1]] = 1
                Kr = cosine(D['xr']) + np.eye(D['nr']); Kd = cb.S_tan + np.eye(D['nd'])
                F_ = Kr @ np.linalg.solve(Kr + lam * np.eye(D['nr']), A_) @ np.linalg.solve(Kd + lam * np.eye(D['nd']), Kd)
                s = F_[te[:, 0], te[:, 1]]; ncol = 0
            elif a.scorer.startswith('rankavg'):
                cols_ = a.scorer.split(':')[1].split('+'); f = cb.transform(tr[ytr == 1], te, fams)
                from scipy.stats import rankdata
                s = np.mean([rankdata(f[c].values) for c in cols_], axis=0); ncol = len(cols_)
            elif a.scorer.startswith('transfer'):
                k_ = int(a.scorer.split(':')[1]) if ':' in a.scorer else 0
                Xtr, names = train_features(cb, tr, ytr, fams, 'warm', seed)
                sup = tr[ytr == 1]; A_ = np.zeros((D['nr'], D['nd']), np.float32); A_[sup[:, 0], sup[:, 1]] = 1
                tr_drugs = np.unique(sup[:, 1])
                q = np.array([(m, j) for j in tr_drugs for m in range(D['nr']) if A_[m, j] == 0], np.int64)
                Xq = cb.transform(sup, q, fams).values
                Xtr_, Xq_ = prep(Xtr, Xq); ncol = Xtr_.shape[1]
                sq, _ = fit_predict(a.model, Xtr_, ytr, Xq_, seed)
                P_ = A_.copy(); P_[q[:, 0], q[:, 1]] = sq             # predicted warm profile matrix (observed positives = 1)
                S_ = cb.S_tan.copy()
                if k_:
                    thr = np.sort(S_, axis=1)[:, -k_][:, None]; S_ = np.where(S_ >= thr, S_, 0)
                W_ = S_[:, tr_drugs]; W_ = W_ / np.maximum(W_.sum(1, keepdims=True), 1e-8)   # [nd, n_train_drugs]
                T_ = P_[:, tr_drugs] @ W_.T                                                     # [nr, nd]
                s = T_[te[:, 0], te[:, 1]]
            else:
                Xtr, names = train_features(cb, tr, ytr, fams, a.fit, seed)
                Xte = cb.transform(tr[ytr == 1], te, fams).values
                Xtr, Xte = prep(Xtr, Xte); ncol = Xtr.shape[1]
                s, _ = fit_predict(a.model, Xtr, ytr, Xte, seed)
            auc, ap_ = roc_auc_score(yte, s), average_precision_score(yte, s); mauc, nd_ = macro_auc(te, yte, s, 1 if side == 'drug' else 0)
            rows.append(dict(seed=seed, fold=fi, n_test_drugs=len(test_drugs), n_test=len(yte), auc=auc, aupr=ap_, macro_auc=mauc, n_macro=nd_))
            pooled_s.append(s); pooled_y.append(yte); pooled_te.append(te)
            print(f'  seed {seed} fold {fi}: drugs {len(test_drugs)} pairs {len(yte)} AUC {auc:.4f} AUPR {ap_:.4f} macroAUC {mauc:.4f} ({nd_}) ncol {ncol} {time.time()-t0:.0f}s', flush=True)
        if a.protocol == 'ldo':
            s = np.concatenate(pooled_s); y = np.concatenate(pooled_y); te = np.concatenate(pooled_te)
            rows.append(dict(seed=seed, fold='pooled', n_test_drugs=D['nd'], n_test=len(y), auc=roc_auc_score(y, s), aupr=average_precision_score(y, s), macro_auc=macro_auc(te, y, s)[0], n_macro=macro_auc(te, y, s)[1]))
    df = pd.DataFrame(rows)
    per = df[df.fold != 'pooled'] if a.protocol == 'ldo' else df
    summ = f"## {a.tag or a.families} {a.dataset} {a.protocol} fit={a.fit} {a.model}: AUC {per.auc.mean():.4f}±{per.auc.std():.4f} AUPR {per.aupr.mean():.4f}±{per.aupr.std():.4f} macroAUC {per.macro_auc.mean():.4f}"
    if a.protocol == 'ldo':
        pl = df[df.fold == 'pooled']; summ += f" | pooled AUC {pl.auc.mean():.4f}±{pl.auc.std():.4f} AUPR {pl.aupr.mean():.4f}±{pl.aupr.std():.4f}"
    print(summ, flush=True)
    out = os.path.join(HERE, 'results', 'cold_probe.csv'); new = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['tag', 'dataset', 'protocol', 'fit', 'model', 'families', 'score_col', 'kshot', 'auc', 'auc_sd', 'aupr', 'aupr_sd', 'macro_auc', 'pooled_auc', 'pooled_aupr', 'n_runs'])
        pl = df[df.fold == 'pooled'] if a.protocol == 'ldo' else None
        w.writerow([a.tag or a.families, a.dataset, a.protocol, a.fit, a.model, a.families, a.score_col or a.scorer, a.kshot,
                    f'{per.auc.mean():.6f}', f'{per.auc.std():.6f}', f'{per.aupr.mean():.6f}', f'{per.aupr.std():.6f}', f'{per.macro_auc.mean():.6f}',
                    f'{pl.auc.mean():.6f}' if pl is not None else '', f'{pl.aupr.mean():.6f}' if pl is not None else '', len(per)])


if __name__ == '__main__':
    main()
