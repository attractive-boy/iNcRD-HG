"""Round-3 pilot: (1) content-imputed popularity for cold drugs (LDO pooled metric),
(2) pseudo-support self-training for cold drugs (plmf protocol). Reuses cold_probe machinery."""
import argparse, os, sys, time, csv
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'plmf_frontier_20260922'))
import cold_probe as cp

def impute_degree(S, deg, train_drugs, k=20, mode='knn'):
    """Predict log(1+deg) of every drug from content similarity to *training* drugs only."""
    W = S[:, train_drugs].copy()
    if k:
        thr = np.sort(W, axis=1)[:, -k][:, None]; W = np.where(W >= thr, W, 0)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-8)
    return W @ np.log1p(deg[train_drugs])

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def run_fold(D, cb, a, tr, ytr, te, yte, test_drugs, seed):
    fams = set(a.families.split(','))
    sup = tr[ytr == 1]; deg = np.bincount(sup[:, 1], minlength=D['nd']).astype(float)
    train_drugs = np.unique(sup[:, 1])
    Xtr, names = cp.train_features(cb, tr, ytr, fams, 'warm', seed)
    Xte = cb.transform(sup, te, fams).values
    Xtr_, Xte_ = cp.prep(Xtr, Xte)
    s0, m = cp.fit_predict(a.model, Xtr_, ytr, Xte_, seed)
    out = {'base': s0}
    # --- popularity imputation
    A_full = np.zeros((D['nr'], D['nd'])); A_full[D['pos'][:, 0], D['pos'][:, 1]] = 1; true_deg = A_full.sum(0)
    dhat = impute_degree(cb.S_tan, deg, train_drugs, k=a.k_imp)
    rho = spearmanr(dhat[test_drugs], true_deg[test_drugs]).correlation
    zt = (dhat - dhat[train_drugs].mean()) / (dhat[train_drugs].std() + 1e-8)
    zo = (np.log1p(true_deg) - np.log1p(deg[train_drugs]).mean()) / (np.log1p(deg[train_drugs]).std() + 1e-8)
    l0 = logit(s0)
    for lam in a.lams:
        out[f'shift{lam}'] = l0 + lam * zt[te[:, 1]]
        out[f'oracle{lam}'] = l0 + lam * zo[te[:, 1]]
    # feature-level imputation: put dhat into degree_d / degree_product columns of the cold rows
    if 'degree_d' in names:
        Xte2 = Xte.copy(); j = names.index('degree_d'); Xte2[:, j] = np.expm1(dhat)[te[:, 1]]
        if 'degree_product' in names:
            Xte2[:, names.index('degree_product')] = Xte2[:, j] * Xte2[:, names.index('degree_r')]
        _, Xte2_ = cp.prep(Xtr, Xte2); out['featimp'] = m.predict_proba(Xte2_)[:, 1]
    # --- pseudo-support self-training (no retraining)
    if a.pseudo:
        q_all = np.array([(r, d) for d in test_drugs for r in range(D['nr'])], np.int64)
        sup_p = sup.copy(); s_prev = None
        for it in range(a.pseudo_iters):
            in_sup = np.array([tuple(e) in set(map(tuple, sup_p)) for e in q_all])
            Xq = cb.transform(sup_p, q_all[~in_sup], fams).values; _, Xq_ = cp.prep(Xtr, Xq)
            sq = np.full(len(q_all), -1.0); sq[~in_sup] = m.predict_proba(Xq_)[:, 1]
            add = []
            for d in test_drugs:
                mk = np.where(q_all[:, 1] == d)[0]
                nq = a.pseudo_q if a.pseudo_q > 0 else max(1, int(round(np.expm1(dhat[d]))))
                top = mk[np.argsort(-sq[mk])[:nq]]; add.append(q_all[top])
            sup_p = np.concatenate([sup_p] + add)
            # a test pair whose own edge was pseudo-added must not see itself: score hit pairs with the
            # pseudo set minus all test-colliding edges (leave-self-out approximation)
            sp = set(map(tuple, sup_p[len(sup):])); hit = np.array([(r, d) in sp for r, d in te])
            s3 = np.zeros(len(te))
            if (~hit).any():
                X_ = cb.transform(sup_p, te[~hit], fams).values; _, X_ = cp.prep(Xtr, X_); s3[~hit] = m.predict_proba(X_)[:, 1]
            if hit.any():
                keep = np.array([tuple(e) not in set(map(tuple, te[hit])) for e in sup_p])
                X_ = cb.transform(sup_p[keep], te[hit], fams).values; _, X_ = cp.prep(Xtr, X_); s3[hit] = m.predict_proba(X_)[:, 1]
            out[f'pseudo_it{it+1}'] = s3
            out[f'pseudo_it{it+1}_selfhit'] = hit.mean()
    return out, rho

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--protocol', default='ldo', choices=['plmf', 'ldo'])
    ap.add_argument('--families', default='ctx,knn'); ap.add_argument('--model', default='lr:0.3')
    ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--k_imp', type=int, default=20)
    ap.add_argument('--lams', default='0.25,0.5,1,1.5,2'); ap.add_argument('--pseudo', action='store_true')
    ap.add_argument('--pseudo_q', type=int, default=0, help='0 = use imputed degree'); ap.add_argument('--pseudo_iters', type=int, default=2)
    ap.add_argument('--tag', default='')
    a = ap.parse_args(); a.lams = [float(x) for x in a.lams.split(',')]
    D = cp.load_all(a.dataset); cb = cp.ColdBank(D, 'kernel', 10)
    A = np.zeros((D['nr'], D['nd']), np.int8); A[D['pos'][:, 0], D['pos'][:, 1]] = 1; ddeg = A.sum(0)
    rows = []
    for seed in [int(s) for s in a.seeds.split(',')]:
        if a.protocol == 'plmf':
            folds = [np.where((ddeg >= 11) & (ddeg <= 20))[0]]
        else:
            bins = np.digitize(ddeg, [2, 6, 21, 101])
            folds = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(D['nd']), bins)]
        pooled = {}; pooled_y, pooled_te = [], []; rhos = []
        for fi, test_drugs in enumerate(folds):
            t0 = time.time()
            tr, ytr, te, yte = cp.cold_split(D, test_drugs, seed * 10 + fi, 'drug', 0)
            out, rho = run_fold(D, cb, a, tr, ytr, te, yte, test_drugs, seed); rhos.append(rho)
            for k, s in out.items():
                if k.endswith('selfhit'):
                    rows.append(dict(seed=seed, fold=fi, variant=k, auc=s, aupr=np.nan, macro=np.nan)); continue
                pooled.setdefault(k, []).append(s)
                rows.append(dict(seed=seed, fold=fi, variant=k, auc=roc_auc_score(yte, s), aupr=average_precision_score(yte, s), macro=cp.macro_auc(te, yte, s)[0]))
            pooled_y.append(yte); pooled_te.append(te)
            print(f'  seed {seed} fold {fi}: rho(dhat,deg)={rho:.3f} ' + ' '.join(f"{k}={roc_auc_score(yte, s):.3f}" for k, s in out.items() if not k.endswith('selfhit')) + f' {time.time()-t0:.0f}s', flush=True)
        y = np.concatenate(pooled_y); te = np.concatenate(pooled_te)
        for k, ss in pooled.items():
            s = np.concatenate(ss)
            rows.append(dict(seed=seed, fold='pooled', variant=k, auc=roc_auc_score(y, s), aupr=average_precision_score(y, s), macro=cp.macro_auc(te, y, s)[0]))
        print(f'seed {seed} mean rho {np.mean(rhos):.3f}', flush=True)
    df = pd.DataFrame(rows)
    per = df[df.fold != 'pooled'].groupby('variant')[['auc', 'aupr', 'macro']].agg(['mean', 'std'])
    pl = df[df.fold == 'pooled'].groupby('variant')[['auc', 'aupr']].agg(['mean', 'std'])
    print(f'## {a.tag or a.families} {a.dataset} {a.protocol} per-fold:'); print(per.round(4).to_string())
    if a.protocol == 'ldo':
        print('## pooled:'); print(pl.round(4).to_string())
    df.to_csv(os.path.join(HERE, 'results', f'cold_pop_{a.dataset}_{a.protocol}_{a.tag or "run"}.csv'), index=False)

if __name__ == '__main__':
    main()
