"""Round-3 pilot 2: which side information predicts a cold drug's popularity (degree), and how much
pooled leave-drugs-out AUC/AUPR that buys. score = logit(compat) + lam * z(pop_hat)."""
import argparse, os, sys, time, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, KFold
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'plmf_frontier_20260922')); sys.path.insert(0, os.path.join(BASE, 'plmf_xds_20260922'))
import cold_probe as cp
from xds_probe import other_matrix

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def side_info(ds, D, cb):
    g = np.load(os.path.join(BASE, 'mphgnn_plmf_20260921', 'cache', f'graph_{ds}.npz'), allow_pickle=True); ids = g['drug_ids'].tolist()
    Ax, _, pd_ = other_matrix(ds); xdeg = Ax.sum(0)
    pm = pd.read_csv(os.path.join(HERE, 'data', 'pubmed_drug_plmf.csv')).set_index('drugbank_id')
    pub = np.full((D['nd'], 4), np.nan)
    for i, db in enumerate(ids):
        if db in pm.index and pm.loc[db, 'n_all'] >= 0: pub[i] = pm.loc[db, ['n_all', 'n_mirna', 'n_resist', 'n_cancer']].values
    pub = np.log1p(np.where(np.isnan(pub), np.nanmedian(pub, 0), pub))
    S = dict(tan=cb.S_tan, cb=cb.S_cb, tgt=np.log1p(cb.gd_cnt), xdeg=np.log1p(xdeg), xpresent=pd_, pub=pub)
    return S

def knn_pred(S, y, train_drugs, k=20):
    W = S[:, train_drugs].copy(); thr = np.sort(W, axis=1)[:, -k][:, None]; W = np.where(W >= thr, W, 0)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-8); return W @ y[train_drugs]

def predictors(SI, D, deg, train_drugs):
    """Every predictor is fitted on training drugs' log-degree only."""
    y = np.log1p(deg); P = {}
    P['tan_knn'] = knn_pred(SI['tan'], y, train_drugs); P['cb_knn'] = knn_pred(SI['cb'], y, train_drugs)
    X = np.concatenate([D['xd'], D['xcb']], 1); r = Ridge(alpha=100.0).fit(X[train_drugs], y[train_drugs]); P['content_ridge'] = r.predict(X)
    P['tgt_cnt'] = SI['tgt']; P['xdeg'] = SI['xdeg']
    for j, n in enumerate(['pub_all', 'pub_mirna', 'pub_resist', 'pub_cancer']): P[n] = SI['pub'][:, j]
    # combined ridge over the side-info columns (calibrated to log-degree on training drugs)
    for name, cols in (('combo_nolit', ['tan_knn', 'cb_knn', 'tgt_cnt', 'xdeg']), ('combo_lit', ['pub_all', 'pub_mirna', 'pub_resist', 'pub_cancer']),
                       ('combo_all', ['tan_knn', 'cb_knn', 'tgt_cnt', 'xdeg', 'pub_all', 'pub_mirna', 'pub_resist', 'pub_cancer'])):
        Z = np.stack([P[c] for c in cols], 1); mu, sd = Z[train_drugs].mean(0), Z[train_drugs].std(0) + 1e-8; Z = (Z - mu) / sd
        # inner CV over training drugs for honest combo (knn preds of train drugs are in-sample -> use inner LOO-ish KFold)
        r = Ridge(alpha=10.0).fit(Z[train_drugs], y[train_drugs]); P[name] = r.predict(Z)
    return P

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--families', default='ctx,knn'); ap.add_argument('--model', default='lr:0.3')
    ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--lams', default='0.5,1,1.5,2,3,4,6'); ap.add_argument('--protocol', default='ldo', choices=['ldo','plmf']); ap.add_argument('--shuffle_side', action='store_true', help='control: permute pub counts and xdeg across drugs'); ap.add_argument('--tag', default='pop2')
    a = ap.parse_args(); lams = [float(x) for x in a.lams.split(',')]; fams = set(a.families.split(','))
    D = cp.load_all(a.dataset); cb = cp.ColdBank(D, 'kernel', 10); SI = side_info(a.dataset, D, cb)
    if a.shuffle_side:
        prm = np.random.RandomState(123).permutation(D['nd']); SI['pub'] = SI['pub'][prm]; SI['xdeg'] = SI['xdeg'][prm]; SI['tgt'] = SI['tgt'][prm]
    A = np.zeros((D['nr'], D['nd']), np.int8); A[D['pos'][:, 0], D['pos'][:, 1]] = 1; ddeg = A.sum(0).astype(float)
    rows, rhos = [], []
    for seed in [int(s) for s in a.seeds.split(',')]:
        bins = np.digitize(ddeg, [2, 6, 21, 101])
        folds = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(D['nd']), bins)] if a.protocol == 'ldo' else [np.where((ddeg >= 11) & (ddeg <= 20))[0]]
        pooled, pooled_y, pooled_te = {}, [], []
        for fi, test_drugs in enumerate(folds):
            tr, ytr, te, yte = cp.cold_split(D, test_drugs, seed * 10 + fi, 'drug', 0)
            sup = tr[ytr == 1]; deg = np.bincount(sup[:, 1], minlength=D['nd']).astype(float); train_drugs = np.unique(sup[:, 1])
            Xtr, names = cp.train_features(cb, tr, ytr, fams, 'warm', seed); Xte = cb.transform(sup, te, fams).values
            Xtr_, Xte_ = cp.prep(Xtr, Xte); s0, m = cp.fit_predict(a.model, Xtr_, ytr, Xte_, seed); l0 = logit(s0)
            P = predictors(SI, D, deg, train_drugs); ytrue = np.log1p(ddeg)
            out = {'base': s0}
            # --- inner drug-grouped CV on training drugs to pick lambda per predictor (never looks at test drugs)
            from sklearn.model_selection import GroupKFold
            rng_in = np.random.RandomState(seed * 100 + fi); ud = np.unique(tr[:, 1]); grp_of = dict(zip(ud, rng_in.permutation(len(ud)) % 3))
            groups = np.array([grp_of[x] for x in tr[:, 1]]); inner = {k: {lam: [[], []] for lam in lams} for k in P}
            for itr, iva in GroupKFold(3).split(tr, ytr, groups):
                sup_i = tr[itr][ytr[itr] == 1]; deg_i = np.bincount(sup_i[:, 1], minlength=D['nd']).astype(float); td_i = np.unique(sup_i[:, 1])
                Xb_cold = cb.transform(sup_i, tr[iva], fams).values   # held-out drugs must be cold in the inner fold
                Xa, Xb = cp.prep(Xtr[itr], Xb_cold); sb, _ = cp.fit_predict(a.model, Xa, ytr[itr], Xb, seed); lb = logit(sb)
                Pi = predictors(SI, D, deg_i, td_i)
                for k, p in Pi.items():
                    z = (p - p[td_i].mean()) / (p[td_i].std() + 1e-8)
                    for lam in lams: inner[k][lam][0].append(lb + lam * z[tr[iva, 1]]); inner[k][lam][1].append(ytr[iva])
            lam_sel = {}
            for k in P:
                sc = {lam: roc_auc_score(np.concatenate(inner[k][lam][1]), np.concatenate(inner[k][lam][0])) for lam in lams}
                lam_sel[k] = max(lams, key=lambda l: sc[l])
            print('   lam_sel', {k: lam_sel[k] for k in ('combo_lit', 'combo_all', 'xdeg', 'pub_mirna')}, flush=True)
            for k, p in P.items():
                rho = spearmanr(p[test_drugs], ytrue[test_drugs]).correlation; rhos.append(dict(seed=seed, fold=fi, pred=k, rho=rho))
                z = (p - p[train_drugs].mean()) / (p[train_drugs].std() + 1e-8)
                for lam in lams: out[f'{k}@{lam}'] = l0 + lam * z[te[:, 1]]
                out[f'{k}@sel'] = l0 + lam_sel[k] * z[te[:, 1]]
            zo = (ytrue - ytrue[train_drugs].mean()) / (ytrue[train_drugs].std() + 1e-8)
            for lam in lams: out[f'oracle@{lam}'] = l0 + lam * zo[te[:, 1]]
            for k, s in out.items():
                pooled.setdefault(k, []).append(s)
                rows.append(dict(seed=seed, fold=fi, variant=k, auc=roc_auc_score(yte, s), aupr=average_precision_score(yte, s), macro=cp.macro_auc(te, yte, s)[0]))
            pooled_y.append(yte); pooled_te.append(te)
            print(f'  seed {seed} fold {fi} base {roc_auc_score(yte, s0):.3f} ' + ' '.join(f"{r['pred']}={r['rho']:.2f}" for r in rhos if r['seed'] == seed and r['fold'] == fi), flush=True)
        y = np.concatenate(pooled_y); te = np.concatenate(pooled_te)
        for k, ss in pooled.items():
            s = np.concatenate(ss); rows.append(dict(seed=seed, fold='pooled', variant=k, auc=roc_auc_score(y, s), aupr=average_precision_score(y, s), macro=cp.macro_auc(te, y, s)[0]))
    df = pd.DataFrame(rows); rh = pd.DataFrame(rhos)
    print(f'## {a.dataset} rho(pop_hat, true log-degree) on cold drugs, mean over folds:'); print(rh.groupby('pred').rho.agg(['mean', 'std']).round(3).sort_values('mean', ascending=False).to_string())
    pl = df[df.fold == 'pooled'].groupby('variant')[['auc', 'aupr', 'macro']].agg(['mean', 'std']).round(4)
    pl['pred'] = [v.split('@')[0] for v in pl.index]; pl['lam'] = [v.split('@')[1] if '@' in v else '' for v in pl.index]
    print(f'## {a.dataset} pooled LDO (3 seeds):'); print(pl.sort_values(('auc', 'mean'), ascending=False).to_string())
    df.to_csv(os.path.join(HERE, 'results', f'cold_pop2_{a.dataset}_{a.protocol}_{a.tag}.csv'), index=False); rh.to_csv(os.path.join(HERE, 'results', f'cold_pop2_rho_{a.dataset}_{a.protocol}_{a.tag}.csv'), index=False)

if __name__ == '__main__':
    main()
