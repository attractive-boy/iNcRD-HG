"""Screen: do the LR-stack `diff` features (PPR / Katz / path7 + row-z / row-pct / col-z)
add information on top of the round-8 full model's test scores?  Within-test 2-fold CV
logistic combiner, repeated 10x.  Test pairs are unobserved in the support graph, so the
features are leak-free by construction (same convention as plmf_innov bank.py)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
sys.path.insert(0, '../plmf_innov_20260921')
sys.path.insert(0, '../dlstmda_strongest_20260920')
from plmf_splits import get_split, load_dataset
from bank import diffusion, DIFF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'F_full'
mir, dru, pos, neg = load_dataset(ds); nr, nd = len(mir), len(dru)
sc = pd.read_csv(f'../plmf_nolqo_20260923/results/{tag}_{ds}_5cv_seed42_scores.csv')
rows = []
for f in range(5):
    tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
    A = np.zeros((nr, nd)); A[tr[ytr == 1][:, 0], tr[ytr == 1][:, 1]] = 1
    gm = {}
    for al, be in DIFF:
        for key, D in diffusion(A, al, be).items():
            gm[f'{key}_{int(al*100)}_{int(be*1000)}'] = D
    r, d = te[:, 0], te[:, 1]
    feats = {}
    for k, D in gm.items():
        v = D[r, d]
        feats[f'd_{k}'] = v
        feats[f'd_{k}_rz'] = (v - D[r].mean(1)) / (D[r].std(1) + 1e-8)
        feats[f'd_{k}_rpc'] = (D[r] < v[:, None]).mean(1)
        feats[f'd_{k}_dz'] = (v - D[:, d].mean(0)) / (D[:, d].std(0) + 1e-8)
    df = pd.DataFrame(feats); df['y'] = yte
    s = sc[sc.fold == f]; assert (s.m.values == r).all() and (s.d.values == d).all()
    df['logit'] = np.log(s.score.values.clip(1e-6, 1 - 1e-6) / (1 - s.score.values.clip(1e-6, 1 - 1e-6)))
    df['fold'] = f
    rows.append(df)
df = pd.concat(rows, ignore_index=True)
feats = [c for c in df.columns if c not in ('y', 'logit', 'fold')]
print(ds, tag, 'base', round(roc_auc_score(df.y, df.logit), 4), round(average_precision_score(df.y, df.logit), 4))

def cv(cols, reps=10):
    ga, gp = [], []
    for rep in range(reps):
        rng = np.random.RandomState(rep); h = rng.rand(len(df)) < 0.5
        out = np.zeros(len(df))
        for tr_m in (h, ~h):
            X = df[cols].values; mu, sd = X[tr_m].mean(0), X[tr_m].std(0) + 1e-6
            clf = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr_m] - mu) / sd, df.y[tr_m])
            out[~tr_m] = clf.decision_function((X[~tr_m] - mu) / sd)
        ga.append(roc_auc_score(df.y, out) - roc_auc_score(df.y, df.logit))
        gp.append(average_precision_score(df.y, out) - average_precision_score(df.y, df.logit))
    return np.mean(ga), np.mean(gp)

raw = [c for c in feats if not (c.endswith('_rz') or c.endswith('_rpc') or c.endswith('_dz'))]
norm = [c for c in feats if c not in raw]
print('  +raw all   ', cv(['logit'] + raw))
print('  +norm all  ', cv(['logit'] + norm))
print('  +all       ', cv(['logit'] + feats))
for c in feats:
    g = cv(['logit', c])
    if abs(g[0]) > 0.0003 or abs(g[1]) > 0.0005:
        print(f'  +{c:22s} dAUC {g[0]:+.4f} dAUPR {g[1]:+.4f}')
