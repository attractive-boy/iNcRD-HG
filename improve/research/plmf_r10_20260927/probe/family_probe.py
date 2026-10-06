"""Fast screen: graph score maps computed on the fold's training graph (test pairs are never in it), query value +
row / column context of test pairs; in-test 2-fold logistic combiner over the GNN logit (calibrated against the
GNN --xfeat runs: diff_all probe +0.0021 vs trained +0.0020)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
ds, tag = sys.argv[1], sys.argv[2]
mir, dru, pos, neg = load_dataset(ds); nr, nd = len(mir), len(dru)
g = np.load(f'../mphgnn_plmf_20260921/cache/graph_{ds}.npz', allow_pickle=True)
def rmm(x): return (x - x.min(1, keepdims=True)) / (x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-8)
def knn(x, k=10):
    u = x / np.linalg.norm(x, axis=1, keepdims=True); s = u @ u.T; np.fill_diagonal(s, -9)
    A = np.zeros_like(s); idx = np.argsort(-s, 1)[:, :k]; A[np.arange(len(s))[:, None], idx] = 1; return ((A + A.T) > 0).astype(float)
Sr, Sd = knn(rmm(g['mirna_kmer'])), knn(rmm(g['drug_x']))
sc = pd.read_csv(f'results/{tag}_{ds}_5cv_seed42_scores.csv')
def ctxcols(M, r, d, name):
    v = M[r, d]; R = M[r]; C = M[:, d].T
    return {name: v, name + '_rz': (v - R.mean(1)) / (R.std(1) + 1e-9), name + '_dz': (v - C.mean(1)) / (C.std(1) + 1e-9),
            name + '_rp': (R < v[:, None]).mean(1), name + '_dp': (C < v[:, None]).mean(1)}
def maps(A):
    out = {}
    dr = np.maximum(A.sum(1, keepdims=True), 1); dd = np.maximum(A.sum(0, keepdims=True), 1)
    An = A / dr; And = (A / dd).T
    G = And @ An
    for al in (0.3, 0.6):
        P = np.zeros_like(A); p = An.copy()
        for k in range(12): P += al * (1 - al) ** (2 * k + 1) * p; p = p @ G
        out[f'ppr{al}'] = P
    Gs = An @ And
    for al in (0.3,):
        P = np.zeros((nd, nr)); q = And.copy()
        for k in range(12): P += al * (1 - al) ** (2 * k + 1) * q; q = q @ Gs
        out[f'pprT{al}'] = P.T
    smax = np.linalg.norm(A, 2)
    for c in (0.3, 0.7):
        b = c / smax; out[f'katz{c}'] = b * A @ np.linalg.inv(np.eye(nd) - b * b * A.T @ A)
    for lam in (10.,):
        P = np.linalg.inv(A.T @ A + lam * np.eye(nd)); B = -P / np.diag(P)[None]; np.fill_diagonal(B, 0); out['easeD'] = A @ B
    Asym = A / np.sqrt(dr) / np.sqrt(dd)
    out['sym3'] = Asym @ Asym.T @ Asym
    out['rw3'] = An @ And.T.T.T if False else An @ (A.T / np.maximum(A.T.sum(1, keepdims=True), 1)) @ An     # m->d->m->d random walk
    out['srA'] = Sr @ A; out['ASd'] = A @ Sd
    out['srAn'] = (Sr / np.maximum(Sr.sum(1, keepdims=True), 1)) @ A
    out['ASdn'] = An @ (Sd / np.maximum(Sd.sum(0, keepdims=True), 1))
    return out
rows = []
for f in range(5):
    tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
    A = np.zeros((nr, nd)); tp = tr[ytr == 1]; A[tp[:, 0], tp[:, 1]] = 1
    s = sc[sc.fold == f]; r, d = te[:, 0], te[:, 1]
    p = s.score.clip(1e-6, 1 - 1e-6).values
    cols = dict(y=yte, logit=np.log(p / (1 - p)))
    for k, M in maps(A).items(): cols.update(ctxcols(M, r, d, k))
    rows.append(pd.DataFrame(cols))
df = pd.concat(rows, ignore_index=True)
Y = df.y.values; L = df.logit.values
def cv(cols, reps=6):
    ga, gp = [], []
    for rep in range(reps):
        h = np.random.RandomState(rep).rand(len(Y)) < 0.5; out = np.zeros(len(Y))
        M = np.column_stack([L] + [np.sign(df[c].values) * np.log1p(np.abs(df[c].values)) for c in cols])
        for m_ in (h, ~h):
            mu, sd = M[m_].mean(0), M[m_].std(0) + 1e-9
            out[~m_] = LogisticRegression(C=0.3, max_iter=4000).fit((M[m_] - mu) / sd, Y[m_]).decision_function((M[~m_] - mu) / sd)
        ga.append(roc_auc_score(Y, out) - roc_auc_score(Y, L)); gp.append(average_precision_score(Y, out) - average_precision_score(Y, L))
    return round(np.mean(ga), 4), round(np.mean(gp), 4)
names = sorted({c.split('_')[0] for c in df.columns if c not in ('y', 'logit')})
print(ds, tag, 'base', round(roc_auc_score(Y, L), 4))
for n in names:
    base = [n]; ctx = [n + s for s in ('_rz', '_dz', '_rp', '_dp')]
    print(f'  {n:8s} raw {cv(base)}  ctx {cv(ctx)}  both {cv(base + ctx)}')
allc = [c for c in df.columns if c not in ('y', 'logit')]
print('  ALL', cv(allc))
print('  ALL ppr', cv([c for c in allc if c.startswith('ppr')]))
print('  ALL non-ppr', cv([c for c in allc if not c.startswith('ppr')]))
