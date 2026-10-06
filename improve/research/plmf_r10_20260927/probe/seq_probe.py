"""Screen: do dataset-internal pairwise-similarity-to-partners signals add information on top of the
round-8 full model's test scores?  Within-test 2-fold CV logistic combiner, repeated 10x."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'F_full'
mir, dru, pos, neg = load_dataset(ds)
seqs = mir['miRNA_Sequence'].astype(str).str.upper().str.replace('T', 'U').tolist()
nr, nd = len(seqs), len(dru)
L = 22
nt = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
OH = np.zeros((nr, L, 4), np.float32)
for i, s in enumerate(seqs):
    for p, c in enumerate(s[:L]):
        if c in nt: OH[i, p, nt[c]] = 1
def ident(p0, p1):
    X = OH[:, p0:p1].reshape(nr, -1); return X @ X.T
I_seed = ident(1, 8); I_full = ident(0, L); I_3p = ident(12, L)
np.fill_diagonal(I_seed, -1); np.fill_diagonal(I_full, -1); np.fill_diagonal(I_3p, -1)
fps = []
for s in dru['SMILES'].astype(str):
    m = Chem.MolFromSmiles(s)
    fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m is not None else None)
T = np.zeros((nd, nd), np.float32)
for i in range(nd):
    for j in range(nd):
        if fps[i] is not None and fps[j] is not None and i != j:
            T[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
sc = pd.read_csv(f'../plmf_nolqo_20260923/results/{tag}_{ds}_5cv_seed42_scores.csv')
rows = []
for f in range(5):
    tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
    A = np.zeros((nr, nd), np.float32); tp = tr[ytr == 1]; A[tp[:, 0], tp[:, 1]] = 1
    s = sc[sc.fold == f]; assert (s.m.values == te[:, 0]).all()
    r, d = te[:, 0], te[:, 1]
    P = A[:, d].T                                  # [B, nr] partners of d (test pairs are unobserved -> hollow)
    Q = A[r]                                       # [B, nd] partners of m
    def mx(M, W):
        v = np.where(W > 0, M, -1).max(1); return v
    feat = dict(
        seed_max=mx(I_seed[r], P), full_max=mx(I_full[r], P), p3_max=mx(I_3p[r], P),
        seed_mean=(I_seed[r] * P).sum(1) / (P.sum(1) + 1), full_mean=(I_full[r] * P).sum(1) / (P.sum(1) + 1),
        seed7cnt=((I_seed[r] >= 7) * P).sum(1), seed6cnt=((I_seed[r] >= 6) * P).sum(1),
        tan_max=mx(T[d], Q), tan_mean=(T[d] * Q).sum(1) / (Q.sum(1) + 1),
    )
    # positional profile (PWM) of d's partners, log-odds of m vs background
    prof = np.einsum('bn,npk->bpk', P, OH) + 0.5
    prof = prof / prof.sum(2, keepdims=True); bg = OH.mean(0) + 1e-3; bg = bg / bg.sum(1, keepdims=True)
    ll = (OH[r] * np.log(prof / bg[None])).sum(2)
    feat['pwm_seed'] = ll[:, 1:8].sum(1); feat['pwm_full'] = ll.sum(1)
    df = pd.DataFrame(feat); df['y'] = yte; df['logit'] = np.log(s.score.values.clip(1e-6, 1 - 1e-6) / (1 - s.score.values.clip(1e-6, 1 - 1e-6)))
    df['degm'] = A[r].sum(1); df['fold'] = f
    rows.append(df)
df = pd.concat(rows, ignore_index=True)
feats = [c for c in df.columns if c not in ('y', 'logit', 'degm', 'fold')]
print(ds, 'base', round(roc_auc_score(df.y, df.logit), 4), round(average_precision_score(df.y, df.logit), 4))
for c in feats:
    print(f'  {c:10s} alone AUC {roc_auc_score(df.y, df[c]):.4f}  | pos mean {df[df.y==1][c].mean():.3f} neg {df[df.y==0][c].mean():.3f}')
def cv(cols, reps=10):
    ga, gp = [], []
    for rep in range(reps):
        rng = np.random.RandomState(rep); h = rng.rand(len(df)) < 0.5
        out = np.zeros(len(df))
        for tr_m in (h, ~h):
            X = df[cols].values; mu, sd = X[tr_m].mean(0), X[tr_m].std(0) + 1e-6
            clf = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr_m] - mu) / sd, df.y[tr_m])
            out[~tr_m] = clf.decision_function((X[~tr_m] - mu) / sd)
        ga.append(roc_auc_score(df.y, out) - roc_auc_score(df.y, df.logit)); gp.append(average_precision_score(df.y, out) - average_precision_score(df.y, df.logit))
    return np.mean(ga), np.mean(gp)
for c in feats:
    print(f'  +{c:10s} dAUC {cv(["logit", c])[0]:+.4f} dAUPR {cv(["logit", c])[1]:+.4f}')
print('  +seq all', cv(['logit'] + [c for c in feats if not c.startswith('tan')]))
print('  +tan all', cv(['logit', 'tan_max', 'tan_mean']))
print('  +all', cv(['logit'] + feats))
