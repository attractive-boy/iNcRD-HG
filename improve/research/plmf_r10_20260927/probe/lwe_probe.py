"""Screen: locally-weighted, self-excluded EASE (LW-EASE) -- drug-drug regression fitted per miRNA on the rows
of sequence-similar miRNAs (own row weight 0), and miRNA-miRNA regression fitted per drug on similar drugs.
Incremental value over the round-8 full model test scores, within-test 2-fold logistic combiner."""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
ds = sys.argv[1]; tag = sys.argv[2]; dev = 'cuda:1'
g = np.load(f'../mphgnn_plmf_20260921/cache/graph_{ds}.npz', allow_pickle=True)
def rmm(x):
    x = torch.as_tensor(x, dtype=torch.float); return (x - x.min(1, keepdim=True).values) / (x.max(1, keepdim=True).values - x.min(1, keepdim=True).values + 1e-8)
xr = F.normalize(rmm(g['mirna_kmer']), dim=1).to(dev); xd = F.normalize(rmm(g['drug_x']), dim=1).to(dev)
Kr = xr @ xr.t(); Kd = xd @ xd.t()
mir, dru, pos, neg = load_dataset(ds); nr, nd = len(mir), len(dru)
sc = pd.read_csv(f'results/{tag}_{ds}_5cv_seed42_scores.csv')
def weights(K, tau, topk, eps):
    W = torch.exp((K - 1) / tau); W.fill_diagonal_(0)
    if topk:
        thr = W.topk(topk, 1).values[:, -1:]; W = torch.where(W >= thr, W, torch.zeros_like(W))
    W = W / W.sum(1, keepdim=True) * W.shape[0]            # mean weight 1 (scale comparable to the global Gram)
    return W + eps
def lwe_drugside(A, W, lam):         # per miRNA m: G_m = A^T diag(W[m]) A + lam I ; score(m, d)
    G = torch.einsum('mk,kd,ke->mde', W, A, A) + lam * torch.eye(A.shape[1], device=dev)
    P = torch.linalg.inv(G); dg = torch.diagonal(P, dim1=1, dim2=2)
    B = -P / dg[:, None, :]; B = B - torch.diag_embed(torch.diagonal(B, dim1=1, dim2=2))
    return torch.einsum('md,mde->me', A, B)
rows = []
cfgs = {'r_t005_k50': ('r', 0.05, 50, 0.0), 'r_t01_k100': ('r', 0.1, 100, 0.0), 'r_t005_k50_e01': ('r', 0.05, 50, 0.1), 'r_glob': ('r', 1e9, 0, 0.0),
        'd_t005_k20': ('d', 0.05, 20, 0.0), 'd_t01_k40': ('d', 0.1, 40, 0.0), 'd_glob': ('d', 1e9, 0, 0.0)}
for f in range(5):
    tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
    A = torch.zeros(nr, nd, device=dev); tp = tr[ytr == 1]; A[tp[:, 0], tp[:, 1]] = 1
    s = sc[sc.fold == f]; r, d = te[:, 0], te[:, 1]
    df = pd.DataFrame(dict(y=yte, logit=np.log(s.score.clip(1e-6, 1 - 1e-6) / (1 - s.score.clip(1e-6, 1 - 1e-6))).values))
    for name, (side, tau, k, eps) in cfgs.items():
        for lam in (10.0,):
            if side == 'r':
                S = lwe_drugside(A, weights(Kr, tau, k, eps), lam)
            else:
                S = torch.zeros(nr, nd, device=dev)
                W = weights(Kd, tau, k, eps)
                for j in range(nd):     # per drug: miRNA-miRNA Gram on drugs similar to j (own column weight 0)
                    G = (A * W[j][None, :]) @ A.t() + lam * torch.eye(nr, device=dev)
                    P = torch.linalg.inv(G); B = -P / torch.diagonal(P)[None, :]; B.fill_diagonal_(0)
                    S[:, j] = A[:, j] @ B
            v = S[r, d].cpu().numpy(); df[name] = np.sign(v) * np.log1p(np.abs(v) * 10)
    rows.append(df)
df = pd.concat(rows, ignore_index=True)
print(ds, tag, 'base', round(roc_auc_score(df.y, df.logit), 4), round(average_precision_score(df.y, df.logit), 4))
def cv(cols, reps=10):
    ga, gp = [], []
    for rep in range(reps):
        h = np.random.RandomState(rep).rand(len(df)) < 0.5; out = np.zeros(len(df))
        for m_ in (h, ~h):
            X = df[cols].values; mu, sd = X[m_].mean(0), X[m_].std(0) + 1e-6
            out[~m_] = LogisticRegression(max_iter=3000).fit((X[m_] - mu) / sd, df.y[m_]).decision_function((X[~m_] - mu) / sd)
        ga.append(roc_auc_score(df.y, out) - roc_auc_score(df.y, df.logit)); gp.append(average_precision_score(df.y, out) - average_precision_score(df.y, df.logit))
    return round(np.mean(ga), 4), round(np.mean(gp), 4)
for c in cfgs:
    print(f'  {c:16s} alone {roc_auc_score(df.y, df[c]):.4f}  +base {cv(["logit", c])}  +base+glob {cv(["logit", c, "r_glob", "d_glob"])}')
print('  globs', cv(['logit', 'r_glob', 'd_glob']))
