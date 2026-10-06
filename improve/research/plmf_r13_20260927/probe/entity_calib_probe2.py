"""Offline probe 2: per-entity random-effect biases fitted on labelled training+val pairs on top of the F logit,
optionally graph-smoothed (Laplacian penalty over miRNA relations kNN/fam/pre/seed and drug kNN)."""
import sys, re, numpy as np, torch
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import load_dataset
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
g = np.load(f'../mphgnn_plmf_20260921/cache/graph_{ds}.npz', allow_pickle=True)
mir, dru, pos, neg = load_dataset(ds); nm, nd = len(mir), len(dru)
def rowmm(x): mn = x.min(1, keepdims=True); mx = x.max(1, keepdims=True); return (x - mn) / (mx - mn + 1e-8)
def knn(x, k):
    x = rowmm(x); xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12); s = xn @ xn.T; np.fill_diagonal(s, -1e9)
    nb = np.argsort(-s, 1)[:, :k]; W = np.zeros((len(x), len(x))); W[np.repeat(np.arange(len(x)), k), nb.ravel()] = 1; return np.maximum(W, W.T)
def stem(n): return re.sub(r'-(3p|5p)$', '', n.lower().strip())
def family(n):
    s = stem(n).replace('hsa-', ''); s = re.sub(r'-[12]$', '', s); return re.sub(r'([0-9])[a-z]$', r'\1', s)
names = mir.miRNA_Name.astype(str).values; seqs = mir.miRNA_Sequence.astype(str).str.upper().str.replace('T', 'U').values
def same(lab): lab = np.asarray(lab); R = (lab[:, None] == lab[None, :]).astype(float); np.fill_diagonal(R, 0); return R
Wr = {'knn': knn(g['mirna_kmer'], 10), 'fam': same([family(n) for n in names]), 'pre': same([stem(n) for n in names]), 'seed': same([q[1:8] for q in seqs])}
Wd = knn(g['drug_x'], 10)
def lap(W): W = torch.tensor(W); return torch.diag(W.sum(1)) - W
def fit(L, Y, dI, mI, lam_b, lam_m, lap_r=None, lap_d=None, gam=0.0):
    L = torch.tensor(L, dtype=torch.float64); Y = torch.tensor(Y, dtype=torch.float64); dI = torch.tensor(dI); mI = torch.tensor(mI)
    b = torch.zeros(nd, dtype=torch.float64, requires_grad=True); bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([b, bm], max_iter=400, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        z = L + b[dI] + bm[mI]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction='sum') + lam_b * (b ** 2).sum() + lam_m * (bm ** 2).sum()
        if gam:
            if lap_r is not None: loss = loss + gam * bm @ (lap_r @ bm)
            if lap_d is not None: loss = loss + gam * b @ (lap_d @ b)
        loss.backward(); return loss
    opt.step(closure); return b.detach().numpy(), bm.detach().numpy()
LR = {k: lap(W).double() for k, W in Wr.items()}; LR['all'] = sum(LR.values()); LD = lap(Wd).double()
res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    for key, lr_, ld_, gam in (('plain', None, None, 0), ('lap_all_1', LR['all'], LD, 1.0), ('lap_all_3', LR['all'], LD, 3.0),
                               ('lap_fam_3', LR['fam'], None, 3.0), ('lap_knn_3', LR['knn'], LD, 3.0), ('lap_seed_pre_3', LR['seed'] + LR['pre'], None, 3.0)):
        b, bm = fit(L, Y, P[:, 1], P[:, 0], 10, 10, lr_, ld_, gam)
        s = lte + b[te[:, 1]] + bm[te[:, 0]]
        res.setdefault(key, []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base': continue
    m = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])]); w2 = np.mean([x[1] > y[1] for x, y in zip(v, res['base'])])
    print(f'  {k:16s} {m[0]:.4f}/{m[1]:.4f}  d {m[0]-bs[0]:+.4f}/{m[1]-bs[1]:+.4f}  win {w:.1f}/{w2:.1f}')
