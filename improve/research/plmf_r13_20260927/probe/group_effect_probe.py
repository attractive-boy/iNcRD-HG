"""Offline probe 7: crossed random effects beyond entity biases -- drug x miRNA-family, drug x miRNA-seed,
miRNA x drug-cluster (kmeans on ChemBERTa), drug x miRNA-arm -- fitted with ridge on labelled pairs."""
import sys, re, numpy as np, torch
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import load_dataset
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
from sklearn.cluster import KMeans
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
g = np.load(f'../mphgnn_plmf_20260921/cache/graph_{ds}.npz', allow_pickle=True)
mir, dru, pos, neg = load_dataset(ds); nm, nd = len(mir), len(dru)
names = mir.miRNA_Name.astype(str).values; seqs = mir.miRNA_Sequence.astype(str).str.upper().str.replace('T', 'U').values


def stem(n): return re.sub(r'-(3p|5p)$', '', n.lower().strip())


def family(n):
    s = stem(n).replace('hsa-', ''); s = re.sub(r'-[12]$', '', s); return re.sub(r'([0-9])[a-z]$', r'\1', s)


def codes(lab):
    u = {k: i for i, k in enumerate(sorted(set(lab)))}; return np.array([u[k] for k in lab]), len(u)


fam, nfam = codes([family(n) for n in names]); seed, nseed = codes([q[1:8] for q in seqs])
arm, narm = codes([('5p' if n.endswith('5p') else '3p' if n.endswith('3p') else 'x') for n in names])
x = g['drug_x']; x = (x - x.min(1, keepdims=True)) / (x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-8)
dcl = KMeans(8, n_init=10, random_state=0).fit_predict(x / np.linalg.norm(x, axis=1, keepdims=True)); ndcl = 8


def fit(L, Y, m, d, groups, lam_b=10.0):
    T = lambda v: torch.tensor(v, dtype=torch.float64)
    L, Y = T(L), T(Y); m = torch.tensor(m); d = torch.tensor(d)
    b = torch.zeros(nd, dtype=torch.float64, requires_grad=True); bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True)
    gp = [(torch.zeros(n, dtype=torch.float64, requires_grad=True), torch.tensor(idx), lam) for (idx, n, lam) in groups]
    opt = torch.optim.LBFGS([b, bm] + [p for p, _, _ in gp], max_iter=500, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        zz = L + b[d] + bm[m]
        for p, i, _ in gp:
            zz = zz + p[i]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(zz, Y, reduction='sum') + lam_b * ((b ** 2).sum() + (bm ** 2).sum())
        for p, _, l in gp:
            loss = loss + l * (p ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return b.detach().numpy(), bm.detach().numpy(), [p.detach().numpy() for p, _, _ in gp]


res = {}
specs = {'bias': lambda M, D: [],
         'd x fam': lambda M, D: [(D * nfam + fam[M], nd * nfam, 10.0)],
         'd x seed': lambda M, D: [(D * nseed + seed[M], nd * nseed, 10.0)],
         'd x arm': lambda M, D: [(D * narm + arm[M], nd * narm, 10.0)],
         'm x dcl': lambda M, D: [(M * ndcl + dcl[D], nm * ndcl, 10.0)]}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    for k, sp in specs.items():
        b, bm, gps = fit(L, Y, P[:, 0], P[:, 1], sp(P[:, 0], P[:, 1]))
        s = lte + b[te[:, 1]] + bm[te[:, 0]]
        for (idx, n, lam), gv in zip(sp(te[:, 0], te[:, 1]), gps):
            s = s + gv[idx]
        res.setdefault(k, []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base':
        continue
    m_ = np.mean(v, 0); w = np.mean([a[0] > b[0] for a, b in zip(v, res['base'])]); w2 = np.mean([a[1] > b[1] for a, b in zip(v, res['base'])])
    print(f'  {k:10s} {m_[0]:.4f}/{m_[1]:.4f}  d {m_[0]-bs[0]:+.4f}/{m_[1]-bs[1]:+.4f}  win {w:.1f}/{w2:.1f}')
