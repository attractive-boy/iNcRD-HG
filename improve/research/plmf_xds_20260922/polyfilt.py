# -*- coding: utf-8 -*-
"""CPGF: Content-conditioned learnable Polynomial Graph Filter for miRNA-drug association.

Unified warm/cold link scorer (HL-GNN-style learnable heuristic on a content-augmented
heterogeneous adjacency):
    A_hat = [[g_r * S_r, A], [A^T, g_d * S_d]]          (A = support positives, S = content kNN kernels)
    A_n   = D^{-p} A_hat D^{-(1-p)}                      (degree-PRESERVING asymmetric normalisation, p learnable)
    score(r,d) = sum_k beta_k * z_k( [A_n^k]_{r,d} ) + w . [log deg_r, log deg_d, log cdeg_r, log cdeg_d] + b
All coefficients (beta_k, g_r, g_d, p, w, b) are learned end-to-end with BCE; the readout is linear so every
beta_k is a walk-type weight.  Cold drugs have no edge in A but reach the graph through S_d -> inductive.
Training uses the same 5-fold cross-fitted supports as plmf_innov (query pairs never sit in their own support).

Protocols: warm (PLMF 5cv, seed 42), cold (PLMF Table-2 cold-drug, degree 11-20, 5 seeds), cold_mirna.
Optional --export writes the learned filter columns so they can be stacked with ctx/diff x LR.
"""
import argparse, os, sys, time, csv, math
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
for p in ('mphgnn_plmf_20260921', 'dlstmda_strongest_20260920', 'plmf_innov_20260921', 'plmf_frontier_20260922'):
    sys.path.insert(0, os.path.join(BASE, p))
from plmf_splits import get_split  # noqa
import run_probe as RP  # noqa
import cold_probe as CP  # noqa
from bank import Bank  # noqa

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CPGF(torch.nn.Module):
    def __init__(self, Sr, Sd, K=6, per_k_norm=True, init_p=0.5):
        super().__init__()
        self.Sr, self.Sd = Sr, Sd; self.nr, self.nd = Sr.shape[0], Sd.shape[0]; self.K = K
        self.beta = torch.nn.Parameter(torch.zeros(K)); self.beta.data[0] = 1.0
        self.lg_r = torch.nn.Parameter(torch.tensor(math.log(0.5))); self.lg_d = torch.nn.Parameter(torch.tensor(math.log(0.5)))
        self.p_logit = torch.nn.Parameter(torch.tensor(math.log(init_p / (1 - init_p))))
        self.w = torch.nn.Parameter(torch.zeros(4)); self.b = torch.nn.Parameter(torch.zeros(1))
        self.per_k_norm = per_k_norm

    def powers(self, A):
        """A: [nr, nd] support (float tensor). Returns list of K rd-blocks of A_n^k and degree vectors."""
        g_r, g_d = torch.exp(self.lg_r), torch.exp(self.lg_d); p = torch.sigmoid(self.p_logit)
        top = torch.cat([g_r * self.Sr, A], 1); bot = torch.cat([A.t(), g_d * self.Sd], 1)
        Ah = torch.cat([top, bot], 0)
        deg = Ah.sum(1).clamp_min(1e-6)
        An = (deg ** (-p))[:, None] * Ah * (deg ** (-(1 - p)))[None, :]
        blocks = []; P = An
        for k in range(self.K):
            blocks.append(P[:self.nr, self.nr:])
            if k < self.K - 1:
                P = P @ An
        degr = A.sum(1); degd = A.sum(0)
        return blocks, degr, degd

    def features(self, A, q):
        blocks, degr, degd = self.powers(A)
        r, d = q[:, 0], q[:, 1]
        cols = []
        for B in blocks:
            v = B[r, d]
            if self.per_k_norm:
                v = v / (B.std() + 1e-8)
            cols.append(v)
        Z = torch.stack(cols, 1)
        side = torch.stack([torch.log1p(degr[r]), torch.log1p(degd[d]),
                            torch.log1p(self.Sr.sum(1)[r]), torch.log1p(self.Sd.sum(1)[d])], 1)
        return Z, side

    def forward(self, A, q):
        Z, side = self.features(A, q)
        return (Z * self.beta).sum(1) + side @ self.w + self.b


def to_A(sup, nr, nd):
    A = torch.zeros(nr, nd, device=dev); A[sup[:, 0], sup[:, 1]] = 1.0; return A


def crossfit_groups(tr, ytr, seed, fit='warm'):
    """warm: pair-level 5-fold (query pairs absent from support, their drugs present).
    drugcv: drug-grouped 5-fold (query drugs have NO edge in support -> simulates cold drugs).
    both: union of the two group sets (unified warm/cold training)."""
    from sklearn.model_selection import GroupKFold
    groups = []
    if fit in ('warm', 'both'):
        for f_, qy in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, ytr.astype(int)):
            groups.append((tr[f_[ytr[f_] == 1]], qy))
    if fit in ('drugcv', 'both'):
        drugs = tr[:, 1]; ud = np.unique(drugs); rng = np.random.RandomState(seed)
        grp_of = dict(zip(ud, rng.permutation(len(ud)) % 5)); g = np.array([grp_of[x] for x in drugs])
        for f_, qy in GroupKFold(5).split(tr, ytr, g):
            groups.append((tr[f_[ytr[f_] == 1]], qy))
    return groups


def train_eval(model, tr, ytr, te, nr, nd, seed, epochs=300, lr=0.02, wd=1e-4, verbose=False, fit='warm'):
    torch.manual_seed(seed)
    groups = [(to_A(s, nr, nd), torch.as_tensor(tr[q], device=dev), torch.as_tensor(ytr[q], device=dev, dtype=torch.float32))
              for s, q in crossfit_groups(tr, ytr, seed, fit)]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        model.train(); opt.zero_grad(); loss = 0
        for A, q, y in groups:
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(model(A, q), y)
        loss = loss / len(groups); loss.backward(); opt.step()
        if verbose and ep % 50 == 0:
            print(f'    ep {ep} loss {loss.item():.4f} beta {model.beta.data.cpu().numpy().round(3)} g_r {model.lg_r.exp().item():.3f} g_d {model.lg_d.exp().item():.3f} p {torch.sigmoid(model.p_logit).item():.3f}', flush=True)
    model.eval()
    with torch.no_grad():
        Afull = to_A(tr[ytr == 1], nr, nd); qt = torch.as_tensor(te, device=dev)
        s = torch.sigmoid(model(Afull, qt)).cpu().numpy()
        # exported columns (train cross-fitted, test full support) for stacking
        Ztr = np.zeros((len(ytr), model.K + 4), np.float32)
        exp_groups = crossfit_groups(tr, ytr, seed, 'drugcv' if fit == 'drugcv' else 'warm')
        for (sup, qy) in exp_groups:
            A = to_A(sup, nr, nd); q = torch.as_tensor(tr[qy], device=dev)
            Z, side = model.features(A, q); Ztr[qy] = torch.cat([Z, side], 1).cpu().numpy()
        Z, side = model.features(Afull, qt); Zte = torch.cat([Z, side], 1).cpu().numpy()
    return s, Ztr, Zte


def stack_lr(bank, tr, ytr, te, fams, seed, Ztr, Zte, C=0.3):
    Xtr, Xte = RP.build_xy(bank, tr, ytr, te, fams, seed)
    both = np.concatenate([Ztr, Zte]); both = np.log1p(np.abs(both)) * np.sign(both)
    mu, sd = both[:len(Ztr)].mean(0), both[:len(Ztr)].std(0) + 1e-6
    Xtr = np.concatenate([Xtr, (both[:len(Ztr)] - mu) / sd], 1); Xte = np.concatenate([Xte, (both[len(Ztr):] - mu) / sd], 1)
    m = LogisticRegression(max_iter=5000, C=C).fit(Xtr, ytr.astype(int)); return m.predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--protocol', default='warm', choices=['warm', 'cold', 'cold_mirna'])
    ap.add_argument('--K', type=int, default=6); ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--wd', type=float, default=1e-4); ap.add_argument('--graph', default='kernel'); ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--no_content', action='store_true', help='ablation: S_r=S_d=0 (plain bipartite HL-GNN)')
    ap.add_argument('--shuffle_content', action='store_true', help='control: permute S_d rows/cols')
    ap.add_argument('--fix_p', type=float, default=-1, help='freeze normalisation exponent (0.5 = symmetric)')
    ap.add_argument('--stack', default='', help="families to stack with the learned filter columns in an LR, e.g. 'ctx,diff'")
    ap.add_argument('--fit', default='warm', choices=['warm', 'drugcv', 'both']); ap.add_argument('--seeds', default='42'); ap.add_argument('--tag', default=''); ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    xr, xd = RP.load_nodes(a.dataset); nr, nd = xr.shape[0], xd.shape[0]
    sr = RP.build_adj(xr, a.knn, 'cosine', a.graph); sd = RP.build_adj(xd, a.knn, 'tanimoto', a.graph)
    bank = Bank(xr, xd, sr, sd) if a.stack else None
    if a.no_content:
        sr = np.zeros_like(sr); sd = np.zeros_like(sd)
    if a.shuffle_content:
        rng = np.random.RandomState(0); perm = rng.permutation(nd); sd = sd[perm][:, perm]
    Sr = torch.as_tensor(sr, device=dev, dtype=torch.float32); Sd = torch.as_tensor(sd, device=dev, dtype=torch.float32)
    rows = []
    if a.protocol == 'warm':
        splits = [(f, 42, get_split(a.dataset, '5cv', f, 42)) for f in range(5)]
    else:
        D = CP.load_all(a.dataset); A = np.zeros((nr, nd), np.int8); A[D['pos'][:, 0], D['pos'][:, 1]] = 1
        if a.protocol == 'cold':
            ddeg = A.sum(0); ents = np.where((ddeg >= 11) & (ddeg <= 20))[0]; side = 'drug'
        else:
            mdeg = A.sum(1); ents = np.where((mdeg >= 5) & (mdeg <= 6))[0]; side = 'mirna'
        splits = [(0, s, CP.cold_split(D, ents, s * 10, side, 0)) for s in [int(x) for x in a.seeds.split(',')]]
    for f, seed, (tr, ytr, te, yte) in splits:
        t0 = time.time()
        model = CPGF(Sr, Sd, K=a.K).to(dev)
        if a.fix_p >= 0:
            model.p_logit.data.fill_(math.log(a.fix_p / (1 - a.fix_p))); model.p_logit.requires_grad_(False)
        s, Ztr, Zte = train_eval(model, tr, ytr, te, nr, nd, seed * 100 + f, a.epochs, a.lr, a.wd, a.verbose, a.fit)
        auc, ap_ = roc_auc_score(yte, s), average_precision_score(yte, s)
        rec = dict(fold=f, seed=seed, auc=auc, aupr=ap_)
        if a.protocol != 'warm':
            rec['macro_auc'] = CP.macro_auc(te, yte, s, 1 if side == 'drug' else 0)[0]
        msg = f'  fold {f} seed {seed}: AUC {auc:.4f} AUPR {ap_:.4f}'
        if a.stack:
            s2 = stack_lr(bank, tr, ytr, te, set(a.stack.split(',')), seed * 100 + f, Ztr, Zte)
            rec['auc_stack'], rec['aupr_stack'] = roc_auc_score(yte, s2), average_precision_score(yte, s2)
            msg += f' | stack {rec["auc_stack"]:.4f} {rec["aupr_stack"]:.4f}'
        msg += f' | beta {model.beta.data.cpu().numpy().round(2)} g_r {model.lg_r.exp().item():.2f} g_d {model.lg_d.exp().item():.2f} p {torch.sigmoid(model.p_logit).item():.2f} w {model.w.data.cpu().numpy().round(2)} ({time.time()-t0:.0f}s)'
        print(msg, flush=True); rows.append(rec)
    df = pd.DataFrame(rows)
    tag = a.tag or f'cpgf_K{a.K}_{a.fit}' + ('_nocontent' if a.no_content else '') + ('_shufS' if a.shuffle_content else '') + (f'_p{a.fix_p}' if a.fix_p >= 0 else '') + (f'_stack[{a.stack}]' if a.stack else '')
    line = f'## {tag} {a.dataset} {a.protocol}: AUC {df.auc.mean():.4f}±{df.auc.std():.4f} AUPR {df.aupr.mean():.4f}±{df.aupr.std():.4f}'
    if 'macro_auc' in df: line += f' macroAUC {df.macro_auc.mean():.4f}'
    if a.stack: line += f' | stack AUC {df.auc_stack.mean():.4f}±{df.auc_stack.std():.4f} AUPR {df.aupr_stack.mean():.4f}±{df.aupr_stack.std():.4f}'
    print(line, flush=True)
    out = os.path.join(HERE, 'results', 'polyfilt.csv'); new = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new: w.writerow(['tag', 'dataset', 'protocol', 'K', 'epochs', 'auc', 'auc_sd', 'aupr', 'aupr_sd', 'macro_auc', 'auc_stack', 'aupr_stack', 'n'])
        w.writerow([tag, a.dataset, a.protocol, a.K, a.epochs, f'{df.auc.mean():.6f}', f'{df.auc.std():.6f}', f'{df.aupr.mean():.6f}', f'{df.aupr.std():.6f}',
                    f'{df.macro_auc.mean():.6f}' if 'macro_auc' in df else '', f'{df.auc_stack.mean():.6f}' if a.stack else '', f'{df.aupr_stack.mean():.6f}' if a.stack else '', len(df)])


if __name__ == '__main__':
    main()
