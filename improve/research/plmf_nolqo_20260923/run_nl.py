# -*- coding: utf-8 -*-
"""iNcRD-HG GNN body on PLMF-MDA -- round 8: architecture modules WITHOUT leave-query-out (LQO) training.

Training / evaluation protocol is exactly the plain base: one message graph = all training positives
(same graph in training and inference), full-batch BCE on the training pairs, RMSprop 1e-4, 300 epochs,
10% stratified hold-out of the training pairs for epoch selection (valsel), PLMF-MDA splits.
Only the PLMF-MDA release is read: miRNA names + mature sequences (k-mer features), drug SMILES
(ChemBERTa features, as in the base), and the training part of the association matrix.

Query invariance (instead of LQO): every pair-level channel is defined on quantities that cannot
contain the query edge itself -- simple paths of length >= 2 between the query endpoints, zero-diagonal
relations, and closed-form regressions in which the query cell is masked (exact rank-2 Woodbury
downdate).  For an observed training pair and an unobserved test pair the channels are therefore
computed in the same way; no edge is removed from the message graph and no extra forward pass is made.

Modules (all off = plain base; reproduces ../plmf_gnnarch_20260923/run_gnn.py plain up to CUDA scatter noise)
  M1 --enc mask --exact   query-masked message passing: the 2-layer encoder's endpoint states (and, with
                   --exact, the pooled --qncn intermediates) are replaced in closed form by their values on
                   the graph without the query edge; --enc content drops association edges instead
  M2 --qis [--qis_ext] --qncn
                   simple-path structural counts of the query pair (degree without the partner, simple
                   3-paths, content-bridged paths; ext: masked 5-walks, partner mean degree, masked profile
                   cosines) -> MLP, and neural common neighbours on simple 2-hop paths (m-d'-m'-d) +
                   relation-bridged neighbours (--ncn_rel): pooled encoder states of the intermediates
  M3 --ease LAMS   masked-cell closed-form collaborative regression (EASE, Steck 2019) on both sides,
                   query cell removed from the Gram by a rank-2 Woodbury downdate (--ease_learn: learn lambda)
  M4 --rel LIST / --ease_side LIST
                   dataset-internal miRNA relations used by --qis / --qncn: seed7, seed6 (mature-sequence
                   seed identity), pre (same precursor, 5p/3p arms), fam (name family); miRNA kernels added
                   to the miRNA-side Gram of --ease (kmer cosine Gram, relation Grams)
Controls / diagnostics
  --nomask, --ease_nomask   compute the channels with the query edge included
  --rel_shuffle    permute the internal relations (and the kmer kernel) across miRNAs
  --qis_shuffle    permute the qis rows across pairs of the same split part
  --check_enc      dense encoder == PyG encoder; masked states / masked EASE == explicit edge deletion
  --check_inv N    after training: logit of N training positives vs a full recomputation without their edge
  --lr_probe, --wide, --xr_pos   diagnostics that did not help (see REPORT_NOLQO.md section 8)
Output: results/{tag}_{DS}_{split}_seed{seed}.csv (+ _scores.csv, _curves.json, _args.json)"""
import argparse, json, math, os, random, re, sys, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GeneralConv, HeteroConv
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
from plmf_splits import get_split, load_dataset  # noqa: E402

GRAPH_CACHE = os.path.join(BASE, 'mphgnn_plmf_20260921', 'cache', 'graph_{}.npz')   # kmer(seq) + ChemBERTa(SMILES)

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['MDR', 'MDS'])
ap.add_argument('--split', required=True, choices=['5cv', 'single'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--knn', type=int, default=10)
ap.add_argument('--hidden', type=int, default=256)
ap.add_argument('--aggr', default='mean')
ap.add_argument('--layers', type=int, default=2)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--val', action='store_true')
ap.add_argument('--val_frac', type=float, default=0.1)
ap.add_argument('--enc', default='plain', choices=['plain', 'content', 'mask'])
ap.add_argument('--qis', action='store_true')
ap.add_argument('--qis_ext', action='store_true', help='add masked 5-paths, partner mean degree and masked content-profile cosines to --qis')
ap.add_argument('--qncn', action='store_true')
ap.add_argument('--ncn_dim', type=int, default=64)
ap.add_argument('--ncn_rel', default='co,knn', help='relations pooled by --qncn: co (bipartite 2-hop), knn, drug knn (dknn), and any --rel')
ap.add_argument('--rel', default='', help='comma list of seed7,seed6,pre,fam')
ap.add_argument('--rel_shuffle', action='store_true')
ap.add_argument('--qis_shuffle', action='store_true')
ap.add_argument('--ease', default='', help='comma list of ridge strengths, e.g. 10,50 (empty = off)')
ap.add_argument('--ease_side', default='', help='miRNA kernels added to the miRNA-side Gram: kmer and/or relation names')
ap.add_argument('--ease_alpha', type=float, default=1.0)
ap.add_argument('--ease_nomask', action='store_true', help='control: do not mask the query cell (inclusive fit)')
ap.add_argument('--nomask', action='store_true', help='control: compute qis/qncn with the query edge included')
ap.add_argument('--wide', action='store_true', help='wide & deep: linear logit on the query-invariant pair statistics (qis, ease) added to the MLP logit')
ap.add_argument('--wide_lr', type=float, default=1.0, help='learning-rate multiplier of the wide head')
ap.add_argument('--xr_pos', type=int, default=0, help='append one-hot nucleotides of the first N positions of the mature sequence to the miRNA node features (kNN graph unchanged)')
ap.add_argument('--ease_learn', action='store_true', help='learn the ridge strengths and side-kernel weights of --ease end-to-end (differentiable masked-cell EASE layer)')
ap.add_argument('--ease_lr', type=float, default=30.0, help='learning-rate multiplier of the EASE-layer parameters')
ap.add_argument('--lr_probe', action='store_true', help='diagnostic: logistic regression on the qis/ease pair statistics only (same pairs), no GNN')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--tag', default='plain')
ap.add_argument('--folds', default='')
ap.add_argument('--model_seed', type=int, default=0)
ap.add_argument('--check_enc', action='store_true', help='debug: verify the dense encoder equals the PyG encoder')
ap.add_argument('--exact', action='store_true', help='with --enc mask: also shift the pooled NCN intermediate states to their edge-removed values (exact invariance)')
ap.add_argument('--check_inv', type=int, default=0, help='debug: after training, compare the output of N training positives with a recomputation on the graph without their edge')
a = ap.parse_args()
DEV = torch.device(a.device)
REL = [k for k in a.rel.split(',') if k]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def rowminmax(x):
    x = torch.as_tensor(x, dtype=torch.float)
    mn = x.min(1, keepdim=True).values; mx = x.max(1, keepdim=True).values
    return (x - mn) / (mx - mn + 1e-8)


def knn_edges(x, k):
    x = torch.as_tensor(x, dtype=torch.float)
    xn = F.normalize(x, dim=1); s = xn @ xn.t(); s.fill_diagonal_(-1e9)
    nb = s.topk(k, dim=1).indices
    src = torch.arange(x.shape[0]).repeat_interleave(k); dst = nb.reshape(-1)
    return torch.unique(torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]), dim=1)


def stem(n):
    return re.sub(r'-(3p|5p)$', '', n.lower().strip())


def family(n):
    s = stem(n).replace('hsa-', ''); s = re.sub(r'-[12]$', '', s); return re.sub(r'([0-9])[a-z]$', r'\1', s)


def same_label(lab):
    lab = np.asarray(lab); R = (lab[:, None] == lab[None, :]).astype(np.float32); np.fill_diagonal(R, 0); return R


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, ds):
        g = np.load(GRAPH_CACHE.format(ds), allow_pickle=True)
        mir, dru, pos, neg = load_dataset(ds)
        self.names = mir['miRNA_Name'].astype(str).values; self.dbs = dru['DrugBank_ID'].astype(str).values
        self.seqs = mir['miRNA_Sequence'].astype(str).str.upper().str.replace('T', 'U').tolist()
        assert list(self.names) == list(g['mirna_names']), 'miRNA order mismatch'
        self.n_mir, self.n_drug = len(self.names), len(self.dbs)
        self.xr = rowminmax(g['mirna_kmer']); self.xd = rowminmax(g['drug_x'])
        self.sim_r = knn_edges(self.xr, a.knn); self.sim_d = knn_edges(self.xd, a.knn)
        self.xk = self.xr                                             # k-mer part (kNN graph, kernels, profiles)
        if a.xr_pos > 0:
            nt = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
            oh = np.zeros((self.n_mir, a.xr_pos, 4), np.float32)
            for i, s in enumerate(self.seqs):
                for p, c in enumerate(s[:a.xr_pos]):
                    if c in nt:
                        oh[i, p, nt[c]] = 1
            self.xr = torch.cat([self.xr, torch.as_tensor(oh.reshape(self.n_mir, -1))], 1)
        self.Sr = torch.zeros(self.n_mir, self.n_mir); self.Sr[self.sim_r[0], self.sim_r[1]] = 1
        self.Sd = torch.zeros(self.n_drug, self.n_drug); self.Sd[self.sim_d[0], self.sim_d[1]] = 1
        self.Sr, self.Sd = self.Sr.to(DEV), self.Sd.to(DEV)            # zero diagonal (kNN excludes self)
        lab = {'seed7': [q[1:8] for q in self.seqs], 'seed6': [q[1:7] for q in self.seqs],
               'pre': [stem(n) for n in self.names], 'fam': [family(n) for n in self.names]}
        self.rel = {}
        for k in REL:
            self.rel[k] = torch.as_tensor(same_label(lab[k]), device=DEV)
            R = self.rel[k]
            print(f'[rel] {k}: coverage {(R.sum(1) > 0).float().mean():.2f}, mean degree {R.sum(1).mean():.2f}', flush=True)
        xn = F.normalize(self.xk, dim=1)
        self.kmer_gram = (xn @ xn.t()).to(DEV)
        self.lab = lab


def build_graph(D, msg_pos, with_assoc=True):
    data = HeteroData()
    data['miRNA'].x = D.xr; data['drug'].x = D.xd
    data['miRNA', 'MiMi', 'miRNA'].edge_index = D.sim_r
    data['drug', 'DrugDrug', 'drug'].edge_index = D.sim_d
    if with_assoc:
        data['miRNA', 'MiDrug', 'drug'].edge_index = torch.as_tensor(msg_pos.T, dtype=torch.long)
    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)
    return data


# ----------------------------------------------------------------------------- encoder
class GNNEncoder(torch.nn.Module):
    """plain iNcRD-HG encoder: HeteroConv(GeneralConv(-1,-1,h, aggr=add)) x L, relation mean, leaky_relu.
    forward_dense() recomputes it with dense operators (identical output) and, given query pairs,
    returns their endpoint states with the query edge's messages removed (--enc mask)."""

    def __init__(self, data, hidden, layers):
        super().__init__()
        def convs():
            return HeteroConv({et: GeneralConv((-1, -1), hidden, aggr='add') for et in data.edge_types}, aggr=a.aggr)
        self.convs = torch.nn.ModuleList([convs() for _ in range(layers)])

    def forward(self, x_dict, ei_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, ei_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.leaky_relu(v, negative_slope=0.1) for k, v in x_dict.items()}
        return x_dict

    def forward_dense(self, xr, xd, Sr1, Sd1, A, r=None, d=None, q=None):
        """Sr1 / Sd1 = kNN + I, A = message-graph biadjacency.  If (r, d, q) are given, also returns the
        layer-L states of r (without the message from d) and of d (without the message from r) for the
        pairs whose query edge is present (q = A[r, d])."""
        assert len(self.convs) == 2
        c1, c2 = self.convs
        k_mm, k_dd = ('miRNA', 'MiMi', 'miRNA'), ('drug', 'DrugDrug', 'drug')
        k_md, k_dm = ('miRNA', 'MiDrug', 'drug'), ('drug', 'rev_MiDrug', 'miRNA')

        def layer(c, hr, hd):
            mm, dd, md, dm = c.convs[k_mm], c.convs[k_dd], c.convs[k_md], c.convs[k_dm]
            msg_dm = dm.lin_msg(hd); msg_md = md.lin_msg(hr)
            out_r = 0.5 * (Sr1 @ mm.lin_msg(hr) + mm.lin_self(hr) + A @ msg_dm + dm.lin_self(hr))
            out_d = 0.5 * (Sd1 @ dd.lin_msg(hd) + dd.lin_self(hd) + A.t() @ msg_md + md.lin_self(hd))
            return out_r, out_d, msg_dm, msg_md, (mm, dd, md, dm)

        p1r, p1d, m1_dm, m1_md, _ = layer(c1, xr, xd)
        h1r, h1d = F.leaky_relu(p1r, 0.1), F.leaky_relu(p1d, 0.1)
        zr, zd, m2_dm, m2_md, (mm, dd, md, dm) = layer(c2, h1r, h1d)
        if r is None:
            return zr, zd
        qq = q.view(-1, 1)
        # layer 1 without the query edge (pre-activation shift by the removed message)
        h1r_q = F.leaky_relu(p1r[r] - 0.5 * qq * m1_dm[d], 0.1)
        h1d_q = F.leaky_relu(p1d[d] - 0.5 * qq * m1_md[r], 0.1)
        dr_, dd_ = h1r_q - h1r[r], h1d_q - h1d[d]
        lin = lambda L, x: F.linear(x, L.weight)                           # bias-free part (biases are unchanged)
        zr_q = zr[r] + 0.5 * (lin(mm.lin_msg, dr_) + lin(mm.lin_self, dr_) + lin(dm.lin_self, dr_)) - 0.5 * qq * m2_dm[d]
        zd_q = zd[d] + 0.5 * (lin(dd.lin_msg, dd_) + lin(dd.lin_self, dd_) + lin(md.lin_self, dd_)) - 0.5 * qq * m2_md[r]
        corr = dict(dh_m=dr_, dh_d=dd_, W_dm=dm.lin_msg.weight, W_mm=mm.lin_msg.weight, W_md=md.lin_msg.weight, W_dd=dd.lin_msg.weight)
        return zr, zd, zr_q, zd_q, corr


# ----------------------------------------------------------------------------- query-invariant pair channels
def ease_masked(X, P, u, i, q, mask=True):
    """Masked-cell EASE.  items = columns of X, users = rows.  P = (X^T X + lam I + K)^-1.
    score(u, i) = sum_{j != i} X[u, j] B[j, i],  B[:, i] = -P'[:, i] / P'[i, i], where P' is the inverse Gram
    of X with the cell (u, i) set to 0 (only for q = X[u, i] = 1; rank-2 Woodbury downdate)."""
    x = X[u]                                                   # [B, I]
    Pi = P[i]                                                  # [B, I]  (P symmetric)
    ar = torch.arange(len(i), device=X.device)
    vPv = Pi[ar, i]
    plain_col = -Pi / vPv[:, None]
    if not mask:
        col = plain_col.clone(); col[ar, i] = 0
        return (x * col).sum(1)
    Pu = x @ P
    uPu = (x * Pu).sum(1); uPv = Pu[ar, i]
    M11, M12, M22 = -1 + uPu, -1 + uPv, vPv
    det = M11 * M22 - M12 * M12
    s1 = (M22 * uPv - M12 * vPv) / det; s2 = (-M12 * uPv + M11 * vPv) / det
    xcol = Pi - s1[:, None] * Pu - s2[:, None] * Pi
    new_col = -xcol / xcol[ar, i][:, None]
    col = torch.where(q[:, None] > 0.5, new_col, plain_col)
    col[ar, i] = 0
    xu = x.clone(); xu[ar, i] = 0
    return (xu * col).sum(1)


class PairStruct:
    """Structural quantities of the (single) message graph and the query-invariant pair channels."""

    def __init__(self, D, msg_pos, rel_perm=None):
        A = torch.zeros(D.n_mir, D.n_drug, device=DEV); A[msg_pos[:, 0], msg_pos[:, 1]] = 1.0
        self.A = A; self.deg_r = A.sum(1); self.deg_d = A.sum(0)
        self.Cm = A @ A.t(); self.Cd = A.t() @ A
        self.P3 = self.Cm @ A
        self.AG = A @ self.Cd; self.P5 = self.AG @ self.Cd
        self.ur = F.normalize(D.xk, dim=1).to(DEV); self.ud = F.normalize(D.xd, dim=1).to(DEV)
        self.Sr, self.Sd = D.Sr, D.Sd
        self.SrA = D.Sr @ A; self.ASd = A @ D.Sd; self.SrASd = D.Sr @ A @ D.Sd
        self.rel = {}
        for k, R in D.rel.items():
            if rel_perm is not None:
                R = R[rel_perm][:, rel_perm]
            self.rel[k] = R
        self.RA = {k: R @ A for k, R in self.rel.items()}
        self.ease_P = []
        self.ease_K = []
        for k in [s for s in a.ease_side.split(',') if s]:
            K = D.kmer_gram if k == 'kmer' else (self.rel[k] + torch.diag((self.rel[k].sum(1) > 0).float()))
            if k == 'kmer' and rel_perm is not None:
                K = K[rel_perm][:, rel_perm]
            self.ease_K.append(K)
        if a.ease and not a.ease_learn:
            for lam in [float(x) for x in a.ease.split(',')]:
                Gd = self.Cd + lam * torch.eye(D.n_drug, device=DEV)
                Gr = self.Cm + lam * torch.eye(D.n_mir, device=DEV)
                for k in [s for s in a.ease_side.split(',') if s]:
                    K = D.kmer_gram if k == 'kmer' else (self.rel[k] + torch.diag((self.rel[k].sum(1) > 0).float()))
                    if k == 'kmer' and rel_perm is not None:
                        K = K[rel_perm][:, rel_perm]
                    Gr = Gr + a.ease_alpha * K
                self.ease_P.append((torch.linalg.inv(Gd), torch.linalg.inv(Gr)))

    def qvals(self, r, d):
        return self.A[r, d] if not a.nomask else torch.zeros(len(r), device=DEV)

    def qis(self, r, d):
        q = self.qvals(r, d)
        dr = self.deg_r[r] - q; dd = self.deg_d[d] - q
        # simple 3-paths m-d'-m'-d (d' != d, m' != m): walks through the query edge are removed
        p3 = self.P3[r, d] - q * (self.deg_r[r] + self.deg_d[d] - 1)
        sra, asd, srasd = self.SrA[r, d], self.ASd[r, d], self.SrASd[r, d]      # zero-diagonal bridges
        f = [dr, dd, p3, p3 / (dr * dd + 1.0), p3 / (dr + 1.0), p3 / (dd + 1.0), sra, asd, srasd, sra / (dd + 1.0), asd / (dr + 1.0)]
        for k, R in self.rel.items():
            c = self.RA[k][r, d]; f += [c, c / (R[r].sum(1) + 1.0), c / (dd + 1.0)]
        out = torch.log1p(torch.stack(f, 1).clamp(min=0))
        if a.qis_ext:
            # masked 5-paths: with u = A[m], G = A^T A, the query edge removed gives
            # P5' = (G u - G e_d - (deg(m)-1) e_d)^T (G e_d - u)   (exact, verified against explicit removal)
            u = self.A[r]; Gd_ = self.Cd[d]; Gu = self.AG[r]
            ed = F.one_hot(d, self.A.shape[1]).float()
            p5m = ((Gu - Gd_ - (self.deg_r[r] - 1).unsqueeze(1) * ed) * (Gd_ - u)).sum(1)
            p5 = torch.where(q > 0.5, p5m, self.P5[r, d])
            # mean degree of the other partners of m / of d (the query edge excluded)
            nbr_r = (u @ self.deg_d - q * self.deg_d[d]) / (dr).clamp(min=1)
            nbr_d = (self.A[:, d].t() @ self.deg_r - q * self.deg_r[r]) / (dd).clamp(min=1)
            # content profile of the other partners vs the query node (masked)
            pd_ = u @ self.ud - q.unsqueeze(1) * self.ud[d]; pr_ = self.A[:, d].t() @ self.ur - q.unsqueeze(1) * self.ur[r]
            cos_d = (F.normalize(pd_, dim=1) * self.ud[d]).sum(1); cos_r = (F.normalize(pr_, dim=1) * self.ur[r]).sum(1)
            ext = torch.stack([torch.log1p(p5.clamp(min=0)), torch.log1p((p5 / (dr * dd + 1.0)).clamp(min=0)),
                               torch.log1p(nbr_r), torch.log1p(nbr_d), cos_d, cos_r], 1)
            out = torch.cat([out, ext], 1)
        return out

    def ease(self, r, d):
        q = self.qvals(r, d); out = []
        for Pd, Pr in self.ease_P:
            out.append(ease_masked(self.A, Pd, r, d, q, not a.ease_nomask))
            out.append(ease_masked(self.A.t(), Pr, d, r, q, not a.ease_nomask))
        x = torch.stack(out, 1)
        return torch.sign(x) * torch.log1p(x.abs() * 10.0)

    def ncn_rels(self):
        out = []
        for k in a.ncn_rel.split(','):
            if k in ('co', 'knn', 'dknn') or k in self.rel:
                out.append(k)
        return out

    def qncn(self, r, d, z, corr=None):
        """pooled encoder states of the intermediate nodes of simple 2-hop paths between r and d.
        corr (from GNNEncoder.forward_dense, --exact): the intermediates' states are shifted to their values on
        the graph without the query edge -- an intermediate miRNA m' hears d (A[m', d] = 1) and, if m is one of its
        kNN neighbours, m; an intermediate drug d' hears m (A[m, d'] = 1) and, if d is a kNN neighbour, d."""
        q = self.qvals(r, d).view(-1, 1)
        out = []
        if corr is not None:
            lin = lambda W, x: F.linear(x, W)
            cm_dm = 0.5 * lin(corr['W_dm'], corr['dh_d']); cm_mm = 0.5 * lin(corr['W_mm'], corr['dh_m'])
            cd_md = 0.5 * lin(corr['W_md'], corr['dh_m']); cd_dd = 0.5 * lin(corr['W_dd'], corr['dh_d'])

        def pool_m(w):
            den = w.sum(1, keepdim=True) + 1.0; e = w @ z['miRNA'] / den
            if corr is not None:
                e = e + cm_dm * (w.sum(1, keepdim=True) / den) + cm_mm * ((w * self.Sr[:, r].t()).sum(1, keepdim=True) / den)
            return e

        def pool_d(w):
            den = w.sum(1, keepdim=True) + 1.0; e = w @ z['drug'] / den
            if corr is not None:
                e = e + cd_md * (w.sum(1, keepdim=True) / den) + cd_dd * ((w * self.Sd[:, d].t()).sum(1, keepdim=True) / den)
            return e

        for k in self.ncn_rels():
            if k == 'co':
                # miRNA side: m' != m with A[m', d] = 1 and a shared drug d'' != d with m
                Ad = self.A[:, d].t()                                    # [B, nm]
                w = Ad * (self.Cm[r] - q * Ad)
                w[torch.arange(len(r), device=DEV), r] = 0
                out.append(pool_m(w))
                # drug side: d' != d with A[m, d'] = 1 and a shared miRNA m'' != m with d
                Ar = self.A[r]                                           # [B, nd]
                w = Ar * (self.Cd[:, d].t() - q * Ar)
                w[torch.arange(len(d), device=DEV), d] = 0
                out.append(pool_d(w))
            elif k == 'dknn':
                out.append(pool_d(self.A[r] * self.Sd[:, d].t()))
            else:
                R = self.Sr if k == 'knn' else self.rel[k]
                out.append(pool_m(self.A[:, d].t() * R[r]))
        return out

    def ncn_widths(self, hidden):
        w = []
        for k in self.ncn_rels():
            w += [a.ncn_dim, a.ncn_dim] if k == 'co' else [a.ncn_dim]
        return w


class EaseLayer(torch.nn.Module):
    """Differentiable masked-cell EASE (both sides): ridge strengths lambda and side-kernel weights alpha are
    learned end-to-end (gradients through the Cholesky inverse and the Woodbury downdate)."""

    def __init__(self, lams, n_k):
        super().__init__()
        self.H = len(lams)
        self.tl_d = torch.nn.Parameter(torch.log(torch.tensor(lams)))
        self.tl_r = torch.nn.Parameter(torch.log(torch.tensor(lams)))
        self.ta = torch.nn.Parameter(torch.full((self.H, max(n_k, 1)), math.log(a.ease_alpha)))

    @staticmethod
    def _inv(G):
        return torch.cholesky_inverse(torch.linalg.cholesky(G))

    def forward(self, ps, r, d):
        q = ps.qvals(r, d); out = []
        Id = torch.eye(ps.A.shape[1], device=DEV); Ir = torch.eye(ps.A.shape[0], device=DEV)
        for h in range(self.H):
            Pd = self._inv(ps.Cd + torch.exp(self.tl_d[h].clamp(-3, 8)) * Id)
            Gr = ps.Cm + torch.exp(self.tl_r[h].clamp(-3, 8)) * Ir
            for i, K in enumerate(ps.ease_K):
                Gr = Gr + torch.exp(self.ta[h, i].clamp(-8, 4)) * K
            Pr = self._inv(Gr)
            out.append(ease_masked(ps.A, Pd, r, d, q, not a.ease_nomask))
            out.append(ease_masked(ps.A.t(), Pr, d, r, q, not a.ease_nomask))
        x = torch.stack(out, 1)
        return torch.sign(x) * torch.log1p(x.abs() * 10.0)

    def describe(self):
        return dict(lam_d=torch.exp(self.tl_d).tolist(), lam_r=torch.exp(self.tl_r).tolist(), alpha=torch.exp(self.ta).tolist())


# ----------------------------------------------------------------------------- model
class Model(torch.nn.Module):
    def __init__(self, data, hidden, ps, n_qis, n_ease):
        super().__init__()
        self.encoder = GNNEncoder(data, hidden, a.layers)
        d = 2 * hidden
        self.ncn_lin = torch.nn.ModuleList()
        if a.qncn:
            for w in ps.ncn_widths(hidden):
                self.ncn_lin.append(Linear(hidden, w)); d += w
        if a.qis:
            self.pc = torch.nn.Sequential(Linear(n_qis, hidden), torch.nn.LeakyReLU(0.1), Linear(hidden, hidden)); d += hidden
        if a.ease_learn:
            self.ease_layer = EaseLayer([float(x) for x in a.ease.split(',')], len(ps.ease_K))
        if a.ease:
            self.eb = torch.nn.BatchNorm1d(n_ease)
            self.em = torch.nn.Sequential(Linear(n_ease, hidden), torch.nn.LeakyReLU(0.1), Linear(hidden, hidden)); d += hidden
        self.lin1 = Linear(d, 2 * hidden); self.lin2 = Linear(2 * hidden, hidden); self.lin3 = Linear(hidden, 1)
        if a.wide:
            self.wide = Linear(n_qis + n_ease, 1)
            torch.nn.init.zeros_(self.wide.weight); torch.nn.init.zeros_(self.wide.bias)

    def forward(self, data, eli, ps=None, qis=None, ease=None, dense=None):
        r, d = eli
        if a.enc == 'mask':
            xr, xd, Sr1, Sd1, A = dense
            zr, zd, zm, zdd, corr = self.encoder.forward_dense(xr, xd, Sr1, Sd1, A, r, d, ps.qvals(r, d))
            z = {'miRNA': zr, 'drug': zd}
            if not a.exact:
                corr = None
        else:
            z = self.encoder(data.x_dict, data.edge_index_dict)
            zm, zdd = z['miRNA'][r], z['drug'][d]; corr = None
        parts = [zm, zdd]
        if a.qncn:
            parts += [lin(e) for lin, e in zip(self.ncn_lin, ps.qncn(r, d, z, corr))]
        if a.qis:
            parts.append(self.pc(qis))
        if a.ease_learn:
            ease = self.ease_layer(ps, r, d)
        if a.ease:
            parts.append(self.em(self.eb(ease)))
        h = torch.cat(parts, -1)
        h = F.leaky_relu(self.lin1(h), 0.1)
        h = F.leaky_relu(self.lin2(h), 0.1)
        out = self.lin3(h).view(-1)
        if a.wide:
            wx = torch.cat([x for x in (qis, self.eb(ease) if a.ease else None) if x is not None], 1)
            out = out + self.wide(wx).view(-1)
        return out


def metrics(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s); p = (s >= 0.5).astype(int)
    return dict(AUC=roc_auc_score(y, s), AUPR=average_precision_score(y, s), ACC=accuracy_score(y, p),
                F1=f1_score(y, p, zero_division=0), MCC=matthews_corrcoef(y, p) if len(set(y)) > 1 else 0.0)


# ----------------------------------------------------------------------------- one fold
def run_fold(D, fold):
    tr, ytr, te, yte = get_split(a.dataset, a.split, fold, a.seed)
    rng = np.random.RandomState(a.seed * 100 + fold)
    val = None
    MS = a.model_seed * 7919
    if a.val:
        vidx = []
        for c in (0, 1):
            ids = np.where(ytr == c)[0]; rng.shuffle(ids); vidx.append(ids[:int(round(a.val_frac * len(ids)))])
        vidx = np.concatenate(vidx); mask = np.ones(len(ytr), bool); mask[vidx] = False
        val = (tr[vidx], ytr[vidx]); tr, ytr = tr[mask], ytr[mask]
    if MS:
        rng = np.random.RandomState(a.seed * 100 + fold + MS)
    set_seed(a.seed * 100 + fold + MS)
    train_pos = tr[ytr == 1]
    rel_perm = torch.as_tensor(rng.permutation(D.n_mir), device=DEV) if a.rel_shuffle else None
    full = build_graph(D, train_pos, with_assoc=(a.enc != 'content')).to(DEV)
    need_ps = a.qis or a.qncn or a.ease or a.enc == 'mask'
    ps = PairStruct(D, torch.as_tensor(train_pos, device=DEV), rel_perm) if need_ps else None
    eli_tr = torch.as_tensor(tr.T, dtype=torch.long, device=DEV); y_tr = torch.as_tensor(ytr, dtype=torch.float, device=DEV)
    eli_te = torch.as_tensor(te.T, dtype=torch.long, device=DEV)
    eli_va = torch.as_tensor(val[0].T, dtype=torch.long, device=DEV) if val is not None else None
    feats = {}
    if a.qis:
        f_tr = ps.qis(*eli_tr); mu, sd = f_tr.mean(0, keepdim=True), f_tr.std(0, keepdim=True) + 1e-6
        feats['qis'] = {k: (ps.qis(*e) - mu) / sd for k, e in (('tr', eli_tr), ('te', eli_te), ('va', eli_va)) if e is not None}
        if a.qis_shuffle:
            for k in feats['qis']:
                feats['qis'][k] = feats['qis'][k][torch.as_tensor(rng.permutation(len(feats['qis'][k])), device=DEV)]
    if a.ease and not a.ease_learn:
        feats['ease'] = {k: ps.ease(*e) for k, e in (('tr', eli_tr), ('te', eli_te), ('va', eli_va)) if e is not None}
    if a.lr_probe:
        from sklearn.linear_model import LogisticRegression
        blocks = [k for k in ('qis', 'ease') if k in feats]
        X = {p: torch.cat([feats[k][p] for k in blocks], 1).cpu().numpy() for p in ('tr', 'te')}
        if 'ease' in feats:
            mu_e, sd_e = feats['ease']['tr'].mean(0), feats['ease']['tr'].std(0) + 1e-6
            ne = feats['ease']['tr'].shape[1]
            for p in ('tr', 'te'):
                X[p][:, -ne:] = ((feats['ease'][p] - mu_e) / sd_e).cpu().numpy()
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(X['tr'], ytr)
        s = clf.predict_proba(X['te'])[:, 1]
        row = dict(dataset=a.dataset, split=a.split, tag=a.tag, fold=fold, seed=a.seed, valsel_auc=roc_auc_score(yte, s), valsel_aupr=average_precision_score(yte, s),
                   best_auc=roc_auc_score(yte, s), best_aupr=average_precision_score(yte, s), last_auc=roc_auc_score(yte, s), last_aupr=average_precision_score(yte, s), best_epoch=0, secs=0)
        print(f'== {a.tag} LR fold {fold}: {row["valsel_auc"]:.4f}/{row["valsel_aupr"]:.4f} ({X["tr"].shape[1]} cols)', flush=True)
        return row, [], pd.DataFrame(dict(fold=fold, m=te[:, 0], d=te[:, 1], y=yte, score=s))
    dense = None
    if a.enc == 'mask':
        Ir = torch.eye(D.n_mir, device=DEV); Id = torch.eye(D.n_drug, device=DEV)
        dense = (D.xr.to(DEV), D.xd.to(DEV), D.Sr + Ir, D.Sd + Id, ps.A)
    g = lambda key, part: feats[key][part] if key in feats else None
    n_qis = feats['qis']['tr'].shape[1] if a.qis else 0
    n_ease = (feats['ease']['tr'].shape[1] if not a.ease_learn else 2 * len(a.ease.split(','))) if a.ease else 0
    model = Model(full, a.hidden, ps, n_qis, n_ease).to(DEV)
    model.eval()
    with torch.no_grad():
        model.encoder(full.x_dict, full.edge_index_dict)                      # materialise lazy layers
        if a.check_enc and a.enc != 'content':
            z = model.encoder(full.x_dict, full.edge_index_dict)
            Ir = torch.eye(D.n_mir, device=DEV); Id = torch.eye(D.n_drug, device=DEV)
            A = torch.zeros(D.n_mir, D.n_drug, device=DEV); A[train_pos[:, 0], train_pos[:, 1]] = 1
            zr, zd = model.encoder.forward_dense(D.xr.to(DEV), D.xd.to(DEV), D.Sr + Ir, D.Sd + Id, A)
            print('[check] dense vs pyg max|diff|', float((zr - z['miRNA']).abs().max()), float((zd - z['drug']).abs().max()), flush=True)
            # masked states of a positive pair == dense states on the graph without that edge
            k = 0; m0, d0 = int(train_pos[k, 0]), int(train_pos[k, 1])
            _, _, zm_q, zd_q, _ = model.encoder.forward_dense(D.xr.to(DEV), D.xd.to(DEV), D.Sr + Ir, D.Sd + Id, A,
                                                           torch.tensor([m0], device=DEV), torch.tensor([d0], device=DEV), torch.ones(1, device=DEV))
            A2 = A.clone(); A2[m0, d0] = 0
            zr2, zd2 = model.encoder.forward_dense(D.xr.to(DEV), D.xd.to(DEV), D.Sr + Ir, D.Sd + Id, A2)
            print('[check] masked vs edge-removed max|diff|', float((zm_q[0] - zr2[m0]).abs().max()), float((zd_q[0] - zd2[d0]).abs().max()),
                  '| effect of the edge', float((zr[m0] - zr2[m0]).abs().max()), flush=True)
            if ps is not None and a.ease:
                # masked EASE of that positive == plain EASE on the graph without the edge
                lam = float(a.ease.split(',')[0])
                Gd = A2.t() @ A2 + lam * torch.eye(D.n_drug, device=DEV); Pd2 = torch.linalg.inv(Gd)
                s2 = ease_masked(A2, Pd2, torch.tensor([m0], device=DEV), torch.tensor([d0], device=DEV), torch.zeros(1, device=DEV), False)
                s1 = ease_masked(A, ps.ease_P[0][0], torch.tensor([m0], device=DEV), torch.tensor([d0], device=DEV), torch.ones(1, device=DEV), True)
                Gr = A2 @ A2.t() + lam * torch.eye(D.n_mir, device=DEV)
                for kk in [s for s in a.ease_side.split(',') if s]:
                    K = D.kmer_gram if kk == 'kmer' else (ps.rel[kk] + torch.diag((ps.rel[kk].sum(1) > 0).float()))
                    Gr = Gr + a.ease_alpha * K
                Pr2 = torch.linalg.inv(Gr)
                t2 = ease_masked(A2.t(), Pr2, torch.tensor([d0], device=DEV), torch.tensor([m0], device=DEV), torch.zeros(1, device=DEV), False)
                t1 = ease_masked(A.t(), ps.ease_P[0][1], torch.tensor([d0], device=DEV), torch.tensor([m0], device=DEV), torch.ones(1, device=DEV), True)
                print('[check] masked EASE vs edge-removed EASE', float(s1), float(s2), '|', float(t1), float(t2), flush=True)
    wide_p = list(model.wide.parameters()) if a.wide else []
    ease_p = list(model.ease_layer.parameters()) if a.ease_learn else []
    wid = {id(p) for p in wide_p + ease_p}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in wid])]
    if wide_p:
        groups.append(dict(params=wide_p, lr=a.lr * a.wide_lr))
    if ease_p:
        groups.append(dict(params=ease_p, lr=a.lr * a.ease_lr))
    opt = torch.optim.RMSprop(groups, lr=a.lr, alpha=0.99)
    hist = []; best = None; best_val = (-1, None); t0 = time.time(); best_scores = None

    def score(eli, part):
        return torch.sigmoid(model(full, eli, ps, g('qis', part), g('ease', part), dense)).cpu().numpy()

    for ep in range(1, a.epochs + 1):
        model.train(); opt.zero_grad()
        pred = model(full, eli_tr, ps, g('qis', 'tr'), g('ease', 'tr'), dense)
        loss = F.binary_cross_entropy_with_logits(pred, y_tr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s_te = score(eli_te, 'te')
            m = metrics(yte, s_te); m['epoch'] = ep; m['loss'] = float(loss)
            if eli_va is not None:
                s_va = score(eli_va, 'va')
                m['val_auc'] = roc_auc_score(val[1], s_va); m['val_aupr'] = average_precision_score(val[1], s_va)
                if m['val_auc'] > best_val[0]:
                    best_val = (m['val_auc'], dict(m)); best_scores = s_te.copy()
        hist.append({k: float(v) for k, v in m.items()})
        if best is None or m['AUC'] > best['AUC']:
            best = dict(m)
        last = dict(m)
        if ep % 50 == 0 or ep == 1:
            print(f'  ep{ep:3d} loss {m["loss"]:.4f} test AUC {m["AUC"]:.4f} AUPR {m["AUPR"]:.4f} best {best["AUC"]:.4f}'
                  + (f' val {m["val_auc"]:.4f}' if eli_va is not None else ''), flush=True)
    secs = time.time() - t0
    if a.ease_learn:
        print('   ease params', json.dumps(model.ease_layer.describe())[:300], flush=True)
    if a.check_inv:
        # invariance test: output for a training positive on the full graph (masked computation) vs a plain
        # recomputation on the graph from which that single edge has been deleted
        model.eval(); pos_idx = np.where(ytr == 1)[0][:a.check_inv]; diffs = []
        with torch.no_grad():
            e1 = eli_tr[:, pos_idx]
            o1 = model(full, e1, ps, feats['qis']['tr'][pos_idx] if a.qis else None, feats['ease']['tr'][pos_idx] if 'ease' in feats else None, dense)
            for j, k in enumerate(pos_idx):
                keep = ~((train_pos[:, 0] == tr[k, 0]) & (train_pos[:, 1] == tr[k, 1]))
                msg2 = train_pos[keep]
                full2 = build_graph(D, msg2, with_assoc=(a.enc != 'content')).to(DEV)
                ps2 = PairStruct(D, torch.as_tensor(msg2, device=DEV), rel_perm)
                e = eli_tr[:, [k]]
                q2 = (ps2.qis(*e) - mu) / sd if a.qis else None
                es2 = ps2.ease(*e) if 'ease' in feats else None
                dense2 = (dense[0], dense[1], dense[2], dense[3], ps2.A) if dense is not None else None
                o2 = model(full2, e, ps2, q2, es2, dense2)
                diffs.append(float((o1[j] - o2[0]).abs()))
        print(f'[check_inv] |logit(full graph, masked) - logit(edge deleted)| over {len(diffs)} training positives: '
              f'max {max(diffs):.2e} mean {np.mean(diffs):.2e}', flush=True)
    row = dict(dataset=a.dataset, split=a.split, tag=a.tag, fold=fold, seed=a.seed,
               best_auc=best['AUC'], best_aupr=best['AUPR'], best_epoch=best['epoch'],
               last_auc=last['AUC'], last_aupr=last['AUPR'], secs=secs)
    if best_val[1] is not None:
        b = best_val[1]
        row.update(valsel_epoch=b['epoch'], valsel_auc=b['AUC'], valsel_aupr=b['AUPR'], valsel_acc=b['ACC'], valsel_f1=b['F1'], valsel_mcc=b['MCC'], val_auc=best_val[0])
    print(f'== {a.tag} {a.dataset} {a.split} fold={fold} seed={a.seed} best_ep={best["epoch"]} AUC={best["AUC"]:.4f} AUPR={best["AUPR"]:.4f}'
          f' | last {last["AUC"]:.4f}/{last["AUPR"]:.4f}'
          + (f' | valsel ep{best_val[1]["epoch"]} {best_val[1]["AUC"]:.4f}/{best_val[1]["AUPR"]:.4f}' if best_val[1] else '')
          + f' ({secs:.0f}s)', flush=True)
    sc = pd.DataFrame(dict(fold=fold, m=te[:, 0], d=te[:, 1], y=yte, score=best_scores if best_scores is not None else s_te))
    return row, hist, sc


def main():
    D = Data(a.dataset)
    print(f'{a.dataset}: miRNA {D.n_mir} drug {D.n_drug} MiMi {D.sim_r.shape[1]} DrugDrug {D.sim_d.shape[1]} | {vars(a)}', flush=True)
    folds = [0] if a.split == 'single' else list(range(5))
    if a.folds:
        folds = [int(f) for f in a.folds.split(',')]
    rows, curves, scs = [], {}, []
    for f in folds:
        row, hist, sc = run_fold(D, f)
        rows.append(row); curves[str(f)] = hist; scs.append(sc)
    out = os.path.join(HERE, 'results', f'{a.tag}_{a.dataset}_{a.split}_seed{a.seed}.csv')
    d = pd.DataFrame(rows); d.to_csv(out, index=False)
    json.dump(curves, open(out.replace('.csv', '_curves.json'), 'w'))
    json.dump(vars(a), open(out.replace('.csv', '_args.json'), 'w'), indent=1)
    pd.concat(scs).to_csv(out.replace('.csv', '_scores.csv'), index=False)
    print(f'## {a.tag} {a.dataset} {a.split} seed{a.seed}: best {d.best_auc.mean():.4f}/{d.best_aupr.mean():.4f}'
          f' | last {d.last_auc.mean():.4f}/{d.last_aupr.mean():.4f}'
          + (f' | valsel {d.valsel_auc.mean():.4f}/{d.valsel_aupr.mean():.4f}' if 'valsel_auc' in d else ''), flush=True)


if __name__ == '__main__':
    main()
