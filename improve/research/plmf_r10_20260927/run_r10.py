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
ap.add_argument('--unl', type=float, default=0.0, help='round 10: each epoch add unl x |train pairs| cells sampled from the unlabeled pool (not a training positive, not a labelled / val / test pair) as label-0 pairs')
ap.add_argument('--unl_w', type=float, default=1.0, help='weight of the unlabeled-cell BCE term')
ap.add_argument('--unl_pop', default='0,0', help='a,b: sample unlabeled cells with prob ~ (deg_drug+1)^a (deg_miRNA+1)^b (training graph degrees)')
ap.add_argument('--hpa', action='store_true', help='round 10: hollow partner attention -- m attends over the other partners of d (and d over the other partners of m) with content keys')
ap.add_argument('--hpa_seq', action='store_true', help='--hpa: add positional one-hot of the mature sequence to the miRNA attention input')
ap.add_argument('--hpa_dim', type=int, default=64)
ap.add_argument('--lw_d', type=float, default=0.0, help='round 10: weight of the drug-wise softmax (listwise) loss: each positive against the same-drug negatives of the batch')
ap.add_argument('--lw_m', type=float, default=0.0, help='round 10: weight of the miRNA-wise softmax loss')
ap.add_argument('--dump', action='store_true', help='save the hollow qis / ease features of train / val / test pairs to cache/ and exit')
ap.add_argument('--xfeat', default='', help='diagnostic: cached cross-fitted LR-pipeline pair features (ctx+diff, K=10) as an extra channel; value = column index spec all|diffrz')
ap.add_argument('--hac', default='', help='round 10: hollow axial context -- comma list of closed-form score maps (ease, katz, bridge) whose query-hollow row / column context (z-scores, percentiles) is fed to the decoder')
ap.add_argument('--hac_katz', default='0.3,0.7', help='Katz decay as a fraction of 1/sigma_max(A)')
ap.add_argument('--hac_stats', default='z,p', help='context statistics: z (row/col z-score), p (row/col percentile)')
ap.add_argument('--hac_nomask', action='store_true', help='control: row / column context taken from the full training graph (query edge included)')
ap.add_argument('--hac_shuffle', action='store_true', help='control: permute the hac feature rows across pairs of the same split part')
ap.add_argument('--check_hac', type=int, default=0, help='debug: per-feature |hac(masked) - hac(edge deleted)| over N training positives, then exit')
ap.add_argument('--unl_strict', action='store_true', help='--unl pool = every cell that is not a training positive (val / test cells included, so the pool reveals nothing about the evaluation pairs)')
ap.add_argument('--hdc', default='', help='round 10: hollow diffusion context -- comma list of degree-normalised diffusions computed exactly on the per-query graph A - A[m,d] e_m e_d^T: ppr30,ppr60,katz30,katz60 (walk from the miRNA, *_rd) and the same from the drug (*_dr), path7')
ap.add_argument('--hdc_steps', type=int, default=12)
ap.add_argument('--xfit', type=int, default=0, help='diagnostic: compute the training pairs\' qis / ease / hdc channels K-fold cross-fitted (graph without the inner fold\'s positives) instead of query-hollow')
ap.add_argument('--val_graph', action='store_true', help='round 10: in-graph (hollow) validation -- the validation positives stay in the message graph; every channel is query-hollow, so a validation pair never sees its own edge, and the test-time graph holds all training positives')
ap.add_argument('--krloo', default='', help='round 10: learned-kernel leave-one-out ridge channel; comma list of ridge strengths (e.g. 1,10)')
ap.add_argument('--krloo_dim', type=int, default=64)
ap.add_argument('--krloo_lr', type=float, default=30.0)
ap.add_argument('--moe', type=int, default=0, help='round 10: K-expert decoder gated by the hollow pair statistics (qis)')
ap.add_argument('--chdrop', type=float, default=0.0, help='round 10: channel-block dropout in the decoder (each channel block of each pair zeroed with prob p during training, rescaled)')
ap.add_argument('--val_graph_test', action='store_true', help='control (OGB practice): validation positives are added to the message graph for test-time inference only')
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
        nt4 = {'A': 0, 'C': 1, 'G': 2, 'U': 3}; oh = np.zeros((self.n_mir, 22, 4), np.float32)
        for i, sq in enumerate(self.seqs):
            for p_, c in enumerate(sq[:22]):
                if c in nt4:
                    oh[i, p_, nt4[c]] = 1
        self.seq_oh = torch.as_tensor(oh.reshape(self.n_mir, -1))
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
        self.hpa_xr = torch.cat([D.xk, D.seq_oh], 1).to(DEV) if a.hpa_seq else D.xk.to(DEV)
        self.hpa_xd = D.xd.to(DEV)
        self.kr_xr = torch.cat([D.xk, D.seq_oh], 1).to(DEV)
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



# ----------------------------------------------------------------------------- round 10: hollow axial context
class AxialContext:
    """Hollow axial context (--hac).  For every closed-form score map F (masked EASE on the drug and miRNA side, bipartite
    Katz, content bridges) and a query (m, d), the value F'[m, d], the row F'[m, :] and the column F'[:, d] are taken on
    the graph without the query cell (exact rank-2 Woodbury downdates; q = A[m, d] = 0 pairs use the full-graph maps).
    Output per map: the query's z-score and percentile within its row and its column."""

    def __init__(self, ps, betas=None):
        self.ps = ps; A = ps.A.double(); self.A = A                         # float64: exact downdates are ill-conditioned for large Katz decay
        self.kinds = [k for k in a.hac.split(',') if k]
        self.maps = []          # (name, kind, params)
        if 'ease' in self.kinds:
            lams = [float(x) for x in a.ease.split(',')]
            for li, lam in enumerate(lams):
                Gr = A @ A.t() + lam * torch.eye(A.shape[0], device=DEV, dtype=torch.float64)
                for k in ps.ease_K:
                    Gr = Gr + a.ease_alpha * k.double()
                Pd = torch.linalg.inv(A.t() @ A + lam * torch.eye(A.shape[1], device=DEV, dtype=torch.float64))
                self.maps.append(('ease_d%d' % li, 'ease', (A, Pd, False)))
                self.maps.append(('ease_r%d' % li, 'ease', (A.t(), torch.linalg.inv(Gr), True)))
        if 'katz' in self.kinds:
            smax = torch.linalg.matrix_norm(A, ord=2)
            nd = A.shape[1]
            self.betas = betas or [c / float(smax) for c in [float(x) for x in a.hac_katz.split(',')]]    # global scalars, fixed per fold
            for c, beta in zip([float(x) for x in a.hac_katz.split(',')], self.betas):
                R = torch.linalg.inv(torch.eye(nd, device=DEV, dtype=torch.float64) - beta ** 2 * (A.t() @ A))
                self.maps.append(('katz%g' % c, 'katz', (A, R, beta)))
        if 'bridge' in self.kinds:
            self.maps += [('srA', 'srA', None), ('ASd', 'ASd', None), ('srASd', 'srASd', None)]
        self.full = {n: self._full(k, p) for n, k, p in self.maps}
        st = a.hac_stats.split(',')
        self.width = len(self.maps) * (2 * ('z' in st) + 2 * ('p' in st))

    def _full(self, kind, p):
        ps = self.ps
        if kind == 'ease':
            X, P, tr = p
            B = -P / torch.diagonal(P)[None, :]; B = B - torch.diag(torch.diagonal(B))
            S = X @ B
            return S.t() if tr else S
        if kind == 'katz':
            X, R, beta = p; return beta * X @ R
        return {'srA': ps.SrA, 'ASd': ps.ASd, 'srASd': ps.SrASd}[kind].double()

    # exact query-hollow (value, row over drugs, column over miRNAs) for q = 1 queries
    def _ease_exact(self, X, P, u, i):
        """users x items X, P = (X^T X + lam I + K)^-1; query cell (u, i).  Returns value, user row over items, item column over users."""
        ar = torch.arange(len(i), device=DEV)
        x = X[u]; Pu = x @ P; Pi = P[i]
        uPu = (x * Pu).sum(1); uPv = Pu[ar, i]; vPv = Pi[ar, i]
        M = torch.stack([torch.stack([-1 + uPu, -1 + uPv], 1), torch.stack([-1 + uPv, vPv], 1)], 1)
        Minv = torch.linalg.inv(M)
        W = torch.stack([Pu, Pi], 2)                                            # [B, I, 2]
        dg = torch.diagonal(P)[None, :] - torch.einsum('bka,bac,bkc->bk', W, Minv, W)
        xp = x.clone(); xp[ar, i] = 0
        xP = xp @ P - torch.einsum('ba,bac,bkc->bk', torch.einsum('bk,bka->ba', xp, W), Minv, W)
        row = -xP / dg + xp                                                    # sum_{j != k} x'_j B'[j, k]
        coli = Pi - torch.einsum('bka,bac,bc->bk', W, Minv, W[ar, i])          # P'[:, i]
        bcol = -coli / coli[ar, i][:, None]; bcol[ar, i] = 0
        col = bcol @ X.t()                                                     # [B, users]; X' B'[:, i] = X B'[:, i]
        return row[ar, i], row, col

    def _katz_exact(self, X, R, beta, u, i):
        ar = torch.arange(len(i), device=DEV)
        x = X[u]; Ru = x @ R; Ri = R[i]
        ib = 1.0 / beta ** 2
        M = torch.stack([torch.stack([ib + (x * Ru).sum(1), ib + Ru[ar, i]], 1), torch.stack([ib + Ru[ar, i], Ri[ar, i]], 1)], 1)
        Minv = torch.linalg.inv(M)
        W = torch.stack([Ru, Ri], 2)
        xp = x.clone(); xp[ar, i] = 0
        row = beta * (xp @ R - torch.einsum('ba,bac,bkc->bk', torch.einsum('bk,bka->ba', xp, W), Minv, W))
        coli = Ri - torch.einsum('bka,bac,bc->bk', W, Minv, W[ar, i])          # R'[:, i]
        col = beta * (coli @ X.t()); col[ar, u] = col[ar, u] - beta * coli[ar, i]
        return row[ar, i], row, col

    def _stats(self, v, row, col):
        st = a.hac_stats.split(','); out = []
        if 'z' in st:
            out += [(v - row.mean(1)) / (row.std(1) + 1e-6), (v - col.mean(1)) / (col.std(1) + 1e-6)]
        if 'p' in st:                                                          # mid-rank percentile, ties within a relative tolerance
            for vec in (row, col):
                eps = 1e-6 * (vec.abs().max(1).values + 1e-12)[:, None]
                out.append(((vec < v[:, None] - eps).double() + 0.5 * ((vec - v[:, None]).abs() <= eps).double()).mean(1))
        return out

    def features(self, r, d, chunk=4096):
        outs = []
        for s0 in range(0, len(r), chunk):
            outs.append(self._features(r[s0:s0 + chunk], d[s0:s0 + chunk]))
        return torch.cat(outs)

    def _features(self, r, d):
        q = self.A[r, d] if not a.hac_nomask else torch.zeros(len(r), device=DEV)
        hot = q > 0.5; ih = hot.nonzero().view(-1)
        cols = []
        for name, kind, p in self.maps:
            F_ = self.full[name]
            v = F_[r, d].clone(); row = F_[r].clone(); col = F_[:, d].t().clone()
            if len(ih):
                rh, dh = r[ih], d[ih]
                if kind == 'ease':
                    X, P, tr = p
                    if not tr:
                        vv, rr_, cc_ = self._ease_exact(X, P, rh, dh)             # user = miRNA, item = drug
                        v[ih], row[ih], col[ih] = vv, rr_, cc_
                    else:
                        vv, rr_, cc_ = self._ease_exact(X, P, dh, rh)             # user = drug, item = miRNA
                        v[ih], col[ih], row[ih] = vv, rr_, cc_
                elif kind == 'katz':
                    X, R, beta = p
                    vv, rr_, cc_ = self._katz_exact(X, R, beta, rh, dh)
                    v[ih], row[ih], col[ih] = vv, rr_, cc_
                elif kind == 'srA':                                             # column d of S_r A' loses S_r[:, m]
                    col[ih] = col[ih] - self.ps.Sr[:, rh].t().double()
                elif kind == 'ASd':                                             # row m of A' S_d loses S_d[d, :]
                    row[ih] = row[ih] - self.ps.Sd[dh].double()
            cols += self._stats(v, row, col)
        return torch.stack(cols, 1).float()


# ----------------------------------------------------------------------------- round 10: hollow diffusion context
class HollowDiffusion:
    """--hdc.  Degree-normalised bipartite diffusions (PPR / Katz / path-7 of ../plmf_innov_20260921/bank.py::diffusion)
    evaluated on the per-query graph A' = A - q e_m e_d^T (q = A[m, d]) exactly, batched: the walk operator
    G' = Dd'^-1 A'^T Dr'^-1 A' differs from G only by the rank-2 change of row m and the rescaling of the degree of d,
    so every step is a shared dense product plus two per-query corrections.  For each map D' the query value, its row
    D'[m, :] and its column D'[:, d] give (value, row z, column z, row percentile, column percentile)."""
    CFG = {'ppr30': (0.3, 'ppr'), 'ppr60': (0.6, 'ppr'), 'katz30': (0.01, 'katz'), 'katz60': (0.05, 'katz')}

    def __init__(self, A):
        self.A = A.double()
        self.maps = [k for k in a.hdc.split(',') if k]
        self.width = 5 * sum(2 if k in self.CFG else 1 for k in self.maps)

    @staticmethod
    def coeffs(kind, x, K):
        return [x * (1 - x) ** (2 * k + 1) if kind == 'ppr' else x ** (2 * k + 1) for k in range(K)]

    def _walk(self, A, u, i, q, cs, left='norm'):
        """rows / columns of D = L Σ_k c_k G^k with L = Dr^-1 A (left='norm') or A (left='raw'), G = Dd^-1 A^T Dr^-1 A,
        on A' = A - q e_u e_i^T.  Returns value D'[u, i], row D'[u, :] (len I), column D'[:, i] (len U)."""
        B = len(u); ar = torch.arange(B, device=DEV); qq = q.view(-1, 1)
        dr = A.sum(1); dd = A.sum(0)
        a = A[u]                                                         # [B, I]
        ap = a.clone(); ap[ar, i] = ap[ar, i] - q                          # a'
        dru = (dr[u] - q).clamp(min=1).view(-1, 1)                         # dr'_u
        ddp = dd.clamp(min=1).expand(B, -1).clone(); ddp[ar, i] = (dd[i] - q).clamp(min=1)   # dd' per query
        drc = dr.clamp(min=1)
        H = A.t() @ (A / drc.view(-1, 1))                                  # A^T Dr^-1 A (nd x nd)
        au = a / dr[u].clamp(min=1).view(-1, 1)                            # removed contribution a a^T / dr_u  -> use (a, au)
        apu = ap / dru

        def Hx(x):                                                         # x H' for row vectors x [B, I]  (H' symmetric)
            return x @ H - (x * a).sum(1, keepdim=True) * au + (x * ap).sum(1, keepdim=True) * apu

        # row of D': start = L'[u] (row u of Dr'^-1 A' or A')
        x = apu.clone() if left == 'norm' else ap.clone()
        row = cs[0] * x
        for c in cs[1:]:
            x = Hx(x / ddp)                                                # x G' = (x Dd'^-1) H'
            row = row + c * x
        # column of D': D'[:, i] = L' g, g = Σ c_k G'^k e_i ; G' y = Dd'^-1 H' y
        y = F.one_hot(i, A.shape[1]).double()
        g = cs[0] * y
        for c in cs[1:]:
            y = Hx(y) / ddp
            g = g + c * y
        col = g @ A.t()                                                    # A g  [B, U]
        col[ar, u] = col[ar, u] - q * g[ar, i]                             # A' g
        if left == 'norm':
            den = drc.expand(B, -1).clone(); den[ar, u] = dru.view(-1); col = col / den
        return row[ar, i], row, col

    def _stats(self, v, row, col):
        out = [v, (v - row.mean(1)) / (row.std(1) + 1e-12), (v - col.mean(1)) / (col.std(1) + 1e-12)]
        for vec in (row, col):
            eps = 1e-9 * (vec.abs().max(1).values + 1e-30)[:, None]
            out.append(((vec < v[:, None] - eps).double() + 0.5 * ((vec - v[:, None]).abs() <= eps).double()).mean(1))
        return out

    def features(self, r, d, chunk=4096):
        return torch.cat([self._features(r[s0:s0 + chunk], d[s0:s0 + chunk]) for s0 in range(0, len(r), chunk)])

    def _features(self, r, d):
        A = self.A; q = A[r, d]; K = a.hdc_steps; cols = []
        for k in self.maps:
            if k == 'path7':
                v, row, col = self._walk(A, r, d, q, [0, 0, 0, 1], 'raw')
                cols += self._stats(v, row, col)
                continue
            x, kind = self.CFG[k]; cs = self.coeffs(kind, x, K)
            v, row, col = self._walk(A, r, d, q, cs)                        # walk from the miRNA  (D_rd)
            cols += self._stats(v, row, col)
            v, col2, row2 = self._walk(A.t().contiguous(), d, r, q, cs)     # walk from the drug  (D_dr^T): its row is our column
            cols += self._stats(v, row2, col2)
        x = torch.stack(cols, 1)
        x = torch.where(x.abs() > 0, torch.sign(x) * torch.log1p(x.abs() * 1e2), x) if False else x
        return x.float()


class KRLoo(torch.nn.Module):
    """--krloo.  Learned-kernel closed-form ridge with exact leave-one-out, both sides.  miRNA side: phi = MLP(k-mer, mature
    sequence one-hot); for every drug column the ridge fit of A[:, d] on phi is evaluated at m with row m left out
    (LOO identity (yhat_m - h_mm y_m) / (1 - h_mm)); drug side: psi = MLP(ChemBERTa), ridge of A[m, :] on psi with column d
    left out.  Both outputs are exactly independent of A[m, d]; the kernels are learned end-to-end through the solve."""

    def __init__(self, dr, dd, k, lams):
        super().__init__()
        mk = lambda i: torch.nn.Sequential(Linear(i, 256), torch.nn.LeakyReLU(0.1), Linear(256, k))
        self.fr = mk(dr)
        self.fd = torch.nn.Sequential(Linear(dd, 256), torch.nn.LeakyReLU(0.1), Linear(256, 16))
        self.register_buffer('ll', torch.log(torch.tensor(lams)))              # fixed ridge strengths (learned ones collapse to interpolation)
        self.width = 2 * len(lams)

    @staticmethod
    def loo(Phi, Y, lam):
        """rows of Y (n x c) regressed on Phi (n x k) per column; LOO prediction matrix (n x c)"""
        Phi = torch.cat([Phi, torch.ones(len(Phi), 1, device=Phi.device)], 1)
        G = Phi.t() @ Phi + lam * torch.eye(Phi.shape[1], device=Phi.device)
        Gi = torch.linalg.inv(G)
        PG = Phi @ Gi
        h = (PG * Phi).sum(1, keepdim=True)
        Yh = PG @ (Phi.t() @ Y)
        return (Yh - h * Y) / (1 - h).clamp(min=0.05)

    def forward(self, xr, xd, A, r, d):
        pr = F.normalize(self.fr(xr), dim=1); pd_ = F.normalize(self.fd(xd), dim=1)
        out = []
        for lam in torch.exp(self.ll):
            out.append(self.loo(pr, A, lam)[r, d])
            out.append(self.loo(pd_, A.t(), lam)[d, r])
        return torch.stack(out, 1)

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


class PartnerAttn(torch.nn.Module):
    """round 10 --hpa: hollow partner attention.  For a query (m, d), m attends over the other partners of d
    (miRNAs m' != m with A[m', d] = 1) and d over the other partners of m; keys, queries and values come from node
    content only, so the output never depends on A[m, d] (exactly query-hollow).  Returns the attention-pooled value,
    the max and the log-mean-exp of the attention logits, and an empty-set flag, per side."""

    def __init__(self, dr, dd, k=32, out=64):
        super().__init__()
        self.k = k
        self.qr, self.kr, self.vr = Linear(dr, k), Linear(dr, k), Linear(dr, out)
        self.qd, self.kd, self.vd = Linear(dd, k), Linear(dd, k), Linear(dd, out)
        self.width = 2 * (out + 3)

    def side(self, Qx, Kx, Vx, mask):
        s = (Qx @ Kx.t()) / math.sqrt(self.k)
        cnt = mask.sum(1, keepdim=True); has = (cnt > 0).float()
        s = s.masked_fill(mask <= 0, -1e4)
        att = torch.softmax(s, 1) * has
        mx = s.max(1, keepdim=True).values * has
        lme = (torch.logsumexp(s, 1, keepdim=True) - torch.log(cnt.clamp(min=1))) * has
        return torch.cat([att @ Vx, mx, lme, has], 1)

    def forward(self, xr, xd, A, r, d):
        ar = torch.arange(len(r), device=r.device)
        mr = A[:, d].t().clone(); mr[ar, r] = 0                     # other partners of d
        md = A[r].clone(); md[ar, d] = 0                            # other partners of m
        o1 = self.side(self.qr(xr[r]), self.kr(xr), self.vr(xr), mr)
        o2 = self.side(self.qd(xd[d]), self.kd(xd), self.vd(xd), md)
        return torch.cat([o1, o2], 1)


# ----------------------------------------------------------------------------- model
class Model(torch.nn.Module):
    def __init__(self, data, hidden, ps, n_qis, n_ease, n_hac=0, n_hdc=0):
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
        if a.hac:
            self.hc = torch.nn.Sequential(Linear(n_hac, hidden), torch.nn.LeakyReLU(0.1), Linear(hidden, hidden)); d += hidden
        if a.krloo:
            self.kr = KRLoo(ps.hpa_xr.shape[1] + 88, ps.hpa_xd.shape[1], a.krloo_dim, [float(x) for x in a.krloo.split(',')])
            self.krb = torch.nn.BatchNorm1d(self.kr.width)
            self.krm = torch.nn.Sequential(Linear(self.kr.width, hidden), torch.nn.LeakyReLU(0.1), Linear(hidden, hidden)); d += hidden
        if a.hdc:
            self.dc = torch.nn.Sequential(Linear(n_hdc, hidden), torch.nn.LeakyReLU(0.1), Linear(hidden, hidden)); d += hidden
        if a.xfeat:
            nx = 85 if a.xfeat == 'all' else 39
            self.xf = torch.nn.Sequential(Linear(nx, hidden), torch.nn.LeakyReLU(0.1), Linear(hidden, hidden)); d += hidden
        if a.hpa:
            self.hpa = PartnerAttn(ps.hpa_xr.shape[1], ps.hpa_xd.shape[1], 32, a.hpa_dim); d += self.hpa.width
        self.lin1 = Linear(d, 2 * hidden); self.lin2 = Linear(2 * hidden, hidden); self.lin3 = Linear(hidden, 1)
        if a.moe:
            self.lin2e = torch.nn.ModuleList([Linear(2 * hidden, hidden) for _ in range(a.moe - 1)])
            self.lin3e = torch.nn.ModuleList([Linear(hidden, 1) for _ in range(a.moe - 1)])
            self.gate = torch.nn.Sequential(Linear(n_qis, 64), torch.nn.LeakyReLU(0.1), Linear(64, a.moe))
        if a.wide:
            self.wide = Linear(n_qis + n_ease, 1)
            torch.nn.init.zeros_(self.wide.weight); torch.nn.init.zeros_(self.wide.bias)

    def forward(self, data, eli, ps=None, qis=None, ease=None, dense=None, xf=None, hac=None, hdc=None):
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
        if a.hac:
            parts.append(self.hc(hac))
        if a.hdc:
            parts.append(self.dc(hdc))
        if a.krloo:
            parts.append(self.krm(self.krb(self.kr(ps.kr_xr, ps.hpa_xd, ps.A, r, d))))
        if a.xfeat:
            parts.append(self.xf(xf))
        if a.hpa:
            parts.append(self.hpa(ps.hpa_xr, ps.hpa_xd, ps.A, r, d))
        if a.chdrop and self.training:
            parts = [p_ * (torch.rand(len(p_), 1, device=p_.device) > a.chdrop).float() / (1 - a.chdrop) for p_ in parts]
        h = torch.cat(parts, -1)
        h = F.leaky_relu(self.lin1(h), 0.1)
        if a.moe:
            outs = [self.lin3(F.leaky_relu(self.lin2(h), 0.1))] + [l3(F.leaky_relu(l2(h), 0.1)) for l2, l3 in zip(self.lin2e, self.lin3e)]
            g_ = torch.softmax(self.gate(qis), 1)
            return (torch.cat(outs, 1) * g_).sum(1)
        h = F.leaky_relu(self.lin2(h), 0.1)
        out = self.lin3(h).view(-1)
        if a.wide:
            wx = torch.cat([x for x in (qis, self.eb(ease) if a.ease else None) if x is not None], 1)
            out = out + self.wide(wx).view(-1)
        return out


def hdc_t2(hd, e):
    x = hd.features(*e); return torch.sign(x) * torch.log1p(x.abs())


def listwise(pred, y, idx, n):
    """mean over positives i of -log softmax of s_i against the negatives of the batch sharing its node idx[i]"""
    neg = y < 0.5; pos = ~neg
    sn, inn = pred[neg], idx[neg]
    m = torch.full((n,), -1e4, device=pred.device).scatter_reduce(0, inn, sn.detach(), 'amax', include_self=True)
    se = torch.zeros(n, device=pred.device).scatter_add(0, inn, torch.exp(sn - m[inn]))
    lse = m + torch.log(se.clamp(min=1e-30))
    lse = torch.where(se > 0, lse, torch.full_like(lse, -1e4))
    sp = pred[pos]
    return (torch.logaddexp(sp, lse[idx[pos]]) - sp).mean()


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
    if a.val_graph and val is not None:
        train_pos = np.concatenate([train_pos, val[0][val[1] == 1]])          # message graph = all training positives
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
    full_te, ps_te = full, ps
    if a.val_graph_test and val is not None:
        all_pos = np.concatenate([train_pos, val[0][val[1] == 1]])
        full_te = build_graph(D, all_pos, with_assoc=(a.enc != 'content')).to(DEV)
        ps_te = PairStruct(D, torch.as_tensor(all_pos, device=DEV), rel_perm) if need_ps else None
        if a.qis:
            feats['qis']['te'] = (ps_te.qis(*eli_te) - mu) / sd
        if 'ease' in feats:
            feats['ease']['te'] = ps_te.ease(*eli_te)
    if a.hac:
        ac = AxialContext(ps)
        h_tr = ac.features(*eli_tr); hmu, hsd = h_tr.mean(0, keepdim=True), h_tr.std(0, keepdim=True) + 1e-6
        feats['hac'] = {k: (ac.features(*e) - hmu) / hsd for k, e in (('tr', eli_tr), ('te', eli_te), ('va', eli_va)) if e is not None}
        if a.check_hac:
            pidx = np.where(ytr == 1)[0][:a.check_hac]; diffs = []; raw = ac.features(*eli_tr[:, pidx])
            for j, k in enumerate(pidx):
                keep = ~((train_pos[:, 0] == tr[k, 0]) & (train_pos[:, 1] == tr[k, 1]))
                ac2 = AxialContext(PairStruct(D, torch.as_tensor(train_pos[keep], device=DEV), rel_perm), getattr(ac, 'betas', None))
                diffs.append((raw[j] - ac2.features(*eli_tr[:, [k]])[0]).abs())
            dm = torch.stack(diffs).max(0).values
            names = [f'{n}_{s_}' for n, _, _ in ac.maps for s_ in ('zr', 'zc', 'pr', 'pc')]
            print('[check_hac] max |diff| per feature:', {n: f'{float(v):.1e}' for n, v in zip(names, dm)}, flush=True)
            print('[check_hac] effect of the edge (nomask vs masked):', flush=True)
            sys.exit(0)
        if a.hac_shuffle:
            for k in feats['hac']:
                feats['hac'][k] = feats['hac'][k][torch.as_tensor(rng.permutation(len(feats['hac'][k])), device=DEV)]
    if a.hdc:
        hd = HollowDiffusion(ps.A)
        def hdc_t(e):
            x = hd.features(*e); return torch.sign(x) * torch.log1p(x.abs())
        c_tr = hdc_t(eli_tr); cmu, csd = c_tr.mean(0, keepdim=True), c_tr.std(0, keepdim=True) + 1e-6
        feats['hdc'] = {k: (hdc_t(e) - cmu) / csd for k, e in (('tr', eli_tr), ('te', eli_te), ('va', eli_va)) if e is not None}
        if a.check_hac:
            pidx = np.where(ytr == 1)[0][:a.check_hac]; raw = hd.features(*eli_tr[:, pidx]); diffs = []
            for j, k in enumerate(pidx):
                keep = ~((train_pos[:, 0] == tr[k, 0]) & (train_pos[:, 1] == tr[k, 1]))
                A2 = torch.zeros_like(ps.A); A2[train_pos[keep][:, 0], train_pos[keep][:, 1]] = 1
                diffs.append((raw[j] - HollowDiffusion(A2).features(*eli_tr[:, [k]])[0]).abs() / (raw[j].abs() + 1e-3))
            print('[check_hdc] max rel |diff| per feature:', [f'{float(v):.1e}' for v in torch.stack(diffs).max(0).values], flush=True)
            if not a.hac:
                sys.exit(0)
    if a.xfit:
        from sklearn.model_selection import StratifiedKFold
        for fit_i, q_i in StratifiedKFold(a.xfit, shuffle=True, random_state=a.seed * 100 + fold).split(tr, ytr.astype(int)):
            sub = tr[fit_i[ytr[fit_i] == 1]]
            ps_k = PairStruct(D, torch.as_tensor(sub, device=DEV), rel_perm)
            e = eli_tr[:, torch.as_tensor(q_i, device=DEV)]
            qi = torch.as_tensor(q_i, device=DEV)
            if a.qis:
                feats['qis']['tr'][qi] = (ps_k.qis(*e) - mu) / sd
            if 'ease' in feats:
                feats['ease']['tr'][qi] = ps_k.ease(*e)
            if a.hdc:
                feats['hdc']['tr'][qi] = (hdc_t2(HollowDiffusion(ps_k.A), e) - cmu) / csd
        print(f'[xfit] training-pair channels cross-fitted with K={a.xfit}', flush=True)
    if a.xfeat:
        zz = np.load(os.path.join(BASE, 'plmf_warm_20260922', 'cache', f'{a.dataset}_{"single" if a.split == "single" else ""}s{a.seed}_f{fold}_ctx+diffK10_kernel10.npz'))
        cols = np.arange(85) if a.xfeat == 'all' else np.arange(33, 85).reshape(13, 4)[:, 1:].ravel()
        key = lambda P: P[:, 0] * 100000 + P[:, 1]
        pos_ = pd.Series(np.arange(len(zz['tr'])), index=key(zz['tr']))
        assert (zz['te'] == te).all()
        feats['xf'] = {'tr': torch.as_tensor(zz['Xtr'][pos_[key(tr)].values][:, cols], device=DEV), 'te': torch.as_tensor(zz['Xte'][:, cols], device=DEV)}
        if val is not None:
            feats['xf']['va'] = torch.as_tensor(zz['Xtr'][pos_[key(val[0])].values][:, cols], device=DEV)
    eli_u = None
    if a.unl > 0:
        occ = np.zeros((D.n_mir, D.n_drug), bool)
        for arr in ((train_pos,) if a.unl_strict else (tr, te) + ((val[0],) if val is not None else ())):
            occ[arr[:, 0], arr[:, 1]] = True
        U = np.argwhere(~occ)
        eli_u = torch.as_tensor(U.T, dtype=torch.long, device=DEV)
        chunks = lambda fn: torch.cat([fn(eli_u[0, i:i + 20000], eli_u[1, i:i + 20000]) for i in range(0, eli_u.shape[1], 20000)])
        if a.qis:
            feats['qis']['u'] = (chunks(ps.qis) - mu) / sd
        if a.hac:
            feats['hac']['u'] = (ac.features(*eli_u, chunk=8192) - hmu) / hsd
        if a.hdc:
            feats['hdc']['u'] = (hdc_t(eli_u) - cmu) / csd
        if a.ease and not a.ease_learn:
            feats['ease']['u'] = chunks(ps.ease)
        n_u = int(round(a.unl * len(tr)))
        print(f'[unl] pool {len(U)} cells, {n_u} sampled per epoch, weight {a.unl_w}', flush=True)
        u_gen = torch.Generator(device='cpu'); u_gen.manual_seed(a.seed * 100 + fold + MS)
        pa, pb = [float(x) for x in a.unl_pop.split(',')]
        u_prob = ((ps.deg_d[eli_u[1]] + 1) ** pa * (ps.deg_r[eli_u[0]] + 1) ** pb).cpu() if (pa or pb) else None
    if a.dump:
        os.makedirs(os.path.join(HERE, 'cache'), exist_ok=True)
        np.savez_compressed(os.path.join(HERE, 'cache', f'feats_{a.dataset}_{a.split}_s{a.seed}_f{fold}.npz'),
                            **{f'{k}_{p}': v.cpu().numpy() for k in feats for p, v in feats[k].items()}, tr=tr, ytr=ytr, te=te, yte=yte,
                            va=val[0] if val is not None else np.zeros((0, 2)), yva=val[1] if val is not None else np.zeros(0))
        return dict(dataset=a.dataset, split=a.split, tag=a.tag, fold=fold, seed=a.seed), [], pd.DataFrame()
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
    model = Model(full, a.hidden, ps, n_qis, n_ease, feats['hac']['tr'].shape[1] if a.hac else 0,
                  feats['hdc']['tr'].shape[1] if a.hdc else 0).to(DEV)
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
    kr_p = list(model.kr.parameters()) if a.krloo else []
    wid = {id(p) for p in wide_p + ease_p + kr_p}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in wid])]
    if wide_p:
        groups.append(dict(params=wide_p, lr=a.lr * a.wide_lr))
    if ease_p:
        groups.append(dict(params=ease_p, lr=a.lr * a.ease_lr))
    if kr_p:
        groups.append(dict(params=kr_p, lr=a.lr * a.krloo_lr))
    opt = torch.optim.RMSprop(groups, lr=a.lr, alpha=0.99)
    hist = []; best = None; best_val = (-1, None); t0 = time.time(); best_scores = None

    dense_te = (dense[0], dense[1], dense[2], dense[3], ps_te.A) if (dense is not None and ps_te is not ps) else dense

    def score(eli, part):
        fg, pp, dn = (full_te, ps_te, dense_te) if part == 'te' else (full, ps, dense)
        return torch.sigmoid(model(fg, eli, pp, g('qis', part), g('ease', part), dn, g('xf', part), g('hac', part), g('hdc', part))).cpu().numpy()

    for ep in range(1, a.epochs + 1):
        model.train(); opt.zero_grad()
        if eli_u is not None:
            ui = (torch.randint(0, eli_u.shape[1], (n_u,), generator=u_gen) if u_prob is None else
                  torch.multinomial(u_prob, n_u, replacement=True, generator=u_gen)).to(DEV)
            cat = lambda key: torch.cat([feats[key]['tr'], feats[key]['u'][ui]]) if key in feats else None
            pred = model(full, torch.cat([eli_tr, eli_u[:, ui]], 1), ps, cat('qis'), cat('ease'), dense, None, cat('hac'), cat('hdc'))
            nt_ = len(y_tr)
            loss = F.binary_cross_entropy_with_logits(pred[:nt_], y_tr) + a.unl_w * F.binary_cross_entropy_with_logits(pred[nt_:], torch.zeros(n_u, device=DEV))
        else:
            pred = model(full, eli_tr, ps, g('qis', 'tr'), g('ease', 'tr'), dense, g('xf', 'tr'), g('hac', 'tr'), g('hdc', 'tr'))
            loss = F.binary_cross_entropy_with_logits(pred, y_tr)
        if a.lw_d or a.lw_m:
            yall = torch.cat([y_tr, torch.zeros(len(pred) - len(y_tr), device=DEV)])
            eall = torch.cat([eli_tr, eli_u[:, ui]], 1) if eli_u is not None else eli_tr
            for w_, idx, n_ in ((a.lw_d, eall[1], D.n_drug), (a.lw_m, eall[0], D.n_mir)):
                if w_:
                    loss = loss + w_ * listwise(pred, yall, idx, n_)
        loss.backward(); opt.step()
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
            o1 = model(full, e1, ps, feats['qis']['tr'][pos_idx] if a.qis else None, feats['ease']['tr'][pos_idx] if 'ease' in feats else None, dense,
                       None, feats['hac']['tr'][pos_idx] if a.hac else None, feats['hdc']['tr'][pos_idx] if a.hdc else None)
            for j, k in enumerate(pos_idx):
                keep = ~((train_pos[:, 0] == tr[k, 0]) & (train_pos[:, 1] == tr[k, 1]))
                msg2 = train_pos[keep]
                full2 = build_graph(D, msg2, with_assoc=(a.enc != 'content')).to(DEV)
                ps2 = PairStruct(D, torch.as_tensor(msg2, device=DEV), rel_perm)
                e = eli_tr[:, [k]]
                q2 = (ps2.qis(*e) - mu) / sd if a.qis else None
                es2 = ps2.ease(*e) if 'ease' in feats else None
                dense2 = (dense[0], dense[1], dense[2], dense[3], ps2.A) if dense is not None else None
                h2 = (AxialContext(ps2, getattr(ac, 'betas', None)).features(*e) - hmu) / hsd if a.hac else None
                c2 = (hdc_t2(HollowDiffusion(ps2.A), e) - cmu) / csd if a.hdc else None
                o2 = model(full2, e, ps2, q2, es2, dense2, None, h2, c2)
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
