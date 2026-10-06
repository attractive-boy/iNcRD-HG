# -*- coding: utf-8 -*-
"""iNcRD-HG GNN body on PLMF-MDA -- round 7: architecture modules that use *only* the PLMF-MDA
release (miRNA sequences + names, drug SMILES -> ChemBERTa as in PLMF-MDA itself, and the
training part of the association matrix).  No external database, literature record or
knowledge graph is read anywhere in this file.

Base = plain iNcRD-HG (GeneralConv x2 via HeteroConv + MLP edge decoder, RMSprop 1e-4, 300 epochs,
full batch), bit-identical to ../plmf_gnnarch_20260923/run_gnn.py without --prov/--side.

Inherited internal modules (round 6)
  --lqo K         M1 leave-query-out message passing
  --ncn --cnt     M2 relation-typed neural-common-neighbour pair decoder + structural counts
New modules (round 7)
  --nbf           drug-conditioned path encoder: NBFNet-style generalised Bellman-Ford propagation
                  from every drug over the message graph (labelling trick; pair state h_d(m))
  --crlt          retrieval label transfer: learned content attention of the query miRNA over all
                  message-graph miRNAs (and of the query drug over drugs), values = their links to
                  the query partner, attention logits biased by the label-free miRNA relations
  --irel LIST     sequence/name-derived miRNA relations (seed7, seed6, pre, fam, lev) as extra
                  relation channels of the RT-NCN pair decoder
  --tta           LQO-consistent inference: val/test pairs are scored on each of the K pass
                  message graphs and averaged (removes the train/test message-degree shift)
  --rowctx        row/column-contextual encodings of the structural pair counts
  --diffc         PPR / Katz diffusion counts on the message graph
Controls
  --irel_shuffle  permute the relation rows/cols across miRNAs (keeps relation statistics)
  --crlt_shuffle  permute the content features used by the retrieval head across miRNAs/drugs
Output results/{tag}_{DS}_{split}_seed{seed}.csv (+ _scores.csv, _curves.json, _args.json)"""
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
ap.add_argument('--dropout', type=float, default=0.0)
ap.add_argument('--val', action='store_true')
ap.add_argument('--refit', action='store_true', help='after val epoch selection, retrain on train+val for that many epochs; report refit_*')
ap.add_argument('--val_frac', type=float, default=0.1)
ap.add_argument('--lqo', type=int, default=0)
ap.add_argument('--ncn', action='store_true')
ap.add_argument('--cnt', action='store_true')
ap.add_argument('--ncn_dim', type=int, default=64)
# round 7
ap.add_argument('--irel', default='', help='comma list of seed7,seed6,pre,fam,lev')
ap.add_argument('--irel_shuffle', action='store_true')
ap.add_argument('--nbf', action='store_true')
ap.add_argument('--nbf_dim', type=int, default=32)
ap.add_argument('--nbf_layers', type=int, default=3)
ap.add_argument('--nbf_rel', default='A,Sr,Sd', help='relations the path encoder walks on')
ap.add_argument('--crlt', action='store_true')
ap.add_argument('--crlt_heads', type=int, default=4)
ap.add_argument('--crlt_dim', type=int, default=32)
ap.add_argument('--crlt_feat', default='kmer,seed,pos', help='miRNA key/query features: kmer,seed,pos')
ap.add_argument('--crlt_bias', type=int, default=1, help='1: relation priors (kNN + --irel) bias the retrieval logits')
ap.add_argument('--crlt_emb', type=int, default=0, help='1: also retrieve value embeddings (link-weighted z of retrieved nodes)')
ap.add_argument('--crlt_shuffle', action='store_true')
ap.add_argument('--tta', type=int, default=0, help='1: average val/test scores over the K pass graphs; 2: K pass graphs + full graph')
ap.add_argument('--rowctx', action='store_true')
ap.add_argument('--nameprior', action='store_true', help='nomenclature priors (miRNA number, arm, let-7, DrugBank number) as pair-decoder inputs')
ap.add_argument('--diffc', action='store_true')
ap.add_argument('--ease', type=float, default=0.0, help='EASE ridge lambda (>0 adds closed-form item-item regression channels)')
ap.add_argument('--ridge', action='store_true', help='differentiable closed-form collaborative ridge layer (side-information EASE) in the pair decoder')
ap.add_argument('--ridge_lams', default='10,50,200', help='initial ridge strengths, one head each')
ap.add_argument('--ridge_learn', type=int, default=1, help='1: ridge strengths and kernel weights are learned end-to-end')
ap.add_argument('--ridge_side', default='kmer', help='miRNA kernels in the Gram: kmer (dense k-mer cosine), knn, seed7, fam, pre, lev; "none" = plain EASE')
ap.add_argument('--ridge_alpha', type=float, default=0.3, help='initial kernel weight')
ap.add_argument('--ridge_drug', default='knn', help='drug kernel in the drug-side Gram: knn | none')
ap.add_argument('--ridge_ncn', type=int, default=0, help='1: ridge coefficients also act as RT-NCN relations (hidden-state aggregation)')
ap.add_argument('--ridge_shuffle', action='store_true', help='control: permute the miRNA kernels across miRNAs')
ap.add_argument('--bank', default='', help='diagnostic: pair-feature families of the LR pipeline (ctx,diff,...) fed to the decoder')
ap.add_argument('--wide', action='store_true', help='wide & deep: linear head on the tabular pair features added to the logit')
ap.add_argument('--wide_lr', type=float, default=1.0, help='learning-rate multiplier of the wide head')
ap.add_argument('--mod_lr', type=float, default=1.0, help='learning-rate multiplier of the path encoder / retrieval head')
ap.add_argument('--dec', default='mlp', choices=['mlp', 'nam'])
ap.add_argument('--nam_rank', type=int, default=64)
ap.add_argument('--nam_h', type=int, default=16)
ap.add_argument('--wd', type=float, default=0.0)
ap.add_argument('--zdrop', type=float, default=0.0, help='channel dropout on the query-node embeddings z_m, z_d in the decoder input (1.0 = drop them entirely)')
ap.add_argument('--lqo_resample', type=int, default=0, help='re-draw the LQO groups every R epochs (0 = fixed)')
ap.add_argument('--neg_mult', type=float, default=0.0, help='extra negatives per positive resampled each epoch from the unlabeled pool')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--tag', default='plain')
ap.add_argument('--folds', default='')
ap.add_argument('--model_seed', type=int, default=0, help='offset of the model / LQO-group seed (split unchanged)')
a = ap.parse_args()
DEV = torch.device(a.device)
IREL = [k for k in a.irel.split(',') if k]


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


def lev(s, t):
    prev = list(range(len(t) + 1))
    for i, cs in enumerate(s, 1):
        cur = [i]
        for j, ct in enumerate(t, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cs != ct)))
        prev = cur
    return prev[-1]


def lev_knn(seqs, k, cache):
    if os.path.exists(cache):
        E = np.load(cache)
    else:
        n = len(seqs); E = np.zeros((n, n), np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                E[i, j] = E[j, i] = 1 - lev(seqs[i], seqs[j]) / max(len(seqs[i]), len(seqs[j]))
        np.save(cache, E)
    S = E.copy(); np.fill_diagonal(S, -1e9); idx = np.argsort(-S, 1)[:, :k]
    R = np.zeros_like(S); np.put_along_axis(R, idx, 1.0, 1); return np.maximum(R, R.T).astype(np.float32)


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
        self.Sr = torch.zeros(self.n_mir, self.n_mir); self.Sr[self.sim_r[0], self.sim_r[1]] = 1
        self.Sd = torch.zeros(self.n_drug, self.n_drug); self.Sd[self.sim_d[0], self.sim_d[1]] = 1
        self.Sr, self.Sd = self.Sr.to(DEV), self.Sd.to(DEV)
        # label-free miRNA relations derived from the dataset's own sequences / names
        # internal miRNA / drug kernels for the ridge layer; every kernel is a Gram matrix F F^T (PSD), i.e. the
        # side information enters as extra pseudo-user rows of the item matrix (Jeunen et al. 2020)
        if a.ridge:
            xn = F.normalize(self.xr, dim=1)
            kern = {'kmer': xn @ xn.t()}
            lab = {'seed7': [q[1:8] for q in self.seqs], 'fam': [family(n) for n in self.names], 'pre': [stem(n) for n in self.names],
                   'seed6': [q[1:7] for q in self.seqs]}
            for k, v in lab.items():
                codes = pd.factorize(pd.Series(v))[0]
                M = torch.zeros(self.n_mir, codes.max() + 1); M[torch.arange(self.n_mir), torch.as_tensor(codes)] = 1.0
                cnt = M.sum(0); M = M[:, cnt > 1]                       # only groups with >= 2 members
                kern[k] = M @ M.t()
            self.ridge_kr = [kern[k].float().to(DEV) for k in a.ridge_side.split(',') if k and k != 'none']
            xdn = F.normalize(self.xd, dim=1)
            self.ridge_kd = [(xdn @ xdn.t()).to(DEV)] if a.ridge_drug != 'none' else []
        # nomenclature priors parsed from the dataset's own miRNA names / DrugBank ids (no lookup)
        num = np.array([int(m.group(1)) if (m := re.search(r'(?:mir|let)-(\d+)', n.lower())) else 0 for n in self.names], np.float32)
        arm = np.array([[n.lower().endswith('-5p'), n.lower().endswith('-3p')] for n in self.names], np.float32)
        let = np.array(['let-7' in n.lower() for n in self.names], np.float32)
        dbn = np.array([int(re.sub(r'\D', '', x) or 0) for x in self.dbs], np.float32)
        self.name_r = torch.as_tensor(np.column_stack([np.log1p(num), arm, let]), device=DEV)
        self.name_d = torch.as_tensor(np.log1p(dbn)[:, None], device=DEV)
        self.irel = {}
        for k in IREL:
            if k == 'seed7':
                R = same_label([s[1:8] for s in self.seqs])
            elif k == 'seed6':
                R = same_label([s[1:7] for s in self.seqs])
            elif k == 'pre':
                R = same_label([stem(n) for n in self.names])
            elif k == 'fam':
                R = same_label([family(n) for n in self.names])
            elif k == 'lev':
                R = lev_knn(self.seqs, a.knn, os.path.join(HERE, 'results', f'lev_{ds}.npy'))
            else:
                raise ValueError(k)
            self.irel[k] = torch.as_tensor(R, device=DEV)
            print(f'[irel] {k}: coverage {(R.sum(1) > 0).mean():.2f}, mean degree {R.sum(1).mean():.2f}', flush=True)
        # internal sequence features for the retrieval head
        if a.crlt:
            parts = []
            fe = a.crlt_feat.split(',')
            if 'kmer' in fe:
                parts.append(self.xr.numpy())
            nt = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
            if 'seed' in fe:
                oh = np.zeros((self.n_mir, 7, 4), np.float32)
                for i, s in enumerate(self.seqs):
                    for p, c in enumerate(s[1:8]):
                        if c in nt:
                            oh[i, p, nt[c]] = 1
                parts.append(oh.reshape(self.n_mir, -1))
            if 'pos' in fe:
                oh = np.zeros((self.n_mir, 24, 4), np.float32)
                for i, s in enumerate(self.seqs):
                    for p, c in enumerate(s[:24]):
                        if c in nt:
                            oh[i, p, nt[c]] = 1
                parts.append(oh.reshape(self.n_mir, -1))
            self.fr = torch.as_tensor(np.concatenate(parts, 1), device=DEV)
            self.fd = self.xd.to(DEV)
        if a.bank:
            sys.path.insert(0, os.path.join(BASE, 'plmf_innov_20260921'))
            sys.path.insert(0, os.path.join(BASE, 'dlstmda_strongest_20260920'))
            from run_probe import load_nodes, make_bank
            xr_np, xd_np = load_nodes(ds)
            self.bank = make_bank(xr_np, xd_np, 'kernel', 10)
            self.bank_fams = set(a.bank.split(','))

    def bank_feats(self, msg, pairs):
        return self.bank.transform(np.asarray(msg), np.asarray(pairs), self.bank_fams).values.astype(np.float32)


def build_graph(D, msg_pos):
    data = HeteroData()
    data['miRNA'].x = D.xr; data['drug'].x = D.xd
    data['miRNA', 'MiMi', 'miRNA'].edge_index = D.sim_r
    data['drug', 'DrugDrug', 'drug'].edge_index = D.sim_d
    data['miRNA', 'MiDrug', 'drug'].edge_index = torch.as_tensor(msg_pos.T, dtype=torch.long)
    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)
    return data


# ----------------------------------------------------------------------------- model
class GNNEncoder(torch.nn.Module):
    def __init__(self, data, hidden, layers, dropout):
        super().__init__()
        def convs():
            return HeteroConv({et: GeneralConv((-1, -1), hidden, aggr='add') for et in data.edge_types}, aggr=a.aggr)
        self.convs = torch.nn.ModuleList([convs() for _ in range(layers)])
        self.dropout = dropout

    def forward(self, x_dict, ei_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, ei_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.leaky_relu(v, negative_slope=0.1) for k, v in x_dict.items()}
                if self.dropout > 0:
                    x_dict = {k: F.dropout(v, self.dropout, self.training) for k, v in x_dict.items()}
        return x_dict


CNT_NAMES = ['deg_r', 'deg_d', 'p3', 'p3_n', 'p3_r', 'p3_d', 'ASd', 'SrA', 'SrASd', 'ASd_n', 'SrA_n', 'p5', 'p5_n']


def sym_norm(M):
    r = M.sum(1).clamp(min=1).pow(-0.5); c = M.sum(0).clamp(min=1).pow(-0.5)
    return r[:, None] * M * c[None, :]


def diffusion_t(A, alpha, beta, K=12):
    """PPR / Katz from miRNAs to drugs (and drugs to miRNAs) on the bipartite message graph (bank.py::diffusion)."""
    An = A / A.sum(1, keepdim=True).clamp(min=1); And = (A / A.sum(0, keepdim=True).clamp(min=1)).t()
    G = And @ An; Gs = An @ And
    ppr = torch.zeros_like(A); katz = torch.zeros_like(A); p = An
    for k in range(K):
        ppr = ppr + alpha * (1 - alpha) ** (2 * k + 1) * p; katz = katz + beta ** (2 * k + 1) * p; p = p @ G
    pd_ = torch.zeros_like(A.t()); kd = torch.zeros_like(A.t()); q = And
    for k in range(K):
        pd_ = pd_ + alpha * (1 - alpha) ** (2 * k + 1) * q; kd = kd + beta ** (2 * k + 1) * q; q = q @ Gs
    return {'ppr_rd': ppr, 'katz_rd': katz, 'ppr_dr': pd_.t(), 'katz_dr': kd.t()}


def ease_scores(A, lam):
    """EASE (Steck 2019) closed-form item-item ridge regression on the message graph, both directions:
    drug side predicts A[m, d] from m's other drugs, miRNA side predicts it from d's other miRNAs."""
    out = {}
    for name, X in (('ease_d', A), ('ease_r', A.t())):
        G = X.t() @ X
        P = torch.linalg.inv(G + lam * torch.eye(G.shape[0], device=G.device))
        B = -P / torch.diag(P)[None, :]
        B.fill_diagonal_(0)
        S = X @ B
        out[name] = S if name == 'ease_d' else S.t()
    return out


class RidgeLayer(torch.nn.Module):
    """Differentiable closed-form collaborative ridge layer (side-information EASE, Steck 2019 / Jeunen 2020)
    evaluated on the current message graph A.  miRNA side (items = miRNAs, users = drugs):
        G_r = A A^T + sum_k softplus(alpha_k) K_k + lambda I,  P = G_r^-1,  B_r = I - P diag(1/diag P)  (diag B = 0)
        S_r[m, d] = sum_m' B_r[m', m] A[m', d]      (predict A[m, d] from d's other miRNAs)
    drug side (items = drugs, users = miRNAs):  S_d = A B_d  with G_d = A^T A + beta K_d + lambda I.
    One head per initial lambda; lambdas / kernel weights are learned end-to-end (gradients through the
    Cholesky inverse).  Outputs the 2H score matrices and the coefficient matrices (for RT-NCN aggregation)."""

    def __init__(self, lams, kr, kd, learn):
        super().__init__()
        H = len(lams)
        self.kr = kr; self.kd = kd
        # log-parameterised strengths (lambda = exp(theta)); trained with the --mod_lr multiplier
        self.lam_r = torch.nn.Parameter(torch.tensor([math.log(l - 1.0) for l in lams]), requires_grad=bool(learn))
        self.lam_d = torch.nn.Parameter(torch.tensor([math.log(l - 1.0) for l in lams]), requires_grad=bool(learn))
        self.al_r = torch.nn.Parameter(torch.full((H, max(len(kr), 1)), math.log(a.ridge_alpha)), requires_grad=bool(learn))
        self.al_d = torch.nn.Parameter(torch.full((H, max(len(kd), 1)), 0.0), requires_grad=bool(learn))
        self.H = H

    @staticmethod
    def _ease(G):
        L = torch.linalg.cholesky(G)
        P = torch.cholesky_inverse(L)
        B = -P / torch.diagonal(P)[None, :]
        return B * (1 - torch.eye(B.shape[0], device=B.device))

    def forward(self, A, perm=None):
        nm, nd = A.shape
        Gr0 = A @ A.t(); Gd0 = A.t() @ A
        Ir = torch.eye(nm, device=A.device); Id = torch.eye(nd, device=A.device)
        S, Bs = [], []
        kr = self.kr if perm is None else [K[perm][:, perm] for K in self.kr]
        for h in range(self.H):
            G = Gr0 + (1.0 + torch.exp(self.lam_r[h].clamp(-6, 8))) * Ir
            for i, K in enumerate(kr):
                G = G + torch.exp(self.al_r[h, i].clamp(-8, 4)) * K
            Br = self._ease(G); S.append(Br.t() @ A); Bs.append(('r', Br))
            G = Gd0 + (1.0 + torch.exp(self.lam_d[h].clamp(-6, 8))) * Id
            for i, K in enumerate(self.kd):
                G = G + torch.exp(self.al_d[h, i].clamp(-8, 4)) * K
            Bd = self._ease(G); S.append(A @ Bd); Bs.append(('d', Bd))
        return S, Bs

    def describe(self):
        return dict(lam_r=(1 + torch.exp(self.lam_r)).tolist(), lam_d=(1 + torch.exp(self.lam_d)).tolist(),
                    al_r=torch.exp(self.al_r).tolist(), al_d=torch.exp(self.al_d).tolist())


class PairStruct:
    """Dense structural quantities of one message graph + relation-typed common-neighbour channels.
    Everything is computed from the message-graph positives and label-free relations only."""

    def __init__(self, D, msg_pos, irel_perm=None):
        A = torch.zeros(D.n_mir, D.n_drug, device=DEV); A[msg_pos[:, 0], msg_pos[:, 1]] = 1.0
        self.A = A; self.deg_r = A.sum(1); self.deg_d = A.sum(0)
        self.Cm = A @ A.t(); self.Cd = A.t() @ A
        self.P3 = self.Cm @ A; self.P5 = self.Cm @ self.P3
        self.ASd = A @ D.Sd; self.SrA = D.Sr @ A; self.SrASd = D.Sr @ A @ D.Sd
        self.mir_rel, self.drug_rel = {}, {}
        if a.ncn:
            self.mir_rel['co'] = self.Cm; self.drug_rel['co'] = self.Cd
        self.irel = {}
        for k, R in D.irel.items():
            if irel_perm is not None:
                R = R[irel_perm][:, irel_perm]
            self.irel[k] = R
            if a.ncn or a.cnt:
                self.mir_rel[k] = R
        self.mir_rel = {k: (R, R.sum(1)) for k, R in self.mir_rel.items()}
        self.drug_rel = {k: (R, R.sum(1)) for k, R in self.drug_rel.items()}
        self.RA = {k: R @ A for k, (R, _) in self.mir_rel.items()}
        self.AR = {k: A @ R for k, (R, _) in self.drug_rel.items()}
        self.diff = diffusion_t(A, 0.3, 0.01) if a.diffc else {}
        self.ease = ease_scores(A, a.ease) if a.ease > 0 else {}
        self.names = CNT_NAMES + [f'{k}_{t}' for k in list(self.mir_rel) + list(self.drug_rel) for t in ('c', 'cr', 'cd')] + list(self.diff) + list(self.ease)
        self.name_r, self.name_d = D.name_r, D.name_d
        if a.nameprior:
            self.names = self.names + ['mir_num', 'arm5p', 'arm3p', 'let7', 'db_num']
        # score matrices whose row/column context is encoded by --rowctx
        self.ctx_mats = {'p3': self.P3, 'p5': self.P5, 'ASd': self.ASd, 'SrA': self.SrA, 'SrASd': self.SrASd}
        self.ctx_mats.update({f'RA_{k}': v for k, v in self.RA.items() if k != 'co'})
        self.ctx_mats.update(self.diff)
        self.ctx_mats.update(self.ease)
        if a.rowctx:
            self.ctx_stats = {k: (M.mean(1), M.std(1) + 1e-6, M.mean(0), M.std(0) + 1e-6) for k, M in self.ctx_mats.items()}
            self.names = self.names + [f'{k}_{t}' for k in self.ctx_mats for t in ('rz', 'rp', 'cz', 'cp')]
        # normalised relation operators for the drug-conditioned path encoder
        if a.nbf:
            nm, nd = D.n_mir, D.n_drug
            ops = []
            rels = a.nbf_rel.split(',')
            if 'A' in rels:
                An = sym_norm(A)
                ops.append(('m2d', An.t().to_sparse()))   # drug <- miRNA
                ops.append(('d2m', An.to_sparse()))       # miRNA <- drug
            if 'Sr' in rels:
                ops.append(('mm', sym_norm(D.Sr).to_sparse()))
            if 'Sd' in rels:
                ops.append(('dd', sym_norm(D.Sd).to_sparse()))
            for k in IREL:
                if k in rels:
                    ops.append((f'mm_{k}', sym_norm(self.irel[k]).to_sparse()))
            self.nbf_ops = ops

    def counts(self, r, d):
        dr, dd = self.deg_r[r], self.deg_d[d]; p3 = self.P3[r, d]; p5 = self.P5[r, d]
        f = [dr, dd, p3, p3 / (dr * dd + 1.0), p3 / (dr + 1.0), p3 / (dd + 1.0), self.ASd[r, d], self.SrA[r, d], self.SrASd[r, d],
             self.ASd[r, d] / (dr + 1.0), self.SrA[r, d] / (dd + 1.0), p5, p5 / (dr * dd + 1.0)]
        for k, (R, rs) in self.mir_rel.items():
            c = self.RA[k][r, d]; f += [c, c / (rs[r] + 1.0), c / (dd + 1.0)]
        for k, (R, rs) in self.drug_rel.items():
            c = self.AR[k][r, d]; f += [c, c / (rs[d] + 1.0), c / (dr + 1.0)]
        for k, M in self.diff.items():
            f.append(M[r, d] * 100.0)
        out = torch.log1p(torch.stack(f, 1))
        if self.ease:
            e = torch.stack([M[r, d] for M in self.ease.values()], 1)
            out = torch.cat([out, torch.sign(e) * torch.log1p(e.abs() * 10.0)], 1)
        if a.nameprior:
            out = torch.cat([out, self.name_r[r], self.name_d[d]], 1)
        if a.rowctx:
            g = []
            for k, M in self.ctx_mats.items():
                mr, sr, mc, sc = self.ctx_stats[k]; v = M[r, d]
                g += [(v - mr[r]) / sr[r], (M[r] < v[:, None]).float().mean(1), (v - mc[d]) / sc[d], (M[:, d].t() < v[:, None]).float().mean(1)]
            out = torch.cat([out, torch.stack(g, 1)], 1)
        return out

    def ncn(self, r, d, z):
        out = []
        for k, (R, _) in self.mir_rel.items():
            w = self.A[:, d].t() * R[r]; out.append(w @ z['miRNA'] / (w.sum(1, keepdim=True) + 1.0))
        for k, (R, _) in self.drug_rel.items():
            w = self.A[r] * R[:, d].t(); out.append(w @ z['drug'] / (w.sum(1, keepdim=True) + 1.0))
        return out

    def channels(self):
        return (list(self.mir_rel) + list(self.drug_rel)) if a.ncn else []


class DrugPath(torch.nn.Module):
    """Drug-conditioned generalised Bellman-Ford propagation (NBFNet-style labelling trick).
    State H[s, v] for every source drug s and node v (miRNAs then drugs); boundary H0[s, s] = q;
    layer: H <- H + ReLU(LN(W [ sum_r P_r (H * w_r) + H0 ])).  h_s(m) after T layers summarises all
    s ~> m walks through associations (message graph only) and similarity edges."""

    def __init__(self, n_rel, dim, layers):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(dim) * 0.5)
        self.w = torch.nn.ParameterList([torch.nn.Parameter(1.0 + 0.1 * torch.randn(n_rel, dim)) for _ in range(layers)])
        self.lin = torch.nn.ModuleList([Linear(dim, dim) for _ in range(layers)])
        self.ln = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for _ in range(layers)])
        self.dim = dim

    def forward(self, ops, nm, nd):
        dim = self.dim; n = nm + nd
        H0 = torch.zeros(nd, n, dim, device=DEV)
        idx = torch.arange(nd, device=DEV)
        H0[idx, nm + idx] = self.q
        H = H0
        for t in range(len(self.lin)):
            M = H0.clone()
            for i, (name, P) in enumerate(ops):
                Hw = H * self.w[t][i]                                  # DistMult message
                if name == 'm2d':
                    src, dst = Hw[:, :nm], slice(nm, n)
                elif name == 'd2m':
                    src, dst = Hw[:, nm:], slice(0, nm)
                elif name.startswith('mm'):
                    src, dst = Hw[:, :nm], slice(0, nm)
                else:
                    src, dst = Hw[:, nm:], slice(nm, n)
                ns = src.shape[1]
                msg = torch.sparse.mm(P, src.permute(1, 0, 2).reshape(ns, -1)).reshape(-1, nd, dim).permute(1, 0, 2)
                M[:, dst] = M[:, dst] + msg
            H = H + F.relu(self.ln[t](self.lin[t](M)))
        return H[:, :nm]                                               # [drug, miRNA, dim]


class Retrieval(torch.nn.Module):
    """Retrieval label transfer.  miRNA side: attention of the query miRNA over every message-graph
    miRNA m' (m' != m, deg(m') > 0) with learned content keys and relation-prior biases; the value
    of m' is its link A[m', d] to the query drug.  Drug side: same over drugs, value A[m, d']."""

    def __init__(self, in_r, in_d, heads, dim, n_bias):
        super().__init__()
        self.h, self.dh = heads, dim
        self.qr = Linear(in_r, heads * dim); self.kr = Linear(in_r, heads * dim)
        self.qd = Linear(in_d, heads * dim); self.kd = Linear(in_d, heads * dim)
        self.br = torch.nn.Parameter(torch.zeros(heads, max(n_bias, 1)))
        self.n_bias = n_bias
        self.out_dim = 2 * heads + (2 * heads if a.crlt_emb else 0)

    def _side(self, Q, K, bias_mats, bias_w, idx, V, avail):
        B = Q.shape[0]
        L = torch.einsum('bhd,nhd->bhn', Q, K) / math.sqrt(self.dh)
        for i, R in enumerate(bias_mats):
            L = L + bias_w[:, i].view(1, -1, 1) * R[idx].unsqueeze(1)
        mask = ~avail.unsqueeze(0).expand(B, -1).clone()
        mask[torch.arange(B, device=DEV), idx] = True
        L = L.masked_fill(mask.unsqueeze(1), -1e9)
        att = torch.softmax(L, -1)
        return att, (att * V.unsqueeze(1)).sum(-1)

    def forward(self, fr, fd, ps, r, d, bias_mats, z=None):
        B = len(r); h, dh = self.h, self.dh
        att_r, o_r = self._side(self.qr(fr[r]).view(B, h, dh), self.kr(fr).view(-1, h, dh), bias_mats, self.br,
                                r, ps.A[:, d].t(), ps.deg_r > 0)
        att_d, o_d = self._side(self.qd(fd[d]).view(B, h, dh), self.kd(fd).view(-1, h, dh), [], self.br,
                                d, ps.A[r], ps.deg_d > 0)
        out = [o_r, o_d]
        if a.crlt_emb and z is not None:
            wr = att_r * ps.A[:, d].t().unsqueeze(1); wd = att_d * ps.A[r].unsqueeze(1)
            out += [(wr @ z['miRNA']).mean(-1), (wd @ z['drug']).mean(-1)]
        return torch.cat(out, -1)


class Model(torch.nn.Module):
    def __init__(self, data, hidden, ps, D):
        super().__init__()
        self.encoder = GNNEncoder(data, hidden, a.layers, a.dropout)
        d = 2 * hidden
        self.chan = ps.channels() if ps is not None else []
        self.ncn_lin = torch.nn.ModuleDict()
        for k in self.chan:
            w = hidden if k == 'co' else a.ncn_dim
            self.ncn_lin[k] = Linear(hidden, w); d += w
        if a.cnt:
            self.pc = torch.nn.Sequential(Linear(len(ps.names), hidden), torch.nn.LeakyReLU(0.1), Linear(hidden, hidden)); d += hidden
        if a.nbf:
            self.path = DrugPath(len(ps.nbf_ops), a.nbf_dim, a.nbf_layers)
            self.path_lin = torch.nn.Sequential(Linear(a.nbf_dim, hidden), torch.nn.LeakyReLU(0.1)); d += hidden
        if a.crlt:
            nb = (1 + len(IREL)) if a.crlt_bias else 0
            self.ret = Retrieval(D.fr.shape[1], D.fd.shape[1], a.crlt_heads, a.crlt_dim, nb)
            self.ret_lin = torch.nn.Sequential(Linear(self.ret.out_dim, a.ncn_dim), torch.nn.LeakyReLU(0.1)); d += a.ncn_dim
        if a.bank:
            self.bk = torch.nn.Sequential(Linear(D.n_bank, hidden), torch.nn.LeakyReLU(0.1), Linear(hidden, hidden)); d += hidden
        if a.ridge:
            lams = [float(x) for x in a.ridge_lams.split(',')]
            self.ridge = RidgeLayer(lams, D.ridge_kr, D.ridge_kd, a.ridge_learn)
            n_rf = 2 * len(lams) * 3
            self.ridge_bn = torch.nn.BatchNorm1d(n_rf)
            self.ridge_mlp = torch.nn.Sequential(Linear(n_rf, hidden), torch.nn.LeakyReLU(0.1), Linear(hidden, hidden)); d += hidden
            if a.ridge_ncn:
                self.ridge_ncn_lin = torch.nn.ModuleList([Linear(hidden, a.ncn_dim) for _ in range(2)]); d += 2 * a.ncn_dim
        self.lin1 = Linear(d, 2 * hidden); self.lin2 = Linear(2 * hidden, hidden); self.lin3 = Linear(hidden, 1)
        n_tab = (len(ps.names) if a.cnt else 0) + (D.n_bank if a.bank else 0)
        if a.wide:
            self.wide = Linear(n_tab, 1)
        if a.dec == 'nam':
            # neural additive link decoder: logit = <P z_m, Q z_d> + u(z_m) + v(z_d) + sum_c f_c(ncn_c) + sum_j g_j(x_j)
            self.P = Linear(hidden, a.nam_rank, bias=False); self.Q = Linear(hidden, a.nam_rank, bias=False)
            self.um = Linear(hidden, 1); self.vd = Linear(hidden, 1)
            self.fc = torch.nn.ModuleDict({k: torch.nn.Sequential(torch.nn.LeakyReLU(0.1), Linear(self.ncn_lin[k].out_features, 1)) for k in self.chan})
            if n_tab:
                self.shape = ShapeFns(n_tab, a.nam_h)

    def tab(self, cnt, bank):
        return torch.cat([x for x in (cnt if a.cnt else None, bank if a.bank else None) if x is not None], -1)

    def forward(self, data, eli, ps=None, cnt=None, D=None, feats=None, bank=None):
        if a.dec == 'nam':
            return self.forward_nam(data, eli, ps, cnt, bank)
        z = self.encoder(data.x_dict, data.edge_index_dict)
        r, d = eli
        zm, zd = z['miRNA'][r], z['drug'][d]
        if a.zdrop >= 1.0:
            zm, zd = zm * 0, zd * 0
        elif a.zdrop > 0 and self.training:
            keep = (torch.rand(zm.shape[0], 1, device=DEV) > a.zdrop).float()
            zm, zd = zm * keep / (1 - a.zdrop), zd * keep / (1 - a.zdrop)
        parts = [zm, zd]
        if self.chan:
            parts += [self.ncn_lin[k](e) for k, e in zip(self.chan, ps.ncn(r, d, z))]
        if a.cnt:
            parts.append(self.pc(cnt))
        if a.nbf:
            Hp = self.path(ps.nbf_ops, D.n_mir, D.n_drug)
            parts.append(self.path_lin(Hp[d, r]))
        if a.crlt:
            fr, fd = feats
            bias = ([D.Sr] + [ps.irel[k] for k in IREL]) if a.crlt_bias else []
            parts.append(self.ret_lin(self.ret(fr, fd, ps, r, d, bias, z)))
        if a.bank:
            parts.append(self.bk(bank))
        if a.ridge:
            S, Bs = self.ridge(ps.A, getattr(ps, 'ridge_perm', None))
            fts = []
            for M in S:
                v = M[r, d]
                fts += [v, (v - M[r].mean(1)) / (M[r].std(1) + 1e-6), (v - M[:, d].mean(0)) / (M[:, d].std(0) + 1e-6)]
            x = torch.nan_to_num(torch.stack(fts, 1))
            x = torch.sign(x) * torch.log1p(x.abs() * 10.0)
            parts.append(self.ridge_mlp(self.ridge_bn(x)))
            if a.ridge_ncn:
                mid = len(Bs) // 2 // 2 * 2          # middle head, miRNA side then drug side
                (_, Br), (_, Bd) = Bs[mid], Bs[mid + 1]
                w = Br[:, r].t() * ps.A[:, d].t()                         # m' -> m ridge weight x link(m', d)
                parts.append(self.ridge_ncn_lin[0](w @ z['miRNA'] / (w.abs().sum(1, keepdim=True) + 1.0)))
                w = ps.A[r] * Bd[:, d].t()                                # link(m, d') x ridge weight d' -> d
                parts.append(self.ridge_ncn_lin[1](w @ z['drug'] / (w.abs().sum(1, keepdim=True) + 1.0)))
        h = torch.cat(parts, -1)
        h = F.leaky_relu(self.lin1(h), 0.1)
        if a.dropout > 0:
            h = F.dropout(h, a.dropout, self.training)
        h = F.leaky_relu(self.lin2(h), 0.1)
        out = self.lin3(h).view(-1)
        if a.wide:
            out = out + self.wide(self.tab(cnt, bank)).view(-1)
        return out

    def forward_nam(self, data, eli, ps, cnt, bank):
        z = self.encoder(data.x_dict, data.edge_index_dict)
        r, d = eli
        zm, zd = z['miRNA'][r], z['drug'][d]
        out = (self.P(zm) * self.Q(zd)).sum(-1) / math.sqrt(a.nam_rank) + self.um(zm).view(-1) + self.vd(zd).view(-1)
        if self.chan:
            for k, e in zip(self.chan, ps.ncn(r, d, z)):
                out = out + self.fc[k](self.ncn_lin[k](e)).view(-1)
        if a.cnt or a.bank:
            out = out + self.shape(self.tab(cnt, bank))
        return out


class ShapeFns(torch.nn.Module):
    """One small MLP shape function per tabular pair feature (NAM) + a linear term; returns their sum."""

    def __init__(self, n, h):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(n, h) * 0.5); self.b1 = torch.nn.Parameter(torch.zeros(n, h))
        self.w2 = torch.nn.Parameter(torch.randn(n, h) * (1.0 / math.sqrt(h)) * 0.1); self.lin = torch.nn.Parameter(torch.zeros(n))
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        hdn = F.leaky_relu(x.unsqueeze(-1) * self.w1 + self.b1, 0.1)
        return (hdn * self.w2).sum(-1).sum(-1) + (x * self.lin).sum(-1) + self.b


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
    def fit(tr, ytr, val, epochs, rng):
        if MS:
            rng = np.random.RandomState(a.seed * 100 + fold + MS)
        set_seed(a.seed * 100 + fold + MS)
        train_pos = tr[ytr == 1]
        irel_perm = rng.permutation(D.n_mir) if a.irel_shuffle else None
        ridge_perm = torch.as_tensor(rng.permutation(D.n_mir), device=DEV) if a.ridge_shuffle else None
        feats = None
        if a.crlt:
            fr, fd = D.fr, D.fd
            if a.crlt_shuffle:
                fr = fr[torch.as_tensor(rng.permutation(D.n_mir), device=DEV)]; fd = fd[torch.as_tensor(rng.permutation(D.n_drug), device=DEV)]
            feats = (fr, fd)
        need_ps = a.ncn or a.cnt or a.nbf or a.crlt or a.ridge
        K = max(a.lqo, 1)

        def make_passes(rg):
            grp_pos = rg.randint(K, size=len(train_pos)) if a.lqo else np.zeros(len(train_pos), int)
            grp_all = np.zeros(len(tr), int)
            grp_all[ytr == 1] = grp_pos
            grp_all[ytr == 0] = rg.randint(K, size=int((ytr == 0).sum())) if a.lqo else 0
            out = []
            for j in range(K):
                msg = train_pos[grp_pos != j] if a.lqo else train_pos
                sup = np.where(grp_all == j)[0]
                g = build_graph(D, msg).to(DEV)
                ps = PairStruct(D, torch.as_tensor(msg, device=DEV), irel_perm) if need_ps else None
                if ps is not None:
                    ps.ridge_perm = ridge_perm
                out.append(dict(data=g, ps=ps, eli=torch.as_tensor(tr[sup].T, dtype=torch.long, device=DEV),
                                y=torch.as_tensor(ytr[sup], dtype=torch.float, device=DEV), sup=sup, msg=msg))
            return out
        passes = make_passes(rng)
        full = build_graph(D, train_pos).to(DEV)
        ps_full = PairStruct(D, torch.as_tensor(train_pos, device=DEV), irel_perm) if need_ps else None
        if ps_full is not None:
            ps_full.ridge_perm = ridge_perm
        eli_te = torch.as_tensor(te.T, dtype=torch.long, device=DEV)
        eli_va = torch.as_tensor(val[0].T, dtype=torch.long, device=DEV) if val is not None else None
        # evaluation graphs: full training graph, or (--tta) the K pass graphs (+ full)
        evals = [dict(data=full, ps=ps_full, msg=train_pos)]
        if a.tta and a.lqo:
            evals = [dict(data=p['data'], ps=p['ps'], msg=p['msg']) for p in passes] + ([dict(data=full, ps=ps_full, msg=train_pos)] if a.tta == 2 else [])
        mu = sd = None
        if a.cnt:
            ctr = torch.cat([p['ps'].counts(p['eli'][0], p['eli'][1]) for p in passes], 0)
            mu, sd = ctr.mean(0, keepdim=True), ctr.std(0, keepdim=True) + 1e-6
            for p in passes:
                p['cnt'] = (p['ps'].counts(p['eli'][0], p['eli'][1]) - mu) / sd
            for e in evals:
                e['cnt_te'] = (e['ps'].counts(eli_te[0], eli_te[1]) - mu) / sd
                e['cnt_va'] = (e['ps'].counts(eli_va[0], eli_va[1]) - mu) / sd if eli_va is not None else None
        else:
            for p in passes:
                p['cnt'] = None
            for e in evals:
                e['cnt_te'] = e['cnt_va'] = None
        # diagnostic: LR-pipeline pair features (ctx/diff ...) computed on each message graph
        for p in passes:
            p['bank'] = None
        for e in evals:
            e['bank_te'] = e['bank_va'] = None
        if a.bank:
            tf = lambda x: np.sign(x) * np.log1p(np.abs(x))
            for p in passes:
                p['bank'] = tf(D.bank_feats(p['msg'], tr[p['sup']]))
            bmu = np.concatenate([p['bank'] for p in passes]).mean(0, keepdims=True); bsd = np.concatenate([p['bank'] for p in passes]).std(0, keepdims=True) + 1e-6
            to_t = lambda x: torch.as_tensor((x - bmu) / bsd, dtype=torch.float, device=DEV)
            for p in passes:
                p['bank'] = to_t(p['bank'])
            for e in evals:
                msg = e.get('msg', train_pos)
                e['bank_te'] = to_t(tf(D.bank_feats(msg, te)))
                e['bank_va'] = to_t(tf(D.bank_feats(msg, val[0]))) if val is not None else None
            D.n_bank = passes[0]['bank'].shape[1]
        if a.neg_mult > 0:
            known = np.zeros((D.n_mir, D.n_drug), bool)
            known[tr[:, 0], tr[:, 1]] = True; known[te[:, 0], te[:, 1]] = True
            if val is not None:
                known[val[0][:, 0], val[0][:, 1]] = True
            pool = np.argwhere(~known); rng_neg = np.random.RandomState(a.seed * 100 + fold + 7 + MS)
        set_seed(a.seed * 100 + fold + MS)
        model = Model(full, a.hidden, ps_full, D).to(DEV)
        model.eval()
        with torch.no_grad():
            model(full, eli_te[:, :2], ps_full, evals[0]['cnt_te'][:2] if a.cnt else None, D, feats,
                  evals[0]['bank_te'][:2] if a.bank else None)
        wide_p = list(model.wide.parameters()) if a.wide else []
        mod_p = [q for n_, q in model.named_parameters() if n_.split('.')[0] in ('path', 'path_lin', 'ret', 'ret_lin', 'ridge')]
        sp_ids = {id(q) for q in wide_p + mod_p}
        groups = [dict(params=[q for q in model.parameters() if id(q) not in sp_ids])]
        if wide_p:
            groups.append(dict(params=wide_p, lr=a.lr * a.wide_lr))
        if mod_p:
            groups.append(dict(params=mod_p, lr=a.lr * a.mod_lr))
        opt = torch.optim.RMSprop(groups, lr=a.lr, alpha=0.99, weight_decay=a.wd)
        rng_grp = np.random.RandomState(a.seed * 100 + fold + 13 + MS)
        hist = []; best = None; best_val = (-1, None); t0 = time.time()

        def score(eli, key):
            s = 0.0
            for e in evals:
                s = s + torch.sigmoid(model(e['data'], eli, e['ps'], e['cnt_' + key], D, feats, e['bank_' + key]))
            return (s / len(evals)).cpu().numpy()

        for ep in range(1, epochs + 1):
            if a.lqo_resample and ep > 1 and (ep - 1) % a.lqo_resample == 0:
                passes = make_passes(rng_grp)            # stochastic LQO: new leave-out groups
                if a.cnt:
                    for p in passes:
                        p['cnt'] = (p['ps'].counts(p['eli'][0], p['eli'][1]) - mu) / sd
                for p in passes:
                    p['bank'] = None
            model.train(); losses = []
            for p in passes:
                opt.zero_grad()
                eli, y, cnt = p['eli'], p['y'], p['cnt']
                if a.neg_mult > 0:
                    n_neg = int(a.neg_mult * int((y > 0.5).sum()))
                    sel = pool[rng_neg.randint(len(pool), size=n_neg)]
                    sel_t = torch.as_tensor(sel.T, dtype=torch.long, device=DEV)
                    eli = torch.cat([eli, sel_t], 1); y = torch.cat([y, torch.zeros(n_neg, device=DEV)])
                    if a.cnt:
                        cnt = torch.cat([cnt, (p['ps'].counts(sel_t[0], sel_t[1]) - mu) / sd], 0)
                pred = model(p['data'], eli, p['ps'], cnt, D, feats, p['bank'])
                loss = F.binary_cross_entropy_with_logits(pred, y); loss.backward(); opt.step(); losses.append(float(loss))
            model.eval()
            with torch.no_grad():
                s_te = score(eli_te, 'te')
                m = metrics(yte, s_te); m['epoch'] = ep; m['loss'] = float(np.mean(losses))
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
        if a.ridge:
            print('   ridge params', json.dumps(model.ridge.describe(), default=lambda o: o)[:400], flush=True)
        return best, last, best_val, s_te, (best_scores if best_val[1] is not None else s_te), hist, time.time() - t0

    best, last, best_val, s_te, scores, hist, secs = fit(tr, ytr, val, a.epochs, rng)
    row = dict(dataset=a.dataset, split=a.split, tag=a.tag, fold=fold, seed=a.seed,
               best_auc=best['AUC'], best_aupr=best['AUPR'], best_epoch=best['epoch'],
               last_auc=last['AUC'], last_aupr=last['AUPR'], secs=secs)
    if best_val[1] is not None:
        b = best_val[1]
        row.update(valsel_epoch=b['epoch'], valsel_auc=b['AUC'], valsel_aupr=b['AUPR'], valsel_acc=b['ACC'], valsel_f1=b['F1'], valsel_mcc=b['MCC'], val_auc=best_val[0])
        if a.refit:
            # refit on the full training set (train + val) for the validation-selected number of epochs
            trf = np.concatenate([tr, val[0]]); ytrf = np.concatenate([ytr, val[1]])
            _, rl, _, rs, _, _, _ = fit(trf, ytrf, None, b['epoch'], np.random.RandomState(a.seed * 100 + fold + 1000))
            row.update(refit_auc=rl['AUC'], refit_aupr=rl['AUPR'], refit_acc=rl['ACC'], refit_f1=rl['F1'], refit_mcc=rl['MCC'])
            scores = rs
            print(f'   refit ep{b["epoch"]} {rl["AUC"]:.4f}/{rl["AUPR"]:.4f}', flush=True)
    print(f'== {a.tag} {a.dataset} {a.split} fold={fold} seed={a.seed} best_ep={best["epoch"]} AUC={best["AUC"]:.4f} AUPR={best["AUPR"]:.4f}'
          f' | last {last["AUC"]:.4f}/{last["AUPR"]:.4f}'
          + (f' | valsel ep{best_val[1]["epoch"]} {best_val[1]["AUC"]:.4f}/{best_val[1]["AUPR"]:.4f}' if best_val[1] else '')
          + f' ({secs:.0f}s)', flush=True)
    sc = pd.DataFrame(dict(fold=fold, m=te[:, 0], d=te[:, 1], y=yte, score=scores))
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
          + (f' | valsel {d.valsel_auc.mean():.4f}/{d.valsel_aupr.mean():.4f}' if 'valsel_auc' in d else '')
          + (f' | refit {d.refit_auc.mean():.4f}/{d.refit_aupr.mean():.4f}' if 'refit_auc' in d else ''), flush=True)


if __name__ == '__main__':
    main()
