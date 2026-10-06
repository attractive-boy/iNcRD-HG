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
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--tag', default='plain')
ap.add_argument('--folds', default='')
ap.add_argument('--model_seed', type=int, default=0)
ap.add_argument('--check_enc', action='store_true', help='debug: verify the dense encoder equals the PyG encoder')
ap.add_argument('--exact', action='store_true', help='with --enc mask: also shift the pooled NCN intermediate states to their edge-removed values (exact invariance)')
ap.add_argument('--check_inv', type=int, default=0, help='debug: after training, compare the output of N training positives with a recomputation on the graph without their edge')
# ---- round-11 modules -------------------------------------------------------
ap.add_argument('--idemb', type=int, default=0, help='R11: append a learnable per-node embedding of this width to the miRNA/drug content features')
ap.add_argument('--ease_kd', default='', help='R11: drug-side EASE kernels (comma list of chem, knn, or --rel names)')
ap.add_argument('--ease_kd_a', type=float, default=1.0, help='R11: weight of the drug-side EASE kernels')
ap.add_argument('--hardneg', type=int, default=0, help='R11: number of hard negatives per training positive added to the loss')
ap.add_argument('--hardneg_mode', default='same', choices=['same', 'deg', 'both', 'rand', 'mirna', 'all'], help='R11: same-drug / degree-matched / both / uniform-random control; R12 adds mirna (same-miRNA) and all (same-drug + same-miRNA + degree)')
ap.add_argument('--hardneg_w', type=float, default=1.0, help='R11: loss weight of the hard negatives')
ap.add_argument('--hardneg_loss', default='bce', choices=['bce', 'bpr'], help='R11: bce = binary CE on the hard negatives, bpr = pairwise ranking vs the paired positive')
ap.add_argument('--hardneg_pool', type=int, default=0, help='R11: if >0, pre-sample this many candidates per positive and draw --hardneg of them fresh every epoch')
ap.add_argument('--popdec', action='store_true', help='R11: explicit popularity branch added to the decoder logit (degree features only)')
ap.add_argument('--ppr', action='store_true', help='R11: query-hollow PPR/Katz diffusion pair channel (exact masked rank-2 Woodbury, both directions)')
ap.add_argument('--ppr_norm', action='store_true', help='R11: with --ppr, add exact masked row-z / row-percentile / col-z of every diffusion quantity')
ap.add_argument('--ppr_shuffle', action='store_true', help='R11: control -- permute the diffusion features across pairs of the same split part')
ap.add_argument('--ppr_lr', type=float, default=1.0, help='R11: learning-rate multiplier of the diffusion branch')
ap.add_argument('--tpe', type=int, default=0, help='R11: typed path encoder -- max number of length-3 path instances per pair (0 = off)')
ap.add_argument('--tpe_dim', type=int, default=64, help='R11: path token / GRU width')
ap.add_argument('--tpe_lr', type=float, default=1.0, help='R11: learning-rate multiplier of the path encoder')
ap.add_argument('--tpe_shuffle', action='store_true', help='R11: control -- permute the path instances across pairs')
ap.add_argument('--tpe_notype', action='store_true', help='R11: control -- drop the path type/role embeddings')
# ---- round-12 modules -------------------------------------------------------
ap.add_argument('--val_graph', action='store_true', help='R12 (from R10): the 10%% validation positives stay in the message graph '
                '(every channel is query-hollow, so they never see their own edge); the test-time graph then holds all training positives '
                'exactly as in the official PLMF-MDA protocol, instead of only 90%% of them')
ap.add_argument('--lw_d', type=float, default=0.0, help='R12 (from R10): weight of the drug-wise listwise softmax loss (each positive against the same-drug negatives in the batch)')
ap.add_argument('--lw_m', type=float, default=0.0, help='R12 (from R10): weight of the miRNA-wise listwise softmax loss')
ap.add_argument('--moe', type=int, default=0, help='R12: mixture-of-experts decoder -- this many expert heads on the shared pair MLP state, routed by the query pair degree profile')
ap.add_argument('--moe_dim', type=int, default=32, help='R12: router hidden width')
ap.add_argument('--psa', type=int, default=0, help='R12: hollow attention pooling over the --qncn intermediate sets (learned query/key with the M2 weights as log-prior); value = attention query/key width')
ap.add_argument('--psa_heads', type=int, default=1, help='R12: attention heads of --psa (shared value projection, corrections applied to the pooled hidden state)')
ap.add_argument('--psa_lr', type=float, default=30.0, help='R12: learning-rate multiplier of the --psa parameters')
ap.add_argument('--psa_uniform', action='store_true', help='R12: control -- drop the log(w) prior of --psa (uniform attention over the same candidate sets)')
ap.add_argument('--knn2', type=int, default=0, help='R12: union the kNN graph with a second scale (k = this value) on both sides, for the encoder relation, M2 and the EASE kernels')
ap.add_argument('--aa', action='store_true', help='R12: degree-corrected (Adamic-Adar / resource-allocation weighted) masked 3-path counts added to the --qis block')
# ---- round-13 -------------------------------------------------------------------
ap.add_argument('--ebias', type=float, default=0.0, help='R13: entity random effects -- learnable per-miRNA and per-drug logit biases with ridge strength LAM (sum-BCE scale); 0 = off')
ap.add_argument('--ebias_lr', type=float, default=100.0, help='R13: learning-rate multiplier of the entity biases')
ap.add_argument('--ebias_mode', default='lab', choices=['lab', 'all'], help='R13: lab = biases receive gradient only from the labelled-pair BCE (detached in the UNL / HN terms); all = from every term')
ap.add_argument('--ebpost', type=float, default=0.0, help='R13: post-hoc entity biases fitted at the valsel epoch on the labelled train+val pairs (ridge LAM), reported as ebp_*')
ap.add_argument('--here', type=float, default=0.0, help='R13: hollow empirical-Bayes random-effects layer -- per-miRNA / per-drug logit intercepts profiled out '
                'each epoch by ridge-logistic Newton steps (strength LAM, sum-BCE scale) on the labelled training pairs with the network logit as offset; '
                'a training pair is scored with leave-its-own-label-out intercepts (one-step influence correction), val / test with the full ones; 0 = off')
ap.add_argument('--here_iters', type=int, default=4, help='R13: block-Newton sweeps per epoch (warm-started)')
ap.add_argument('--here_eval', action='store_true', help='R13: with --here, the intercepts are refitted every epoch but used only to score val / test '
                '(epoch selection on the calibrated scores); the network is trained without them')
ap.add_argument('--here_noloo', action='store_true', help='R13: control -- score training pairs with the full (label-inclusive) intercepts')
ap.add_argument('--znorm', action='store_true', help='R13: LayerNorm the endpoint states and the pooled NCN vectors before the decoder (magnitude of add-aggregated states grows with degree)')
ap.add_argument('--dropout', type=float, default=0.0, help='R13: dropout on the decoder input and hidden layers')
ap.add_argument('--wd', type=float, default=0.0, help='R13: weight decay of the main parameter group')
ap.add_argument('--strict_pool', action='store_true', help='R13 audit: draw hard negatives and unlabeled cells from every cell outside the message graph and '
                'the training / validation pairs (test cells included, the official negative-pool file not used), i.e. without any knowledge '
                'of which cells are test pairs or positives')
ap.add_argument('--cpr', type=int, default=0, help='R13: cross-pair rectangle contrast -- for each training positive (m1,d1) draw K partner positives (m2,d2) '
                'whose swapped cells (m1,d2),(m2,d1) are outside the message graph; loss -log sigmoid(s11 + s22 - s12 - s21), invariant to any '
                'per-miRNA / per-drug offset (pure interaction signal); 0 = off')
ap.add_argument('--cpr_w', type=float, default=1.0, help='R13: weight of the --cpr term')
ap.add_argument('--aux_head', action='store_true', help='R13: decoupled auxiliary output head -- the unlabeled-cell BCE and the hard-negative BPR '
                'are computed on a second final linear layer over the shared pair state, so they shape the representation but not the '
                'calibration of the evaluation head (labelled-pair BCE only)')
ap.add_argument('--hn_dns', type=float, default=0.0, help='R13: dynamic hard-negative selection -- draw the --hardneg negatives of each positive from its --hardneg_pool candidates with prob. softmax(logit / TAU) under the current model (0 = uniform, R11)')
ap.add_argument('--ema', type=float, default=0.0, help='R13: exponential moving average of the parameters (decay per epoch); validation / test are scored with the EMA weights')
ap.add_argument('--dump', action='store_true', help='R13: at the valsel epoch save logits + penultimate states of train/val/test pairs and the hollow features to cache/')
a = ap.parse_args()
DEV = torch.device(a.device)
REL = [k for k in a.rel.split(',') if k]
DIFF_SET = ((0.3, 0.01), (0.6, 0.05))            # R11: (alpha, beta) of the PPR / Katz channel (as in plmf_innov)


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
        self.neg_pool = np.asarray(neg, dtype=np.int64)          # R11: official negative pool (hard negatives)
        self.pos_all = np.asarray(pos, dtype=np.int64)
        self.seqs = mir['miRNA_Sequence'].astype(str).str.upper().str.replace('T', 'U').tolist()
        assert list(self.names) == list(g['mirna_names']), 'miRNA order mismatch'
        self.n_mir, self.n_drug = len(self.names), len(self.dbs)
        self.xr = rowminmax(g['mirna_kmer']); self.xd = rowminmax(g['drug_x'])
        self.sim_r = knn_edges(self.xr, a.knn); self.sim_d = knn_edges(self.xd, a.knn)
        if a.knn2:                                                                  # R12: multi-scale kNN union
            self.sim_r = torch.unique(torch.cat([self.sim_r, knn_edges(self.xr, a.knn2)], 1), dim=1)
            self.sim_d = torch.unique(torch.cat([self.sim_d, knn_edges(self.xd, a.knn2)], 1), dim=1)
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
        # R11: drug-side kernels (ChemBERTa cosine gram / drug kNN) for the drug-side Gram
        self.ease_KD = []
        for k in [s for s in a.ease_kd.split(',') if s]:
            if k == 'chem':
                K = self.ud @ self.ud.t()
            elif k == 'knn':
                K = D.Sd
            else:
                K = self.rel[k] + torch.diag((self.rel[k].sum(1) > 0).float())
            self.ease_KD.append(K)
        if a.ease and not a.ease_learn:
            for lam in [float(x) for x in a.ease.split(',')]:
                Gd = self.Cd + lam * torch.eye(D.n_drug, device=DEV)
                Gr = self.Cm + lam * torch.eye(D.n_mir, device=DEV)
                for k in [s for s in a.ease_side.split(',') if s]:
                    K = D.kmer_gram if k == 'kmer' else (self.rel[k] + torch.diag((self.rel[k].sum(1) > 0).float()))
                    if k == 'kmer' and rel_perm is not None:
                        K = K[rel_perm][:, rel_perm]
                    Gr = Gr + a.ease_alpha * K
                for K in self.ease_KD:
                    Gd = Gd + a.ease_kd_a * K
                self.ease_P.append((torch.linalg.inv(Gd), torch.linalg.inv(Gr)))

    def _diff_setup(self):
        """PPR / Katz resolvents of the bipartite random walk, both directions."""
        self.diff_sets = []
        An = self.A / self.deg_r.clamp(min=1)[:, None]                  # miRNA -> drug
        And = self.A.t() / self.deg_d.clamp(min=1)[:, None]             # drug -> miRNA
        Gd = And @ An; Gr = An @ And
        Id = torch.eye(Gd.shape[0], device=DEV); Ir = torch.eye(Gr.shape[0], device=DEV)
        for al, be in DIFF_SET:
            for tag, lam, pref in (('ppr', (1 - al) ** 2, al * (1 - al)), ('katz', be ** 2, be ** 2)):
                Md = torch.linalg.inv(Id - lam * Gd); Mr = torch.linalg.inv(Ir - lam * Gr)
                self.diff_sets.append(dict(tag=tag, al=al, be=be, lam=lam, pref=pref, An=An, And=And, Md=Md, Mr=Mr,
                                           AnMd=An @ Md, AndMr=And @ Mr))

    def _diff_side(self, S, r, d, q):
        """masked (query edge removed) and plain drug-side / miRNA-side values for the pairs (r, d).

        Deleting (m, d) removes the edge *and* renormalises the row/column degrees, so both An and
        And change by one nonzero row each; G' = And' An' is then an exact rank-2 update of G and the
        masked resolvent is a rank-2 Woodbury downdate of the cached inverse (verified against
        explicit edge deletion to float32 precision)."""
        lam, pref, An, And, Md, Mr = S['lam'], S['pref'], S['An'], S['And'], S['Md'], S['Mr']
        B = len(r)
        ar = torch.arange(B, device=DEV)
        dr = self.deg_r[r]; dd = self.deg_d[d]
        dr1 = (dr - 1).clamp(min=1); dd1 = (dd - 1).clamp(min=1)
        drs = dr.clamp(min=1); dds = dd.clamp(min=1)
        e_d = torch.zeros(An.shape[1], B, device=DEV); e_d[d, ar] = 1.0        # [nd, B]
        e_m = torch.zeros(An.shape[0], B, device=DEV); e_m[r, ar] = 1.0        # [nm, B]
        eye2 = torch.eye(2, device=DEV)

        # ---- miRNA -> drug: value = pref * An'[m,:] @ M'[:,d]
        w1 = ((And[d] - e_m.t()) / dd1[:, None]) @ An                          # [B, nd]
        v3 = -((An[r] - e_d.t()) / (dds[:, None] * dr1[:, None]))              # [B, nd]
        U = torch.stack([e_d, And[:, r]], 1)                                   # [2, nd, B]
        V = torch.stack([(w1 + v3).t(), ((An[r] - e_d.t()) / dr1[:, None]).t()], 1)   # [2, nd, B]
        Med = Md[:, d]                                                         # [nd, B]
        MU = torch.einsum('ij,jkb->ikb', Md, U)                                # [nd, 2, B]
        VtMU = torch.einsum('ijb,ikb->jkb', V, MU)                             # [2, 2, B]
        VtMe = torch.einsum('ijb,ib->jb', V, Med)                              # [2, B]
        Binv = torch.linalg.inv((eye2[..., None] / lam - VtMU).permute(2, 0, 1)).permute(1, 2, 0)
        c = torch.einsum('jkb,kb->jb', Binv, VtMe)
        w = Med + torch.einsum('ikb,kb->ib', MU, c)
        Anm = (dr[:, None] * An[r] - e_d.t()) / dr1[:, None]                   # [B, nd]
        val_m = pref * (Anm * w.t()).sum(1)
        val_p = pref * (An[r] * Med.t()).sum(1)
        rd = torch.where(q > 0.5, val_m, val_p)
        # ---- drug -> miRNA: value = pref * And'[d,:] @ M'[:,m]
        u3 = An[:, d]                                                          # [nm, B]
        v4 = ((And[d] - e_m.t()) / dd1[:, None]).t()                           # [nm, B]
        v5 = ((((An[r] - e_d.t()) / dr1[:, None])) @ And).t()                  # [nm, B]
        v6 = -((And[d] - e_m.t()) / (drs[:, None] * dd1[:, None])).t()         # [nm, B]
        U2 = torch.stack([u3, e_m], 1)                                         # [2, nm, B]
        V2 = torch.stack([v4, v5 + v6], 1)                                     # [2, nm, B]
        Mre = Mr[:, r]                                                         # [nm, B]
        MU2 = torch.einsum('ij,jkb->ikb', Mr, U2)
        VtMU2 = torch.einsum('ijb,ikb->jkb', V2, MU2)
        VtMe2 = torch.einsum('ijb,ib->jb', V2, Mre)
        Binv2 = torch.linalg.inv((eye2[..., None] / lam - VtMU2).permute(2, 0, 1)).permute(1, 2, 0)
        c2 = torch.einsum('jkb,kb->jb', Binv2, VtMe2)
        w2 = Mre + torch.einsum('ikb,kb->ib', MU2, c2)
        Andm = (dd[:, None] * And[d] - e_m.t()) / dd1[:, None]                 # [B, nm]
        val_m2 = pref * (Andm * w2.t()).sum(1)
        val_p2 = pref * (And[d] * Mre.t()).sum(1)
        drv = torch.where(q > 0.5, val_m2, val_p2)
        return rd, drv

    def _diff_rowcol(self, S, r, d, q):
        """exact masked full row (over drugs) and column (over miRNAs) of the two diffusion directions."""
        lam, pref, An, And, Md, Mr = S['lam'], S['pref'], S['An'], S['And'], S['Md'], S['Mr']
        B = len(r)
        ar = torch.arange(B, device=DEV)
        dr = self.deg_r[r]; dd = self.deg_d[d]
        dr1 = (dr - 1).clamp(min=1); dd1 = (dd - 1).clamp(min=1)
        dds = dd.clamp(min=1)
        e_d = torch.zeros(An.shape[1], B, device=DEV); e_d[d, ar] = 1.0
        e_m = torch.zeros(An.shape[0], B, device=DEV); e_m[r, ar] = 1.0
        eye2 = torch.eye(2, device=DEV)
        # ---- miRNA -> drug: masked row over d'
        w1 = ((And[d] - e_m.t()) / dd1[:, None]) @ An
        v3 = -((An[r] - e_d.t()) / (dds[:, None] * dr1[:, None]))
        U = torch.stack([e_d, And[:, r]], 1)
        V = torch.stack([(w1 + v3).t(), ((An[r] - e_d.t()) / dr1[:, None]).t()], 1)
        row0 = (dr[:, None] * (S['AnMd'][r]) - Md[d]) / dr1[:, None]              # An'[m,:] @ M
        MU = torch.einsum('ijb,bi->jb', U, row0)                                  # row0 @ U   [2, B]
        VtM = torch.einsum('ijb,ik->jkb', V, Md)                                  # V^T M      [2, nd, B]
        VtMU = torch.einsum('ijb,ikb->jkb', V, torch.einsum('ij,jkb->ikb', Md, U))
        Binv = torch.linalg.inv((eye2[..., None] / lam - VtMU).permute(2, 0, 1)).permute(1, 2, 0)
        BinvVtM = torch.einsum('jkb,kib->jib', Binv, VtM)                         # [2, nd, B]
        row_d = row0.t() + torch.einsum('jb,jib->ib', MU, BinvVtM)                # [nd, B]
        row_d = torch.where(q[:, None] > 0.5, pref * row_d.t(), pref * S['AnMd'][r])
        # ---- drug -> miRNA: masked row over m'
        u3 = An[:, d]
        v4 = ((And[d] - e_m.t()) / dd1[:, None]).t()
        v5 = (((An[r] - e_d.t()) / dr1[:, None]) @ And).t()
        v6 = -((And[d] - e_m.t()) / (dr[:, None] * dd1[:, None])).t()
        U2 = torch.stack([u3, e_m], 1)
        V2 = torch.stack([v4, v5 + v6], 1)
        col0 = (dd[:, None] * (S['AndMr'][d]) - Mr[r]) / dd1[:, None]            # And'[d,:] @ Mr
        MU2 = torch.einsum('ijb,bi->jb', U2, col0)
        VtM2 = torch.einsum('ijb,ik->jkb', V2, Mr)
        VtMU2 = torch.einsum('ijb,ikb->jkb', V2, torch.einsum('ij,jkb->ikb', Mr, U2))
        Binv2 = torch.linalg.inv((eye2[..., None] / lam - VtMU2).permute(2, 0, 1)).permute(1, 2, 0)
        BinvVtM2 = torch.einsum('jkb,kib->jib', Binv2, VtM2)                      # [2, nm, B]
        row_r = col0.t() + torch.einsum('jb,jib->ib', MU2, BinvVtM2)              # [nm, B]
        row_r = torch.where(q[:, None] > 0.5, pref * row_r.t(), pref * S['AndMr'][d])
        return row_d, row_r

    def diff(self, r, d):
        q = self.qvals(r, d)
        ar = torch.arange(len(r), device=DEV)
        out = []
        for S in self.diff_sets:
            a_, b_ = self._diff_side(S, r, d, q)
            out += [a_, b_]
        x = torch.stack(out, 1)
        raw = torch.log1p(x.clamp(min=0) * 20.0)
        if not a.ppr_norm:
            return raw
        norm = []
        for S in self.diff_sets:
            rd, rv = self._diff_side(S, r, d, q)
            row_d, row_r = self._diff_rowcol(S, r, d, q)
            for v, row in ((rd, row_d), (rv, row_r)):
                mu = row.mean(1); sd = row.std(1) + 1e-8
                norm.append((v - mu) / sd)                                     # row z (exactly hollow; pct is tie-unstable)
        return torch.cat([raw, torch.stack(norm, 1)], 1)

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
        if a.aa:
            # R12: degree-corrected simple 3-path (= butterfly) counts.  Intermediates' degrees do not change
            # when the query edge is deleted (it only touches m and d), so these stay exactly query-hollow.
            ar = torch.arange(len(r), device=DEV)
            cd = self.Cd[:, d].t() - q[:, None]                                    # [B, nd]  |N(d') cap N(d) \ {m}
            cm = self.Cm[r] - q[:, None]                                           # [B, nm]  |N(m') cap N(m) \ {d}
            cd[ar, d] = 0; cm[ar, r] = 0
            wdd = 1.0 / torch.log(2.0 + self.deg_d); wdd2 = 1.0 / self.deg_d.clamp(min=1)
            wmm = 1.0 / torch.log(2.0 + self.deg_r); wmm2 = 1.0 / self.deg_r.clamp(min=1)
            a3d = ((self.A[r] * wdd) * cd.clamp(min=0)).sum(1)                     # Adamic-Adar via drug intermediates
            r3d = ((self.A[r] * wdd2) * cd.clamp(min=0)).sum(1)                    # resource allocation via drug intermediates
            a3m = ((self.A[:, d].t() * wmm) * cm.clamp(min=0)).sum(1)              # Adamic-Adar via miRNA intermediates
            r3m = ((self.A[:, d].t() * wmm2) * cm.clamp(min=0)).sum(1)             # resource allocation via miRNA intermediates
            out = torch.cat([out, torch.log1p(torch.stack([a3d, r3d, a3m, r3m], 1).clamp(min=0))], 1)
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

    def qncn_attn_data(self, r, d):
        """R12 (--psa): the hollow weight matrices w (B x N) feeding the --qncn pools, one per relation in
        ncn_rels(), paired with the side ('m' = intermediates are miRNAs, 'd' = drugs) of the states they
        weight.  Same construction as qncn(), so every w is query-hollow by construction."""
        q = self.qvals(r, d).view(-1, 1)
        out = []
        for k in self.ncn_rels():
            if k == 'co':
                Ad = self.A[:, d].t()
                w = Ad * (self.Cm[r] - q * Ad)
                w[torch.arange(len(r), device=DEV), r] = 0
                out.append((w, 'm'))
                Ar = self.A[r]
                w = Ar * (self.Cd[:, d].t() - q * Ar)
                w[torch.arange(len(d), device=DEV), d] = 0
                out.append((w, 'd'))
            elif k == 'dknn':
                out.append((self.A[r] * self.Sd[:, d].t(), 'd'))
            else:
                R = self.Sr if k == 'knn' else self.rel[k]
                out.append((self.A[:, d].t() * R[r], 'm'))
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


# ----------------------------------------------------------------------------- typed path encoder (R11)
class PathData:
    __slots__ = ('node', 'slot', 'typ', 'mask', 'pair_pos', 'n_pairs', 'n_paths')

    def __init__(self, node, slot, typ, mask, pair_pos, n_pairs):
        self.node = node; self.slot = slot; self.typ = typ; self.mask = mask
        self.pair_pos = pair_pos; self.n_pairs = n_pairs; self.n_paths = len(node)


def _cap_by_pair(pp, K, rng):
    """keep at most K paths per part row (deterministic given rng)."""
    if K <= 0 or len(pp) == 0:
        return np.zeros(len(pp), bool)
    order = np.lexsort((rng.random(len(pp)), pp))
    s = pp[order]
    starts = np.r_[0, np.nonzero(np.diff(s))[0] + 1]
    rank = np.arange(len(s)) - np.repeat(starts, np.diff(np.r_[starts, len(s)]))
    keep = np.zeros(len(pp), bool); keep[order] = rank < K
    return keep


def build_paths(D, ps, pairs, K0=8, Kb=8, rng=None):
    """Query-hollow typed path instances for the given (m, d) pairs.

    type 0: association 3-path  m -A- d1 -A- m1 -A- d
    type 1: kNN bridge          m -Sr- m1 -A- d
    type 2: drug-kNN bridge     m -A- d1 -Sd- d
    type 3..: sequence relation bridge  m -R- m1 -A- d  (R in REL)
    All paths are simple and contain no query edge.  Slot 2 is a PAD for the 2-node bridges."""
    if rng is None:
        rng = np.random.RandomState(0)
    An = ps.A.cpu().numpy().astype(np.int8); nm, nd = An.shape
    pos_of = np.full(nm * nd, -1, np.int64)
    pid_pairs = pairs[:, 0].astype(np.int64) * nd + pairs[:, 1]
    pos_of[pid_pairs] = np.arange(len(pairs))
    nbr_d = [np.nonzero(An[:, j])[0] for j in range(nd)]
    nbr_m = [np.nonzero(An[i])[0] for i in range(nm)]
    chunks = []
    # --- type 0
    rows, cols = np.nonzero(An)
    cnt = An.sum(0)[cols]
    m_rep = np.repeat(rows, cnt); d1_rep = np.repeat(cols, cnt)
    m1_rep = np.concatenate([nbr_d[j] for j in cols])
    cnt2 = An.sum(1)[m1_rep]
    m2 = np.repeat(m_rep, cnt2); d1_2 = np.repeat(d1_rep, cnt2); m1_2 = np.repeat(m1_rep, cnt2)
    d2 = np.concatenate([nbr_m[i] for i in m1_rep])
    keep = (d2 != d1_2) & (m1_2 != m2)
    m2, d1_2, m1_2, d2 = m2[keep], d1_2[keep], m1_2[keep], d2[keep]
    pid = m2.astype(np.int64) * nd + d2; pp = pos_of[pid]; sel = pp >= 0
    m2, d1_2, m1_2, d2, pp = m2[sel], d1_2[sel], m1_2[sel], d2[sel], pp[sel]
    keep = _cap_by_pair(pp, K0, rng)
    chunks.append((m2[keep], d1_2[keep], m1_2[keep], d2[keep], pp[keep], np.zeros(int(keep.sum()), np.int64)))
    # --- bridge types
    bridges = [(1, D.Sr)] + [(3 + i, D.rel[k]) for i, k in enumerate(REL)]
    for ti, R in bridges:
        R = R.cpu().numpy()
        rr, cc = np.nonzero(R)
        cnt = An.sum(1)[cc]
        m_r = np.repeat(rr, cnt); m1_r = np.repeat(cc, cnt)
        d_r = np.concatenate([nbr_m[i] for i in cc])
        pid = m_r.astype(np.int64) * nd + d_r; pp = pos_of[pid]; sel = pp >= 0
        m_r, m1_r, d_r, pp = m_r[sel], m1_r[sel], d_r[sel], pp[sel]
        keep = _cap_by_pair(pp, Kb, rng)
        chunks.append((m_r[keep], m1_r[keep], np.full(int(keep.sum()), -1, np.int64), d_r[keep], pp[keep], np.full(int(keep.sum()), ti, np.int64)))
    Sd = D.Sd.cpu().numpy(); rr, cc = np.nonzero(Sd)
    cnt = An.sum(0)[rr]
    m_r = np.concatenate([nbr_d[j] for j in rr]); d1_r = np.repeat(rr, cnt); d_r = np.repeat(cc, cnt)
    pid = m_r.astype(np.int64) * nd + d_r; pp = pos_of[pid]; sel = pp >= 0
    m_r, d1_r, d_r, pp = m_r[sel], d1_r[sel], d_r[sel], pp[sel]
    keep = _cap_by_pair(pp, Kb, rng)
    chunks.append((m_r[keep], d1_r[keep], np.full(int(keep.sum()), -1, np.int64), d_r[keep], pp[keep], np.full(int(keep.sum()), 2, np.int64)))
    n_paths = sum(len(c[0]) for c in chunks)
    if n_paths == 0:
        return PathData(torch.zeros(0, 4, dtype=torch.long, device=DEV), torch.zeros(0, 4, dtype=torch.long, device=DEV),
                        torch.zeros(0, dtype=torch.long, device=DEV), torch.zeros(0, 4, device=DEV),
                        torch.zeros(0, dtype=torch.long, device=DEV), len(pairs))
    x1 = np.concatenate([c[1] for c in chunks]); x2 = np.concatenate([c[2] for c in chunks])
    d_ = np.concatenate([c[3] for c in chunks]); pp = np.concatenate([c[4] for c in chunks])
    m_ = np.concatenate([c[0] for c in chunks]); ty = np.concatenate([c[5] for c in chunks])
    node = np.stack([m_, x1, x2, d_], 1)
    mask = np.ones_like(node, np.float32); mask[:, 2] = (x2 >= 0)
    slot = np.tile(np.arange(4, dtype=np.int64), (len(node), 1))
    return PathData(torch.as_tensor(node, dtype=torch.long, device=DEV), torch.as_tensor(slot, device=DEV),
                    torch.as_tensor(ty, dtype=torch.long, device=DEV), torch.as_tensor(mask, device=DEV),
                    torch.as_tensor(pp, dtype=torch.long, device=DEV), len(pairs))


class PathEncoder(torch.nn.Module):
    """GRU over the typed path tokens + attention pooling over the path set of each pair."""

    def __init__(self, dim, n_types, hidden, nm, nd, Sr, Sd):
        super().__init__()
        self.inp = Linear(hidden, dim)
        self.role = torch.nn.Embedding(4, dim)
        self.typ = torch.nn.Embedding(n_types, dim)
        self.pad = torch.nn.Parameter(torch.zeros(hidden))
        self.gru = torch.nn.GRU(dim, dim, batch_first=True)
        self.att = torch.nn.Sequential(Linear(dim, dim), torch.nn.Tanh(), Linear(dim, 1))
        self.out = Linear(dim, hidden)
        self.nm, self.nd = nm, nd; self.Sr, self.Sd = Sr, Sd

    def forward(self, z_all, pd, eli, corr):
        P = pd.n_paths
        if P == 0:
            return torch.zeros(pd.n_pairs, self.out.out_features, device=z_all.device)
        node = pd.node; mask = pd.mask
        zt = z_all[node.clamp(min=0, max=z_all.shape[0] - 1)]
        zt = torch.where(mask[..., None] > 0, zt, self.pad.view(1, 1, -1))
        if corr is not None:
            r = eli[0][pd.pair_pos]; d = eli[1][pd.pair_pos]
            cm_dm, cm_mm, cd_md, cd_dd = (x[pd.pair_pos] for x in corr)
            inter = (((pd.slot == 1) | (pd.slot == 2)) & (mask > 0)).float()[..., None]
            is_m = (node < self.nm)
            mi = node.clamp(min=0, max=self.nm - 1)
            sr_val = self.Sr[mi, r[:, None]]
            add_m = cm_dm[:, None, :] + cm_mm[:, None, :] * sr_val[..., None]
            di = (node - self.nm).clamp(min=0, max=self.nd - 1)
            sd_val = self.Sd[di, d[:, None]]
            add_d = cd_md[:, None, :] + cd_dd[:, None, :] * sd_val[..., None]
            zt = zt + torch.where(is_m[..., None], add_m, add_d) * inter
        tok = self.inp(zt) + self.role.weight[pd.slot]
        if not a.tpe_notype:
            tok = tok + self.typ(pd.typ)[:, None, :]
        h, _ = self.gru(tok)
        h = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        w = self.att(h).view(-1)
        pmax = torch.full((pd.n_pairs,), -1e9, device=z_all.device)
        pmax = pmax.scatter_reduce(0, pd.pair_pos, w.detach(), reduce='amax', include_self=True)
        e = torch.exp(w - pmax[pd.pair_pos])
        den = torch.zeros(pd.n_pairs, device=z_all.device).index_add_(0, pd.pair_pos, e).clamp(min=1e-9)
        emb = torch.zeros(pd.n_pairs, self.out.in_features, device=z_all.device).index_add_(0, pd.pair_pos, e[:, None] * h)
        return self.out(emb / den[:, None])


class PairSetAttention(torch.nn.Module):
    """R12 (--psa): learned attention pooling over the hollow 2-hop intermediate sets of --qncn.

    For a relation set with hollow weights w (B x N) and intermediate states Z (N x H):
        logits = q([z_m, z_d]) . k(Z) / sqrt(dim) + log(w + eps)     (w = 0 masked out)
        att    = softmax(logits)
        pooled = att @ Z + cm_dm * (att . 1[intermediate hears d]) + cm_mm * (att . 1[intermediate hears m])
        e      = o(pooled)
    The two correction terms are exactly the query-hollow shifts of the intermediates that the fixed --qncn
    pools already use, so with --exact the module stays exactly query-hollow (--check_inv verifies it).
    This replaces the fixed weights w/(sum w + 1) of M2 by a learned, pair-conditioned reweighting of the
    same hollow candidate sets, i.e. a neural common-neighbour with attention instead of a fixed mean."""

    def __init__(self, n_sets, hidden, dim, A, Sr, Sd):
        super().__init__()
        self.q = torch.nn.ModuleList([Linear(2 * hidden, dim) for _ in range(n_sets)])
        self.k = torch.nn.ModuleList([Linear(hidden, dim) for _ in range(n_sets)])
        self.o = torch.nn.ModuleList([Linear(hidden, dim) for _ in range(n_sets)])
        self.A, self.Sr, self.Sd = A, Sr, Sd

    def forward(self, pair, sets, z, corr, r, d):
        outs = []
        for s_, (w, side) in enumerate(sets):
            Z = z['miRNA'] if side == 'm' else z['drug']
            mask = w > 0
            logits = (self.q[s_](pair) @ self.k[s_](Z).t()) / math.sqrt(self.q[s_].out_features)
            if not a.psa_uniform:
                logits = logits + torch.log(w.clamp(min=1e-9))
            logits = logits.masked_fill(~mask, -1e9)
            att = torch.softmax(logits, 1)
            pooled = att @ Z
            if corr is not None:
                cm_dm = 0.5 * F.linear(corr['dh_d'], corr['W_dm']); cm_mm = 0.5 * F.linear(corr['dh_m'], corr['W_mm'])
                cd_md = 0.5 * F.linear(corr['dh_m'], corr['W_md']); cd_dd = 0.5 * F.linear(corr['dh_d'], corr['W_dd'])
                if side == 'm':
                    sel1 = (self.A[:, d] > 0.5).t().float(); sel2 = self.Sr[r]
                    pooled = pooled + cm_dm * (att * sel1).sum(1, keepdim=True) + cm_mm * (att * sel2).sum(1, keepdim=True)
                else:
                    sel1 = (self.A[r] > 0.5).float(); sel2 = self.Sd[d]
                    pooled = pooled + cd_md * (att * sel1).sum(1, keepdim=True) + cd_dd * (att * sel2).sum(1, keepdim=True)
            empty = (~mask).all(1, keepdim=True)
            outs.append(self.o[s_]((pooled * (~empty).float())))
        return outs


class MoEDecoder(torch.nn.Module):
    """R12 (--moe): mixture of expert heads on the shared pair MLP state, routed by the query pair degree
    profile [log1p(deg_m), log1p(deg_d), product, |difference|].  The router output layer is zero-initialised
    so training starts from the uniform mixture = the average expert (a single head); experts may then
    specialize across the degree regimes where the gains of the other modules are known to concentrate."""

    def __init__(self, hidden, K, dim=32):
        super().__init__()
        self.experts = torch.nn.ModuleList([Linear(hidden, 1) for _ in range(K)])
        self.router = torch.nn.Sequential(Linear(4, dim), torch.nn.LeakyReLU(0.1), Linear(dim, K))
        torch.nn.init.zeros_(self.router[-1].weight); torch.nn.init.zeros_(self.router[-1].bias)

    def forward(self, h, degf):
        g = torch.softmax(self.router(degf), 1)
        out = torch.zeros(h.shape[0], device=h.device)
        for k in range(len(self.experts)):
            out = out + g[:, k] * self.experts[k](h).view(-1)
        return out


# ----------------------------------------------------------------------------- model
class Model(torch.nn.Module):
    def __init__(self, data, hidden, ps, n_qis, n_ease, n_diff=0):
        super().__init__()
        self.encoder = GNNEncoder(data, hidden, a.layers)
        self.id_r = torch.nn.Embedding(ps.A.shape[0], a.idemb) if a.idemb else None
        self.id_d = torch.nn.Embedding(ps.A.shape[1], a.idemb) if a.idemb else None
        if self.id_r is not None:
            torch.nn.init.normal_(self.id_r.weight, std=0.01); torch.nn.init.normal_(self.id_d.weight, std=0.01)
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
        if a.ppr:
            self.db = torch.nn.BatchNorm1d(n_diff)
            self.dm = torch.nn.Sequential(Linear(n_diff, hidden), torch.nn.LeakyReLU(0.1), Linear(hidden, hidden)); d += hidden
        if a.tpe:
            self.tpe = PathEncoder(a.tpe_dim, 3 + len(REL), hidden, ps.A.shape[0], ps.A.shape[1], ps.Sr, ps.Sd); d += hidden
        if a.psa:
            assert a.qncn and a.exact and a.enc == 'mask', '--psa requires --qncn --enc mask --exact'
            n_sets = sum(2 if k == 'co' else 1 for k in ps.ncn_rels())
            self.psa = PairSetAttention(n_sets, hidden, a.psa, ps.A, ps.Sr, ps.Sd); d += n_sets * a.psa
        self.lin1 = Linear(d, 2 * hidden); self.lin2 = Linear(2 * hidden, hidden)
        if a.znorm:
            self.zn = torch.nn.ModuleList([torch.nn.LayerNorm(hidden), torch.nn.LayerNorm(hidden)] +
                                          [torch.nn.LayerNorm(w) for w in (ps.ncn_widths(hidden) if a.qncn else [])])
        self.lin3 = None if a.moe else Linear(hidden, 1)
        if a.aux_head:
            self.lin3_aux = Linear(hidden, 1)
        self._last_aux = None
        if a.moe:
            self.moe = MoEDecoder(hidden, a.moe, a.moe_dim)
        if a.popdec:
            self.pop = torch.nn.Sequential(Linear(4, 16), torch.nn.LeakyReLU(0.1), Linear(16, 1))
            torch.nn.init.zeros_(self.pop[-1].weight); torch.nn.init.zeros_(self.pop[-1].bias)
        if a.wide:
            self.wide = Linear(n_qis + n_ease, 1)
            torch.nn.init.zeros_(self.wide.weight); torch.nn.init.zeros_(self.wide.bias)
        if a.ebias > 0:                                                            # R13: entity random effects
            self.eb_m = torch.nn.Parameter(torch.zeros(ps.A.shape[0])); self.eb_d = torch.nn.Parameter(torch.zeros(ps.A.shape[1]))

    def ebias_term(self, eli):
        return self.eb_m[eli[0]] + self.eb_d[eli[1]]

    def forward(self, data, eli, ps=None, qis=None, ease=None, dense=None, paths=None, diff=None, return_h=False):
        r, d = eli
        if a.enc == 'mask':
            xr, xd, Sr1, Sd1, A = dense
            if self.id_r is not None:
                xr = torch.cat([xr, self.id_r.weight], 1); xd = torch.cat([xd, self.id_d.weight], 1)
            zr, zd, zm, zdd, corr = self.encoder.forward_dense(xr, xd, Sr1, Sd1, A, r, d, ps.qvals(r, d))
            z = {'miRNA': zr, 'drug': zd}
            if not a.exact:
                corr = None
        else:
            z = self.encoder(data.x_dict, data.edge_index_dict)
            zm, zdd = z['miRNA'][r], z['drug'][d]; corr = None
        if a.tpe and paths is not None:
            if corr is not None:
                lin = lambda W, x: F.linear(x, W)
                c4 = (0.5 * lin(corr['W_dm'], corr['dh_d']), 0.5 * lin(corr['W_mm'], corr['dh_m']),
                      0.5 * lin(corr['W_md'], corr['dh_m']), 0.5 * lin(corr['W_dd'], corr['dh_d']))
            else:
                c4 = None
            parts_tpe = self.tpe(torch.cat([z['miRNA'], z['drug']], 0), paths, eli, c4)
        else:
            parts_tpe = None
        parts = [zm, zdd]
        if a.qncn:
            parts += [lin(e) for lin, e in zip(self.ncn_lin, ps.qncn(r, d, z, corr))]
        if a.znorm:
            parts = [ln(x) for ln, x in zip(self.zn, parts)]
        if a.psa:
            parts += self.psa(torch.cat([zm, zdd], -1), ps.qncn_attn_data(r, d), z, corr, r, d)
        if a.qis:
            parts.append(self.pc(qis))
        if a.ease_learn:
            ease = self.ease_layer(ps, r, d)
        if a.ease:
            parts.append(self.em(self.eb(ease)))
        if a.ppr:
            parts.append(self.dm(self.db(diff)))
        if parts_tpe is not None:
            parts.append(parts_tpe)
        h = torch.cat(parts, -1)
        if a.dropout > 0:
            h = F.dropout(h, a.dropout, self.training)
        h = F.leaky_relu(self.lin1(h), 0.1)
        if a.dropout > 0:
            h = F.dropout(h, a.dropout, self.training)
        h = F.leaky_relu(self.lin2(h), 0.1)
        if a.moe:
            q_ = ps.qvals(r, d)
            dr_ = torch.log1p((ps.deg_r[r] - q_).clamp(min=0)); dd_ = torch.log1p((ps.deg_d[d] - q_).clamp(min=0))
            out = self.moe(h, torch.stack([dr_, dd_, dr_ * dd_, (dr_ - dd_).abs()], 1))
        else:
            out = self.lin3(h).view(-1)
            if a.aux_head:
                self._last_aux = self.lin3_aux(h).view(-1)
        if a.popdec:
            q = ps.qvals(r, d)
            dr = torch.log1p((ps.deg_r[r] - q).clamp(min=0)); dd = torch.log1p((ps.deg_d[d] - q).clamp(min=0))
            out = out + self.pop(torch.stack([dr, dd, dr * dd, dr + dd], 1)).view(-1)
        if a.wide:
            wx = torch.cat([x for x in (qis, self.eb(ease) if a.ease else None) if x is not None], 1)
            out = out + self.wide(wx).view(-1)
        if a.ebias > 0:
            out = out + self.ebias_term(eli)
        if return_h:
            return out, h
        return out


def listwise(pred, y, idx, n):
    """R12 (from R10): mean over positives of -log softmax of s_i against the negatives of the batch
    sharing its node idx[i] (listwise / whole-row-style objective)."""
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


def here_fit(L, y, mI, dI, bm, bd, lam, iters):
    """R13 --here: block-coordinate Newton for ridge-logistic intercepts with offset L (sum-BCE + lam * ||b||^2).
    Returns the intercepts and, per labelled pair, its leave-own-label-out intercepts (one Newton step without the pair)."""
    nm, nd = bm.shape[0], bd.shape[0]
    for _ in range(iters):                                                # damped block Newton (|step| <= 1: saturated logits make
        p = torch.sigmoid(L + bm[mI] + bd[dI]); w = p * (1 - p)           # the undamped step overshoot and oscillate)
        bm = bm - ((torch.zeros(nm, device=L.device).index_add_(0, mI, p - y) + 2 * lam * bm) / (torch.zeros(nm, device=L.device).index_add_(0, mI, w) + 2 * lam)).clamp(-1, 1)
        p = torch.sigmoid(L + bm[mI] + bd[dI]); w = p * (1 - p)
        bd = bd - ((torch.zeros(nd, device=L.device).index_add_(0, dI, p - y) + 2 * lam * bd) / (torch.zeros(nd, device=L.device).index_add_(0, dI, w) + 2 * lam)).clamp(-1, 1)
    p = torch.sigmoid(L + bm[mI] + bd[dI]); w = p * (1 - p)
    Hm = torch.zeros(nm, device=L.device).index_add_(0, mI, w) + 2 * lam; Hd = torch.zeros(nd, device=L.device).index_add_(0, dI, w) + 2 * lam
    loo = bm[mI] + (p - y) / (Hm[mI] - w) + bd[dI] + (p - y) / (Hd[dI] - w)
    return bm, bd, loo


def ebias_post(keep, tr, ytr, val, te, nm, nd, lam):
    """R13 --ebpost: per-miRNA / per-drug logit offsets fitted (L-BFGS, ridge lam, sum-BCE) on the labelled train + val
    pairs with the network logit as a fixed offset, then added to the test logits."""
    if val is not None:
        P = np.concatenate([tr, val[0]]); Y = torch.tensor(np.concatenate([ytr, val[1]]), dtype=torch.float64)
        L = torch.tensor(np.concatenate([keep['tr'], keep['va']]), dtype=torch.float64)
    else:                                                                          # official protocol: all training pairs
        P = tr; Y = torch.tensor(ytr, dtype=torch.float64); L = torch.tensor(keep['tr'], dtype=torch.float64)
    mI = torch.as_tensor(P[:, 0]); dI = torch.as_tensor(P[:, 1])
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([bm, bd], max_iter=500, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(L + bm[mI] + bd[dI], Y, reduction='sum') + lam * (bm.pow(2).sum() + bd.pow(2).sum())
        loss.backward(); return loss
    opt.step(closure)
    return keep['te'] + bm.detach().numpy()[te[:, 0]] + bd.detach().numpy()[te[:, 1]]


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
        train_pos = np.concatenate([train_pos, val[0][val[1] == 1]])          # R12: message graph = all training positives (hollow)
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
    if a.ppr:
        ps._diff_setup()
        feats['diff'] = {k: ps.diff(*e) for k, e in (('tr', eli_tr), ('te', eli_te), ('va', eli_va)) if e is not None}
        if a.ppr_shuffle:
            for k in feats['diff']:
                feats['diff'][k] = feats['diff'][k][torch.as_tensor(rng.permutation(len(feats['diff'][k])), device=DEV)]
    # R11: hard negatives -----------------------------------------------------
    eli_hn = None; hn_anchor = None
    if a.hardneg > 0:
        occ = np.zeros((D.n_mir, D.n_drug), bool)
        if a.strict_pool:                                                   # R13 audit: no test / positive knowledge
            for arr in (tr,) + ((val[0],) if val is not None else ()):
                occ[arr[:, 0], arr[:, 1]] = True
            occ[train_pos[:, 0], train_pos[:, 1]] = True
            pool = np.argwhere(~occ)
        else:
            for arr in (tr, te) + ((val[0],) if val is not None else ()):
                occ[arr[:, 0], arr[:, 1]] = True
            occ[D.pos_all[:, 0], D.pos_all[:, 1]] = True
            pool = D.neg_pool[~occ[D.neg_pool[:, 0], D.neg_pool[:, 1]]]
        by_drug = {int(dd): pool[pool[:, 1] == dd] for dd in np.unique(pool[:, 1])}
        by_mir = {int(mm): pool[pool[:, 0] == mm] for mm in np.unique(pool[:, 0])}
        degd_train = np.bincount(train_pos[:, 1], minlength=D.n_drug)
        qs = np.quantile(degd_train, np.linspace(0, 1, 6)); dbins = np.digitize(degd_train, qs[1:-1])
        by_bin = {}
        for b in np.unique(dbins):
            ds = np.where(dbins == b)[0]
            by_bin[int(b)] = pool[np.isin(pool[:, 1], ds)]
        pos_idx = np.where(ytr == 1)[0]
        cand_list = []
        for i in pos_idx:
            m0, d0 = tr[i, 0], tr[i, 1]
            cands = []
            if a.hardneg_mode == 'rand':
                cands.append(pool)
            if a.hardneg_mode in ('same', 'both', 'all') and int(d0) in by_drug:
                cands.append(by_drug[int(d0)])
            if a.hardneg_mode in ('mirna', 'all') and int(m0) in by_mir:
                cands.append(by_mir[int(m0)])
            if a.hardneg_mode in ('deg', 'both', 'all'):
                c = by_bin.get(int(dbins[d0]))
                if c is not None and len(c):
                    cands.append(c)
            cand_list.append(np.concatenate(cands) if cands else np.zeros((0, 2), np.int64))
        if a.hardneg_pool > 0:
            P = a.hardneg_pool
            cand_arr = np.zeros((len(pos_idx), P, 2), np.int64); cand_ok = np.zeros((len(pos_idx), P), bool)
            for j, c in enumerate(cand_list):
                if len(c) == 0:
                    continue
                pick = rng.choice(len(c), size=P, replace=len(c) < P)
                cand_arr[j] = c[pick]; cand_ok[j] = True
            hn_cand = torch.as_tensor(cand_arr, dtype=torch.long, device=DEV)      # [n_pos, P, 2]
            hn_ok = torch.as_tensor(cand_ok, device=DEV)
            hn_pos = torch.as_tensor(pos_idx, dtype=torch.long, device=DEV)
            hn_gen = np.random.RandomState(a.seed * 977 + fold + MS)
            print(f'[hn] resampled pool {P} per positive ({cand_ok.any(1).sum()} with candidates), K={a.hardneg}, '
                  f'mode={a.hardneg_mode} w={a.hardneg_w} loss={a.hardneg_loss}', flush=True)
        else:
            hn = np.zeros((len(pos_idx) * a.hardneg, 2), np.int64); anc = np.zeros(len(pos_idx) * a.hardneg, np.int64)
            cnt = 0
            for i, cands in zip(pos_idx, cand_list):
                if len(cands) == 0:
                    continue
                pick = rng.choice(len(cands), size=a.hardneg, replace=len(cands) < a.hardneg)
                for p in pick:
                    hn[cnt] = cands[p]; anc[cnt] = i; cnt += 1
            hn, anc = hn[:cnt], anc[:cnt]
            eli_hn = torch.as_tensor(hn.T, dtype=torch.long, device=DEV)
            hn_anchor = torch.as_tensor(anc, dtype=torch.long, device=DEV)
            if a.qis:
                feats['qis']['hn'] = (ps.qis(*eli_hn) - mu) / sd
            if a.ease and not a.ease_learn:
                feats['ease']['hn'] = ps.ease(*eli_hn)
            if a.ppr:
                feats['diff']['hn'] = ps.diff(*eli_hn)
            print(f'[hn] {cnt} hard negatives mode={a.hardneg_mode} w={a.hardneg_w} loss={a.hardneg_loss}', flush=True)
    eli_u = None
    if a.unl > 0:
        occ = np.zeros((D.n_mir, D.n_drug), bool)
        for arr in ((tr,) if a.strict_pool else (tr, te)) + ((val[0],) if val is not None else ()):
            occ[arr[:, 0], arr[:, 1]] = True
        if a.strict_pool:
            occ[train_pos[:, 0], train_pos[:, 1]] = True
        U = np.argwhere(~occ)
        eli_u = torch.as_tensor(U.T, dtype=torch.long, device=DEV)
        chunks = lambda fn: torch.cat([fn(eli_u[0, i:i + 20000], eli_u[1, i:i + 20000]) for i in range(0, eli_u.shape[1], 20000)])
        if a.qis:
            feats['qis']['u'] = (chunks(ps.qis) - mu) / sd
        if a.ease and not a.ease_learn:
            feats['ease']['u'] = chunks(ps.ease)
        if a.ppr:
            feats['diff']['u'] = chunks(ps.diff)
        n_u = int(round(a.unl * len(tr)))
        print(f'[unl] pool {len(U)} cells, {n_u} sampled per epoch, weight {a.unl_w}', flush=True)
        u_gen = torch.Generator(device='cpu'); u_gen.manual_seed(a.seed * 100 + fold + MS)
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
    pds = {}
    if a.tpe:
        for part, eli in (('tr', eli_tr), ('te', eli_te), ('va', eli_va), ('hn', eli_hn)):
            if eli is not None:
                pairs_np = np.stack([eli[0].cpu().numpy(), eli[1].cpu().numpy()], 1)
                pds[part] = build_paths(D, ps, pairs_np, K0=a.tpe, Kb=a.tpe, rng=rng)
        if a.tpe_shuffle:
            for part, pd_ in pds.items():
                perm = torch.as_tensor(rng.permutation(pd_.n_paths), device=DEV)
                pd_.node = pd_.node[perm]; pd_.typ = pd_.typ[perm]; pd_.mask = pd_.mask[perm]
        print(f'[tpe] paths ' + ' '.join(f'{k} {v.n_paths}' for k, v in pds.items()), flush=True)
    g = lambda key, part: pds.get(part) if key == 'tpe' else (feats[key][part] if key in feats else None)
    n_qis = feats['qis']['tr'].shape[1] if a.qis else 0
    n_ease = (feats['ease']['tr'].shape[1] if not a.ease_learn else 2 * len(a.ease.split(','))) if a.ease else 0
    n_diff = feats['diff']['tr'].shape[1] if a.ppr else 0
    model = Model(full, a.hidden, ps, n_qis, n_ease, n_diff).to(DEV)
    model.eval()
    with torch.no_grad():
        if a.idemb:                                                           # lazy layers must see the ID-extended dims
            full['miRNA'].x = torch.cat([full['miRNA'].x, model.id_r.weight], 1)
            full['drug'].x = torch.cat([full['drug'].x, model.id_d.weight], 1)
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
    tpe_p = list(model.tpe.parameters()) if a.tpe else []
    ppr_p = (list(model.db.parameters()) + list(model.dm.parameters())) if a.ppr else []
    psa_p = list(model.psa.parameters()) if a.psa else []
    eb_p = [model.eb_m, model.eb_d] if a.ebias > 0 else []
    wid = {id(p) for p in wide_p + ease_p + tpe_p + ppr_p + psa_p + eb_p}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in wid], weight_decay=a.wd)]
    if wide_p:
        groups.append(dict(params=wide_p, lr=a.lr * a.wide_lr))
    if ease_p:
        groups.append(dict(params=ease_p, lr=a.lr * a.ease_lr))
    if tpe_p:
        groups.append(dict(params=tpe_p, lr=a.lr * a.tpe_lr))
    if ppr_p:
        groups.append(dict(params=ppr_p, lr=a.lr * a.ppr_lr))
    if psa_p:
        groups.append(dict(params=psa_p, lr=a.lr * a.psa_lr))
    if eb_p:
        groups.append(dict(params=eb_p, lr=a.lr * a.ebias_lr))
    opt = torch.optim.RMSprop(groups, lr=a.lr, alpha=0.99)
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if a.ema > 0 else None
    here_bm = torch.zeros(D.n_mir, device=DEV); here_bd = torch.zeros(D.n_drug, device=DEV); here_L = None
    here_loo = torch.zeros(len(y_tr), device=DEV)
    # R13: dynamic hard negatives need the hollow channels of every candidate once
    if a.hn_dns != 0:
        assert a.hardneg > 0 and a.hardneg_pool > 0
        cand_flat = hn_cand.reshape(-1, 2).T
        chunks2 = lambda fn: torch.cat([fn(cand_flat[0, i:i + 20000], cand_flat[1, i:i + 20000]) for i in range(0, cand_flat.shape[1], 20000)])
        feats['qis']['cand'] = (chunks2(ps.qis) - mu) / sd if a.qis else None
        feats['ease']['cand'] = chunks2(ps.ease) if (a.ease and not a.ease_learn) else None
        feats['diff']['cand'] = chunks2(ps.diff) if a.ppr else None
        P_ = hn_cand.shape[1]
    hist = []; best = None; best_val = (-1, None); t0 = time.time(); best_scores = None

    def score(eli, part):
        o = model(full, eli, ps, g('qis', part), g('ease', part), dense, g('tpe', part), g('diff', part))
        if a.here > 0:
            o = o + here_bm[eli[0]] + here_bd[eli[1]]
        return torch.sigmoid(o).cpu().numpy()

    hn_active = (a.hardneg > 0 and (a.hardneg_pool > 0 or eli_hn is not None))
    if a.cpr > 0:
        assert eli_u is not None or hn_active, '--cpr is wired into the extended-batch branch (use with --unl or --hardneg)'
        cpr_pos = torch.as_tensor(np.where(ytr == 1)[0], device=DEV)
        cpr_gen = torch.Generator(device='cpu'); cpr_gen.manual_seed(a.seed * 31 + fold + MS)
    for ep in range(1, a.epochs + 1):
        model.train(); opt.zero_grad()
        if eli_u is not None or hn_active:
            eli_parts = [eli_tr]
            q_parts = [feats['qis']['tr']] if a.qis else None
            e_parts = [feats['ease']['tr']] if (a.ease and not a.ease_learn) else None
            d_parts = [feats['diff']['tr']] if a.ppr else None
            if eli_u is not None:
                ui = torch.randint(0, eli_u.shape[1], (n_u,), generator=u_gen).to(DEV)
                eli_parts.append(eli_u[:, ui])
                if a.qis:
                    q_parts.append(feats['qis']['u'][ui])
                if a.ease and not a.ease_learn:
                    e_parts.append(feats['ease']['u'][ui])
                if a.ppr:
                    d_parts.append(feats['diff']['u'][ui])
            anchor_ep = None; n_hn = 0
            if hn_active:
                if a.hardneg_pool > 0:
                    n_pos = hn_cand.shape[0]
                    if a.hn_dns != 0:
                        model.eval()
                        with torch.no_grad():
                            lc = model(full, cand_flat, ps, feats['qis']['cand'], feats['ease']['cand'] if 'ease' in feats else None, dense, None,
                                       feats['diff']['cand'] if 'diff' in feats else None).view(n_pos, P_)
                        model.train()
                        lc = lc.masked_fill(~hn_ok, -1e4)
                        gumb = -torch.log(-torch.log(torch.as_tensor(hn_gen.rand(n_pos, P_), device=DEV, dtype=torch.float).clamp(1e-9, 1 - 1e-9)))
                        sel = (lc / a.hn_dns + gumb).topk(a.hardneg, dim=1).indices          # Gumbel top-k = sampling w/o replacement ~ softmax(l/tau)
                    else:
                        sel = torch.as_tensor(hn_gen.randint(0, hn_cand.shape[1], size=(n_pos, a.hardneg)), device=DEV)
                    ok = hn_ok.gather(1, sel)
                    eli_hn_ep = hn_cand.gather(1, sel[..., None].expand(-1, -1, 2)).reshape(-1, 2).T
                    anchor_ep = hn_pos[:, None].expand(-1, a.hardneg)[ok]
                    eli_hn_ep = eli_hn_ep[:, ok.reshape(-1)]
                    if a.hn_dns != 0:
                        fi = (hn_pos.new_tensor(np.arange(n_pos))[:, None] * P_ + sel)[ok]
                        q_hn = feats['qis']['cand'][fi] if a.qis else None
                        e_hn = feats['ease']['cand'][fi] if (a.ease and not a.ease_learn) else None
                        d_hn = feats['diff']['cand'][fi] if a.ppr else None
                    else:
                        q_hn = (ps.qis(*eli_hn_ep) - mu) / sd if a.qis else None
                        e_hn = ps.ease(*eli_hn_ep) if (a.ease and not a.ease_learn) else None
                        d_hn = ps.diff(*eli_hn_ep) if a.ppr else None
                else:
                    eli_hn_ep, anchor_ep = eli_hn, hn_anchor
                    q_hn = feats['qis']['hn'] if a.qis else None
                    e_hn = feats['ease']['hn'] if 'ease' in feats else None
                    d_hn = feats['diff']['hn'] if 'diff' in feats else None
                n_hn = eli_hn_ep.shape[1]
                eli_parts.append(eli_hn_ep)
                if a.qis:
                    q_parts.append(q_hn)
                if a.ease and not a.ease_learn:
                    e_parts.append(e_hn)
                if a.ppr:
                    d_parts.append(d_hn)
            n_c = 0
            if a.cpr > 0:
                n_p = len(cpr_pos)
                anc = cpr_pos.repeat(a.cpr)
                par = cpr_pos[torch.randint(0, n_p, (n_p * a.cpr,), generator=cpr_gen).to(DEV)]
                m1, d1, m2, d2 = eli_tr[0, anc], eli_tr[1, anc], eli_tr[0, par], eli_tr[1, par]
                okc = (m1 != m2) & (d1 != d2) & (ps.A[m1, d2] < 0.5) & (ps.A[m2, d1] < 0.5)
                anc, par, m1, d1, m2, d2 = anc[okc], par[okc], m1[okc], d1[okc], m2[okc], d2[okc]
                n_c = len(anc)
                sw = torch.stack([torch.cat([m1, m2]), torch.cat([d2, d1])])
                eli_parts.append(sw)
                if a.qis:
                    q_parts.append((ps.qis(*sw) - mu) / sd)
                if a.ease and not a.ease_learn:
                    e_parts.append(ps.ease(*sw))
                if a.ppr:
                    d_parts.append(ps.diff(*sw))
            pred = model(full, torch.cat(eli_parts, 1), ps,
                         torch.cat(q_parts) if q_parts is not None else None,
                         torch.cat(e_parts) if e_parts is not None else None,
                         dense, None, torch.cat(d_parts) if d_parts is not None else None)
            nt_ = len(y_tr)
            if a.here > 0:
                here_L = pred[:nt_].detach()
                if not a.here_eval:
                    eall_ = torch.cat(eli_parts, 1)
                    pred = torch.cat([pred[:nt_] + here_loo, pred[nt_:] + here_bm[eall_[0, nt_:]] + here_bd[eall_[1, nt_:]]])
            if a.ebias > 0 and a.ebias_mode == 'lab':
                ebt = model.ebias_term(torch.cat(eli_parts, 1))
                pred = torch.cat([pred[:nt_], pred[nt_:] - ebt[nt_:] + ebt[nt_:].detach()])
            loss = F.binary_cross_entropy_with_logits(pred[:nt_], y_tr)
            if a.aux_head:
                pred = model._last_aux                                        # auxiliary terms read the auxiliary head
            off = nt_
            if eli_u is not None:
                loss = loss + a.unl_w * F.binary_cross_entropy_with_logits(pred[off:off + n_u], torch.zeros(n_u, device=DEV))
                off += n_u
            if hn_active and n_hn > 0:
                if a.hardneg_loss == 'bpr':
                    loss = loss + a.hardneg_w * (-F.logsigmoid(pred[:nt_][anchor_ep] - pred[off:off + n_hn])).mean()
                else:
                    loss = loss + a.hardneg_w * F.binary_cross_entropy_with_logits(pred[off:off + n_hn], torch.zeros(n_hn, device=DEV))
                off += n_hn
            if n_c > 0:
                s12, s21 = pred[off:off + n_c], pred[off + n_c:off + 2 * n_c]
                loss = loss + a.cpr_w * (-F.logsigmoid(pred[:nt_][anc] + pred[:nt_][par] - s12 - s21)).mean()
                off += 2 * n_c
        else:
            pred = model(full, eli_tr, ps, g('qis', 'tr'), g('ease', 'tr'), dense, g('tpe', 'tr'), g('diff', 'tr'))
            if a.here > 0:
                here_L = pred.detach()
                if not a.here_eval:
                    pred = pred + here_loo
            loss = F.binary_cross_entropy_with_logits(pred, y_tr)
        if a.ebias > 0:
            loss = loss + a.ebias / len(y_tr) * (model.eb_m.pow(2).sum() + model.eb_d.pow(2).sum())
        if a.lw_d or a.lw_m:
            eall = torch.cat(eli_parts, 1) if (eli_u is not None or hn_active) else eli_tr
            yall = torch.cat([y_tr, torch.zeros(len(eall[0]) - len(y_tr), device=DEV)])
            for w_, idx, n_ in ((a.lw_d, eall[1], D.n_drug), (a.lw_m, eall[0], D.n_mir)):
                if w_:
                    loss = loss + w_ * listwise(pred, yall, idx, n_)
        loss.backward(); opt.step()
        if a.here > 0:
            with torch.no_grad():
                if a.here_eval:                                               # intercepts of the current (post-step) network, eval mode
                    model.eval(); here_L = model(full, eli_tr, ps, g('qis', 'tr'), g('ease', 'tr'), dense, g('tpe', 'tr'), g('diff', 'tr')); model.train()
                here_bm, here_bd, loo_ = here_fit(here_L, y_tr, eli_tr[0], eli_tr[1], here_bm, here_bd, a.here, a.here_iters)
                here_loo = (here_bm[eli_tr[0]] + here_bd[eli_tr[1]]) if a.here_noloo else loo_
        model.eval()
        if ema_state is not None:
            with torch.no_grad():
                cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
                for k, v in cur.items():
                    if v.dtype.is_floating_point:
                        ema_state[k].mul_(a.ema).add_(v, alpha=1 - a.ema)
                    else:
                        ema_state[k].copy_(v)
                model.load_state_dict(ema_state)
        with torch.no_grad():
            s_te = score(eli_te, 'te')
            m = metrics(yte, s_te); m['epoch'] = ep; m['loss'] = float(loss)
            if eli_va is not None:
                s_va = score(eli_va, 'va')
                m['val_auc'] = roc_auc_score(val[1], s_va); m['val_aupr'] = average_precision_score(val[1], s_va)
                if m['val_auc'] > best_val[0]:
                    best_val = (m['val_auc'], dict(m)); best_scores = s_te.copy()
                    if a.ebpost > 0:
                        ebp_keep = {part: model(full, eli, ps, g('qis', part), g('ease', part), dense, g('tpe', part), g('diff', part)).cpu().numpy()
                                    for part, eli in (('tr', eli_tr), ('va', eli_va), ('te', eli_te))}
                        if a.here > 0:                                                # intercepts refitted from zero on train + val (post-hoc form)
                            ebp_keep = {k: v for k, v in ebp_keep.items()}
                    if a.dump:
                        dump = {}
                        for part, eli in (('tr', eli_tr), ('va', eli_va), ('te', eli_te)):
                            o_, h_ = model(full, eli, ps, g('qis', part), g('ease', part), dense, g('tpe', part), g('diff', part), return_h=True)
                            dump[f'logit_{part}'] = o_.cpu().numpy(); dump[f'h_{part}'] = h_.cpu().numpy()
                        if eli_u is not None:                                            # every other cell of the matrix
                            lu = []
                            for i0 in range(0, eli_u.shape[1], 20000):
                                sl = slice(i0, i0 + 20000)
                                lu.append(model(full, eli_u[:, sl], ps, feats['qis']['u'][sl] if a.qis else None,
                                                feats['ease']['u'][sl] if 'ease' in feats else None, dense, None,
                                                feats['diff']['u'][sl] if 'diff' in feats else None).cpu().numpy())
                            dump['logit_u'] = np.concatenate(lu); dump['u'] = eli_u.t().cpu().numpy()
        if ema_state is not None:
            model.load_state_dict(cur)
        hist.append({k: float(v) for k, v in m.items()})
        if best is None or m['AUC'] > best['AUC']:
            best = dict(m)
            if a.ebpost > 0 and eli_va is None:                                      # official protocol: logits at the test-best epoch
                with torch.no_grad():
                    ebp_best = {part: model(full, eli, ps, g('qis', part), g('ease', part), dense, g('tpe', part), g('diff', part)).cpu().numpy()
                                for part, eli in (('tr', eli_tr), ('te', eli_te))}
        last = dict(m)
        if ep % 50 == 0 or ep == 1:
            print(f'  ep{ep:3d} loss {m["loss"]:.4f} test AUC {m["AUC"]:.4f} AUPR {m["AUPR"]:.4f} best {best["AUC"]:.4f}'
                  + (f' val {m["val_auc"]:.4f}' if eli_va is not None else ''), flush=True)
    secs = time.time() - t0
    if a.dump:
        os.makedirs(os.path.join(HERE, 'cache'), exist_ok=True)
        extra = {f'{k}_{p}': v.cpu().numpy() for k in feats for p, v in feats[k].items() if p in ('tr', 'va', 'te')}
        np.savez_compressed(os.path.join(HERE, 'cache', f'dump_{a.tag}_{a.dataset}_{a.split}_s{a.seed}_f{fold}.npz'), **dump, **extra,
                            tr=tr, ytr=ytr, te=te, yte=yte, va=val[0], yva=val[1], train_pos=train_pos)
    if a.ease_learn:
        print('   ease params', json.dumps(model.ease_layer.describe())[:300], flush=True)
    if a.check_inv:
        # invariance test: output for a training positive on the full graph (masked computation) vs a plain
        # recomputation on the graph from which that single edge has been deleted
        model.eval(); pos_idx = np.where(ytr == 1)[0][:a.check_inv]; diffs = []
        with torch.no_grad():
            e1 = eli_tr[:, pos_idx]
            p1 = build_paths(D, ps, np.stack([e1[0].cpu().numpy(), e1[1].cpu().numpy()], 1), K0=a.tpe, Kb=a.tpe,
                             rng=np.random.RandomState(0)) if a.tpe else None
            o1 = model(full, e1, ps, feats['qis']['tr'][pos_idx] if a.qis else None, feats['ease']['tr'][pos_idx] if 'ease' in feats else None,
                       dense, p1, ps.diff(*e1) if a.ppr else None)
            for j, k in enumerate(pos_idx):
                keep = ~((train_pos[:, 0] == tr[k, 0]) & (train_pos[:, 1] == tr[k, 1]))
                msg2 = train_pos[keep]
                full2 = build_graph(D, msg2, with_assoc=(a.enc != 'content')).to(DEV)
                ps2 = PairStruct(D, torch.as_tensor(msg2, device=DEV), rel_perm)
                e = eli_tr[:, [k]]
                q2 = (ps2.qis(*e) - mu) / sd if a.qis else None
                es2 = ps2.ease(*e) if 'ease' in feats else None
                dense2 = (dense[0], dense[1], dense[2], dense[3], ps2.A) if dense is not None else None
                p2 = build_paths(D, ps2, np.array([[tr[k, 0], tr[k, 1]]]), K0=a.tpe, Kb=a.tpe,
                                 rng=np.random.RandomState(0)) if a.tpe else None
                if a.ppr:
                    ps2._diff_setup()
                o2 = model(full2, e, ps2, q2, es2, dense2, p2, ps2.diff(*e) if a.ppr else None)
                diffs.append(float((o1[j] - o2[0]).abs()))
        print(f'[check_inv] |logit(full graph, masked) - logit(edge deleted)| over {len(diffs)} training positives: '
              f'max {max(diffs):.2e} mean {np.mean(diffs):.2e}', flush=True)
    ebp = None; ebp_s = None; ebp_off = {}
    if a.ebpost > 0 and best_val[1] is not None:
        ebp_s = ebias_post(ebp_keep, tr, ytr, val, te, D.n_mir, D.n_drug, a.ebpost)
        ebp = (roc_auc_score(yte, ebp_s), average_precision_score(yte, ebp_s))
        print(f'[ebpost] valsel {best_val[1]["AUC"]:.4f}/{best_val[1]["AUPR"]:.4f} -> {ebp[0]:.4f}/{ebp[1]:.4f}', flush=True)
    if a.ebpost > 0 and eli_va is None:
        with torch.no_grad():
            ebp_last = {part: model(full, eli, ps, g('qis', part), g('ease', part), dense, g('tpe', part), g('diff', part)).cpu().numpy()
                        for part, eli in (('tr', eli_tr), ('te', eli_te))}
        for nm_, kp in (('best', ebp_best), ('last', ebp_last)):
            s_ = ebias_post(kp, tr, ytr, None, te, D.n_mir, D.n_drug, a.ebpost)
            ebp_off[f'ebp_{nm_}_auc'] = roc_auc_score(yte, s_); ebp_off[f'ebp_{nm_}_aupr'] = average_precision_score(yte, s_)
        print(f'[ebpost] official best {best["AUC"]:.4f}/{best["AUPR"]:.4f} -> {ebp_off["ebp_best_auc"]:.4f}/{ebp_off["ebp_best_aupr"]:.4f} | '
              f'last {last["AUC"]:.4f}/{last["AUPR"]:.4f} -> {ebp_off["ebp_last_auc"]:.4f}/{ebp_off["ebp_last_aupr"]:.4f}', flush=True)
    row = dict(dataset=a.dataset, split=a.split, tag=a.tag, fold=fold, seed=a.seed,
               best_auc=best['AUC'], best_aupr=best['AUPR'], best_epoch=best['epoch'],
               last_auc=last['AUC'], last_aupr=last['AUPR'], secs=secs)
    if best_val[1] is not None:
        b = best_val[1]
        row.update(valsel_epoch=b['epoch'], valsel_auc=b['AUC'], valsel_aupr=b['AUPR'], valsel_acc=b['ACC'], valsel_f1=b['F1'], valsel_mcc=b['MCC'], val_auc=best_val[0])
    if ebp is not None:
        row.update(ebp_auc=ebp[0], ebp_aupr=ebp[1])
    row.update(ebp_off)
    print(f'== {a.tag} {a.dataset} {a.split} fold={fold} seed={a.seed} best_ep={best["epoch"]} AUC={best["AUC"]:.4f} AUPR={best["AUPR"]:.4f}'
          f' | last {last["AUC"]:.4f}/{last["AUPR"]:.4f}'
          + (f' | valsel ep{best_val[1]["epoch"]} {best_val[1]["AUC"]:.4f}/{best_val[1]["AUPR"]:.4f}' if best_val[1] else '')
          + f' ({secs:.0f}s)', flush=True)
    sc = pd.DataFrame(dict(fold=fold, m=te[:, 0], d=te[:, 1], y=yte, score=best_scores if best_scores is not None else s_te))
    if ebp_s is not None:
        sc['score_eb'] = 1 / (1 + np.exp(-ebp_s))
    return row, hist, sc


def main():
    assert not (a.tpe and (a.unl > 0 or a.hardneg > 0)), 'TPE paths are precomputed per part; not supported with resampled extra pairs'
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
