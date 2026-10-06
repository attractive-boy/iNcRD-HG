# -*- coding: utf-8 -*-
"""iNcRD-HG (GeneralConv x2 via to_hetero + MLP edge decoder, full-batch RMSprop) on
PLMF-MDA with four *architectural* modules that can be switched on independently.
Splits / features / plain recipe are bit-identical to
../incrd_baseline_plmf_20260921/run_v4.py (5cv = paper 5-fold seed 42; single = official 80/20).

Modules
  --lqo K      Leave-query-out message passing.  Training positives are split into K
               groups; in pass j the message graph contains the training positives of the
               other groups only and the loss is taken on group-j pairs (positives + a
               1/K share of the negatives).  Test pairs use the full training graph.
               Removes the "target edge is in the message graph" train/test mismatch.
  --ncn        Bipartite neural-common-neighbour pair decoder: for query (m,d) aggregate the
               hidden states of the drugs d' in N(m) weighted by |N(d') ∩ N(d)| (and the
               symmetric miRNA side), i.e. the embeddings of the nodes on m-d'-m'-d paths,
               and feed them to the edge decoder next to z_m, z_d.
  --cnt        Scalar structural pair counts from the *same* message graph (deg, path3,
               normalised path3, similarity-bridged counts), log1p + z-scored, projected
               and concatenated to the decoder input.
  --prov       Provenance hypergraph node types: 'study' (PMID) and 'ctxtok' (cell line /
               phenotype / disease token) nodes linked to the miRNA and the drug of every
               *message-graph* positive that has a curation record (star expansion of the
               study hyperedge).  Records of query / test pairs are never in the graph.
  --side       Label-free miRNA side-information node types: 'disease' (HMDD v4 profile),
               'family' (name family), 'cluster' (miRBase genomic cluster).
Controls
  --prov_shuffle   permute the curation records across positive pairs
  --side_shuffle   permute the side-information rows across miRNAs
Encoder modules from round 9 (../plmf_enc_20260924/REPORT_ENC.md), added on top of the LQO stack
(copy of ../plmf_gnnarch_20260923/run_gnn.py; with E1-E4 off the code path is unchanged):
  E1 --cka          content-keyed message weights 2 sigmoid(<Q x_i, K x_j>) on MiMi / DrugDrug / MiDrug / rev_MiDrug
  E2 --nbf          drug-source conditional message passing (NBFNet-style, 3 layers), own lr group x --nbf_lr
  E3 --nbf_src both miRNA-source counterpart
  E4 --nbf_jk       conditional readout of the layer-2 and layer-3 states
  Under LQO the query edge is never in the message graph, so E2-E4 need no hollow correction.
Epoch selection: --val holds out 10% of the training pairs (stratified); 'valsel' = test
metrics at the epoch with the best hold-out AUC; 'best' = test-best epoch (board convention);
'last' = final epoch.  Output results/{tag}_{DS}_{split}_seed{seed}.csv"""
import argparse, json, math, os, pickle, random, re, sys, time
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
from plmf_splits import get_split, load_dataset, _balanced_pairs  # noqa: E402

GRAPH_CACHE = os.path.join(BASE, 'mphgnn_plmf_20260921', 'cache', 'graph_{}.npz')
RECORDS = os.path.join(BASE, 'plmf_xrel_20260922', 'data', 'full_{}', 'data', 'pair_records.pkl')
HMDD = '/data1/licheng/incrd-hg/data_external/ncrna_disease_20260917/hmdd_v4_alldata.txt'
MIRBASE_GFF = '/data1/licheng/incrd-hg/data_external/mirbase_20260922/hsa.gff3'

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['MDR', 'MDS'])
ap.add_argument('--split', required=True, choices=['5cv', 'single', 'study'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--knn', type=int, default=10)
ap.add_argument('--hidden', type=int, default=256)
ap.add_argument('--aggr', default='mean')
ap.add_argument('--layers', type=int, default=2)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--dropout', type=float, default=0.0)
ap.add_argument('--val', action='store_true')
ap.add_argument('--val_frac', type=float, default=0.1)
ap.add_argument('--lqo', type=int, default=0)
ap.add_argument('--ncn', action='store_true')
ap.add_argument('--cnt', action='store_true')
ap.add_argument('--prov', action='store_true')
ap.add_argument('--prov_groups', default='study,ctxtok')
ap.add_argument('--prov_shuffle', action='store_true')
ap.add_argument('--side', action='store_true')
ap.add_argument('--side_groups', default='disease,family,cluster')
ap.add_argument('--side_shuffle', action='store_true')
ap.add_argument('--prov_pair', type=int, default=1, help='1: provenance hyperedges also enter the pair decoder (shared-study / shared-token NCN + counts)')
ap.add_argument('--side_pair', type=int, default=1, help='1: side relations also enter the pair decoder (relation-bridged NCN + counts)')
ap.add_argument('--ncn_dim', type=int, default=64, help='width of each auxiliary NCN channel')
ap.add_argument('--neg_mult', type=float, default=0.0, help='extra negatives per positive resampled each epoch from the unlabeled pool')
ap.add_argument('--cka', action='store_true', help='E1 content-keyed message weights on the MiDrug / MiMi / DrugDrug relations')
ap.add_argument('--cka_dim', type=int, default=32)
ap.add_argument('--nbf', action='store_true', help='E2 drug-source conditional message passing (NBFNet-style, 3 layers)')
ap.add_argument('--nbf_dim', type=int, default=128)
ap.add_argument('--nbf_src', default='drug', choices=['drug', 'both'], help='E3 both: add the miRNA-source counterpart')
ap.add_argument('--nbf_mdim', type=int, default=32)
ap.add_argument('--nbf_jk', action='store_true', help='E4 readout = [layer-2, layer-3] conditional states')
ap.add_argument('--nbf_lr', type=float, default=30.0, help='learning-rate multiplier of the NBF parameters')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--tag', default='plain')
ap.add_argument('--folds', default='')
a = ap.parse_args()
DEV = torch.device(a.device)
NEED_PS = a.ncn or a.cnt or a.nbf or a.nbf_src == 'both'
SR = SD = None


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


def study_split(ds, fold, seed, n_folds=5):
    """Primary-study grouped 5-fold split (== ../plmf_xrel_20260922/study_split2.py --records full --group primary):
    each positive goes to the fold of its most specific PMID (fewest reported pairs); negatives as the paper protocol."""
    mir, dru, pos, neg = load_dataset(ds)
    names = mir['miRNA_Name'].astype(str).str.upper().values; dbs = dru['DrugBank_ID'].astype(str).values
    recs = pickle.load(open(RECORDS.format(ds), 'rb'))
    pm = {k: recs[(names[i], dbs[j])]['pmids'] for k, (i, j) in enumerate(map(tuple, pos)) if (names[i], dbs[j]) in recs}
    size = {}
    for v in pm.values():
        for p in v:
            size[p] = size.get(p, 0) + 1
    grp = np.array([min(pm[k], key=lambda p: (size[p], p)) if k in pm else -(k + 1) for k in range(len(pos))])
    rng = np.random.RandomState(seed); sizes = pd.Series(grp).value_counts()
    order = sizes.index.values.copy(); rng.shuffle(order); order = sorted(order, key=lambda c: -sizes[c])
    fold_of, load = {}, np.zeros(n_folds)
    for c in order:
        f = int(np.argmin(load + rng.rand(n_folds) * 0.5)); fold_of[c] = f; load[f] += sizes[c]
    pos_fold = np.array([fold_of[c] for c in grp])
    idx, y = _balanced_pairs(pos, neg, seed); neg_sel = idx[y == 0]; neg_fold = rng.randint(0, n_folds, len(neg_sel))
    tr = np.concatenate([pos[pos_fold != fold], neg_sel[neg_fold != fold]]); ytr = np.concatenate([np.ones((pos_fold != fold).sum()), np.zeros((neg_fold != fold).sum())])
    te = np.concatenate([pos[pos_fold == fold], neg_sel[neg_fold == fold]]); yte = np.concatenate([np.ones((pos_fold == fold).sum()), np.zeros((neg_fold == fold).sum())])
    return tr, ytr, te, yte


def split_of(ds, split, fold, seed):
    return study_split(ds, fold, seed) if split == 'study' else get_split(ds, split, fold, seed)


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, ds):
        g = np.load(GRAPH_CACHE.format(ds), allow_pickle=True)
        mir, dru, pos, neg = load_dataset(ds)
        self.names = mir['miRNA_Name'].astype(str).values; self.dbs = dru['DrugBank_ID'].astype(str).values
        assert list(self.names) == list(g['mirna_names']), 'miRNA order mismatch'
        self.n_mir, self.n_drug = len(self.names), len(self.dbs)
        self.xr = rowminmax(g['mirna_kmer']); self.xd = rowminmax(g['drug_x'])
        self.sim_r = knn_edges(self.xr, a.knn); self.sim_d = knn_edges(self.xd, a.knn)
        self.pos_all = pos
        self.Sr = torch.zeros(self.n_mir, self.n_mir); self.Sr[self.sim_r[0], self.sim_r[1]] = 1
        self.Sd = torch.zeros(self.n_drug, self.n_drug); self.Sd[self.sim_d[0], self.sim_d[1]] = 1
        self.Sr, self.Sd = self.Sr.to(DEV), self.Sd.to(DEV)
        # provenance records keyed by (miRNA idx, drug idx)
        self.rec = {}
        if a.prov:
            recs = pickle.load(open(RECORDS.format(ds), 'rb'))
            up = [n.upper() for n in self.names]
            pm_id, tk_id = {}, {}
            for i, j in pos:
                v = recs.get((up[i], self.dbs[j]))
                if v is None:
                    continue
                self.rec[(i, j)] = dict(pm=[pm_id.setdefault(p, len(pm_id)) for p in v['pmids']],
                                        tk=[tk_id.setdefault(t, len(tk_id)) for t in v['toks']])
            self.n_study, self.n_tok = len(pm_id), len(tk_id)
            print(f'[prov] records for {len(self.rec)}/{len(pos)} positives, {self.n_study} studies, {self.n_tok} context tokens', flush=True)
        # side information (miRNA -> disease / family / cluster)
        self.side = {}
        if a.side:
            groups = a.side_groups.split(',')
            if 'disease' in groups:
                h = pd.read_csv(HMDD, sep='\t', encoding='latin1', on_bad_lines='skip')
                h['m'] = h['miRNA'].astype(str).str.lower().str.strip(); h['dz'] = h['disease'].astype(str).str.strip()
                dz_idx = {d: i for i, d in enumerate(sorted(h['dz'].unique()))}
                st_idx = {}
                for i, n in enumerate(self.names):
                    st_idx.setdefault(stem(n), []).append(i)
                rows, cols = [], []
                for m, dz in zip(h['m'], h['dz']):
                    for i in st_idx.get(m, []):
                        rows.append(i); cols.append(dz_idx[dz])
                self.side['disease'] = (np.array(rows), np.array(cols), len(dz_idx))
                H = torch.zeros(self.n_mir, len(dz_idx)); H[rows, cols] = 1.0
                Hn = F.normalize(H, dim=1); S = Hn @ Hn.t(); S.fill_diagonal_(0)
                thr = S.topk(20, dim=1).values[:, -1:]
                self.H = H.to(DEV); self.S_h = torch.where(S >= thr, S, torch.zeros_like(S)).to(DEV)
            if 'family' in groups:
                fam = pd.factorize(pd.Series([family(n) for n in self.names]))[0]
                keep = np.bincount(fam) > 1
                idx = np.where(keep[fam])[0]
                self.side['family'] = (idx, fam[idx], int(fam.max()) + 1)
            if 'cluster' in groups:
                self.side['cluster'] = self._cluster()
            for k, (r, c, n) in self.side.items():
                print(f'[side] {k}: {n} nodes, {len(r)} edges, miRNA coverage {len(set(r.tolist())) / self.n_mir:.2f}', flush=True)

    def _cluster(self, window=10000):
        pre, mat2pre = [], {}
        for l in open(MIRBASE_GFF):
            if l.startswith('#'):
                continue
            r = l.rstrip('\n').split('\t'); at = dict(kv.split('=') for kv in r[8].split(';'))
            if r[2] == 'miRNA_primary_transcript':
                pre.append((at['ID'], r[0], r[6], int(r[3]), int(r[4]), at['Name'].lower()))
            else:
                mat2pre.setdefault(at['Name'].lower(), []).append(at['Derives_from'])
        pre_df = pd.DataFrame(pre, columns=['id', 'chr', 'strand', 'start', 'end', 'name']).sort_values(['chr', 'strand', 'start'])
        cl, cur, last = {}, -1, None
        for row in pre_df.itertuples(index=False):
            if last is None or row.chr != last.chr or row.strand != last.strand or row.start - last.end > window:
                cur += 1
            cl[row.id] = cur; last = row
        by_name = {}
        for row in pre_df.itertuples(index=False):
            by_name.setdefault(row.name, []).append(row.id); by_name.setdefault(re.sub(r'-\d+$', '', row.name), []).append(row.id)
        clu = np.full(self.n_mir, -1)
        for i, n in enumerate(self.names):
            n = n.lower(); s = stem(n)
            pids = mat2pre.get(n) or mat2pre.get(s + '-3p') or mat2pre.get(s + '-5p') or by_name.get(s.replace('mir', 'mir')) or by_name.get(s.replace('-mir-', '-mir-'))
            if pids:
                clu[i] = cl[pids[0]]
        ok = clu >= 0
        sizes = np.bincount(clu[ok], minlength=cur + 1)
        idx = np.where(ok & (sizes[np.maximum(clu, 0)] > 1))[0]
        return idx, clu[idx], cur + 1


def build_graph(D, msg_pos, rec_perm=None, side_perm=None):
    """HeteroData with the message-graph positives (+ optional provenance / side node types)."""
    data = HeteroData()
    data['miRNA'].x = D.xr; data['drug'].x = D.xd
    data['miRNA', 'MiMi', 'miRNA'].edge_index = D.sim_r
    data['drug', 'DrugDrug', 'drug'].edge_index = D.sim_d
    data['miRNA', 'MiDrug', 'drug'].edge_index = torch.as_tensor(msg_pos.T, dtype=torch.long)
    if a.prov:
        groups = a.prov_groups.split(',')
        sm, ss, dm, dsx, tm, tt, dt = [], [], [], [], [], [], []
        for i, j in msg_pos:
            key = (i, j) if rec_perm is None else rec_perm.get((i, j))
            v = D.rec.get(key) if key is not None else None
            if v is None:
                continue
            for p in v['pm']:
                sm.append(i); ss.append(p); dm.append(j); dsx.append(p)
            for t in v['tk']:
                tm.append(i); tt.append(t); dt.append(j)
        if 'study' in groups:
            data['study'].x = torch.ones(D.n_study, 1)
            data['miRNA', 'MiStudy', 'study'].edge_index = torch.as_tensor(np.array([sm, ss]), dtype=torch.long).reshape(2, -1)
            data['drug', 'DrugStudy', 'study'].edge_index = torch.as_tensor(np.array([dm, dsx]), dtype=torch.long).reshape(2, -1)
        if 'ctxtok' in groups:
            data['ctxtok'].x = torch.ones(D.n_tok, 1)
            data['miRNA', 'MiTok', 'ctxtok'].edge_index = torch.as_tensor(np.array([tm, tt]), dtype=torch.long).reshape(2, -1)
            data['drug', 'DrugTok', 'ctxtok'].edge_index = torch.as_tensor(np.array([dt, tt]), dtype=torch.long).reshape(2, -1)
    if a.side:
        for k, (r, c, n) in D.side.items():
            rr = r if side_perm is None else side_perm[r]
            data[k].x = torch.ones(n, 1)
            data['miRNA', f'Mi{k}', k].edge_index = torch.as_tensor(np.array([rr, c]), dtype=torch.long).reshape(2, -1)
    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)
    return data


# ----------------------------------------------------------------------------- model
AUX_TYPES = {'study', 'ctxtok', 'disease', 'family', 'cluster'}


CKA_REL = ('MiMi', 'DrugDrug', 'MiDrug', 'rev_MiDrug')


class CKAConv(GeneralConv):
    """GeneralConv whose messages are scaled by a per-edge weight self.ew (None = plain GeneralConv)."""
    ew = None

    def message(self, x_i, x_j, edge_index_i, size_i, edge_attr):
        out = super().message(x_i, x_j, edge_index_i, size_i, edge_attr)
        return out if self.ew is None else out * self.ew.view(-1, 1, 1)


class GNNEncoder(torch.nn.Module):
    """Relation-typed GeneralConv stack (== to_hetero(GeneralConv) of the original recipe:
    one GeneralConv per edge type, sum over neighbours, mean over relation types).
    Relations touching an auxiliary node type (study / token / disease / family / cluster)
    average over neighbours instead of summing, i.e. hyperedge nodes hold the mean state of
    their members and a miRNA/drug receives the mean over its hyperedges.
    E1 --cka: on the MiMi / DrugDrug / MiDrug / rev_MiDrug relations every message j -> i is scaled by
    w_ij = 2 sigmoid(<Q x_i, K x_j> / sqrt(k) + b), keyed on the input content x (k-mer / ChemBERTa) only,
    so the weights never depend on the message graph; Q, K ~ N(0, 0.02), b = 0 -> w ~ 1 (= plain) at start."""

    def __init__(self, data, hidden, layers, dropout):
        super().__init__()
        Conv = CKAConv if a.cka else GeneralConv
        def convs():
            return HeteroConv({et: Conv((-1, -1), hidden, aggr='mean' if (et[0] in AUX_TYPES or et[2] in AUX_TYPES) else 'add')
                               for et in data.edge_types}, aggr=a.aggr)
        self.convs = torch.nn.ModuleList([convs() for _ in range(layers)])
        self.dropout = dropout
        if a.cka:
            k = a.cka_dim; nx = {t: data[t].x.shape[1] for t in ('miRNA', 'drug')}
            self.cka_et = [et for et in data.edge_types if et[1] in CKA_REL]
            self.cka = torch.nn.ModuleDict(); self.cka_b = torch.nn.ParameterDict()
            for l in range(layers):
                for src, rel, dst in self.cka_et:
                    Q = Linear(nx[dst], k, bias=False); K = Linear(nx[src], k, bias=False)
                    torch.nn.init.normal_(Q.weight, std=0.02); torch.nn.init.normal_(K.weight, std=0.02)
                    self.cka[f'{rel}_{l}'] = torch.nn.ModuleList([Q, K]); self.cka_b[f'{rel}_{l}'] = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x_dict, ei_dict):
        x0 = x_dict
        for i, conv in enumerate(self.convs):
            if a.cka:
                for et in self.cka_et:
                    src, rel, dst = et; ei = ei_dict[et]; Q, K = self.cka[f'{rel}_{i}']
                    logit = (Q(x0[dst])[ei[1]] * K(x0[src])[ei[0]]).sum(-1) / math.sqrt(a.cka_dim) + self.cka_b[f'{rel}_{i}']
                    conv.convs[et].ew = 2 * torch.sigmoid(logit)
            x_dict = conv(x_dict, ei_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.leaky_relu(v, negative_slope=0.1) for k, v in x_dict.items()}
                if self.dropout > 0:
                    x_dict = {k: F.dropout(v, self.dropout, self.training) for k, v in x_dict.items()}
        return x_dict


class NBF(torch.nn.Module):
    """E2 source-conditioned message passing (labeling trick / NBFNet, Zhu et al. 2021), same layer as
    ../plmf_enc_20260924/run_enc.py.  For every drug s the states H_s over all nodes start from a learned vector
    at s and are propagated 3 times: P = A-messages + kNN-messages + self + boundary(s), H = leaky_relu(LN(P)).
    The pair (r, d) is read as H_d[r] (E4 --nbf_jk: [layer-2, layer-3] state).  Under LQO the query edge is
    never in the message graph (training pairs: other groups only; val / test: not training positives), so no
    hollow correction is needed.  E3 (--nbf_src both, NBFM): every miRNA a source, read at d."""

    def __init__(self, h, drug_src=True):
        super().__init__()
        L = 3
        mk = lambda: torch.nn.ModuleList([Linear(h, h, bias=False) for _ in range(L)])
        self.Wdm, self.Wmd, self.Wmm, self.Wdd = mk(), mk(), mk(), mk()
        self.Wsm = torch.nn.ModuleList([Linear(h, h) for _ in range(L)]); self.Wsd = torch.nn.ModuleList([Linear(h, h) for _ in range(L)])
        self.q = torch.nn.Parameter(torch.randn(h) * 0.1)
        self.bnd = torch.nn.Parameter(torch.randn(L, h) * 0.1)
        self.lnm = torch.nn.ModuleList([torch.nn.LayerNorm(h) for _ in range(L)]); self.lnd = torch.nn.ModuleList([torch.nn.LayerNorm(h) for _ in range(L)])
        self.h = h; self.drug_src = drug_src

    def forward(self, A, Sr, Sd, r, d, jk=False):
        nm, nd = A.shape; h = self.h; dev = A.device
        S = nd if self.drug_src else nm; ids = torch.arange(S, device=dev)
        Hm = torch.zeros(S, nm, h, device=dev); Hd = torch.zeros(S, nd, h, device=dev)
        if self.drug_src:
            Hd[ids, ids] = self.q
        else:
            Hm[ids, ids] = self.q
        hist = []
        for l in range(3):
            Pm = A @ self.Wdm[l](Hd) + Sr @ self.Wmm[l](Hm) + self.Wsm[l](Hm)
            Pd = A.t() @ self.Wmd[l](Hm) + Sd @ self.Wdd[l](Hd) + self.Wsd[l](Hd)
            if self.drug_src:
                Pd = Pd.clone(); Pd[ids, ids] = Pd[ids, ids] + self.bnd[l]
            else:
                Pm = Pm.clone(); Pm[ids, ids] = Pm[ids, ids] + self.bnd[l]
            Hm, Hd = F.leaky_relu(self.lnm[l](Pm), 0.1), F.leaky_relu(self.lnd[l](Pd), 0.1)
            hist.append((Hm, Hd))
        if self.drug_src:
            out = hist[2][0][d, r]
            return torch.cat([hist[1][0][d, r], out], 1) if jk else out
        return hist[2][1][r, d]


CNT_NAMES = ['deg_r', 'deg_d', 'p3', 'p3_n', 'p3_r', 'p3_d', 'ASd', 'SrA', 'SrASd', 'ASd_n', 'SrA_n', 'p5', 'p5_n']


class PairStruct:
    """Dense structural quantities of one message graph (n_mir x n_drug is tiny) and the
    relation-typed common-neighbour channels of the pair decoder:
      miRNA-side relations R (n_mir x n_mir): paths m -R- m' -A- d        (co-drug, kNN sim, family, cluster, HMDD-sim)
      drug-side relations  R (n_drug x n_drug): paths m -A- d' -R- d      (co-miRNA, kNN sim)
      hyperedges (Mr, Md): m -Mr- h -Md- d                                 (study, context token, HMDD disease)
    Everything is computed from the message-graph positives only."""

    def __init__(self, D, msg_pos, rec_perm=None, side_perm=None):
        A = torch.zeros(D.n_mir, D.n_drug, device=DEV); A[msg_pos[:, 0], msg_pos[:, 1]] = 1.0
        self.A = A; self.deg_r = A.sum(1); self.deg_d = A.sum(0)
        self.Cm = A @ A.t(); self.Cd = A.t() @ A
        self.P3 = self.Cm @ A; self.P5 = self.Cm @ self.P3
        self.ASd = A @ D.Sd; self.SrA = D.Sr @ A; self.SrASd = D.Sr @ A @ D.Sd
        self.mir_rel, self.drug_rel, self.hyper = {}, {}, {}
        if a.ncn:
            self.mir_rel['co'] = self.Cm; self.drug_rel['co'] = self.Cd
        if a.prov and a.prov_pair:
            for name, key, n in (('study', 'pm', D.n_study), ('ctxtok', 'tk', D.n_tok)):
                if name not in a.prov_groups.split(','):
                    continue
                Mr = torch.zeros(D.n_mir, n, device=DEV); Md = torch.zeros(D.n_drug, n, device=DEV)
                for i, j in msg_pos.tolist():
                    k = (i, j) if rec_perm is None else rec_perm.get((i, j))
                    v = D.rec.get(k) if k is not None else None
                    if v is None:
                        continue
                    for h in v[key]:
                        Mr[i, h] = 1.0; Md[j, h] = 1.0
                self.hyper[name] = (Mr, Md)
        if a.side and a.side_pair:
            perm = torch.arange(D.n_mir, device=DEV) if side_perm is None else torch.as_tensor(side_perm, device=DEV)
            for k, (r, c, n) in D.side.items():
                if k == 'disease':
                    Hm = D.H[perm]                       # miRNA x disease (rows permuted under the control)
                    self.hyper['disease'] = (Hm, A.t() @ Hm)   # drug side: diseases of the drug's message-graph miRNAs
                    self.mir_rel['hsim'] = D.S_h[perm][:, perm]
                else:
                    G = torch.zeros(n, D.n_mir, device=DEV); G[torch.as_tensor(c, device=DEV), perm[torch.as_tensor(r, device=DEV)]] = 1.0
                    R = G.t() @ G; R.fill_diagonal_(0); self.mir_rel[k] = R
        for name, (Mr, Md) in self.hyper.items():
            self.hyper[name] = (Mr, Md, Mr.sum(1), Md.sum(1))
        self.mir_rel = {k: (R, R.sum(1)) for k, R in self.mir_rel.items()}
        self.drug_rel = {k: (R, R.sum(1)) for k, R in self.drug_rel.items()}
        self.RA = {k: R @ A for k, (R, _) in self.mir_rel.items()}
        self.AR = {k: A @ R for k, (R, _) in self.drug_rel.items()}
        self.names = CNT_NAMES + [f'{k}_{t}' for k in list(self.mir_rel) + list(self.drug_rel) for t in ('c', 'cr', 'cd')] + \
            [f'{k}_{t}' for k in self.hyper for t in ('c', 'cr', 'cd', 'nr', 'nd')]

    def counts(self, r, d):
        dr, dd = self.deg_r[r], self.deg_d[d]; p3 = self.P3[r, d]; p5 = self.P5[r, d]
        f = [dr, dd, p3, p3 / (dr * dd + 1.0), p3 / (dr + 1.0), p3 / (dd + 1.0), self.ASd[r, d], self.SrA[r, d], self.SrASd[r, d],
             self.ASd[r, d] / (dr + 1.0), self.SrA[r, d] / (dd + 1.0), p5, p5 / (dr * dd + 1.0)]
        for k, (R, rs) in self.mir_rel.items():
            c = self.RA[k][r, d]; f += [c, c / (rs[r] + 1.0), c / (dd + 1.0)]
        for k, (R, rs) in self.drug_rel.items():
            c = self.AR[k][r, d]; f += [c, c / (rs[d] + 1.0), c / (dr + 1.0)]
        for k, (Mr, Md, nr, nd) in self.hyper.items():
            c = (Mr[r] * Md[d]).sum(1); f += [c, c / (nr[r] + 1.0), c / (nd[d] + 1.0), nr[r], nd[d]]
        return torch.log1p(torch.stack(f, 1))

    def ncn(self, r, d, z):
        """weighted mean hidden state of the intermediate nodes on each relation-typed 2-hop path."""
        out = []
        for k, (R, _) in self.mir_rel.items():
            w = self.A[:, d].t() * R[r]; out.append(w @ z['miRNA'] / (w.sum(1, keepdim=True) + 1.0))
        for k, (R, _) in self.drug_rel.items():
            w = self.A[r] * R[:, d].t(); out.append(w @ z['drug'] / (w.sum(1, keepdim=True) + 1.0))
        for k, (Mr, Md, _, _) in self.hyper.items():
            w = Mr[r] * Md[d]; out.append(w @ z[k] / (w.sum(1, keepdim=True) + 1.0))
        return out

    def channels(self):
        return list(self.mir_rel) + list(self.drug_rel) + list(self.hyper)


class Model(torch.nn.Module):
    def __init__(self, data, hidden, ps):
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
            self.nbf = NBF(a.nbf_dim, drug_src=True); d += a.nbf_dim * (2 if a.nbf_jk else 1)
        if a.nbf_src == 'both':
            self.nbfm = NBF(a.nbf_mdim, drug_src=False); d += a.nbf_mdim
        self.lin1 = Linear(d, 2 * hidden); self.lin2 = Linear(2 * hidden, hidden); self.lin3 = Linear(hidden, 1)

    def forward(self, data, eli, ps=None, cnt=None):
        z = self.encoder(data.x_dict, data.edge_index_dict)
        r, d = eli
        parts = [z['miRNA'][r], z['drug'][d]]
        if self.chan:
            parts += [self.ncn_lin[k](e) for k, e in zip(self.chan, ps.ncn(r, d, z))]
        if a.cnt:
            parts.append(self.pc(cnt))
        if a.nbf:
            parts.append(self.nbf(ps.A, SR, SD, r, d, a.nbf_jk))
        if a.nbf_src == 'both':
            parts.append(self.nbfm(ps.A, SR, SD, r, d))
        h = torch.cat(parts, -1)
        h = F.leaky_relu(self.lin1(h), 0.1)
        if a.dropout > 0:
            h = F.dropout(h, a.dropout, self.training)
        h = F.leaky_relu(self.lin2(h), 0.1)
        return self.lin3(h).view(-1)


def metrics(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s); p = (s >= 0.5).astype(int)
    return dict(AUC=roc_auc_score(y, s), AUPR=average_precision_score(y, s), ACC=accuracy_score(y, p),
                F1=f1_score(y, p, zero_division=0), MCC=matthews_corrcoef(y, p) if len(set(y)) > 1 else 0.0)


# ----------------------------------------------------------------------------- one fold
def run_fold(D, fold):
    global SR, SD
    SR, SD = D.Sr, D.Sd
    tr, ytr, te, yte = split_of(a.dataset, a.split, fold, a.seed)
    set_seed(a.seed * 100 + fold)
    rng = np.random.RandomState(a.seed * 100 + fold)
    val = None
    if a.val:
        vidx = []
        for c in (0, 1):
            ids = np.where(ytr == c)[0]; rng.shuffle(ids); vidx.append(ids[:int(round(a.val_frac * len(ids)))])
        vidx = np.concatenate(vidx); mask = np.ones(len(ytr), bool); mask[vidx] = False
        val = (tr[vidx], ytr[vidx]); tr, ytr = tr[mask], ytr[mask]
    train_pos = tr[ytr == 1]
    # controls
    rec_perm = side_perm = None
    if a.prov and a.prov_shuffle:
        keys = [tuple(p) for p in train_pos.tolist() if tuple(p) in D.rec]
        perm = rng.permutation(len(keys)); rec_perm = {keys[i]: keys[perm[i]] for i in range(len(keys))}
    if a.side and a.side_shuffle:
        side_perm = rng.permutation(D.n_mir)
    # passes: (message positives, supervised pair index)
    K = max(a.lqo, 1)
    grp_pos = rng.randint(K, size=len(train_pos)) if a.lqo else np.zeros(len(train_pos), int)
    grp_all = np.zeros(len(tr), int)
    grp_all[ytr == 1] = grp_pos
    grp_all[ytr == 0] = rng.randint(K, size=int((ytr == 0).sum())) if a.lqo else 0
    passes = []
    for j in range(K):
        msg = train_pos[grp_pos != j] if a.lqo else train_pos
        sup = np.where(grp_all == j)[0]
        g = build_graph(D, msg, rec_perm, side_perm).to(DEV)
        ps = PairStruct(D, torch.as_tensor(msg, device=DEV), rec_perm, side_perm) if NEED_PS else None
        passes.append(dict(data=g, ps=ps, eli=torch.as_tensor(tr[sup].T, dtype=torch.long, device=DEV),
                           y=torch.as_tensor(ytr[sup], dtype=torch.float, device=DEV), sup=sup, msg=msg))
    full = build_graph(D, train_pos, rec_perm, side_perm).to(DEV)
    ps_full = PairStruct(D, torch.as_tensor(train_pos, device=DEV), rec_perm, side_perm) if NEED_PS else None
    eli_te = torch.as_tensor(te.T, dtype=torch.long, device=DEV)
    eli_va = torch.as_tensor(val[0].T, dtype=torch.long, device=DEV) if val is not None else None
    # structural counts (z-scored with training-pass statistics)
    mu = sd = None; cnt_te = cnt_va = None
    if a.cnt:
        ctr = torch.cat([p['ps'].counts(p['eli'][0], p['eli'][1]) for p in passes], 0)
        mu, sd = ctr.mean(0, keepdim=True), ctr.std(0, keepdim=True) + 1e-6
        for p in passes:
            p['cnt'] = (p['ps'].counts(p['eli'][0], p['eli'][1]) - mu) / sd
        cnt_te = (ps_full.counts(eli_te[0], eli_te[1]) - mu) / sd
        cnt_va = (ps_full.counts(eli_va[0], eli_va[1]) - mu) / sd if eli_va is not None else None
    else:
        for p in passes:
            p['cnt'] = None
    # unlabeled pool for --neg_mult
    if a.neg_mult > 0:
        known = np.zeros((D.n_mir, D.n_drug), bool)
        known[tr[:, 0], tr[:, 1]] = True; known[te[:, 0], te[:, 1]] = True
        if val is not None:
            known[val[0][:, 0], val[0][:, 1]] = True
        pool = np.argwhere(~known); rng_neg = np.random.RandomState(a.seed * 100 + fold + 7)
    set_seed(a.seed * 100 + fold)
    model = Model(full, a.hidden, ps_full).to(DEV)
    with torch.no_grad():
        model(full, eli_te[:, :2], ps_full, cnt_te[:2] if a.cnt else None)
    nbf_p = [p for n, p in model.named_parameters() if n.startswith(('nbf.', 'nbfm.'))]
    nid = {id(p) for p in nbf_p}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in nid])]
    if nbf_p:
        groups.append(dict(params=nbf_p, lr=a.lr * a.nbf_lr))
    opt = torch.optim.RMSprop(groups, lr=a.lr, alpha=0.99)
    hist = []; best = None; best_val = (-1, None); t0 = time.time()
    for ep in range(1, a.epochs + 1):
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
            pred = model(p['data'], eli, p['ps'], cnt)
            loss = F.binary_cross_entropy_with_logits(pred, y); loss.backward(); opt.step(); losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            s_te = torch.sigmoid(model(full, eli_te, ps_full, cnt_te)).cpu().numpy()
            m = metrics(yte, s_te); m['epoch'] = ep; m['loss'] = float(np.mean(losses))
            if eli_va is not None:
                s_va = torch.sigmoid(model(full, eli_va, ps_full, cnt_va)).cpu().numpy()
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
    row = dict(dataset=a.dataset, split=a.split, tag=a.tag, fold=fold, seed=a.seed,
               best_auc=best['AUC'], best_aupr=best['AUPR'], best_epoch=best['epoch'],
               last_auc=last['AUC'], last_aupr=last['AUPR'], secs=time.time() - t0)
    scores = last_scores = s_te
    if best_val[1] is not None:
        b = best_val[1]
        row.update(valsel_epoch=b['epoch'], valsel_auc=b['AUC'], valsel_aupr=b['AUPR'], valsel_acc=b['ACC'], valsel_f1=b['F1'], valsel_mcc=b['MCC'], val_auc=best_val[0])
        scores = best_scores
    print(f'== {a.tag} {a.dataset} {a.split} fold={fold} seed={a.seed} best_ep={best["epoch"]} AUC={best["AUC"]:.4f} AUPR={best["AUPR"]:.4f}'
          f' | last {last["AUC"]:.4f}/{last["AUPR"]:.4f}'
          + (f' | valsel ep{best_val[1]["epoch"]} {best_val[1]["AUC"]:.4f}/{best_val[1]["AUPR"]:.4f}' if best_val[1] else '')
          + f' ({time.time() - t0:.0f}s)', flush=True)
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
          + (f' | valsel {d.valsel_auc.mean():.4f}/{d.valsel_aupr.mean():.4f}' if 'valsel_auc' in d else ''), flush=True)


if __name__ == '__main__':
    main()
