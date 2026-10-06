# -*- coding: utf-8 -*-
"""The *plain* iNcRD-HG predictor (iNcRD-HG/predict_association.py: GeneralConv x2
wrapped by to_hetero + 3-layer MLP edge decoder, full-batch RMSprop) ported to
PLMF-MDA's MDRdataset / MDSdataset under the two protocols of
improve/research/plmf_baselines_20260920 (bit-identical splits via
../mphgnn_plmf_20260921/plmf_splits.py):
  5cv    : paper protocol, positives + equal negatives, 5 folds, seed 42
  single : released-code 80/20 split, seeds 0-4

What is kept from the original recipe
  * heterogeneous graph  RNA--RNA similarity, RNA--drug (TRAINING positives only),
    drug--drug similarity; T.ToUndirected + T.AddSelfLoops
  * GeneralConv(-1,-1) x2, leaky_relu(0.1), to_hetero(aggr), decoder
    Linear(2h,2h)-lrelu-Linear(2h,h)-lrelu-Linear(h,1), BCE-with-logits
  * miRNA setting: hidden 256, aggr 'mean', RMSprop(lr 1e-4, alpha .99), 300 epochs
  * row min-max node features, full-batch training, test metrics every epoch
What has to change (PLMF-MDA ships only miRNA sequence + drug SMILES + labels)
  * miRNA feature: CCLE expression -> 2+3-mer frequency (80-d), optionally the
    CCLE expression rows where available (42-45% of miRNAs)
  * drug feature: ChemBERTa-zinc-base-v1 (same as the original) or Morgan r2/2048
  * similarity edges: |Kendall| > 0.3 on expression -> top-k cosine kNN on the node
    features (k-mer cosine / ChemBERTa cosine or Morgan Tanimoto), symmetrised
  * single task (no lncRNA nodes, no LncMi bridge)
  * negatives: the PLMF-MDA balanced negatives of the split (1:1) instead of the
    original RandomLinkSplit 2:1 resampling

Optional knobs (all off = plain baseline) for the AUPR study:
  --val         hold out 10% of training pairs (stratified) for epoch selection
  --dropout     dropout after conv1 and inside the decoder
  --wd          weight decay
  --skip        concatenate the raw (projected) node features to the decoder input
  --layers      number of GeneralConv layers (default 2)
  --neg_mult    resample this many negatives per positive from the unlabeled
                (non-test) pool each epoch instead of the fixed split negatives
  --opt         rmsprop (original) | adam
  --loss        bce | focal | bce_pw (pos_weight)
Output: results/incrd_{tag}_{DS}_{split}_seed{seed}.csv (+ _curves.json)."""
import argparse, json, os, random, sys, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GeneralConv, to_hetero
import torch_geometric.transforms as T
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
from plmf_splits import get_split, load_dataset  # noqa: E402

GRAPH_CACHE = os.path.join(BASE, 'mphgnn_plmf_20260921', 'cache', 'graph_{}.npz')
MORGAN_CACHE = os.path.join(BASE, 'plmf_strongest_20260920', 'graph_cache_{}.npz')

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['MDR', 'MDS'])
ap.add_argument('--split', required=True, choices=['5cv', 'single'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--feat', default='kmer', choices=['kmer', 'expr', 'kmer_expr', 'onehot'])
ap.add_argument('--drugfeat', default='chemberta', choices=['chemberta', 'morgan', 'onehot'])
ap.add_argument('--knn', type=int, default=10, help='k for the RNA-RNA / drug-drug similarity kNN (0 = no homo edges)')
ap.add_argument('--hidden', type=int, default=256)
ap.add_argument('--aggr', default='mean')
ap.add_argument('--layers', type=int, default=2)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--opt', default='rmsprop', choices=['rmsprop', 'adam'])
ap.add_argument('--dropout', type=float, default=0.0)
ap.add_argument('--wd', type=float, default=0.0)
ap.add_argument('--skip', action='store_true')
ap.add_argument('--loss', default='bce', choices=['bce', 'focal', 'bce_pw'])
ap.add_argument('--neg_mult', type=float, default=0.0)
ap.add_argument('--val', action='store_true')
ap.add_argument('--val_frac', type=float, default=0.1)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--tag', default='plain')
ap.add_argument('--folds', default='', help='comma list of folds to run (5cv only)')
ap.add_argument('--ens', type=int, default=1, help='train K models (different init seeds) on the same split, average sigmoid scores per epoch')
ap.add_argument('--snap', type=int, default=0, help='also report snapshot ensemble: mean test score over epochs > snap (needs --val for honest use)')
ap.add_argument('--ctx', action='store_true', help='add label-free graph-context pair features (deg, common-neighbour, path3) from the TRAINING positive graph to the decoder input; 5-fold cross-fit for training pairs')
ap.add_argument('--lr_decay', type=float, default=1.0, help='multiply lr by this every 100 epochs')
a = ap.parse_args()
DEV = torch.device(a.device)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def rowminmax(x):
    x = torch.as_tensor(x, dtype=torch.float)
    mn = x.min(1, keepdim=True).values; mx = x.max(1, keepdim=True).values
    return (x - mn) / (mx - mn + 1e-8)


def knn_edges(x, k, metric='cosine'):
    """Symmetrised top-k neighbour graph (no self loops) -> edge_index [2, E]."""
    if k <= 0:
        return torch.zeros((2, 0), dtype=torch.long)
    x = torch.as_tensor(x, dtype=torch.float)
    if metric == 'tanimoto':
        inter = x @ x.t(); cnt = x.sum(1, keepdim=True)
        s = inter / (cnt + cnt.t() - inter + 1e-8)
    else:
        xn = F.normalize(x, dim=1); s = xn @ xn.t()
    s.fill_diagonal_(-1e9)
    nb = s.topk(k, dim=1).indices
    src = torch.arange(x.shape[0]).repeat_interleave(k); dst = nb.reshape(-1)
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    ei = torch.unique(ei, dim=1)
    return ei


def load_features(ds):
    g = np.load(GRAPH_CACHE.format(ds), allow_pickle=True)
    n_mir, n_drug = g['mirna_kmer'].shape[0], g['drug_x'].shape[0]
    if a.feat == 'kmer':
        xr = rowminmax(g['mirna_kmer'])
    elif a.feat == 'expr':
        xr = torch.cat([rowminmax(g['mirna_expr']), torch.as_tensor(g['mirna_has_expr'], dtype=torch.float)[:, None]], 1)
    elif a.feat == 'kmer_expr':
        xr = torch.cat([rowminmax(g['mirna_kmer']), rowminmax(g['mirna_expr']),
                        torch.as_tensor(g['mirna_has_expr'], dtype=torch.float)[:, None]], 1)
    else:
        xr = torch.eye(n_mir)
    sim_r = knn_edges(rowminmax(g['mirna_kmer']), a.knn, 'cosine')
    if a.drugfeat == 'chemberta':
        xd = rowminmax(g['drug_x']); sim_d = knn_edges(xd, a.knn, 'cosine')
    elif a.drugfeat == 'morgan':
        m = np.load(MORGAN_CACHE.format(ds), allow_pickle=True)
        _, dru, _, _ = load_dataset(ds)
        assert list(m['drugs']) == dru['SMILES'].astype(str).tolist(), 'drug order mismatch'
        xd = torch.as_tensor(m['xd'], dtype=torch.float); sim_d = knn_edges(xd, a.knn, 'tanimoto')
    else:
        xd = torch.eye(n_drug); sim_d = knn_edges(rowminmax(g['drug_x']), a.knn, 'cosine')
    return xr, xd, sim_r, sim_d, n_mir, n_drug


def build_graph(xr, xd, sim_r, sim_d, train_pos):
    data = HeteroData()
    data['miRNA'].x = xr
    data['drug'].x = xd
    data['miRNA', 'MiMi', 'miRNA'].edge_index = sim_r
    data['drug', 'DrugDrug', 'drug'].edge_index = sim_d
    data['miRNA', 'MiDrug', 'drug'].edge_index = torch.as_tensor(train_pos.T, dtype=torch.long)
    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)
    return data


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden, layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList([GeneralConv((-1, -1), hidden) for _ in range(layers)])
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.leaky_relu(x, negative_slope=0.1)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, self.training)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden, in_extra=0, dropout=0.0):
        super().__init__()
        d = 2 * hidden + in_extra
        self.lin1 = Linear(d, 2 * hidden)
        self.lin2 = Linear(2 * hidden, hidden)
        self.lin3 = Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, z, eli, extra=None):
        row, col = eli
        z = torch.cat([z['miRNA'][row], z['drug'][col]] + ([extra] if extra is not None else []), -1)
        z = F.leaky_relu(self.lin1(z), 0.1)
        if self.dropout > 0:
            z = F.dropout(z, self.dropout, self.training)
        z = F.leaky_relu(self.lin2(z), 0.1)
        return self.lin3(z).view(-1)


class Model(torch.nn.Module):
    def __init__(self, data, hidden, aggr, layers, dropout, skip, dims):
        super().__init__()
        self.encoder = to_hetero(GNNEncoder(hidden, layers, dropout), data.metadata(), aggr=aggr)
        self.skip = skip
        if skip:
            self.pr = Linear(dims[0], hidden); self.pd = Linear(dims[1], hidden)
        self.decoder = EdgeDecoder(hidden, 2 * hidden if skip else 0, dropout)

    def forward(self, x_dict, ei_dict, eli):
        z = self.encoder(x_dict, ei_dict)
        extra = None
        if self.skip:
            extra = torch.cat([self.pr(x_dict['miRNA'][eli[0]]), self.pd(x_dict['drug'][eli[1]])], -1)
        return self.decoder(z, eli, extra)


def metrics(y, s, thr=0.5):
    y = np.asarray(y).astype(int); s = np.asarray(s); p = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(y, s), AUPR=average_precision_score(y, s), ACC=accuracy_score(y, p),
                Precision=precision_score(y, p, zero_division=0), Recall=recall_score(y, p, zero_division=0),
                Specificity=tn / (tn + fp) if tn + fp else 0.0, F1=f1_score(y, p, zero_division=0),
                MCC=matthews_corrcoef(y, p) if len(set(y)) > 1 else 0.0)


def loss_fn(pred, y):
    if a.loss == 'focal':
        p = torch.sigmoid(pred); ce = F.binary_cross_entropy_with_logits(pred, y, reduction='none')
        pt = p * y + (1 - p) * (1 - y)
        return ((1 - pt) ** 2 * ce).mean()
    if a.loss == 'bce_pw':
        pw = (1 - y).sum() / y.sum().clamp(min=1)
        return F.binary_cross_entropy_with_logits(pred, y, pos_weight=pw)
    return F.binary_cross_entropy_with_logits(pred, y)


def run_fold(ds, fold, seed, feats):
    xr, xd, sim_r, sim_d, n_mir, n_drug = feats
    tr, ytr, te, yte = get_split(ds, a.split, fold, seed)
    set_seed(seed * 100 + fold)
    val = None
    if a.val:
        rng = np.random.RandomState(seed * 100 + fold)
        vidx = []
        for c in (0, 1):
            ids = np.where(ytr == c)[0]; rng.shuffle(ids)
            vidx.append(ids[:int(round(a.val_frac * len(ids)))])
        vidx = np.concatenate(vidx); mask = np.ones(len(ytr), bool); mask[vidx] = False
        val = (tr[vidx], ytr[vidx]); tr, ytr = tr[mask], ytr[mask]
    train_pos = tr[ytr == 1]
    data = build_graph(xr, xd, sim_r, sim_d, train_pos).to(DEV)
    model = Model(data, a.hidden, a.aggr, a.layers, a.dropout, a.skip, (xr.shape[1], xd.shape[1])).to(DEV)
    with torch.no_grad():  # lazy init
        model(data.x_dict, data.edge_index_dict, torch.as_tensor(tr[:2].T, device=DEV))
    if a.opt == 'rmsprop':
        opt = torch.optim.RMSprop(model.parameters(), lr=a.lr, alpha=0.99, weight_decay=a.wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    eli_tr = torch.as_tensor(tr.T, dtype=torch.long, device=DEV); y_tr = torch.as_tensor(ytr, dtype=torch.float, device=DEV)
    eli_te = torch.as_tensor(te.T, dtype=torch.long, device=DEV)
    eli_va = torch.as_tensor(val[0].T, dtype=torch.long, device=DEV) if val is not None else None
    # unlabeled pool for --neg_mult (everything not a train/val/test pair)
    if a.neg_mult > 0:
        known = np.zeros((n_mir, n_drug), bool)
        known[tr[:, 0], tr[:, 1]] = True; known[te[:, 0], te[:, 1]] = True
        if val is not None:
            known[val[0][:, 0], val[0][:, 1]] = True
        pool = np.argwhere(~known); rng_neg = np.random.RandomState(seed * 100 + fold + 7)
        pos_idx = torch.as_tensor(train_pos.T, dtype=torch.long, device=DEV)
    hist = []; best = None; best_val = (-1, None); last = None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); opt.zero_grad()
        if a.neg_mult > 0:
            n_neg = int(a.neg_mult * len(train_pos))
            sel = pool[rng_neg.randint(len(pool), size=n_neg)]
            eli = torch.cat([pos_idx, torch.as_tensor(sel.T, dtype=torch.long, device=DEV)], 1)
            y = torch.cat([torch.ones(len(train_pos), device=DEV), torch.zeros(n_neg, device=DEV)])
        else:
            eli, y = eli_tr, y_tr
        pred = model(data.x_dict, data.edge_index_dict, eli)
        loss = loss_fn(pred, y); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s_te = torch.sigmoid(model(data.x_dict, data.edge_index_dict, eli_te)).cpu().numpy()
            m = metrics(yte, s_te); m['epoch'] = ep; m['loss'] = float(loss)
            if eli_va is not None:
                s_va = torch.sigmoid(model(data.x_dict, data.edge_index_dict, eli_va)).cpu().numpy()
                m['val_auc'] = roc_auc_score(val[1], s_va); m['val_aupr'] = average_precision_score(val[1], s_va)
                if m['val_auc'] > best_val[0]:
                    best_val = (m['val_auc'], dict(m))
        hist.append({k: (float(v) if not isinstance(v, int) else v) for k, v in m.items()})
        if best is None or m['AUC'] > best['AUC']:
            best = dict(m)
        last = dict(m)
        if ep % 50 == 0 or ep == 1:
            print(f'  ep{ep:3d} loss {loss:.4f} test AUC {m["AUC"]:.4f} AUPR {m["AUPR"]:.4f} best {best["AUC"]:.4f}'
                  + (f' val {m["val_auc"]:.4f}' if eli_va is not None else ''), flush=True)
    row = dict(dataset=ds, split=a.split, method=f'iNcRD-HG[{a.tag}]', fold=fold, seed=seed,
               auc=best['AUC'], aupr=best['AUPR'], acc=best['ACC'], precision=best['Precision'], recall=best['Recall'],
               specificity=best['Specificity'], f1=best['F1'], mcc=best['MCC'], best_epoch=best['epoch'],
               last_auc=last['AUC'], last_aupr=last['AUPR'], best_aupr_epoch=max(hist, key=lambda h: h['AUPR'])['AUPR'],
               secs=time.time() - t0)
    if best_val[1] is not None:
        row.update(valsel_epoch=best_val[1]['epoch'], valsel_auc=best_val[1]['AUC'], valsel_aupr=best_val[1]['AUPR'],
                   valsel_acc=best_val[1]['ACC'])
    print(f'== {a.tag} {ds} {a.split} fold={fold} seed={seed} best_ep={best["epoch"]} AUC={best["AUC"]:.4f} '
          f'AUPR={best["AUPR"]:.4f} | last {last["AUC"]:.4f}/{last["AUPR"]:.4f}'
          + (f' | valsel ep{best_val[1]["epoch"]} {best_val[1]["AUC"]:.4f}/{best_val[1]["AUPR"]:.4f}' if best_val[1] else '')
          + f' ({time.time() - t0:.0f}s)', flush=True)
    return row, hist


def main():
    feats = load_features(a.dataset)
    print(f'{a.dataset}: miRNA x {tuple(feats[0].shape)} drug x {tuple(feats[1].shape)} '
          f'MiMi edges {feats[2].shape[1]} DrugDrug edges {feats[3].shape[1]}', flush=True)
    folds = list(range(5)) if a.split == '5cv' else [0]
    if a.folds:
        folds = [int(f) for f in a.folds.split(',')]
    rows, curves = [], {}
    for f in folds:
        row, hist = run_fold(a.dataset, f, a.seed, feats)
        rows.append(row); curves[str(f)] = hist
    out = os.path.join(HERE, 'results', f'incrd_{a.tag}_{a.dataset}_{a.split}_seed{a.seed}.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    json.dump(curves, open(out.replace('.csv', '_curves.json'), 'w'))
    json.dump(vars(a), open(out.replace('.csv', '_args.json'), 'w'), indent=1)
    d = pd.DataFrame(rows)
    print(f'## {a.tag} {a.dataset} {a.split} seed{a.seed}: test-best AUC {d.auc.mean():.4f} AUPR {d.aupr.mean():.4f} '
          f'| last {d.last_auc.mean():.4f}/{d.last_aupr.mean():.4f}'
          + (f' | valsel {d.valsel_auc.mean():.4f}/{d.valsel_aupr.mean():.4f}' if 'valsel_auc' in d else ''), flush=True)


if __name__ == '__main__':
    main()
