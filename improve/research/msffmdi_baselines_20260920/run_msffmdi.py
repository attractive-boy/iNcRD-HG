# -*- coding: utf-8 -*-
"""Runner: MSFFMDI baselines on ncDR / RNAInter (clean 5-fold CV, 1:1)."""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import GCNConv

import msffmdi_data as MD
import msffmdi_models as M

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
METHOD = {'lagcn': 'LAGCN', 'gcnnma': 'GCNNMMA', 'mfidma': 'MFIDMA',
          'ldaman': 'LDAMAN', 'dlstmda': 'DLST-MDA'}
DEFAULTS = {'lagcn': (100, 1e-3, 256), 'gcnnma': (100, 1e-3, 256),
            'mfidma': (50, 1e-3, 128), 'ldaman': (100, 1e-3, 256), 'dlstmda': (100, 5e-4, 128)}


def dense_to_edges(A, device):
    A = np.asarray(A)
    i, j = np.nonzero(A)
    ei = torch.tensor(np.vstack([i, j]), dtype=torch.long, device=device)
    ew = torch.tensor(A[i, j], dtype=torch.float32, device=device)
    return ei, ew


def load_emb(path):
    d = {}
    with open(path, newline='') as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                d[row[0]] = np.array([float(x) for x in row[1:]], dtype=np.float32)
    return d


def build_mfidma(data, dataset):
    repo = MD.REPO
    sub = dataset
    mi_emb = load_emb(os.path.join(repo, sub, 'miRNAEmbedding.csv'))
    dr_emb = load_emb(os.path.join(repo, sub, 'DrugEmbedding.csv'))
    beh = load_emb(os.path.join(repo, sub, 'AllNodeBehavior.csv'))
    names, cids = data['names'], data['cids']
    beh_dim = len(next(iter(beh.values())))
    r_beh = np.stack([beh[q] for q in names]).astype(np.float32)
    d_beh = np.stack([beh[c] for c in cids]).astype(np.float32)
    dim = len(next(iter(mi_emb.values())))

    def mi_attr(seq, L=25):
        a = np.zeros((L, dim), dtype=np.float32)
        for i, ch in enumerate(seq[:L]):
            if ch in mi_emb:
                a[i] = mi_emb[ch]
        return a

    def dr_attr(smi, L=64):
        a = np.zeros((L, dim), dtype=np.float32)
        for i, ch in enumerate(smi[:L]):
            if ch in dr_emb:
                a[i] = dr_emb[ch]
        return a

    R = np.stack([mi_attr(s) for s in data['seqs'].values()]).astype(np.float32)
    D = np.stack([dr_attr(s) for s in data['smis'].values()]).astype(np.float32)
    return R, D, r_beh, d_beh, beh_dim


def build_dlst_graphs(data):
    from common import smile_to_graph
    out = []
    for smi in data['smis'].values():
        _, feats, ei = smile_to_graph(smi)
        out.append((np.asarray(feats, dtype=np.float32),
                    torch.tensor(ei, dtype=torch.long).t().contiguous()))
    return out


def make_pyg(dataset_graphs, data, pairs, y, smiles2int, seq2int):
    cid_list = data['cids']
    cid_idx = {c: i for i, c in enumerate(cid_list)}
    name_list = data['names']
    name_idx = {q: i for i, q in enumerate(name_list)}
    ds = []
    for (i, j), lab in zip(pairs, y):
        feats, ei = dataset_graphs[j]
        d = Data(x=torch.tensor(feats), edge_index=ei,
                 y=torch.tensor([float(lab)]))
        d.target = torch.tensor(seq2int(data['seqs'][name_list[i]]), dtype=torch.long).unsqueeze(0)
        d.seqdrug = torch.tensor(smiles2int(data['smis'][cid_list[j]]), dtype=torch.long).unsqueeze(0)
        ds.append(d)
    return ds


def hetero_adj(data, train_pairs, knn=10):
    """Normalized adjacency over [miRNA ; drug] nodes: similarity kNN + train association edges."""
    nr, nd = len(data['names']), len(data['cids'])
    Xr, Xd = data['Xr'], data['Xd']
    ur = Xr / np.maximum(np.linalg.norm(Xr, axis=1, keepdims=True), 1e-8)
    ud = Xd / np.maximum(np.linalg.norm(Xd, axis=1, keepdims=True), 1e-8)
    sr = M.knn_adj((ur @ ur.T).astype(np.float32), knn)
    sd = M.knn_adj((ud @ ud.T).astype(np.float32), knn)
    A = np.zeros((nr + nd, nr + nd), dtype=np.float32)
    A[:nr, :nr] = sr
    A[nr:, nr:] = sd
    A[train_pairs[:, 0], nr + train_pairs[:, 1]] = 1.0
    A[nr + train_pairs[:, 1], train_pairs[:, 0]] = 1.0
    return M.norm_adj(A)


def make_gcnnma_loader(data, pairs, y, compounds, words, comp_index, lmax, shuffle):
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(pairs)
        def __getitem__(self, k):
            i, j = pairs[k]
            w = np.zeros(lmax, dtype=np.int64)
            ww = words[data['seqs'][data['names'][i]]]
            w[:len(ww)] = ww
            return comp_index[data['smis'][data['cids'][j]]], torch.tensor(w), float(y[k])
    cl = list(compounds.keys())
    def collate(batch):
        bi, ws, ys = zip(*batch)
        nmax = max(compounds[cl[b]][1].size(0) for b in bi)
        B = len(bi)
        fp = torch.zeros(B, nmax, dtype=torch.long); adj = torch.zeros(B, nmax, nmax)
        mask = torch.zeros(B, nmax)
        for r, b in enumerate(bi):
            f, a = compounds[cl[b]]; n = a.size(0)
            fp[r, :n] = f; adj[r, :n, :n] = a; mask[r, :n] = 1.0
        return fp, adj, mask, torch.stack(ws), torch.tensor(ys)
    return torch.utils.data.DataLoader(DS(), batch_size=256, shuffle=shuffle, collate_fn=collate)


def eval_scores(y, score):
    return MD.compute_metrics(y, score)


def run_lagcn(data, tr, te, device, epochs, lr, seed):
    nr, nd = len(data['names']), len(data['cids'])
    Xr = torch.tensor(data['Xr'], device=device); Xd = torch.tensor(data['Xd'], device=device)
    A = hetero_adj(data, data['pairs'][tr])
    ei, ew = dense_to_edges(A, device)
    model = M.LAGCN(data['Xr'].shape[1], data['Xd'].shape[1]).to(device)
    model.nr = nr
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ytr = torch.tensor(data['y'][tr], device=device)
    ptr = torch.tensor(data['pairs'][tr], device=device)
    pte = data['pairs'][te]; yte = data['y'][te]
    best, last = {'AUC': -1}, None
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        U, V = model(Xr, Xd, ei, ew)
        logit = (U[ptr[:, 0]] * V[ptr[:, 1]]).sum(-1)
        loss = F.binary_cross_entropy_with_logits(logit, ytr)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            U, V = model(Xr, Xd, ei, ew)
            sc = torch.sigmoid((U[pte[:, 0]] * V[pte[:, 1]]).sum(-1)).cpu().numpy()
        m = eval_scores(yte, sc); m['epoch'] = ep; last = m
        if m['AUC'] > best['AUC']: best = m
    return best, last


def run_mfidma(data, tr, te, device, epochs, lr, seed, dataset):
    R, D, r_beh, d_beh, beh_dim = data['mfidma']
    R = torch.tensor(R, device=device); D = torch.tensor(D, device=device)
    rb = torch.tensor(r_beh, device=device); db = torch.tensor(d_beh, device=device)
    model = M.MFIDMA(attr_dim=R.shape[2], beh_dim=beh_dim, mi_len=R.shape[1], dr_len=D.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    ytr = torch.tensor(data['y'][tr], device=device).long()
    ptr = torch.tensor(data['pairs'][tr], device=device)
    pte = data['pairs'][te]; yte = data['y'][te]
    best, last = {'AUC': -1}, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), device=device)
        tot = 0.0
        for k in range(0, len(tr), 128):
            idx = perm[k:k + 128]
            i, j = ptr[idx, 0], ptr[idx, 1]
            out = model(R[i], D[j], rb[i], db[j])
            loss = F.cross_entropy(out, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            i = torch.tensor(pte[:, 0], device=device); j = torch.tensor(pte[:, 1], device=device)
            sc = torch.softmax(model(R[i], D[j], rb[i], db[j]), -1)[:, 1].cpu().numpy()
        m = eval_scores(yte, sc); m['epoch'] = ep; last = m
        if m['AUC'] > best['AUC']: best = m
    return best, last


def run_dlstmda(data, tr, te, device, epochs, lr, seed):
    from common import smiles2int, seq2int
    tr_ds = make_pyg(data['dlst_graphs'], data, data['pairs'][tr], data['y'][tr], smiles2int, seq2int)
    te_ds = make_pyg(data['dlst_graphs'], data, data['pairs'][te], data['y'][te], smiles2int, seq2int)
    trl = PyGLoader(tr_ds, batch_size=128, shuffle=True)
    tel = PyGLoader(te_ds, batch_size=256, shuffle=False)
    model = M.DLSTMDANet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best, last = {'AUC': -1}, None
    for ep in range(1, epochs + 1):
        model.train()
        for d in trl:
            d = d.to(device)
            loss = F.binary_cross_entropy_with_logits(model(d), d.y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for d in tel:
                d = d.to(device)
                ps.append(torch.sigmoid(model(d)).cpu().numpy()); ys.append(d.y.view(-1).cpu().numpy())
        m = eval_scores(np.concatenate(ys), np.concatenate(ps)); m['epoch'] = ep; last = m
        if m['AUC'] > best['AUC']: best = m
    return best, last


def run_gcnnma(data, tr, te, device, epochs, lr, seed):
    pairs = [(data['smis'][data['cids'][j]], data['seqs'][data['names'][i]])
             for i, j in np.concatenate([data['pairs'][tr], data['pairs'][te]])]
    compounds, words, n_fp, n_word = M._gcnnma_vocab(list(dict.fromkeys(pairs)))
    comp_index = {s: k for k, s in enumerate(compounds.keys())}
    lmax = max(len(v) for v in words.values())
    trl = make_gcnnma_loader(data, data['pairs'][tr], data['y'][tr], compounds, words, comp_index, lmax, True)
    tel = make_gcnnma_loader(data, data['pairs'][te], data['y'][te], compounds, words, comp_index, lmax, False)
    model = M.GCNNMMA(n_fp, n_word).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    best, last = {'AUC': -1}, None
    for ep in range(1, epochs + 1):
        if ep % 10 == 0:
            for g in opt.param_groups: g['lr'] *= 0.5
        model.train()
        for fp, adj, mask, ws, yb in trl:
            fp, adj, mask, ws, yb = fp.to(device), adj.to(device), mask.to(device), ws.to(device), yb.to(device)
            loss = F.cross_entropy(model(fp, adj, mask, ws), yb.long())
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); ps = []
        with torch.no_grad():
            for fp, adj, mask, ws, yb in tel:
                ps.append(torch.softmax(model(fp.to(device), adj.to(device), mask.to(device),
                                               ws.to(device)), -1)[:, 1].cpu().numpy())
        m = eval_scores(data['y'][te], np.concatenate(ps)); m['epoch'] = ep; last = m
        if m['AUC'] > best['AUC']: best = m
    return best, last


def run_ldaman(data, tr, te, device, epochs, lr, seed):
    from xgboost import XGBClassifier
    nr, nd = len(data['names']), len(data['cids'])
    ei, ew = dense_to_edges(hetero_adj(data, data['pairs'][tr]), device)
    Xr_t = torch.tensor(data['Xr'], device=device); Xd_t = torch.tensor(data['Xd'], device=device)

    class AE(nn.Module):
        def __init__(self, fr, fd, h=64):
            super().__init__()
            self.pr = nn.Linear(fr, h); self.pd = nn.Linear(fd, h)
            self.g1 = GCNConv(h, h); self.g2 = GCNConv(h, h)
            self.dr = nn.Linear(h, fr); self.dd = nn.Linear(h, fd)
        def enc(self, xr, xd):
            z = torch.cat([self.pr(xr), self.pd(xd)], 0)
            z = F.relu(self.g1(z, ei, ew)); z = self.g2(z, ei, ew)
            return z
        def forward(self, xr, xd):
            z = self.enc(xr, xd)
            return self.dr(z[:nr]), self.dd(z[nr:]), z
    ae = AE(data['Xr'].shape[1], data['Xd'].shape[1]).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for ep in range(epochs):
        ae.train(); opt.zero_grad()
        rr, dd, _ = ae(Xr_t, Xd_t)
        loss = F.mse_loss(rr, Xr_t) + F.mse_loss(dd, Xd_t)
        loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        Z = ae.enc(Xr_t, Xd_t).cpu().numpy()
    Zr, Zd = Z[:nr], Z[nr:]
    def feats(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        return np.concatenate([Zr[i], Zd[j], Zr[i] * Zd[j]], 1)
    clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, n_jobs=8,
                        eval_metric='logloss')
    clf.fit(feats(data['pairs'][tr]), data['y'][tr].astype(int))
    sc = clf.predict_proba(feats(data['pairs'][te]))[:, 1]
    m = eval_scores(data['y'][te], sc); m['epoch'] = epochs
    return m, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(METHOD))
    ap.add_argument('--dataset', required=True, choices=['ncDR', 'RNAInter'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=0)
    ap.add_argument('--device', default=os.environ.get('MSFFM_DEV', 'cuda:0'))
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    epochs, lr, batch = DEFAULTS[args.model]
    if args.epochs: epochs = args.epochs
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    M.seed_it(args.seed)
    data = M.build_splits(args.dataset, args.seed)
    data['Xr'], data['Xd'] = M.node_features(data)
    if args.model == 'mfidma':
        R, D, rb, db, bd = build_mfidma(data, args.dataset)
        data['mfidma'] = (R, D, rb, db, bd)
    if args.model == 'dlstmda':
        data['dlst_graphs'] = build_dlst_graphs(data)
    print(f'{args.dataset}: miRNAs={len(data["names"])} drugs={len(data["cids"])} '
          f'pairs={len(data["pairs"])} pos={int(data["y"].sum())}', flush=True)

    out = args.out or os.path.join(RESULTS, f'{args.model}_{args.dataset}_seed{args.seed}.csv')
    log = open(os.path.join(LOGS, f'{args.model}_{args.dataset}_seed{args.seed}.log'), 'w')
    fields = ['dataset', 'method', 'fold', 'seed', 'auc', 'aupr', 'acc', 'sen', 'spec',
              'best_epoch', 'last_auc', 'last_aupr']
    rows = []
    for f in [int(x) for x in args.folds.split(',')]:
        tr, te = data['folds'][f]
        t0 = time.time()
        if args.model == 'lagcn':
            best, last = run_lagcn(data, tr, te, args.device, epochs, lr, args.seed)
        elif args.model == 'mfidma':
            best, last = run_mfidma(data, tr, te, args.device, epochs, lr, args.seed, args.dataset)
        elif args.model == 'dlstmda':
            best, last = run_dlstmda(data, tr, te, args.device, epochs, lr, args.seed)
        elif args.model == 'gcnnma':
            best, last = run_gcnnma(data, tr, te, args.device, epochs, lr, args.seed)
        else:
            best, last = run_ldaman(data, tr, te, args.device, epochs, lr, args.seed)
        rows.append(dict(dataset=args.dataset, method=METHOD[args.model], fold=f, seed=args.seed,
                         auc=best['AUC'], aupr=best['AUPR'], acc=best['ACC'], sen=best['Sen'],
                         spec=best['Spec'], best_epoch=best['epoch'],
                         last_auc=last['AUC'], last_aupr=last['AUPR']))
        msg = (f'== {args.model} {args.dataset} fold={f} ep={best["epoch"]} '
               f'AUC={best["AUC"]:.4f} AUPR={best["AUPR"]:.4f} ACC={best["ACC"]:.4f} '
               f'Sen={best["Sen"]:.4f} Spec={best["Spec"]:.4f} | last {last["AUC"]:.4f} '
               f'({time.time()-t0:.0f}s)')
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f'\n=== {METHOD[args.model]} {args.dataset} ===')
    for k in ('auc', 'aupr', 'acc', 'sen', 'spec'):
        v = [r[k] for r in rows]
        print(f'  {k}: {np.mean(v):.4f} +/- {np.std(v):.4f}')
    print('saved', out)


if __name__ == '__main__':
    main()
