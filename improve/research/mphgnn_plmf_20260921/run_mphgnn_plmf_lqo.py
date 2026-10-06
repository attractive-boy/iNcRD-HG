# -*- coding: utf-8 -*-
"""MPHGNN (Huang et al., Briefings in Bioinformatics 2026; github.com/Gtaizu/MPHGNN)
on PLMF-MDA's MDRdataset / MDSdataset, under the two PLMF-MDA protocols used by
improve/research/plmf_baselines_20260920 (bit-identical splits):
  5cv    : paper protocol, positives + equal negatives, 5 folds, seed 42
  single : released-code 80/20 split, seeds 0-4

Model: the official model.py / layer.py verbatim (mphgnn_model.py only turns the
hard-coded 605-miRNA offset into an argument).  Official hyper-parameters
(args.py / paper): hidden 128, 3 hetero-GCN layers, BiTrans aggregator, MIL +
top-3 instance predictor, k=20 (441 metapath instances per pair), batch 256,
Adam lr 1e-4, wd 0, pos_weight BCE, no dropout / batch-norm.

Graph per split (no test leakage): message-passing relations = the official four
(miRNA-drug and drug-miRNA from TRAINING positives only, miRNA-gene, gene-drug);
gene-gene links are used only to enumerate <miRNA, gene1, gene2, drug> instances,
exactly as in the official load_data.meta_path_instance.  The same graph is used
for training and test-time inference.

--lqo K adds leave-query-out message passing (the analogue of run_gnn.py --lqo):
the training positives are split into K groups, pass j builds the message graph from
the other groups only and supervises group-j pairs, so no query pair is ever in the
message graph that scores it (test uses the full training-positive graph).  The
meta-path instances are label-independent and stay fixed.

Output: results/mphgnn_{feat}_{DS}_{split}_seed{seed}.csv with the same columns as
the PLMF baseline CSVs (test-best epoch metrics + last-epoch metrics)."""
import argparse, json, os, random, sys, time
import numpy as np, pandas as pd, torch, dgl
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mphgnn_model import Model
from plmf_splits import get_split

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['MDR', 'MDS'])
ap.add_argument('--split', required=True, choices=['5cv', 'single', 'study', 'cold'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--feat', default='kmer', choices=['kmer', 'expr', 'kmer_expr'])
ap.add_argument('--gene', default='real', choices=['real', 'none', 'shuffled'],
                help="ablation: 'none' drops miRNA-gene/gene-drug/gene-gene links; 'shuffled' permutes gene ids")
ap.add_argument('--epochs', type=int, default=50)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--batch_size', type=int, default=256)
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--layers', type=int, default=3)
ap.add_argument('--k', type=int, default=20)
ap.add_argument('--topk', type=int, default=3)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--tag', default='')
ap.add_argument('--global_graph', default='batch', choices=['batch', 'train'],
                help="edge_index of the global GCN-GAE branch: 'batch' = official (pairs of the current batch, incl. negatives and test queries); 'train' = fixed undirected training-positive graph")
ap.add_argument('--folds', type=int, nargs='+', default=None, help='5cv only: run this subset of folds (output file gets a _f<k> suffix)')
ap.add_argument('--iso_eval', action='store_true',
                help='also evaluate the test set with batch size 1 (each pair scored in isolation: the global branch sees no other query pairs)')
ap.add_argument('--val_frac', type=float, default=0.0,
                help='>0: hold out this fraction of TRAINING pairs (stratified) for early stopping; reported test metrics are then at the best-val-AUC epoch')
ap.add_argument('--lqo', type=int, default=0,
                help='K>1: leave-query-out message passing.  Training pairs are split into K groups; pass j rebuilds the '
                     'miRNA-drug / drug-miRNA message edges (and, with --global_graph train, the fixed GCN edge_index) from '
                     'the training positives of the OTHER K-1 groups, and the loss is taken on group-j pairs only.  Test uses '
                     'the full training-positive graph.  Removes the "target edge is in the message graph" train/test mismatch. '
                     'Requires --global_graph train to be a clean leave-query-out (the official batch graph connects the query '
                     'pairs themselves and cannot be left out).')
a = ap.parse_args()
DEV = torch.device(a.device)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_instances(pairs, r2g, d2g, gg, k, rng):
    """Official meta_path_instance, vectorised over adjacency lists (same instance
    order: miRNA-gene x gene-drug product, then miRNA-gene->gene-gene, then
    gene-gene<-gene-drug; padded by resampling / truncated at k*(k+2)+1)."""
    target = k * (k + 2) + 1
    out = np.empty((len(pairs), target, 4), np.int64)
    pool = np.unique(np.concatenate([x for x in d2g if len(x)])) if any(len(x) for x in d2g) else None
    if pool is None or len(pool) == 0:
        pool = np.unique(np.concatenate([x for x in r2g if len(x)])) if any(len(x) for x in r2g) else np.array([0])
    for n, (mi, dr) in enumerate(pairs):
        g_m, g_d = r2g[mi], d2g[dr]
        parts = []
        if len(g_m) and len(g_d):
            parts.append(np.stack(np.meshgrid(g_m, g_d, indexing='ij'), -1).reshape(-1, 2))
        if len(g_m):
            a1 = [np.stack([np.full(len(gg[g1]), g1), gg[g1]], 1) for g1 in g_m if len(gg[g1])]
            if a1: parts.append(np.concatenate(a1))
        if len(g_d):
            a2 = [np.stack([gg[g2], np.full(len(gg[g2]), g2)], 1) for g2 in g_d if len(gg[g2])]
            if a2: parts.append(np.concatenate(a2))
        if parts:
            gp = np.concatenate(parts)[:target]
        else:
            rg = rng.choice(pool); gp = np.array([[rg, rg]])
        if len(gp) < target:
            gp = np.concatenate([gp, gp[rng.integers(0, len(gp), target - len(gp))]])
        out[n, :, 0] = mi; out[n, :, 1:3] = gp; out[n, :, 3] = dr
    return out


def metrics(y, p):
    auc = roc_auc_score(y, p); aupr = average_precision_score(y, p)
    yb = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yb, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(y); pre = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    spe = tn / (tn + fp) if tn + fp else 0.0; f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return dict(auc=auc, aupr=aupr, acc=acc, precision=pre, recall=rec, specificity=spe, f1=f1, mcc=mcc)


def run_fold(G, tr, ytr, te, yte, seed, fold):
    set_seed(seed)
    nm, nd, ng = len(G['mirna_names']), len(G['drug_ids']), len(G['genes'])
    mg, gd, gg_e = G['mirna_gene'], G['gene_drug'], G['gene_gene']
    rng = np.random.default_rng(seed)
    if a.gene == 'none':
        mg, gd, gg_e = np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64)
    elif a.gene == 'shuffled':
        perm = rng.permutation(ng)
        mg = np.stack([mg[:, 0], perm[mg[:, 1]]], 1); gd = np.stack([perm[gd[:, 0]], gd[:, 1]], 1)
    r2g = [np.array(sorted(set(mg[mg[:, 0] == i, 1])), np.int64) for i in range(nm)]
    d2g = [np.array(sorted(set(gd[gd[:, 1] == j, 0])), np.int64) for j in range(nd)]
    gg = [np.array(sorted(set(gg_e[gg_e[:, 0] == g, 1])), np.int64) for g in range(ng)]
    va, yva = None, None
    if a.val_frac > 0:
        vr = np.random.default_rng(seed + 100)
        vi = np.concatenate([vr.choice(np.flatnonzero(ytr == c), int(round(a.val_frac * (ytr == c).sum())), replace=False) for c in (0, 1)])
        keep = np.ones(len(ytr), bool); keep[vi] = False
        va, yva = tr[vi], ytr[vi]; tr, ytr = tr[keep], ytr[keep]
    pos = tr[ytr == 1]
    K = max(a.lqo, 1)
    grng = np.random.default_rng(seed * 1000 + fold + 17)
    grp_all = np.empty(len(ytr), np.int64)
    grp_all[ytr == 1] = grng.integers(0, K, size=int((ytr == 1).sum()))
    grp_all[ytr == 0] = grng.integers(0, K, size=int((ytr == 0).sum()))

    def make_graph(pj):
        return dgl.heterograph({
            ('gene', 'gene-drug', 'drug'): (torch.as_tensor(gd[:, 0]), torch.as_tensor(gd[:, 1])),
            ('miRNA', 'miRNA-drug', 'drug'): (torch.as_tensor(pj[:, 0]), torch.as_tensor(pj[:, 1])),
            ('drug', 'drug-miRNA', 'miRNA'): (torch.as_tensor(pj[:, 1]), torch.as_tensor(pj[:, 0])),
            ('miRNA', 'miRNA-gene', 'gene'): (torch.as_tensor(mg[:, 0]), torch.as_tensor(mg[:, 1])),
        }, num_nodes_dict={'miRNA': nm, 'drug': nd, 'gene': ng}).to(DEV)

    def make_ei(pj):
        return torch.as_tensor(np.concatenate([np.stack([pj[:, 0], pj[:, 1] + nm], 0),
                                               np.stack([pj[:, 1] + nm, pj[:, 0]], 0)], 1), device=DEV)

    g = make_graph(pos)
    ei = make_ei(pos) if a.global_graph == 'train' else None
    pos_grp = grp_all[ytr == 1]
    msg = [(make_graph(pos[pos_grp != j]), make_ei(pos[pos_grp != j]) if ei is not None else None) for j in range(K)]
    if a.feat == 'kmer':
        mx = G['mirna_kmer']
    elif a.feat == 'expr':
        mx = np.concatenate([G['mirna_expr'], G['mirna_has_expr'][:, None].astype(np.float32)], 1)
    else:
        mx = np.concatenate([G['mirna_kmer'], G['mirna_expr'], G['mirna_has_expr'][:, None].astype(np.float32)], 1)
    feat = {'miRNA': torch.as_tensor(mx, device=DEV), 'drug': torch.as_tensor(G['drug_x'], device=DEV),
            'gene': torch.as_tensor(G['gene_x'], device=DEV)}
    t0 = time.time()
    tr_mp = torch.as_tensor(build_instances(tr, r2g, d2g, gg, a.k, rng), device=DEV)
    te_mp = torch.as_tensor(build_instances(te, r2g, d2g, gg, a.k, rng), device=DEV)
    va_mp = torch.as_tensor(build_instances(va, r2g, d2g, gg, a.k, rng), device=DEV) if va is not None else None
    t_inst = time.time() - t0
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=DEV)
    model = Model(g.etypes, {k: v.shape[1] for k, v in feat.items()}, hidden_feats=a.hidden, num_emb_layers=a.layers,
                  agg_type='BiTrans', k=a.topk, dropout=0., bn=False, skip=True, mil=True, n_rna=nm).to(DEV)
    if ei is not None:
        model.fixed_edge_index = ei
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=0.)
    pw = torch.tensor(float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0), device=DEV)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(ytr); best = None; hist = []

    def predict(mp=None, y=None):
        mp = te_mp if mp is None else mp; y = yte if y is None else y
        model.eval()
        with torch.no_grad():
            lo = torch.cat([model(g, feat, mp[s:s + 1024]).view(-1) for s in range(0, len(y), 1024)])
        return torch.sigmoid(lo).cpu().numpy()

    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.; nb = 0
        for j in range(K):
            gj, eij = msg[j]
            if eij is not None:
                model.fixed_edge_index = eij
            idx = np.flatnonzero(grp_all == j)
            idx = idx[torch.randperm(len(idx)).numpy()]
            for s in range(0, len(idx) - a.batch_size + 1, a.batch_size):   # drop_last=True as official
                ix = torch.as_tensor(idx[s:s + a.batch_size], device=DEV)
                opt.zero_grad(); out = model(gj, feat, tr_mp[ix]).view(-1)
                loss = crit(out, ytr_t[ix]); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if ei is not None:
            model.fixed_edge_index = ei
        p = predict(); m = metrics(yte, p); m['epoch'] = ep; m['train_loss'] = tot / max(nb, 1)
        if va is not None:
            m['val_auc'] = roc_auc_score(yva, predict(va_mp, yva))
        if a.iso_eval:
            model.eval()
            with torch.no_grad():
                lo = torch.cat([model(g, feat, te_mp[i:i + 1]).view(-1) for i in range(len(yte))])
            pi = torch.sigmoid(lo).cpu().numpy(); mi_ = metrics(yte, pi)
            m['iso_auc'] = mi_['auc']; m['iso_aupr'] = mi_['aupr']; m['iso_acc'] = mi_['acc']
        hist.append(m)
        if best is None or m['auc'] > best['auc']:
            best = dict(m)
        if ep % 10 == 0 or ep == a.epochs or ep <= 5:
            print(f"  {a.dataset} {a.split} s{seed} f{fold} ep{ep:03d} loss {tot / max(nb, 1):.4f} test AUC {m['auc']:.4f} AUPR {m['aupr']:.4f}" + (f" iso {m['iso_auc']:.4f}/{m['iso_aupr']:.4f}" if a.iso_eval else '') + (f" val {m['val_auc']:.4f}" if va is not None else '') + f" | best ep{best['epoch']} {best['auc']:.4f}/{best['aupr']:.4f} [{time.time() - t0:.0f}s]", flush=True)
    last = hist[-1]
    minloss = min(hist, key=lambda h: h['train_loss'])
    valsel = max(hist, key=lambda h: h['val_auc']) if va is not None else None
    row = dict(dataset=a.dataset, split=a.split, method='MPHGNN', fold=fold, seed=seed,
               auc=best['auc'], aupr=best['aupr'], acc=best['acc'], precision=best['precision'], recall=best['recall'],
               specificity=best['specificity'], f1=best['f1'], mcc=best['mcc'], best_epoch=best['epoch'],
               last_auc=last['auc'], last_aupr=last['aupr'], last_acc=last['acc'],
               minloss_epoch=minloss['epoch'], minloss_auc=minloss['auc'], minloss_aupr=minloss['aupr'],
               valsel_epoch=valsel['epoch'] if valsel else -1, valsel_auc=valsel['auc'] if valsel else float('nan'),
               valsel_aupr=valsel['aupr'] if valsel else float('nan'), valsel_acc=valsel['acc'] if valsel else float('nan'), val_frac=a.val_frac, global_graph=a.global_graph,
               iso_best_auc=max(h['iso_auc'] for h in hist) if a.iso_eval else float('nan'),
               iso_valsel_auc=valsel['iso_auc'] if (valsel and a.iso_eval) else float('nan'),
               iso_valsel_aupr=valsel['iso_aupr'] if (valsel and a.iso_eval) else float('nan'),
               iso_last_auc=last['iso_auc'] if a.iso_eval else float('nan'), iso_last_aupr=last['iso_aupr'] if a.iso_eval else float('nan'),
               feat=a.feat, gene=a.gene, n_gene=ng, n_train=len(ytr), n_test=len(yte), lqo=a.lqo,
               inst_build_s=round(t_inst, 1), time_s=round(time.time() - t0, 1))
    return row, hist


def main():
    G = dict(np.load(os.path.join(HERE, 'cache', f'graph_{a.dataset}.npz'), allow_pickle=False))
    tag = f"mphgnn_{a.feat}{'' if a.gene == 'real' else '_gene-' + a.gene}{'_val' if a.val_frac > 0 else ''}{'_gg-train' if a.global_graph == 'train' else ''}{'_iso' if a.iso_eval else ''}{'_lqo' + str(a.lqo) if a.lqo > 1 else ''}{a.tag}_{a.dataset}_{a.split}_seed{a.seed}"
    folds = (a.folds if a.folds else range(5)) if a.split in ('5cv', 'study', 'cold') else [0]
    if a.folds and a.split == '5cv':
        tag += '_f' + ''.join(str(f) for f in a.folds)
    out = os.path.join(HERE, 'results', tag + '.csv')
    rows, hists = [], {}
    for f in folds:
        tr, ytr, te, yte = get_split(a.dataset, a.split, f, a.seed)
        row, hist = run_fold(G, tr, ytr, te, yte, a.seed, f)
        rows.append(row); hists[f] = hist
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"== {tag} fold {f}: best ep{row['best_epoch']} AUC {row['auc']:.4f} AUPR {row['aupr']:.4f} | last AUC {row['last_auc']:.4f} AUPR {row['last_aupr']:.4f} | {row['time_s']}s", flush=True)
    json.dump(hists, open(os.path.join(HERE, 'results', tag + '_curves.json'), 'w'))
    d = pd.DataFrame(rows)
    print(f"#### {tag}: best-epoch AUC {d.auc.mean():.4f}±{d.auc.std(ddof=0):.4f} AUPR {d.aupr.mean():.4f} | last AUC {d.last_auc.mean():.4f} AUPR {d.last_aupr.mean():.4f}", flush=True)


if __name__ == '__main__':
    main()
