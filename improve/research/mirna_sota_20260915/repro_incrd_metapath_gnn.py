"""End-to-end gene-mediated metapath heterogeneous GNN (MPHGNN official model)
on incrd-hg's own graph, evaluated on the accepted inner partition for both tasks.

Graph nodes: RNA (miRNA or lncRNA, named 'miRNA' for the model), drug, gene.
Relations (message passing, official 4): gene-drug, miRNA-drug(train pos),
drug-miRNA, miRNA-gene. gene-gene functional links are used only to build
metapath instances <RNA, gene1, gene2, drug>.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import dgl
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import Model  # official MPHGNN model, patched for variable RNA count

ROOT = '/data1/licheng/incrd-hg/iNcRD-HG'
E = ROOT + '/improve/results/tabicl_pathway_context_inputs_20260913'
NET = '/data1/licheng/incrd-hg/incrd-run/2.construct_network'
REF = ROOT + '/improve/results/tabicl_pathway_context_20260913/predictions/inner'
TIERS = ('functional', 'binding', 'association')
WEIGHTS = dict(functional=1., binding=.25, association=.1)

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, nargs='+', default=[666, 0, 1])
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--layers', type=int, default=3)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--batch_size', type=int, default=256)
ap.add_argument('--k', type=int, default=20)
ap.add_argument('--topk', type=int, default=3)
ap.add_argument('--gg_k', type=int, default=10)
ap.add_argument('--device', default='cuda:0')
a = ap.parse_args()
DEV = torch.device(a.device if torch.cuda.is_available() else 'cpu')


def l2_rows(m):
    m = sparse.csr_matrix(m, dtype=np.float64)
    n = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    inv = np.divide(1., n, out=np.zeros_like(n), where=n > 0)
    return sparse.diags(inv).dot(m).tocsr()


def gene_gene_topk(gp_norm, A, k):
    """Top-k cosine neighbours within A via chunked dense product."""
    P = gp_norm[A]
    n = P.shape[0]
    rows, cols = [], []
    for s in range(0, n, 1500):
        e = min(s + 1500, n)
        sim = np.asarray((P[s:e] @ P.T).todense(), dtype=np.float32)
        for i in range(sim.shape[0]):
            sim[i, s + i] = -1.0
        idx = np.argpartition(-sim, kth=min(k, n - 1), axis=1)[:, :k]
        for i in range(sim.shape[0]):
            for j in idx[i]:
                if sim[i, j] > 0:
                    rows.append(s + i); cols.append(j)
    return np.asarray(rows), np.asarray(cols)


def build_gene_lists(mat, A_index):
    """mat: (nr x n_active) csr; return list of gene-index arrays per RNA."""
    mat = sparse.csr_matrix(mat)
    return [mat.indices[mat.indptr[i]:mat.indptr[i + 1]] for i in range(mat.shape[0])]


def meta_path_instance(mi, dr, r2g, d2g, gg, k, rng):
    mpi = [[mi, g1, g2, dr] for g1 in r2g[mi] for g2 in d2g[dr]]
    mpi += [[mi, g1, g2, dr] for g1 in r2g[mi] for g2 in gg[g1]]
    mpi += [[mi, g1, g2, dr] for g2 in d2g[dr] for g1 in gg[g2]]
    if not mpi:
        pool = np.unique(np.concatenate(d2g)) if any(len(x) for x in d2g) else np.unique(np.concatenate(r2g))
        rg = int(rng.choice(pool)); mpi.append([mi, rg, rg, dr])
    target = k * (k + 2) + 1
    if len(mpi) < target:
        idx = rng.integers(0, len(mpi), size=target - len(mpi))
        mpi = mpi + [mpi[i] for i in idx]
    else:
        mpi = mpi[:target]
    return mpi


def build_instances(pairs, r2g, d2g, gg, k, seed):
    rng = np.random.default_rng(seed)
    out = np.asarray([meta_path_instance(int(m), int(d), r2g, d2g, gg, k, rng) for m, d in pairs.T], dtype=np.int64)
    return out


def load_task(task, seed):
    dictionary = json.loads(open(E + '/dictionary.json').read())
    nr = len(dictionary['rna_names'][task])
    nd = len(dictionary['drug_names'])
    dg = sparse.load_npz(E + '/drug_gene.npz').tocsr()
    gp = sparse.load_npz(E + '/gene_pathway.npz').tocsr()
    tiers = {t: sparse.load_npz(E + '/' + task + '_' + t + '.npz').tocsr() for t in TIERS}
    W = sparse.csr_matrix(tiers['functional'].shape, dtype=np.float64)
    for t in TIERS:
        W = W.maximum(tiers[t] * WEIGHTS[t])
    union = sparse.csr_matrix(tiers['functional'].shape, dtype=np.float64)
    for t in TIERS:
        union = union.maximum(tiers[t])
    # active genes: functional+association+ drug genes
    base = (tiers['functional'] + tiers['association'])
    active = np.asarray((base > 0).sum(0)).ravel() > 0
    active |= (np.asarray((dg > 0).sum(0)).ravel() > 0)
    A = np.flatnonzero(active)
    r2g = union[:, A].tocsr()
    d2g = dg[:, A].tocsr()
    gp_norm = l2_rows(gp)
    gr, gc = gene_gene_topk(gp_norm, A, a.gg_k)
    gg = [[] for _ in range(len(A))]
    for i, j in zip(gr, gc):
        gg[i].append(j)
    gg = [np.asarray(x) if len(x) else np.asarray([], dtype=np.int64) for x in gg]
    r2g_l = build_gene_lists(r2g, A)
    d2g_l = build_gene_lists(d2g, A)
    # node features
    import pandas as pd
    exp = pd.read_csv(NET + ('/MiExp.csv' if task == 'MiDrug' else '/LncExp.csv'), index_col=0)
    exp.columns = [c.upper() for c in exp.columns]
    exp = exp.loc[:, dictionary['rna_names'][task]].to_numpy(dtype=np.float32).T
    rna_feat = (exp - exp.min(1, keepdims=True)) / (exp.max(1, keepdims=True) - exp.min(1, keepdims=True) + 1e-8)
    drug = pd.read_csv(NET + '/drug_features-chemBertA.csv', index_col=0).to_numpy(dtype=np.float32)
    drug_feat = (drug - drug.min(1, keepdims=True)) / (drug.max(1, keepdims=True) - drug.min(1, keepdims=True) + 1e-8)
    gene_feat = l2_rows(gp[A].T).T.toarray().astype(np.float32)  # |A| x pathways
    return dict(nr=nr, nd=nd, n_gene=len(A), r2g=r2g, d2g=d2g, gg=gg, r2g_l=r2g_l, d2g_l=d2g_l,
                rna_feat=rna_feat, drug_feat=drug_feat, gene_feat=gene_feat, A=A)


def run(task, seed):
    inner = np.load('/tmp/opencode/inner_%s.npz' % task)
    ptr, ytr = inner['pairs_train'], inner['y_train']
    pva, yva = inner['pairs_val'], inner['y_val']
    d = load_task(task, seed)
    nr = d['nr']; n_gene = d['n_gene']
    pos = ptr[:, ytr == 1]
    r2g_coo, d2g_coo = d['r2g'].tocoo(), d['d2g'].tocoo()
    g = dgl.heterograph({
        ('miRNA', 'mg', 'gene'): (r2g_coo.row.astype(np.int64), r2g_coo.col.astype(np.int64)),
        ('gene', 'gd', 'drug'): (d2g_coo.col.astype(np.int64), d2g_coo.row.astype(np.int64)),
        ('miRNA', 'md', 'drug'): (pos[0].astype(np.int64), pos[1].astype(np.int64)),
        ('drug', 'dm', 'miRNA'): (pos[1].astype(np.int64), pos[0].astype(np.int64)),
    }, num_nodes_dict={'miRNA': nr, 'drug': d['nd'], 'gene': n_gene})
    feat = {'miRNA': torch.as_tensor(d['rna_feat'], device=DEV),
            'drug': torch.as_tensor(d['drug_feat'], device=DEV),
            'gene': torch.as_tensor(d['gene_feat'], device=DEV)}
    g = g.to(DEV)
    tr_mp = torch.as_tensor(build_instances(ptr, d['r2g_l'], d['d2g_l'], d['gg'], a.k, seed + 2), device=DEV)
    va_mp = torch.as_tensor(build_instances(pva, d['r2g_l'], d['d2g_l'], d['gg'], a.k, seed + 3), device=DEV)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=DEV)
    in_feats = {'miRNA': d['rna_feat'].shape[1], 'drug': d['drug_feat'].shape[1], 'gene': d['gene_feat'].shape[1]}
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = Model(g.etypes, in_feats, hidden_feats=a.hidden, num_emb_layers=a.layers,
                  agg_type='BiTrans', k=a.topk, dropout=0., bn=False, skip=True, mil=True).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=0.)
    pw = torch.tensor(float((ytr == 0).sum() / max((ytr == 1).sum(), 1)), device=DEV)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(ytr)
    best_auc, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        tot = 0.
        for s in range(0, n, a.batch_size):
            ix = perm[s:s + a.batch_size]
            opt.zero_grad()
            out = model(g, feat, tr_mp[ix]).view(-1)
            loss = crit(out, ytr_t[ix])
            loss.backward(); opt.step()
            tot += float(loss)
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(g, feat, va_mp[s:s + 512]).view(-1) for s in range(0, len(yva), 512)]).cpu().numpy()
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yva, logits)
        if auc > best_auc:
            best_auc = auc
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print('  %s seed %d ep %d loss %.4f val_auc %.4f' % (task, seed, ep + 1, tot, auc), flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits = torch.cat([model(g, feat, va_mp[s:s + 512]).view(-1) for s in range(0, len(yva), 512)]).cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logits))
    return prob, yva, pva


def main():
    from sklearn.metrics import roc_auc_score, average_precision_score
    import pandas as pd
    rows = []
    for task in ('MiDrug', 'LncDrug'):
        with np.load(REF + '/%s/s666/scores.npz' % task, allow_pickle=False) as z:
            ref_pairs = z['pairs_val']; ref_y = z['y_val']
        for seed in a.seeds:
            with np.load(REF + '/%s/s%d/scores.npz' % (task, seed), allow_pickle=False) as z:
                ref = z['tiered_0p5000']
                np.testing.assert_array_equal(z['pairs_val'], ref_pairs)
            t0 = time.time()
            prob, yva, pva = run(task, seed)
            np.testing.assert_array_equal(pva, ref_pairs)
            rows.append(dict(task=task, seed=seed, variant='gnn',
                             **dict(auc=roc_auc_score(yva, prob), ap=average_precision_score(yva, prob))))
            rows.append(dict(task=task, seed=seed, variant='accepted_reference',
                             **dict(auc=roc_auc_score(yva, ref), ap=average_precision_score(yva, ref))))
            for w in (0.1, 0.2, 0.3, 0.5):
                comp = (1 - w) * ref + w * prob
                rows.append(dict(task=task, seed=seed, variant='composed_%02d' % int(w * 100),
                                 **dict(auc=roc_auc_score(yva, comp), ap=average_precision_score(yva, comp))))
            print('%s seed %d done in %.0fs  gnn %.4f/%.4f  ref %.4f/%.4f' % (
                task, seed, time.time() - t0, rows[-6]['auc'], rows[-6]['ap'],
                rows[-5]['auc'], rows[-5]['ap']), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv('/tmp/opencode/incrd_metapath/metrics.csv', index=False)
    s = t.groupby(['task', 'variant'])[['auc', 'ap']].mean().reset_index()
    print(s.to_string(index=False))
    gates = {}
    for w in (0.1, 0.2, 0.3, 0.5):
        variant = 'composed_%02d' % int(w * 100)
        gate = {}
        for task in ('MiDrug', 'LncDrug'):
            r = t[t.task == task]
            d = {m: np.array([r[(r.seed == sd) & (r.variant == variant)][m].values[0] -
                              r[(r.seed == sd) & (r.variant == 'accepted_reference')][m].values[0] for sd in a.seeds]) for m in ('auc', 'ap')}
            ok = all(np.all(v >= 0) for v in d.values()) if task == 'LncDrug' else all(v.mean() > 0 and (v > 0).sum() >= 2 for v in d.values())
            gate[task] = dict(passed=bool(ok), delta={k: [round(float(x), 6) for x in v] for k, v in d.items()})
        gates[variant] = gate
    print('GATES', json.dumps(gates))


if __name__ == '__main__':
    main()
