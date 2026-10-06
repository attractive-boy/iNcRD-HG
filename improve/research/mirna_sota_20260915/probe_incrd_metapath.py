"""Probe: MPHGNN-style gene-gene functional metapath features on incrd-hg evidence.
Label-free feature construction; check single-feature AUC on the inner validation fold.
"""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path('/data1/licheng/incrd-hg/iNcRD-HG')
E = ROOT/'improve/results/tabicl_pathway_context_inputs_20260913'
TIERS = ('functional', 'binding', 'association')
WEIGHTS = dict(functional=1., binding=.25, association=.1)


def binary(m):
    m = sparse.csr_matrix(m, dtype=np.float64).copy()
    m.eliminate_zeros()
    m.data[:] = 1.
    return m


def l2_rows(m):
    m = sparse.csr_matrix(m, dtype=np.float64)
    norm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    inv = np.divide(1., norm, out=np.zeros_like(norm), where=norm > 0)
    return sparse.diags(inv).dot(m).tocsr()


dictionary = json.loads((E/'dictionary.json').read_text())
gp = binary(sparse.load_npz(E/'gene_pathway.npz'))
dg = binary(sparse.load_npz(E/'drug_gene.npz'))
G = gp.shape[0]
print('genes', G, 'pathways', gp.shape[1], 'drugs', dg.shape[0], 'drug_gene nnz', dg.nnz)

# gene-gene functional similarity from Reactome pathway co-membership
gp_norm = l2_rows(gp)
path_size = np.asarray(gp.sum(0)).ravel()
print('pathway sizes: mean %.1f max %d, >100: %d' % (path_size.mean(), path_size.max(), (path_size>100).sum()))

task_data = {}
for task in ('MiDrug', 'LncDrug'):
    layers = {t: binary(sparse.load_npz(E/(task+'_'+t+'.npz'))) for t in TIERS}
    weighted = sparse.csr_matrix(layers['functional'].shape, dtype=np.float64)
    for t in TIERS:
        weighted = weighted.maximum(layers[t]*WEIGHTS[t])
    task_data[task] = dict(layers=layers, weighted=weighted)
    print(task, {t: layers[t].nnz for t in TIERS}, 'weighted nnz', weighted.nnz)

# gene support sets
for task in ('MiDrug', 'LncDrug'):
    W = task_data[task]['weighted']
    A = np.flatnonzero(np.asarray((W > 0).sum(0)).ravel() > 0)   # RNA genes
    B = np.flatnonzero(np.asarray((dg > 0).sum(0)).ravel() > 0)  # drug genes
    print(task, 'rna genes', len(A), 'drug genes', len(B), 'drug-gene edges', dg.nnz)
    Pa = gp_norm[A]
    Pb = gp_norm[B]
    C = np.asarray((Pa @ Pb.T).todense())  # |A| x |B| gene-gene pathway cosine
    print('  cross cosine: nnz>0 %d (%.3f), >0.1 %d, max %.3f' % (
        (C > 0).sum(), (C > 0).mean(), (C > 0.1).sum(), C.max()))
    Wa = W[:, A].toarray()            # nr x |A|
    Wb = dg[:, B].toarray()           # nd x |B|
    reach = Wa @ C                    # nr x |B|  RNA functional reach over drug genes
    print('  reach nnz>0 %.3f' % (reach > 0).mean())
    # per-drug metapath score
    score = reach @ Wb.T              # nr x nd
    print('  metapath score: nnz>0 %.3f mean %.4f max %.4f' % ((score > 0).mean(), score[score>0].mean() if (score>0).any() else 0, score.max()))
    task_data[task].update(A=A, B=B, C=C, reach=reach, Wb=Wb, score=score)

# load inner splits via accepted reference
import sys
sys.path.insert(0, str(ROOT))
from improve import tabicl_context_program as accepted
OUT = ROOT/'improve/results/tabicl_pathway_context_20260913'
p = accepted.protocol(OUT)
for task in ('MiDrug', 'LncDrug'):
    a, gpmeta = accepted.stage_data(OUT, p, task, 'inner')
    print(task, 'inner pairs_train', a['pairs_train'].shape, 'pairs_val', a['pairs_val'].shape)
    tr = task_data[task]
    A, B, Wb, reach = tr['A'], tr['B'], tr['Wb'], tr['reach']
    # build per-pair features
    def feats(pairs):
        r, d = pairs[0], pairs[1]
        sc = tr['score'][r, d]
        # normalized cosine over reach vs drug-gene: reach row and Wb row are both length |B|
        rn = np.linalg.norm(reach[r], axis=1); dn = np.linalg.norm(Wb[d], axis=1)
        cos = (reach[r]*Wb[d]).sum(1) / np.maximum(rn*dn, 1e-12)
        rmax = reach[r].max(1) if reach.shape[1] else np.zeros(len(r))
        # shared reach genes: top support overlap
        rs = (reach[r] > 0).astype(np.float32)
        ds = (Wb[d] > 0).astype(np.float32)
        shared = (rs*ds).sum(1)
        return dict(mp_score=sc, mp_cos=cos, mp_max=rmax, mp_shared=shared,
                    mp_rna_genes=(rs.sum(1)), mp_drug_genes=(ds.sum(1)))
    ftr, fval = feats(a['pairs_train']), feats(a['pairs_val'])
    y = a['y_val']
    print('  single-feature AUC/AP on inner val (target %s):' % task)
    for k in ftr:
        v = np.asarray(fval[k], dtype=float)
        if np.isfinite(v).all() and v.std() > 0:
            print('    %-14s auc %.4f  ap %.4f' % (k, roc_auc_score(y, v), average_precision_score(y, v)))
        else:
            print('    %-14s constant' % k)
