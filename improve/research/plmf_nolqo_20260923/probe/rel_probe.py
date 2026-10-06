# relation coverage and label-transfer hit rate of dataset-internal miRNA relations
import sys, re
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from feat_probe import G, rowminmax, knn, stem, same
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset

def family(n):
    s = stem(n).replace('hsa-', ''); s = re.sub(r'-[12]$', '', s); return re.sub(r'([0-9])[a-z]$', r'\1', s)

for ds in ('MDR', 'MDS'):
    g = np.load(G.format(ds), allow_pickle=True)
    mir, dru, pos, neg = load_dataset(ds)
    names = mir['miRNA_Name'].astype(str).values; seqs = mir['miRNA_Sequence'].astype(str).str.upper().str.replace('T', 'U').tolist()
    xr = rowminmax(g['mirna_kmer'].astype(float))
    rels = {'knn10': knn(xr, 10), 'seed7': same([s[1:8] for s in seqs]), 'seed6': same([s[1:7] for s in seqs]),
            'pre': same([stem(n) for n in names]), 'fam': same([family(n) for n in names]),
            'seq_id': same(seqs)}
    tr, ytr, te, yte = get_split(ds, '5cv', 0, 42)
    A = np.zeros((len(names), len(dru))); A[tr[ytr == 1, 0], tr[ytr == 1, 1]] = 1
    for k, R in rels.items():
        cov = (R.sum(1) > 0).mean()
        RA = R @ A
        hp = (RA[te[yte == 1, 0], te[yte == 1, 1]] > 0).mean(); hn = (RA[te[yte == 0, 0], te[yte == 0, 1]] > 0).mean()
        # among test positives whose miRNA has train degree <= 1
        deg = A.sum(1); sp = te[(yte == 1) & (deg[te[:, 0]] <= 1)]; sn = te[(yte == 0) & (deg[te[:, 0]] <= 1)]
        print(f'{ds} {k:7s} cover {cov:.2f} meandeg {R.sum(1).mean():5.2f} | hit(pos) {hp:.3f} hit(neg) {hn:.3f} | sparse(deg<=1): pos {(RA[sp[:,0],sp[:,1]]>0).mean():.3f} neg {(RA[sn[:,0],sn[:,1]]>0).mean():.3f}')
    print('names example', names[:12].tolist())
