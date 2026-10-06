"""Which ctx+diff LR-pipeline feature blocks carry information the GNN lacks? within-test 2-fold combiner."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
ds, tag = sys.argv[1], sys.argv[2]
sc = pd.read_csv(f'results/{tag}_{ds}_5cv_seed42_scores.csv'); X = []; L = []; Y = []
for f in range(5):
    z = np.load(f'../plmf_warm_20260922/cache/{ds}_s42_f{f}_ctx+diffK10_kernel10.npz')
    s = sc[sc.fold == f]; assert (s.m.values == z['te'][:, 0]).all()
    X.append(z['Xte']); p = s.score.clip(1e-6, 1 - 1e-6).values; L.append(np.log(p / (1 - p))); Y.append(s.y.values)
X = np.concatenate(X); L = np.concatenate(L); Y = np.concatenate(Y)
D = np.arange(33, 85).reshape(13, 4)     # diff: 13 matrices x (raw, rz, rpc, dz)
blocks = {'ctx': np.arange(33), 'diff_raw': D[:, 0], 'diff_rz': D[:, 1], 'diff_rpc': D[:, 2], 'diff_dz': D[:, 3], 'diff_all': D.ravel(), 'all': np.arange(85)}
def cv(M, reps=6):
    ga, gp = [], []
    for rep in range(reps):
        h = np.random.RandomState(rep).rand(len(Y)) < 0.5; out = np.zeros(len(Y))
        for m_ in (h, ~h):
            mu, sd = M[m_].mean(0), M[m_].std(0) + 1e-6
            out[~m_] = LogisticRegression(C=0.3, max_iter=4000).fit((M[m_] - mu) / sd, Y[m_]).decision_function((M[~m_] - mu) / sd)
        ga.append(roc_auc_score(Y, out) - roc_auc_score(Y, L)); gp.append(average_precision_score(Y, out) - average_precision_score(Y, L))
    return round(np.mean(ga), 4), round(np.mean(gp), 4)
print(ds, tag, 'base', round(roc_auc_score(Y, L), 4))
for k, idx in blocks.items():
    print(f'  +{k:9s}', cv(np.column_stack([L, X[:, idx]])), ' alone', cv(X[:, idx])[0])
names = ['degree_r', 'degree_d', 'degree_product', 'collab_r', 'collab_d', 'collab_r_mean', 'collab_d_mean', 'collab_both', 'path3', 'path3_norm', 'path5', 'path5_norm', 'svd16', 'neighbor_degree_r', 'neighbor_degree_d', 'bridge', 'bridge_norm', 'rna_similarity_degree', 'drug_similarity_degree', 'collab_r2', 'collab_d2', 'rna_attribute_support_sum', 'drug_attribute_support_sum', 'drug_attribute_profile_cos', 'rna_attribute_profile_cos', 'rna_feature_mean', 'rna_feature_std', 'rna_feature_norm', 'drug_feature_mean', 'drug_feature_std', 'drug_feature_norm', 'degree_MiMi', 'degree_DrugDrug']
dn = [f'{k}_{al}_{be}' for al, be in ((30, 10), (60, 50)) for k in ('ppr_rd', 'katz_rd', 'ppr_dr', 'katz_dr', 'path7')] + ['srA', 'ASd', 'srASd']
res = []
for i, n in enumerate(names):
    if X[:, i].std() > 0: res.append((n, *cv(np.column_stack([L, X[:, i]]), 3)))
for j, n in enumerate(dn):
    res.append(('D:' + n, *cv(np.column_stack([L, X[:, D[j]]]), 3)))
for r in sorted(res, key=lambda t: -t[1])[:18]: print('   ', r)
