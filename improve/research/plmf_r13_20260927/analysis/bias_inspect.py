"""Which entities get large post-hoc random-effect intercepts (Fd dumps, fold 0..4, seed 42)?"""
import sys, numpy as np, torch, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import load_dataset
ds = sys.argv[1]
mir, dru, pos, neg = load_dataset(ds); nm, nd = len(mir), len(dru)
names = mir.miRNA_Name.astype(str).values; dbs = dru.DrugBank_ID.astype(str).values
Af = np.zeros((nm, nd)); Af[pos[:, 0], pos[:, 1]] = 1
BM, BD = [], []
for f in range(5):
    z = np.load(f'cache/dump_Fd_{ds}_5cv_s42_f{f}.npz')
    P = np.concatenate([z['tr'], z['va']]); Y = torch.tensor(np.concatenate([z['ytr'], z['yva']]), dtype=torch.float64)
    L = torch.tensor(np.concatenate([z['logit_tr'], z['logit_va']]), dtype=torch.float64); m = torch.tensor(P[:, 0]); d = torch.tensor(P[:, 1])
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([bm, bd], max_iter=500, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(L + bm[m] + bd[d], Y, reduction='sum') + 10 * (bm.pow(2).sum() + bd.pow(2).sum()); l.backward(); return l
    opt.step(cl); BM.append(bm.detach().numpy()); BD.append(bd.detach().numpy())
BM, BD = np.array(BM), np.array(BD)
dd = pd.DataFrame(dict(drug=dbs, deg=Af.sum(0), bias=BD.mean(0), sd=BD.std(0)))
print('drug bias: corr with log deg %.3f; fold-consistency (mean |mean|/sd) %.2f' % (np.corrcoef(np.log1p(dd.deg), dd.bias)[0, 1], (dd.bias.abs() / (dd.sd + 1e-6)).mean()))
print(dd.sort_values('bias').head(8).round(3).to_string()); print(dd.sort_values('bias').tail(8).round(3).to_string())
mm = pd.DataFrame(dict(mir=names, deg=Af.sum(1), bias=BM.mean(0), sd=BM.std(0)))
print('miRNA bias: corr with log deg %.3f; sd across folds mean %.3f; |bias| mean %.3f' % (np.corrcoef(np.log1p(mm.deg), mm.bias)[0, 1], mm.sd.mean(), mm.bias.abs().mean()))
print(mm.sort_values('bias').head(8).round(3).to_string()); print(mm.sort_values('bias').tail(8).round(3).to_string())
mm['arm'] = [('5p' if n.endswith('5p') else '3p' if n.endswith('3p') else 'bare') for n in names]
print(mm.groupby('arm').bias.mean().round(4))
