import sys
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_r11_20260927')
sys.argv = ['x','--dataset','MDR','--split','5cv','--seed','42','--val','--folds','0','--tag','dbg','--enc','mask','--exact','--qis','--qis_ext','--qncn','--ncn_rel','co,knn,dknn,seed7,seed6,pre,fam','--rel','seed7,seed6,pre,fam','--ease','10,50','--ease_side','kmer','--ppr','--ppr_norm','--device','cuda:0']
import numpy as np, torch
import run_r11 as R
D = R.Data('MDR')
tr, ytr, te, yte = R.get_split('MDR', '5cv', 0, 42)
tp = tr[ytr == 1]
ps = R.PairStruct(D, torch.as_tensor(tp, device=R.DEV)); ps._diff_setup()
rng = np.random.RandomState(1); sel = rng.choice(len(tp), 4, replace=False)
r = torch.as_tensor(tp[sel, 0], device=R.DEV); d = torch.as_tensor(tp[sel, 1], device=R.DEV)
q = torch.ones(len(sel), device=R.DEV)
S = ps.diff_sets[0]
row_d, row_r = ps._diff_rowcol(S, r, d, q)
for j, (m0, d0) in enumerate(tp[sel]):
    A2 = ps.A.clone(); A2[m0, d0] = 0
    ps2 = R.PairStruct(D, torch.as_tensor(np.array([[m0, d0]]), device=R.DEV))
    ps2.A = A2; ps2.deg_r = A2.sum(1); ps2.deg_d = A2.sum(0); ps2._diff_setup()
    S2 = ps2.diff_sets[0]
    row_d2 = S2['pref'] * S2['AnMd'][torch.tensor([m0], device=R.DEV)]
    row_r2 = S2['pref'] * S2['AndMr'][torch.tensor([d0], device=R.DEV)]
    print(f'pair ({m0},{d0}) row_d max|diff| {float((row_d[j]-row_d2[0]).abs().max()):.2e}  row_r max|diff| {float((row_r[j]-row_r2[0]).abs().max()):.2e}')
