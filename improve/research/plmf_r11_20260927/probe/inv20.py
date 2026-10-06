import sys
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_r11_20260927')
sys.argv = ['x','--dataset','MDR','--split','5cv','--seed','42','--val','--folds','0','--tag','dbg','--enc','mask','--exact','--qis','--qis_ext','--qncn','--ncn_rel','co,knn,dknn,seed7,seed6,pre,fam','--rel','seed7,seed6,pre,fam','--ease','10,50','--ease_side','kmer','--ppr','--ppr_norm','--device','cuda:0']
import numpy as np, torch
import run_r11 as R
D = R.Data('MDR')
tr, ytr, te, yte = R.get_split('MDR', '5cv', 0, 42)
tp = tr[ytr == 1]
ps = R.PairStruct(D, torch.as_tensor(tp, device=R.DEV)); ps._diff_setup()
rng = np.random.RandomState(42*100+0)
vidx=[]
for c in (0,1):
    ids=np.where(ytr==c)[0]; rng.shuffle(ids); vidx.append(ids[:int(round(0.1*len(ids)))])
vidx=np.concatenate(vidx); mask=np.ones(len(ytr),bool); mask[vidx]=False
tr,ytr=tr[mask],ytr[mask]
pos_idx = np.where(ytr==1)[0][:20]
e1 = torch.as_tensor(tr[pos_idx].T, dtype=torch.long, device=R.DEV)
v1 = ps.diff(*e1)                       # batched masked
mx = 0
for j,k in enumerate(pos_idx):
    m0,d0 = tr[k]
    keep = ~((tp[:,0]==m0)&(tp[:,1]==d0)); msg2 = tp[keep]
    ps2 = R.PairStruct(D, torch.as_tensor(msg2, device=R.DEV))
    ps2._diff_setup()
    v2 = ps2.diff(torch.tensor([m0],device=R.DEV), torch.tensor([d0],device=R.DEV))
    mx = max(mx, float((v1[j]-v2[0]).abs().max()))
pass
diffs = []
for j,k in enumerate(pos_idx):
    m0,d0 = tr[k]
    keep = ~((tp[:,0]==m0)&(tp[:,1]==d0)); msg2 = tp[keep]
    ps2 = R.PairStruct(D, torch.as_tensor(msg2, device=R.DEV)); ps2._diff_setup()
    v2 = ps2.diff(torch.tensor([m0],device=R.DEV), torch.tensor([d0],device=R.DEV))
    diffs.append((float((v1[j]-v2[0]).abs().max()), int(m0), int(d0)))
diffs.sort(reverse=True)
for d,m0,d0 in diffs[:5]:
    print(f'pair ({m0},{d0}) maxdiff {d:.3e}')
# recompute the worst pair alone (batch of 1)
d,m0,d0 = diffs[0]
v1b = ps.diff(torch.tensor([m0],device=R.DEV), torch.tensor([d0],device=R.DEV))
print('batch-of-1 vs batch-of-20 for worst pair:', float((v1b[0]-v1[pos_idx.tolist().index(np.where((tr[:,0]==m0)&(tr[:,1]==d0))[0][0])]).abs().max()))
