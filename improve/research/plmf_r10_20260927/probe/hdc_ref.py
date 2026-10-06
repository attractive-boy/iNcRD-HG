"""verify HollowDiffusion rows/cols against bank.diffusion on the edge-deleted graph (numpy reference)"""
import sys, numpy as np, torch
sys.argv = ['x', '--dataset', 'MDR', '--split', '5cv', '--hdc', 'ppr30,katz60,path7', '--device', 'cuda:1']
sys.path.insert(0, '.'); sys.path.insert(0, '../plmf_innov_20260921'); sys.path.insert(0, '../dlstmda_strongest_20260920')
import run_r10 as R
from bank import diffusion
from plmf_splits import get_split
tr, ytr, te, yte = get_split('MDR', '5cv', 0, 42)
tp = tr[ytr == 1]
A = torch.zeros(1317, 105, device=R.DEV); A[tp[:, 0], tp[:, 1]] = 1
hd = R.HollowDiffusion(A)
for (m, d) in [tuple(tp[3]), tuple(tp[77]), tuple(te[0])]:
    q = A[m, d]
    A2 = A.clone(); A2[m, d] = 0; D = diffusion(A2.cpu().numpy().astype(float), 0.3, 0.01)
    r, dd = torch.tensor([m], device=R.DEV), torch.tensor([d], device=R.DEV)
    cs = hd.coeffs('ppr', 0.3, 12)
    v, row, col = hd._walk(hd.A, r, dd, A[r, dd].double(), cs)
    print('ppr_rd', float(q), abs(row[0].cpu().numpy() - D['ppr_rd'][m]).max(), abs(col[0].cpu().numpy() - D['ppr_rd'][:, d]).max())
    v, col2, row2 = hd._walk(hd.A.t().contiguous(), dd, r, A[r, dd].double(), cs)
    print('ppr_dr', abs(row2[0].cpu().numpy() - D['ppr_dr'][m]).max(), abs(col2[0].cpu().numpy() - D['ppr_dr'][:, d]).max())
    D2 = diffusion(A2.cpu().numpy().astype(float), 0.6, 0.05)
    v, row, col = hd._walk(hd.A, r, dd, A[r, dd].double(), hd.coeffs('katz', 0.05, 12))
    print('katz_rd', abs(row[0].cpu().numpy() - D2['katz_rd'][m]).max() / abs(D2['katz_rd'][m]).max())
    v, row, col = hd._walk(hd.A, r, dd, A[r, dd].double(), [0, 0, 0, 1], 'raw')
    print('path7', abs(row[0].cpu().numpy() - D['path7'][m]).max() / D['path7'][m].max(), abs(col[0].cpu().numpy() - D['path7'][:, d]).max() / D['path7'][:, d].max())
