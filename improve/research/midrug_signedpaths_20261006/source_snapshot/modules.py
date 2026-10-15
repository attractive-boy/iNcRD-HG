"""Degree-corrected M2 and signed, exactly query-masked M3 channels."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[2] / "release/iNcRD-HG-M1M2M3"
sys.path.insert(0, str(RELEASE))
from incrd_hg_m123.data import load_fold, build_graph, structure_data, sha256
from incrd_hg_m123.pair_channels import PairStruct


def signed_ease(X, P, r, d):
    """EASE after setting X[r,d] to zero for arbitrary signed cell values.

    The Gram update has U=[X[r,:], e_d], C=[[0,-q],[-q,q*q]].
    For q != 0, C^-1=[[-1,-1/q],[-1/q,0]]. q=0 uses the unmodified inverse.
    """
    q = X[r, d]
    ix = torch.arange(len(r), device=X.device)
    x, Pi = X[r], P[d]
    Pii = Pi[ix, d]
    xP = x @ P
    xPx = (x*xP).sum(1)
    xPe = xP[ix, d]
    qs = torch.where(q != 0, q, torch.ones_like(q))
    a, b, c = xPx-1, xPe-1/qs, Pii
    det = a*c-b*b
    c1, c2 = (c*xPe-b*Pii)/det, (-b*xPe+a*Pii)/det
    pnew = Pi-c1[:,None]*xP-c2[:,None]*Pi
    col = torch.where((q != 0)[:,None], -pnew/pnew[ix,d,None], -Pi/Pii[:,None])
    col[ix, d] = 0
    x = x.clone()
    x[ix, d] = 0
    return (x*col).sum(1)


def masked_walks(X, pairs, alpha, beta, lengths=(3,5,7)):
    """Weighted alternating walks on X with each query cell removed.

    Matrix multiplication is corrected at every step; per-query degrees are
    also corrected. No dense (batch, n_rna, n_drug) graph copies are required.
    Degree uses |X| for signed evidence, not a potentially negative row sum.
    """
    r, d = pairs.T
    ix = torch.arange(len(r), device=X.device)
    q = X[r,d]
    dr, dd = X.abs().sum(1), X.abs().sum(0)
    wr = dr.clamp(min=1).pow(-alpha).expand(len(r), -1).clone()
    wd = dd.clamp(min=1).pow(-beta).expand(len(r), -1).clone()
    wr[ix,r] = (dr[r]-q.abs()).clamp(min=1).pow(-alpha)
    wd[ix,d] = (dd[d]-q.abs()).clamp(min=1).pow(-beta)
    v = X[r].clone()
    v[ix,d] = 0
    result = []
    for length in range(3, max(lengths)+1, 2):
        v = v * wd
        u = v @ X.T
        u[ix,r] -= q*v[ix,d]
        u = u * wr
        v = u @ X
        v[ix,d] -= q*u[ix,r]
        if length in lengths:
            result.append(v[ix,d].clone())
    return torch.stack(result, 1)


class Evidence:
    """Frozen training evidence; each query's own supervision is removed."""
    def __init__(self, pairs, labels, shape, device):
        self.A = torch.zeros(shape, dtype=torch.float64, device=device)
        self.N = torch.zeros_like(self.A)
        p = torch.as_tensor(pairs, device=device)
        y = torch.as_tensor(labels, device=device)
        self.A[p[y==1,0],p[y==1,1]] = 1
        self.N[p[y==0,0],p[y==0,1]] = 1

    def features(self, pairs):
        pairs = torch.as_tensor(pairs, device=self.A.device)
        r, d = pairs.T
        names, columns = [], []
        def add(name, v):
            names.append(name)
            columns.append(torch.sign(v)*torch.log1p(v.abs()*10))
        for kind, X in (("positive", self.A), ("negative", self.N),
                        ("signed01", self.A-.1*self.N), ("signed03", self.A-.3*self.N),
                        ("signed1", self.A-self.N)):
            for lam in (1.,10.,50.):
                for side, T, u, i in (("d",X,r,d), ("r",X.T,d,r)):
                    P = torch.linalg.inv(T.T @ T + lam*torch.eye(T.shape[1], device=T.device,dtype=T.dtype))
                    v = signed_ease(T,P,u,i)
                    add(f"ease_{kind}_{side}_{lam}", v)
            for alpha, beta in ((0.,1.), (1.,0.),(.5,.5),(1.,1.)):
                walk = masked_walks(X,pairs,alpha,beta)
                for j, length in enumerate((3,5,7)):
                    add(f"walk_{kind}_{alpha}_{beta}_len{length}",walk[:,j])
        return torch.stack(columns,1).float(), names


class DegreePairStruct(PairStruct):
    """M2 interior path weights discount hubs; M1 corrections stay exact."""
    def __init__(self, data, msg_pos, cfg, power=0.):
        super().__init__(data,msg_pos,cfg)
        self.power = power
        self.wr = self.deg_r.clamp(min=1).pow(-power)
        self.wd = self.deg_d.clamp(min=1).pow(-power)
        self.Cmw = (self.A*self.wd) @ self.A.T
        self.Cdw = self.A.T @ (self.wr[:,None]*self.A)

    def qncn(self,r,d,z,corr=None):
        q = self.qvals(r,d)[:,None]
        if corr is not None:
            cm_dm = .5*F.linear(corr["dh_d"],corr["W_dm"])
            cm_mm = .5*F.linear(corr["dh_m"],corr["W_mm"])
            cd_md = .5*F.linear(corr["dh_m"],corr["W_md"])
            cd_dd = .5*F.linear(corr["dh_d"],corr["W_dd"])
        def pool_m(w):
            den = w.sum(1,keepdim=True)+1
            value = w @ z["miRNA"] / den
            if corr is not None:
                value = value + cm_dm*(w.sum(1,keepdim=True)/den) + cm_mm*((w*self.Sr[:,r].T).sum(1,keepdim=True)/den)
            return value
        def pool_d(w):
            den = w.sum(1,keepdim=True)+1
            value = w @ z["drug"] / den
            if corr is not None:
                value = value + cd_md*(w.sum(1,keepdim=True)/den) + cd_dd*((w*self.Sd[:,d].T).sum(1,keepdim=True)/den)
            return value
        ix = torch.arange(len(r),device=r.device)
        Ad, Ar = self.A[:,d].T, self.A[r]
        wm = Ad*(self.Cmw[r]-q*Ad*self.wd[d,None])*self.wr
        wd = Ar*(self.Cdw[:,d].T-q*Ar*self.wr[r,None])*self.wd
        wm[ix,r], wd[ix,d] = 0, 0
        return [pool_m(wm),pool_d(wd),pool_m(Ad*self.Sr[r]*self.wr),pool_d(Ar*self.Sd[:,d].T*self.wd)]
