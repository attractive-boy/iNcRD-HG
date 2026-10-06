"""Query-invariant pair channels on the shared training graph.

M3 MC-EASE  masked-cell EASE scores (``ease_masked`` / ``PairStruct.ease``)
M2 SP-RNCN  pooled intermediate states of simple 2-hop paths (``PairStruct.qncn``)
M1 support  query-invariant graph statistics (``PairStruct.qis``) and the exact
            query-edge removal that the qncn/self-loop formulas build on.
"""
import torch
import torch.nn.functional as F


def ease_masked(X, P, u, i, q, mask=True):
    """Masked-cell EASE.

    items = columns of X, users = rows, P = (X^T X + lam I)^-1.
    score(u, i) = sum_{j != i} X[u, j] B[j, i], B[:, i] = -P'[:, i] / P'[i, i],
    where P' is the inverse Gram of X with the query cell (u, i) set to 0 (only
    for q = X[u, i] = 1; rank-2 Woodbury downdate).
    """
    x = X[u]                                                   # [B, I]
    Pi = P[i]                                                  # [B, I]  (P symmetric)
    ar = torch.arange(len(i), device=X.device)
    vPv = Pi[ar, i]
    plain_col = -Pi / vPv[:, None]
    if not mask:
        col = plain_col.clone()
        col[ar, i] = 0
        return (x * col).sum(1)
    Pu = x @ P
    uPu = (x * Pu).sum(1)
    uPv = Pu[ar, i]
    M11, M12, M22 = -1 + uPu, -1 + uPv, vPv
    det = M11 * M22 - M12 * M12
    s1 = (M22 * uPv - M12 * vPv) / det
    s2 = (-M12 * uPv + M11 * vPv) / det
    xcol = Pi - s1[:, None] * Pu - s2[:, None] * Pi
    new_col = -xcol / xcol[ar, i][:, None]
    col = torch.where(q[:, None] > 0.5, new_col, plain_col)
    col[ar, i] = 0
    xu = x.clone()
    xu[ar, i] = 0
    return (xu * col).sum(1)


class PairStruct:
    """Structural quantities of the message graph and the query-invariant pair channels."""

    def __init__(self, data, msg_pos, cfg):
        # The released own-data graph carries repeated association pairs, so the
        # dense adjacency counts copies; the mask formulas use q = A[r, d] and
        # stay exact (removing all copies of the query pair).
        A = torch.zeros(data.n_mir, data.n_drug, device=data.xr.device)
        A.index_put_((msg_pos[:, 0], msg_pos[:, 1]), torch.ones(len(msg_pos), device=A.device), accumulate=True)
        self.A = A
        self.deg_r = A.sum(1)
        self.deg_d = A.sum(0)
        self.Cm = A @ A.t()
        self.Cd = A.t() @ A
        self.P3 = self.Cm @ A
        self.AG = A @ self.Cd
        self.P5 = self.AG @ self.Cd
        self.ur = F.normalize(data.xk, dim=1)
        self.ud = F.normalize(data.xd, dim=1)
        self.Sr, self.Sd = data.Sr, data.Sd
        self.SrA = data.Sr @ A
        self.ASd = A @ data.Sd
        self.SrASd = data.Sr @ A @ data.Sd
        self.ease_P = []
        for lam in cfg.ease_lambdas:
            Gd = self.Cd + lam * torch.eye(data.n_drug, device=A.device)
            Gr = self.Cm + lam * torch.eye(data.n_mir, device=A.device)
            self.ease_P.append((torch.linalg.inv(Gd), torch.linalg.inv(Gr)))

    def qvals(self, r, d):
        return self.A[r, d]

    def qis(self, r, d):
        q = self.qvals(r, d)
        dr = self.deg_r[r] - q
        dd = self.deg_d[d] - q
        # simple 3-paths m-d'-m'-d (d' != d, m' != m): walks through the query edge are removed
        p3 = self.P3[r, d] - q * (self.deg_r[r] + self.deg_d[d] - 1)
        sra, asd, srasd = self.SrA[r, d], self.ASd[r, d], self.SrASd[r, d]   # zero-diagonal bridges
        f = [dr, dd, p3, p3 / (dr * dd + 1.0), p3 / (dr + 1.0), p3 / (dd + 1.0),
             sra, asd, srasd, sra / (dd + 1.0), asd / (dr + 1.0)]
        out = torch.log1p(torch.stack(f, 1).clamp(min=0))
        # masked 5-paths: with u = A[m], G = A^T A, removing the query edge gives
        # P5' = (G u - G e_d - (deg(m)-1) e_d)^T (G e_d - u)  (exact)
        u = self.A[r]
        Gd_ = self.Cd[d]
        Gu = self.AG[r]
        ed = F.one_hot(d, self.A.shape[1]).float()
        p5m = ((Gu - Gd_ - (self.deg_r[r] - 1).unsqueeze(1) * ed) * (Gd_ - u)).sum(1)
        p5 = torch.where(q > 0.5, p5m, self.P5[r, d])
        # mean degree of the other partners of m / of d (the query edge excluded)
        nbr_r = (u @ self.deg_d - q * self.deg_d[d]) / dr.clamp(min=1)
        nbr_d = (self.A[:, d].t() @ self.deg_r - q * self.deg_r[r]) / dd.clamp(min=1)
        # content profile of the other partners vs the query node (masked)
        pd_ = u @ self.ud - q.unsqueeze(1) * self.ud[d]
        pr_ = self.A[:, d].t() @ self.ur - q.unsqueeze(1) * self.ur[r]
        cos_d = (F.normalize(pd_, dim=1) * self.ud[d]).sum(1)
        cos_r = (F.normalize(pr_, dim=1) * self.ur[r]).sum(1)
        ext = torch.stack([torch.log1p(p5.clamp(min=0)),
                           torch.log1p((p5 / (dr * dd + 1.0)).clamp(min=0)),
                           torch.log1p(nbr_r), torch.log1p(nbr_d), cos_d, cos_r], 1)
        return torch.cat([out, ext], 1)

    def ease(self, r, d):
        q = self.qvals(r, d)
        out = []
        for Pd, Pr in self.ease_P:
            out.append(ease_masked(self.A, Pd, r, d, q))
            out.append(ease_masked(self.A.t(), Pr, d, r, q))
        x = torch.stack(out, 1)
        return torch.sign(x) * torch.log1p(x.abs() * 10.0)

    def qncn(self, r, d, z, corr=None):
        """Pooled encoder states of the intermediate nodes of simple 2-hop paths between r and d.

        corr (from ``QueryHollowEncoder.forward_dense``) shifts the intermediates'
        states to their values on the graph without the query edge: an
        intermediate miRNA m' hears d (A[m', d] = 1) and, if m is one of its kNN
        neighbours, m; an intermediate drug d' hears m (A[m, d'] = 1) and, if d
        is a kNN neighbour, d.
        """
        q = self.qvals(r, d).view(-1, 1)
        out = []
        if corr is not None:
            lin = lambda W, x: F.linear(x, W)
            cm_dm = 0.5 * lin(corr["W_dm"], corr["dh_d"])
            cm_mm = 0.5 * lin(corr["W_mm"], corr["dh_m"])
            cd_md = 0.5 * lin(corr["W_md"], corr["dh_m"])
            cd_dd = 0.5 * lin(corr["W_dd"], corr["dh_d"])

        def pool_m(w):
            den = w.sum(1, keepdim=True) + 1.0
            e = w @ z["miRNA"] / den
            if corr is not None:
                e = e + cm_dm * (w.sum(1, keepdim=True) / den) \
                      + cm_mm * ((w * self.Sr[:, r].t()).sum(1, keepdim=True) / den)
            return e

        def pool_d(w):
            den = w.sum(1, keepdim=True) + 1.0
            e = w @ z["drug"] / den
            if corr is not None:
                e = e + cd_md * (w.sum(1, keepdim=True) / den) \
                      + cd_dd * ((w * self.Sd[:, d].t()).sum(1, keepdim=True) / den)
            return e

        for k in ("co", "knn", "dknn"):
            if k == "co":
                # miRNA side: m' != m with A[m', d] = 1 and a shared drug d'' != d with m
                Ad = self.A[:, d].t()
                w = Ad * (self.Cm[r] - q * Ad)
                w[torch.arange(len(r), device=self.A.device), r] = 0
                out.append(pool_m(w))
                # drug side: d' != d with A[m, d'] = 1 and a shared miRNA m'' != m with d
                Ar = self.A[r]
                w = Ar * (self.Cd[:, d].t() - q * Ar)
                w[torch.arange(len(d), device=self.A.device), d] = 0
                out.append(pool_d(w))
            elif k == "knn":
                out.append(pool_m(self.A[:, d].t() * self.Sr[r]))
            else:
                out.append(pool_d(self.A[r] * self.Sd[:, d].t()))
        return out

    def ncn_widths(self, cfg):
        w = []
        for k in cfg.ncn_rels:
            w += [cfg.ncn_dim, cfg.ncn_dim] if k == "co" else [cfg.ncn_dim]
        return w
