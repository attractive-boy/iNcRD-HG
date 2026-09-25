"""Compact prior-data fitted LINK predictor and explicit graph priors.

Tokens are typed RNA--drug pairs, context labels are observed associations,
and synthetic query targets are latent complete labels. No entity embedding
table, query-to-query attention, test labels, or GraphPFN weights are used.
This is a small from-scratch PFN prototype, not the official GraphPFN model.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ATTR_DIM = 16
STRUCT_NAMES = (
    'rna_degree', 'drug_degree', 'rna_degree_rate', 'drug_degree_rate',
    'rna_side_degree', 'drug_side_degree', 'cross_rna_degree', 'other_drug_degree',
    'rna_neighbor_support', 'drug_neighbor_support', 'both_neighbor_support',
    'three_walk', 'cross_type_support', 'rna_neighbor_mean', 'drug_neighbor_mean',
    'three_walk_normalized', 'cross_type_mean', 'rna_profile_cos',
    'drug_profile_cos', 'rna_attribute_support', 'drug_attribute_support',
    'rna_neighbor_degree', 'drug_neighbor_degree', 'rna_has_support', 'drug_has_support',
)
FEATURE_NAMES = (list(STRUCT_NAMES) + ['rna_attr_'+str(i) for i in range(ATTR_DIM)]
                 + ['drug_attr_'+str(i) for i in range(ATTR_DIM)] + ['task_lnc', 'task_mi'])
FEATURE_DIM = len(FEATURE_NAMES)
PRIORS = ('random', 'hetero', 'missing')


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def observed_matrix(pairs, labels, nr, nd, allowed_drugs=None):
    """Construct BOTH relation directions from this one positive matrix."""
    a = np.zeros((nr, nd), dtype=np.float32)
    keep = np.asarray(labels) == 1
    if allowed_drugs is not None:
        keep &= np.isin(pairs[1], allowed_drugs)
    a[tuple(pairs[:, keep])] = 1.
    return a


def hide_drugs(associations, hidden):
    """All RNA types lose the same drug's labels, including reverse views."""
    out = []
    for a in associations:
        b = a.copy()
        b[:, hidden] = 0.
        out.append(b)
    return out


class LinkFeatures:
    """Label-dependent descriptors use only explicitly supplied associations.

    Side relations and attributes are fixed inputs. There is no hidden access
    to a raw graph or to a validation partition. Reverse association matrices
    are transposes, so a removed label cannot survive in a reverse store.
    Context tokens may describe known positives; unknown queries may not.
    """
    def __init__(self, rna, drug, rna_side, drug_side, cross, target, other, task):
        self.rna, self.drug = np.asarray(rna, np.float32), np.asarray(drug, np.float32)
        self.a, self.task = np.asarray(target, np.float32), int(task)
        nr, nd = self.a.shape
        if self.rna.shape != (nr, ATTR_DIM) or self.drug.shape != (nd, ATTR_DIM):
            raise ValueError('Endpoint feature dimensions')
        sr, sd, cross = rna_side.copy(), drug_side.copy(), cross.copy()
        np.fill_diagonal(sr, 0.)
        np.fill_diagonal(sd, 0.)
        a = self.a
        dr, dd = a.sum(1), a.sum(0)
        nsr, nsd, nc = sr.sum(1), sd.sum(0), cross.sum(1)
        ra, ad, bridge = sr @ a, a @ sd, cross @ other
        walk = a @ (a.T @ a)
        ur, ud = unit(self.rna), unit(self.drug)
        rprof, dprof = unit(a.T @ ur), unit(a @ ud)
        er, ed = ur @ ur.T, ud @ ud.T
        np.fill_diagonal(er, 0.)
        np.fill_diagonal(ed, 0.)
        self.ur, self.ud = ur.astype(np.float32), ud.astype(np.float32)
        self.r_fields = (dr, dr / nd, nsr, nc, sr @ dr, (dr > 0).astype(np.float32))
        self.d_fields = (dd, dd / nr, nsd, other.sum(0), sd @ dd, (dd > 0).astype(np.float32))
        self.matrices = (ra, ad, ra @ sd, walk, bridge,
                         ra / np.maximum(nsr[:, None], 1),
                         ad / np.maximum(nsd[None, :], 1),
                         walk / np.maximum(dr[:, None]*dd[None, :], 1),
                         bridge / np.maximum(nc[:, None], 1),
                         ur @ rprof.T, dprof @ ud.T, er @ a, a @ ed)

    def transform(self, pairs, context=False):
        pairs = np.asarray(pairs, dtype=np.int64)
        r, d = pairs
        if not context and np.any(self.a[r, d]):
            raise ValueError('A query association is visible to its own predictor')
        rf, df = self.r_fields, self.d_fields
        fields = [rf[0][r], df[0][d], rf[1][r], df[1][d], rf[2][r], df[2][d], rf[3][r], df[3][d]]
        fields.extend(m[r, d] for m in self.matrices)
        fields.extend([rf[4][r], df[4][d], rf[5][r], df[5][d]])
        x = np.stack(fields, axis=1)
        # Signed log compresses counts while preserving negative cosine sums.
        x = np.sign(x) * np.log1p(np.abs(x))
        task = np.zeros((len(r), 2), dtype=np.float32)
        task[:, self.task] = 1.
        return np.concatenate([x, self.rna[r], self.drug[d], task], axis=1).astype(np.float32)


def balanced_indices(y, size, rng, priority=None):
    """Fixed-size context, positive/unobserved 1:2; retain supplied supports."""
    y = np.asarray(y)
    priority = np.unique(np.asarray([] if priority is None else priority, dtype=np.int64))
    if len(priority) > size:
        raise ValueError('Context budget cannot contain all required support rows')
    result = list(priority)
    npositive = max(0, size // 3 - int((y[priority] == 1).sum()))
    for cls, want in ((1, npositive), (0, size-len(result)-npositive)):
        pool = np.setdiff1d(np.flatnonzero(y == cls), priority, assume_unique=False)
        if not len(pool):
            pool = np.flatnonzero(y == cls)
        if len(pool) and want > 0:
            result.extend(rng.choice(pool, want, replace=want > len(pool)).tolist())
    if len(result) < size:
        result.extend(rng.choice(len(y), size-len(result), replace=True).tolist())
    result = np.asarray(result[:size], dtype=np.int64)
    rng.shuffle(result)
    return result


def tensor_batch(cx, cy, cp, qx, qp, device):
    """Support-only normalization; queries cannot affect one another."""
    def tensor(x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    cx, qx = tensor(cx), tensor(qx)
    if cx.ndim == 2:
        cx, qx = cx[None], qx[None]
        cy, cp, qp = np.asarray(cy)[None], np.asarray(cp)[None], np.asarray(qp)[None]
    mean = cx.mean(1, keepdim=True)
    scale = cx.std(1, keepdim=True, unbiased=False).clamp_min(.1)
    nc, nq = ((cx-mean)/scale).clamp(-8, 8), ((qx-mean)/scale).clamp(-8, 8)
    nc[..., -2:], nq[..., -2:] = cx[..., -2:], qx[..., -2:]
    return dict(cx=nc, cy=tensor(cy, torch.long),
                cp=tensor(cp, torch.long), qx=nq,
                qp=tensor(qp, torch.long), raw_cx=cx, raw_qx=qx)


def relation_bias(qp, cp, qraw, craw):
    """Permutation-equivariant endpoint relations, without learned entity IDs."""
    start = len(STRUCT_NAMES)
    qr, cr = F.normalize(qraw[..., start:start+ATTR_DIM], dim=-1), F.normalize(craw[..., start:start+ATTR_DIM], dim=-1)
    qd, cd = F.normalize(qraw[..., start+ATTR_DIM:start+2*ATTR_DIM], dim=-1), F.normalize(craw[..., start+ATTR_DIM:start+2*ATTR_DIM], dim=-1)
    same_r = (qp[:, :, None, 0] == cp[:, None, :, 0]).to(qraw.dtype)
    same_d = (qp[:, :, None, 1] == cp[:, None, :, 1]).to(qraw.dtype)
    rcos, dcos = qr @ cr.transpose(1, 2), qd @ cd.transpose(1, 2)
    return torch.stack([same_r, same_d, rcos, dcos, rcos*dcos], dim=-1)


class LinkAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2*width)
        self.out = nn.Linear(width, width)
        self.relation = nn.Parameter(torch.zeros(heads, 5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, context, bias):
        b, n, w = q.shape
        c = context.shape[1]
        qq = self.q(q).reshape(b, n, self.heads, self.dim).transpose(1, 2)
        k, v = self.kv(context).reshape(b, c, 2, self.heads, self.dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        scores = qq @ k.transpose(-1, -2) / math.sqrt(self.dim)
        scores = scores + torch.einsum('bncf,hf->bhnc', bias, self.relation)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self.out((attn @ v).transpose(1, 2).reshape(b, n, w))


class LinkBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = LinkAttention(width, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2*width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, c, bias):
        q = q + self.drop(self.attention(self.norm1(q), self.norm1(c), bias))
        return q + self.drop(self.ff(self.norm2(q)))


class LinkPFN(nn.Module):
    def __init__(self, width=64, depth=2, heads=4, dropout=.1):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(FEATURE_DIM, width), nn.GELU(), nn.LayerNorm(width))
        self.label = nn.Embedding(3, width)
        self.blocks = nn.ModuleList([LinkBlock(width, heads, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, cx, cy, cp, qx, qp, raw_cx, raw_qx, intervention=None):
        labels = torch.full_like(cy, 2) if intervention == 'hide_context_labels' else cy
        if intervention == 'shuffle_context_labels':
            labels = labels.roll(17, dims=1)
        c, q = self.input(cx) + self.label(labels), self.input(qx) + self.label.weight[2]
        cc = relation_bias(cp, cp, raw_cx, raw_cx)
        qc = relation_bias(qp, cp, raw_qx, raw_cx)
        if intervention == 'no_relational_bias':
            cc, qc = torch.zeros_like(cc), torch.zeros_like(qc)
        for block in self.blocks:
            # Simultaneous update: no query can become another query's context.
            qnew = block(q, c, qc)
            c = block(c, c, cc)
            q = qnew
        return self.head(q).squeeze(-1)


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -25, 25)))


def _graph(z, w, density, rng, coupled=True):
    signal = z @ w.T / math.sqrt(z.shape[1]) if coupled else rng.normal(size=(len(z), len(w)))
    threshold = np.quantile(signal, 1-density)
    return (signal + rng.normal(scale=.4, size=signal.shape) > threshold).astype(np.float32)


def synthetic_episode(seed, prior, context_size=128, query_size=64, shift=False, false_positive=0.):
    """Paired hetero/missing graphs differ only in the observation process.

    Context zeros include unobserved true positives. Query truth is NEVER
    passed to LinkFeatures. Missingness has exactly the same positive count
    as MCAR whenever the nonzero-exposure pool permits it.
    """
    if prior not in PRIORS:
        raise ValueError(prior)
    rng = np.random.RandomState(seed)
    nr, no, nd = int(rng.choice([48, 64, 80])), int(rng.choice([24, 32, 48])), int(rng.choice([24, 32, 40]))
    latent = int(rng.choice([3, 5, 8]))
    clusters = int(rng.choice([3, 4, 6]))
    centers = rng.normal(size=(clusters, latent))
    zs = [centers[rng.randint(clusters, size=n)] + rng.normal(scale=.65, size=(n, latent)) for n in (nr, no, nd)]
    zr, zo, zd = zs
    attributes = []
    coupling = rng.uniform(.4, 1.8) if not shift else rng.uniform(.2, 1.)
    for z in zs:
        x = z @ rng.normal(size=(latent, ATTR_DIM)) / math.sqrt(latent)
        x = np.tanh(x) + rng.normal(scale=coupling, size=x.shape)
        x = (x-x.mean(0)) / np.maximum(x.std(0), .1)
        attributes.append(x.astype(np.float32))
    xr, xo, xd = attributes
    density = rng.uniform(.05, .2) if not shift else rng.uniform(.02, .08)
    coupled = prior != 'random'
    sr, sd, cross = (_graph(zr, zr, density, rng, coupled), _graph(zd, zd, density, rng, coupled),
                     _graph(zr, zo, density, rng, coupled))
    np.fill_diagonal(sr, 0.)
    np.fill_diagonal(sd, 0.)
    rd, od, dd = rng.normal(size=nr), rng.normal(size=no), rng.normal(size=nd)
    prevalence = rng.uniform(.06, .28)
    interaction = rng.normal(size=(latent, latent)) / math.sqrt(latent)
    if prior == 'random':
        wr, wd = rng.normal(size=(ATTR_DIM, 12)), rng.normal(size=(ATTR_DIM, 12))
        srule = np.tanh(xr @ wr[:, :12] / 4) @ np.tanh(xd @ wd[:, :12] / 4).T / math.sqrt(12)
        orule = np.tanh(xo @ wr[:, :12] / 4) @ np.tanh(xd @ wd[:, :12] / 4).T / math.sqrt(12)
    else:
        srule, orule = zr @ interaction @ zd.T / math.sqrt(latent), zo @ interaction @ zd.T / math.sqrt(latent)
    noise = rng.uniform(.5, 1.5)
    true = []
    for rule, rbias in ((srule, rd), (orule, od)):
        logits = noise*rule + rng.uniform(.1, 1.2)*(rbias[:, None]+dd[None])
        logits -= np.quantile(logits, 1-prevalence)
        true.append((rng.rand(*logits.shape) < _sigmoid(logits-1.)).astype(np.float32))
    ytrue, other_true = true
    cold = rng.rand() < .6
    qpool = np.zeros((nr, nd), dtype=bool)
    held = rng.choice(nd, max(2, nd//4), replace=False) if cold else np.array([], dtype=np.int64)
    if cold:
        qpool[:, held] = True
    else:
        qpool = rng.rand(nr, nd) < .25
    # Stratified queries with latent truth; observed graph removes the ENTIRE
    # candidate query pool, not just the sampled evaluation rows.
    qp = []
    for cls, count in ((1, query_size//3), (0, query_size-query_size//3)):
        pool = np.flatnonzero(qpool & (ytrue == cls))
        if not len(pool):
            return synthetic_episode(seed+10000019, prior, context_size, query_size, shift, false_positive)
        qp.extend(rng.choice(pool, count, replace=len(pool)<count))
    qp = np.asarray(qp, dtype=np.int64)
    rng.shuffle(qp)
    qp = np.stack(np.unravel_index(qp, ytrue.shape), axis=1)
    retention = rng.uniform(.2, .8) if not shift else rng.uniform(.06, .25)
    # A separate observation RNG leaves hetero and missing latent tasks paired.
    obs_rng = np.random.RandomState(seed ^ 193091)
    exposure_strength = (3. if shift else 1.8)
    observed, observation = [], []
    for i, (truth, rbias) in enumerate(((ytrue, rd), (other_true, od))):
        available = np.ones_like(truth, dtype=bool)
        if i == 0:
            available &= ~qpool
        if cold:
            available[:, held] = False
        pos = np.flatnonzero((truth == 1) & available)
        budget = int(round(retention*len(pos)))
        uniform = obs_rng.random_sample(len(pos))
        rpos, dpos = np.unravel_index(pos, truth.shape)
        weights = np.ones(len(pos))
        if prior == 'missing':
            weights = np.exp(np.clip(exposure_strength*(dd[dpos]+.5*rbias[rpos]) + .5*attributes[i][rpos, 0], -8, 8))
            # Some drug studies have no exposure. At least one pool remains.
            no_study = np.argsort(dd)[:max(1, nd//8)]
            weights[np.isin(dpos, no_study)] = 0.
        key = np.log(np.maximum(uniform, 1e-12)) / np.maximum(weights, 1e-12)
        actual = min(budget, int((weights > 0).sum()))
        chosen = pos[np.argsort(key)[-actual:]] if actual else np.array([], dtype=np.int64)
        a = np.zeros_like(truth)
        a.ravel()[chosen] = 1.
        if false_positive:
            eligible_zero = np.flatnonzero((truth == 0) & available)
            corrupt = int(round(false_positive*actual))
            if corrupt:
                a.ravel()[obs_rng.choice(eligible_zero, min(corrupt,len(eligible_zero)),replace=False)] = 1.
        observed.append(a)
        observation.append(dict(available_true_positive=len(pos),observed_positive=actual,requested_positive=budget))
    a, other = observed
    context_pool = np.flatnonzero(~qpool)
    cx_labels = a.ravel()[context_pool]
    ci = balanced_indices(cx_labels, context_size, rng)
    cp = np.stack(np.unravel_index(context_pool[ci], a.shape), axis=1)
    task = seed % 2
    f = LinkFeatures(xr, xd, sr, sd, cross, a, other, task)
    cx, qx = f.transform(cp.T, context=True), f.transform(qp.T)
    cy, qy = a[tuple(cp.T)].astype(np.int64), ytrue[tuple(qp.T)].astype(np.float32)
    meta = dict(seed=seed,prior=prior,cold=cold,query_support_overlap=int(a[tuple(qp.T)].sum()),
                hidden_drugs=held.tolist(),context_false_zero=int(((cy == 0) & (ytrue[tuple(cp.T)] == 1)).sum()),
                shape=[nr,no,nd],observation=observation,truth_checksum=float(ytrue.sum()+other_true.sum()))
    return dict(cx=cx,cy=cy,cp=cp,qx=qx,qp=qp,qy=qy), meta
