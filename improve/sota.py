# LightGCN and reliable negative sampling, from this literature's 2022-2025 line.
#
# LightGCN (He et al. 2020; used for ncRNA-drug by GSLRDA 2022) strips the GNN
# to symmetric-normalised propagation over learnable ID embeddings -- no feature
# transform, no nonlinearity, layer outputs averaged, inner-product decoder.
# It is the strongest bet available on THIS protocol precisely because it is
# closest to a smoothed degree/co-occurrence model, and 99.7% of the metric is
# a degree comparison.
#
# Reliable negative sampling (DMGAT 2025 uses a random forest; LSNSCDA 2024
# calls it credible negative sampling) replaces uniform negatives with the
# candidates a preliminary model scores most confidently negative.
import numpy as np
import torch
import torch.nn.functional as F


class LightGCN(torch.nn.Module):
    """LightGCN over the full heterogeneous graph, flattened to one adjacency.

    Every node type gets its own embedding table; all relation types are poured
    into a single symmetric-normalised adjacency, which is what LightGCN's
    propagation assumes.  Relation identity is therefore discarded on purpose --
    that is the method, and it is the control the paper's relation-specific
    parameters have to beat.
    """

    def __init__(self, data, hidden_channels, rna_type, n_layers=3,
                 decoder='dot'):
        super().__init__()
        self.node_types = list(data.node_types)
        self.rna_type = rna_type
        self.n_layers = n_layers
        self.decoder_mode = decoder
        self.offset, off = {}, 0
        for nt in self.node_types:
            self.offset[nt] = off
            off += data[nt].num_nodes
        self.n_total = off
        self.emb = torch.nn.Embedding(off, hidden_channels)
        torch.nn.init.normal_(self.emb.weight, std=0.1)
        if decoder == 'mlp':
            from improve.model import EdgeDecoder
            self.mlp = EdgeDecoder(hidden_channels, rna_type, mode='concat')
        self._adj = None

    def _build_adj(self, data):
        rows, cols = [], []
        for (src, _, dst), ei in data.edge_index_dict.items():
            r = ei[0] + self.offset[src]
            c = ei[1] + self.offset[dst]
            rows += [r, c]          # symmetric
            cols += [c, r]
        row = torch.cat(rows)
        col = torch.cat(cols)
        val = torch.ones(row.numel(), device=row.device)
        deg = torch.zeros(self.n_total, device=row.device).scatter_add_(
            0, row, val)
        d = deg.clamp(min=1).pow(-0.5)
        w = d[row] * d[col]
        return torch.sparse_coo_tensor(
            torch.stack([row, col]), w,
            (self.n_total, self.n_total)).coalesce()

    def forward(self, x_dict, edge_index_dict, edge_label_index, data=None):
        if self._adj is None:
            self._adj = self._build_adj(data)
        e = self.emb.weight
        out = e
        for _ in range(self.n_layers):
            e = torch.sparse.mm(self._adj, e)
            out = out + e
        out = out / (self.n_layers + 1)          # layer-averaged, as in LightGCN
        z = {nt: out[self.offset[nt]:self.offset[nt] + n]
             for nt, n in ((nt, self._num(nt)) for nt in self.node_types)}
        row, col = edge_label_index
        if self.decoder_mode == 'mlp':
            return self.mlp(z, edge_label_index)
        return (z[self.rna_type][row] * z['drug'][col]).sum(-1)

    def _num(self, nt):
        nts = self.node_types
        i = nts.index(nt)
        nxt = self.offset[nts[i + 1]] if i + 1 < len(nts) else self.n_total
        return nxt - self.offset[nt]


@torch.no_grad()
def reliable_negatives(model_fn, pos_index, n_src, n_dst, n_neg, device,
                       pool_mult=10, generator=None):
    """DMGAT/LSNSCDA-style: keep the candidates scored most confidently negative.

    A pool of `pool_mult * n_neg` uniform candidates is scored by the current
    model and the lowest-scoring `n_neg` are kept.  Known positives are excluded
    from the pool; only training positives are known, so nothing held out leaks.
    """
    known = set(map(tuple, pos_index.t().cpu().numpy().tolist()))
    pool = []
    need = pool_mult * n_neg
    while len(pool) < need:
        r = torch.randint(0, n_src, (need,), device=device, generator=generator)
        d = torch.randint(0, n_dst, (need,), device=device, generator=generator)
        cand = torch.stack([r, d])
        keep = [i for i, p in enumerate(map(tuple, cand.t().cpu().numpy().tolist()))
                if p not in known]
        pool.append(cand[:, keep])
        if sum(p.size(1) for p in pool) >= need:
            break
    cand = torch.cat(pool, dim=1)[:, :need]
    scores = model_fn(cand)
    idx = torch.argsort(scores)[:n_neg]
    return cand[:, idx]
