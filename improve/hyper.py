# Hypergraph convolution and a DGI-style contrastive auxiliary loss.
#
# The last two untested architecture families in this literature.  Hypergraph
# (HGNNLDA 2023, LHRCDA 2025, HGCJAMH 2025): a KNN hypergraph over node
# features, so a hyperedge links a node to its k nearest neighbours at once
# rather than decomposing into pairwise edges.  It is the only component that
# came out significantly positive in this repo's CDSA experiments (+0.051 on
# cold_rna), and it was scored WORST there by the random-split protocol -- so
# testing it here is testing that contradiction directly.
#
# Contrastive (DMAGCL 2025, GSLRDA 2022, HECLCDA 2026): DGI's corruption
# discriminator as an auxiliary loss on top of the released encoder.
import torch
import torch.nn.functional as F


def knn_incidence(x, k=10):
    """KNN hypergraph incidence: each node is the centroid of one hyperedge."""
    xn = F.normalize(x, dim=1)
    sim = xn @ xn.t()
    idx = sim.topk(k + 1, dim=1).indices          # includes self
    H = torch.zeros(x.size(0), x.size(0), device=x.device)
    H.scatter_(1, idx, 1.0)
    return H


def hgnn_adj(H):
    """HGNN propagation matrix Dv^-1/2 H W De^-1 H^T Dv^-1/2 (Feng et al. 2019)."""
    Dv = H.sum(1).clamp(min=1).pow(-0.5)
    De = H.sum(0).clamp(min=1).pow(-1)
    return (Dv.unsqueeze(1) * H) @ (De.unsqueeze(1) * H.t()) * Dv.unsqueeze(0)


class HyperEncoder(torch.nn.Module):
    """Per-node-type hypergraph convolution; the pairwise graph is not used.

    Each node type gets its own KNN hypergraph built from its own features, so
    lncRNA neighbourhoods come from expression and drug neighbourhoods from
    ChemBERTa -- the same blocks the released model consumes, propagated
    differently.
    """

    def __init__(self, node_types, hidden, k=10):
        super().__init__()
        self.k = k
        self.lin1 = torch.nn.ModuleDict(
            {nt: torch.nn.LazyLinear(hidden) for nt in node_types})
        self.lin2 = torch.nn.ModuleDict(
            {nt: torch.nn.Linear(hidden, hidden) for nt in node_types})
        self._G = {}

    def forward(self, x_dict, edge_index_dict):
        out = {}
        for nt, x in x_dict.items():
            if nt not in self._G:
                with torch.no_grad():
                    self._G[nt] = hgnn_adj(knn_incidence(x, self.k))
            G = self._G[nt]
            h = F.leaky_relu(G @ self.lin1[nt](x), 0.1)
            out[nt] = G @ self.lin2[nt](h)
        return out


def dgi_loss(z_dict, x_dict, encoder, edge_index_dict, disc):
    """DGI: discriminate real node embeddings from feature-shuffled ones."""
    corrupt = {nt: x[torch.randperm(x.size(0), device=x.device)]
               for nt, x in x_dict.items()}
    z_neg = encoder(corrupt, edge_index_dict)
    loss = 0.0
    for nt, z in z_dict.items():
        if nt not in z_neg:
            continue
        s = torch.sigmoid(z.mean(0))
        pos = disc(z, s)
        neg = disc(z_neg[nt], s)
        loss = loss + F.binary_cross_entropy_with_logits(
            torch.cat([pos, neg]),
            torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
    return loss
