# Input-side transforms: relation-wise degree encoding and feature scaling.
#
# Both read only from the split they are handed.  For the train split the
# target relation's edge_index holds the training message-passing edges with
# val/test edges already removed by RandomLinkSplit, so a degree computed here
# cannot see a held-out label.  The val/test splits carry the same
# training-only message-passing edges, so the encoding is identical across the
# three -- it is a property of the training graph, not of the split.
import torch


def relation_degree_features(data, node_types):
    """log1p degree of each node, one column per incident relation type.

    Richer than a single total degree and it matches what the architecture is
    already trying to be: relation-type aware.  Columns are scaled to [0, 1]
    by their own max so they enter the network on the same scale as the
    existing min-max-normalised node features.
    """
    cols = {nt: [] for nt in node_types}
    for (src, rel, dst), edge_index in sorted(data.edge_index_dict.items()):
        for role, nt in ((0, src), (1, dst)):
            if nt not in cols:
                continue
            deg = torch.zeros(data[nt].num_nodes, device=edge_index.device)
            deg.scatter_add_(0, edge_index[role],
                             torch.ones_like(edge_index[role], dtype=torch.float))
            deg = torch.log1p(deg)
            hi = deg.max()
            cols[nt].append(deg / hi if hi > 0 else deg)
    return {nt: torch.stack(v, dim=1) for nt, v in cols.items() if v}


def add_degree_encoding(splits):
    """Append the training graph's degree encoding to x for every split.

    `splits` is (train, val, test); the encoding is built once from train.
    """
    train = splits[0]
    enc = relation_degree_features(train, list(train.node_types))
    for data in splits:
        for nt, e in enc.items():
            data[nt].x = torch.cat([data[nt].x, e.to(data[nt].x.device)], dim=1)
    return splits


def normalise_features(splits):
    """Row-wise L2 normalisation of every node feature block.

    The released features are already min-max scaled into [0, 1], so this is
    not a rescue of an unscaled block.  What it does fix is that the mean row
    norm differs by 2x across node types (lncRNA 6.6, drug 13.9), which biases
    the first convolution towards the drug side before any learning happens.
    """
    for data in splits:
        for nt in data.node_types:
            x = data[nt].x
            data[nt].x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return splits
