"""Pinned original iNcRD-HG encoder with the M1 QMP query-edge correction.

``OriginalGNNEncoder`` is the released network (repository commit eb8de624):
two ``GeneralConv`` layers, no explicit aggregator (add over neighbours), wrapped
by ``to_hetero(..., aggr='mean')`` over the relation types of the graph.

``QueryHollowEncoder`` reimplements the same network with sparse relation
operators.  Beyond being numerically identical to the PyG encoder, its dense
forward returns, for query pairs, the layer-2 endpoint states of the message
graph *with the query edge's messages removed* (M1).  The correction is exact:
it shifts the pre-activations of layer 1 by the removed message and propagates
that shift through layer 2, including the self-loop multiplicity of the
same-type relations and every ``lin_self`` term of the target node.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv, to_hetero


class OriginalGNNEncoder(torch.nn.Module):
    """Released iNcRD-HG encoder, copied verbatim from the original script."""

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GeneralConv((-1, -1), hidden_channels)
        self.conv2 = GeneralConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x, edge_index)
        return x


class QueryHollowEncoder(torch.nn.Module):
    def __init__(self, graph, hidden, layers):
        super().__init__()
        assert layers == 2
        self.original = to_hetero(OriginalGNNEncoder(hidden, hidden), graph.metadata(), aggr="mean")
        self.graph = graph
        self.counts = {n: sum(e[2] == n for e in graph.edge_types) for n in graph.node_types}
        self.adj = {}
        for e in graph.edge_types:
            idx = graph[e].edge_index
            values = torch.ones(idx.shape[1], device=idx.device)
            self.adj[e] = torch.sparse_coo_tensor(
                idx.flip(0), values, (graph[e[2]].num_nodes, graph[e[0]].num_nodes)
            ).coalesce()

    def forward(self, x, edges):
        return self.original(x, edges)

    def layer(self, convs, x):
        """One message-passing layer: sum over neighbours, mean over incoming relations."""
        out = {}
        for e in self.graph.edge_types:
            c = convs["__".join(e)]
            value = torch.sparse.mm(self.adj[e], c.lin_msg(x[e[0]])) + c.lin_self(x[e[2]])
            out[e[2]] = out.get(e[2], 0) + value
        return {n: value / self.counts[n] for n, value in out.items()}

    def forward_dense(self, xr, xd, Sr1, Sd1, A, r=None, d=None, q=None):
        """Dense reimplementation of the encoder; exact query-edge removal (M1).

        Sr1 / Sd1 are the same-type adjacency matrices with their released
        self-loop multiplicity, A is the message-graph biadjacency.  With
        (r, d, q) given, the layer-2 states of r (without the message from d)
        and of d (without the message from r) are returned as well, together
        with the layer-1 shifts and relation operators M2 needs.
        """
        p1 = self.layer(self.original.conv1, self.graph.x_dict)
        h1 = {n: F.leaky_relu(v, 0.1) for n, v in p1.items()}
        z = self.layer(self.original.conv2, h1)
        if r is None:
            return z["miRNA"], z["drug"]
        c1, c2 = self.original.conv1, self.original.conv2
        mm, dd = "miRNA__MiMi__miRNA", "drug__DrugDrug__drug"
        md, dm = "miRNA__MiDrug__drug", "drug__rev_MiDrug__miRNA"
        nr, nd = self.counts["miRNA"], self.counts["drug"]
        qq = q[:, None]
        dh_m = F.leaky_relu(p1["miRNA"][r] - qq * c1[dm].lin_msg(xd)[d] / nr, 0.1) - h1["miRNA"][r]
        dh_d = F.leaky_relu(p1["drug"][d] - qq * c1[md].lin_msg(xr)[r] / nd, 0.1) - h1["drug"][d]

        def self_shift(node, index, delta, same, mat):
            shift = mat.diagonal()[index, None] * F.linear(delta, c2[same].lin_msg.weight)
            for e in self.graph.edge_types:
                if e[2] == node:
                    shift = shift + F.linear(delta, c2["__".join(e)].lin_self.weight)
            return shift / self.counts[node]

        zm = z["miRNA"][r] + self_shift("miRNA", r, dh_m, mm, Sr1) - qq * c2[dm].lin_msg(h1["drug"])[d] / nr
        zd = z["drug"][d] + self_shift("drug", d, dh_d, dd, Sd1) - qq * c2[md].lin_msg(h1["miRNA"])[r] / nd
        # M2 scales these operators by 1/2; the 2/n_rel factor restores the
        # exact target-specific relation mean.
        corr = dict(
            dh_m=dh_m,
            dh_d=dh_d,
            W_dm=c2[dm].lin_msg.weight * (2 / nr),
            W_mm=c2[mm].lin_msg.weight * (2 / nr),
            W_md=c2[md].lin_msg.weight * (2 / nd),
            W_dd=c2[dd].lin_msg.weight * (2 / nd),
        )
        return z["miRNA"], z["drug"], zm, zd, corr
