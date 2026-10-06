"""M1/M2/M3 on the pinned original iNcRD-HG encoder and shared train-only graph.

The existing two-type QMP correction is generalized to the actual number of
incoming relations, retaining all released lncRNA auxiliary information.
M2/M3 and the fusion head are loaded verbatim from the existing module source.
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import to_hetero

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / 'mirna_baselines_20260928'
sys.path.insert(0, str(BENCH))
from bench_data import load, sha
from incrd_adapter import build_graph, original_classes

SOURCE = HERE.parent / 'own_midrug_pilot_20260928/run_own.py'


class OriginalQueryEncoder(torch.nn.Module):
    def __init__(self, graph, hidden, layers):
        super().__init__()
        assert layers == 2
        self.original = to_hetero(original_classes().GNNEncoder(hidden, hidden), graph.metadata(), aggr='mean')
        self.graph = graph
        self.counts = {n: sum(e[2] == n for e in graph.edge_types) for n in graph.node_types}
        self.adj = {}
        for e in graph.edge_types:
            idx = graph[e].edge_index
            values = torch.ones(idx.shape[1], device=idx.device)
            self.adj[e] = torch.sparse_coo_tensor(idx.flip(0), values,
                (graph[e[2]].num_nodes, graph[e[0]].num_nodes)).coalesce()

    def forward(self, x, edges):
        return self.original(x, edges)

    def layer(self, convs, x):
        out = {}
        for e in self.graph.edge_types:
            c = convs['__'.join(e)]
            value = torch.sparse.mm(self.adj[e], c.lin_msg(x[e[0]])) + c.lin_self(x[e[2]])
            out[e[2]] = out.get(e[2], 0) + value
        return {n: value / self.counts[n] for n, value in out.items()}

    def forward_dense(self, xr, xd, Sr1, Sd1, A, r=None, d=None, q=None):
        # Full sparse relation operators reproduce the original GeneralConv(add).
        p1 = self.layer(self.original.conv1, self.graph.x_dict)
        h1 = {n: F.leaky_relu(v, .1) for n, v in p1.items()}
        z = self.layer(self.original.conv2, h1)
        if r is None:
            return z['miRNA'], z['drug']
        c1, c2 = self.original.conv1, self.original.conv2
        mm, dd = 'miRNA__MiMi__miRNA', 'drug__DrugDrug__drug'
        md, dm = 'miRNA__MiDrug__drug', 'drug__rev_MiDrug__miRNA'
        nr, nd = self.counts['miRNA'], self.counts['drug']
        qq = q[:, None]
        dh_m = F.leaky_relu(p1['miRNA'][r] - qq*c1[dm].lin_msg(xd)[d]/nr, .1) - h1['miRNA'][r]
        dh_d = F.leaky_relu(p1['drug'][d] - qq*c1[md].lin_msg(xr)[r]/nd, .1) - h1['drug'][d]

        def self_shift(node, index, delta, same, mat):
            shift = mat.diagonal()[index, None] * F.linear(delta, c2[same].lin_msg.weight)
            for e in self.graph.edge_types:
                if e[2] == node:
                    shift = shift + F.linear(delta, c2['__'.join(e)].lin_self.weight)
            return shift / self.counts[node]

        zm = z['miRNA'][r] + self_shift('miRNA', r, dh_m, mm, Sr1) - qq*c2[dm].lin_msg(h1['drug'])[d]/nr
        zd = z['drug'][d] + self_shift('drug', d, dh_d, dd, Sd1) - qq*c2[md].lin_msg(h1['miRNA'])[r]/nd
        # The unchanged M2 implementation multiplies these matrices by 1/2.
        # Scaling by 2/n_rel supplies the exact target-specific relation mean.
        corr = dict(dh_m=dh_m, dh_d=dh_d,
            W_dm=c2[dm].lin_msg.weight*(2/nr), W_mm=c2[mm].lin_msg.weight*(2/nr),
            W_md=c2[md].lin_msg.weight*(2/nd), W_dd=c2[dd].lin_msg.weight*(2/nd))
        return z['miRNA'], z['drug'], zm, zd, corr


def module_namespace(device):
    config = SimpleNamespace(layers=2, enc='mask', exact=True, qis=True, qis_ext=True,
        qncn=True, ncn_dim=64, ncn_rel='co,knn,dknn', ease='10,50', ease_side='',
        ease_alpha=1., ease_learn=False, ease_nomask=False, nomask=False, wide=False)
    ns = dict(torch=torch, F=F, Linear=Linear, a=config, DEV=torch.device(device),
              GNNEncoder=OriginalQueryEncoder)
    tree = ast.parse(SOURCE.read_text())
    names = {'ease_masked', 'PairStruct', 'Model'}
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return SimpleNamespace(**ns), config


def structure_data(graph):
    def adjacency(edge):
        idx = graph[edge].edge_index
        out = torch.zeros(graph[edge[0]].num_nodes, graph[edge[2]].num_nodes, device=idx.device)
        out.index_put_((idx[0], idx[1]), torch.ones(idx.shape[1], device=idx.device), accumulate=True)
        return out
    sr1 = adjacency(('miRNA','MiMi','miRNA'))
    sd1 = adjacency(('drug','DrugDrug','drug'))
    sr, sd = sr1.clone(), sd1.clone()
    sr.fill_diagonal_(0); sd.fill_diagonal_(0)
    return SimpleNamespace(xr=graph['miRNA'].x, xd=graph['drug'].x, xk=graph['miRNA'].x,
        n_mir=graph['miRNA'].num_nodes, n_drug=graph['drug'].num_nodes,
        Sr=sr, Sd=sd, Sr1=sr1, Sd1=sd1, rel={}, kmer_gram=None)


class FullAdapter:
    def __init__(self, data, args):
        self.data = data
        self.graph = build_graph(data, args.dataset).to(args.device)
        self.module, config = module_namespace(args.device)
        self.D = structure_data(self.graph)
        pos = self.graph['miRNA','MiDrug','drug'].edge_index.t()
        self.ps = self.module.PairStruct(self.D, pos)
        self.pairs = {p: torch.as_tensor(data[p+'_pairs'].T, dtype=torch.long, device=args.device)
                      for p in ('train','val','test')}
        self.y = torch.as_tensor(data['train_y'], dtype=torch.float, device=args.device)
        raw = self.ps.qis(*self.pairs['train'])
        self.mu, self.sd = raw.mean(0, keepdim=True), raw.std(0, keepdim=True)+1e-6
        self.qis = {p: (self.ps.qis(*idx)-self.mu)/self.sd for p, idx in self.pairs.items()}
        self.ease = {p: self.ps.ease(*idx) for p, idx in self.pairs.items()}
        self.dense = (self.D.xr, self.D.xd, self.D.Sr1, self.D.Sd1, self.ps.A)
        self.model = self.module.Model(self.graph, 256, self.ps, raw.shape[1], self.ease['train'].shape[1]).to(args.device)
        with torch.no_grad():
            self.model.encoder(self.graph.x_dict, self.graph.edge_index_dict)
        args.optimizer = 'RMSprop'; args.optimizer_alpha = .99
        args.hidden = 256; args.hetero_aggr = 'mean'; args.batch = 0
        args.modules = vars(config); args.module_source_sha256 = sha(SOURCE)
        self.note = ('Pinned original iNcRD-HG encoder, input features and ALL auxiliary relations retained; '
            'M1 QMP generalized to actual incoming relation counts; unchanged M2 SP-RNCN '
            '(co,knn,dknn; extended path statistics), M3 MC-EASE (lambda=10,50, both sides); '
            'no M4; target graph uses shared training positives only; validation-AUC selection; '
            'RMSprop 1e-4, 300 epochs, hidden=256, full batch.')

    def make_optimizer(self, args):
        return torch.optim.RMSprop(self.model.parameters(), lr=args.lr, alpha=.99, weight_decay=args.wd)

    def logits(self, part):
        return self.model(self.graph, self.pairs[part], self.ps, self.qis[part], self.ease[part], self.dense)

    def train_epoch(self, optimizer, epoch):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(self.logits('train'), self.y)
        loss.backward(); optimizer.step()
        return float(loss.detach())

    def predict(self, part):
        return self.logits(part).sigmoid().cpu().numpy()
