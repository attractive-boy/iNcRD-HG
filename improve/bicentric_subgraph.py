"""Query-marked one-hop subgraphs with bicentric sphere sum attention.

Independent typed-attribute adaptation of BS-SubGNN (AAAI 2026), equations
6--13. Every subgraph is extracted AFTER target query deletion. The pooling
uses exp(tanh(score)) without softmax normalization, as in the paper/code.
No PyG dependency, numeric node-ID feature, or new tabular column is used.
"""
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil

import numpy as np
from scipy import sparse
import torch
from torch import nn

from improve import overlap_block_experiment as common
from improve import overlap_block_models as graph_lib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/bicentric_core_20260915'
KINDS = ('full', 'no_mark', 'flat_pool', 'no_message')
_CONTEXTS = OrderedDict()


class SubgraphContext:
    def __init__(self, edges, n):
        row, col = np.concatenate([edges, edges[::-1]], axis=1)
        keep = row != col
        a = sparse.csr_matrix((np.ones(keep.sum(), dtype=np.float32),
                               (row[keep], col[keep])), shape=(n, n))
        a.sum_duplicates()
        a.data[:] = 1.
        a.sort_indices()
        self.adjacency, self.n = a, n
        self.items, self.packed = {}, OrderedDict()

    def extract(self, u, v):
        if not (0 <= u < self.n and 0 <= v < self.n) or u == v:
            raise ValueError('Expected distinct in-range query endpoints')
        if self.adjacency[u, v] != 0 or self.adjacency[v, u] != 0:
            raise ValueError('Query fact must be deleted before subgraph extraction')
        key = (int(u), int(v))
        if key in self.items:
            return self.items[key]
        a = self.adjacency
        nu, nv = a.indices[a.indptr[u]:a.indptr[u + 1]], a.indices[a.indptr[v]:a.indptr[v + 1]]
        neighbors = np.union1d(nu, nv)
        neighbors = neighbors[(neighbors != u) & (neighbors != v)]
        nodes = np.r_[u, v, neighbors].astype(np.int64)
        labels = np.zeros(len(nodes), dtype=np.int64)
        labels[1] = 1
        in_u, in_v = np.isin(neighbors, nu), np.isin(neighbors, nv)
        # Zero-based equation 6 for h=1: centers 0/1, u-only 2,
        # common 3, v-only 4. Missing distances have the bound h+1=2.
        labels[2:] = np.where(in_u & in_v, 3, np.where(in_u, 2, 4))
        induced = a[nodes][:, nodes] + sparse.eye(len(nodes), format='csr', dtype=np.float32)
        degree = np.asarray(induced.sum(axis=1)).reshape(-1)
        coo = induced.tocoo()
        weights = (1. / np.sqrt(degree[coo.row] * degree[coo.col])).astype(np.float32)
        edge = np.stack([coo.row, coo.col]).astype(np.int64)
        result = (nodes, labels, edge, weights)
        self.items[key] = result
        return result

    def batch(self, pairs):
        array = pairs.detach().cpu().numpy()
        if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] == 0:
            raise ValueError('Expected a nonempty pair matrix')
        key = (str(pairs.device), array.tobytes())
        if key in self.packed:
            self.packed.move_to_end(key)
            return self.packed[key]
        ns, ls, es, ws, bs, starts = [], [], [], [], [], []
        offset = 0
        for i, (u, v) in enumerate(array.T):
            nodes, labels, edges, weights = self.extract(int(u), int(v))
            starts.append(offset)
            ns.append(nodes)
            ls.append(labels)
            es.append(edges + offset)
            ws.append(weights)
            bs.append(np.full(len(nodes), i, dtype=np.int64))
            offset += len(nodes)
        result = dict(nodes=torch.as_tensor(np.concatenate(ns), device=pairs.device),
            labels=torch.as_tensor(np.concatenate(ls), device=pairs.device),
            edges=torch.as_tensor(np.concatenate(es, axis=1), device=pairs.device),
            weights=torch.as_tensor(np.concatenate(ws), device=pairs.device),
            batch=torch.as_tensor(np.concatenate(bs), device=pairs.device),
            starts=torch.as_tensor(starts, device=pairs.device), count=len(starts))
        self.packed[key] = result
        while len(self.packed) > 2:
            self.packed.popitem(last=False)
        return result


def context(edges, n):
    key = (int(n), hashlib.sha256(np.ascontiguousarray(edges).tobytes()).hexdigest())
    if key not in _CONTEXTS:
        _CONTEXTS[key] = SubgraphContext(edges, n)
    _CONTEXTS.move_to_end(key)
    while len(_CONTEXTS) > 10:
        _CONTEXTS.popitem(last=False)
    return _CONTEXTS[key]


class BicentricGraph(nn.Module):
    def __init__(self, dimensions, width=32, depth=3, dropout=.1, kind='full'):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown bicentric subgraph arm')
        self.kind = kind
        self.project = nn.ModuleList([nn.Linear(d, width) for d in dimensions])
        self.mark = nn.Embedding(5, width)
        self.linear = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.update = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(depth)])
        self.pool = nn.ModuleList([nn.Linear(depth * width, 1) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(nn.Linear((3 * depth + 2) * width, 64),
                                    nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        # Residual likelihood starts at the unchanged reference in every arm.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        # Compatibility with the shared residual executor; no degree readout.
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def forward(self, xs, graph, pairs):
        packed = graph.batch(pairs)
        initial = torch.cat([layer(x) for layer, x in zip(self.project, xs)], dim=0)
        label = self.mark(packed['labels'])
        if self.kind == 'no_mark':
            # Retain the same parameter module; the labels are unavailable to
            # propagation. Sphere labels remain available only to pooling.
            label = self.mark.weight.mean(dim=0).expand_as(label)
        h = initial[packed['nodes']] + label
        source, target = packed['edges']
        depth_outputs = []
        for linear, update in zip(self.linear, self.update):
            values = linear(h)
            if self.kind == 'no_message':
                aggregate = values
            else:
                aggregate = torch.zeros_like(values).index_add(0, target,
                    values[source] * packed['weights'][:, None])
            h = self.dropout(torch.tanh(update(torch.cat([values, aggregate], dim=1))))
            depth_outputs.append(h)
        z = torch.cat(depth_outputs, dim=1)
        pooled = []
        for i, gate in enumerate(self.pool):
            ids = (packed['labels'] >= 2 if self.kind == 'flat_pool' else packed['labels'] == i + 2)
            value = z[ids]
            weights = torch.exp(torch.tanh(gate(value)))
            pooled.append(z.new_zeros((packed['count'], z.shape[1])).index_add(
                0, packed['batch'][ids], value * weights))
        # Local adaptation: endpoint states support isolated pairs, while all
        # arms share this fallback. The paper's pooling excludes centers.
        output = torch.cat(pooled + [h[packed['starts']], h[packed['starts'] + 1]], dim=1)
        return self.output(output).squeeze(-1)


def setup(a, p, kind, device):
    n = int(a['node_counts'].sum())
    graphs, folds = [], []
    for f in range(p['folds']):
        ix = np.flatnonzero(a['folds'] == f)
        edges, _ = graph_lib.masked_physical(a, ix)
        graphs.append(context(edges, n))
        folds.append(torch.as_tensor(ix, device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = BicentricGraph([x.shape[1] for x in xs], p['width'], p['depth'], p['dropout'], kind).to(device)
    return model, xs, graphs, folds


def prepare(out=OUT):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a fresh bicentric core directory')
    source_names = ('bicentric_subgraph.py', 'test_bicentric_subgraph.py',
                    'overlap_block_experiment.py', 'overlap_block_models.py', 'non_tabular_experiment.py')
    literature = ROOT / 'research/new_mechanisms_20260915/residual_frontier'
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), tasks=['MiDrug', 'LncDrug'],
        kinds=list(KINDS), primary_kind='full', primary_controls=['no_mark', 'flat_pool', 'no_message'],
        width=32, depth=3, dropout=.1, folds=5, hop=1,
        source_sha256={n: common.base.digest(ROOT / n) for n in source_names},
        inputs={str(literature / n): common.base.digest(literature / n) for n in
                ('bs_subgnn.pdf', 'AUTHOR_FILES.json')},
        literature=dict(doi='10.1609/aaai.v40i17.38490', venue='AAAI 2026',
            title='Subgraph Encoding with Bicentric Sphere Node Labeling and Pooling for Link Prediction'),
        selection='Two fixed learning rates selected jointly across tasks at seed 666; each arm uses its own best admissible mean MiDrug AUC/AP; primary is fixed to full before screening.',
        mechanism='Query-specific induced one-hop graph, bicentric marking before propagation, three separate unnormalized sphere attention sums. Raw typed attributes only; no pair feature expansion.',
        controls='no_mark withholds sphere identity from propagation; flat_pool pools the whole neighbor union through each of the same three gates; no_message replaces normalized adjacency by identity with the same linear modules. No claim of equal effective capacity or identical computations.',
        adaptation='Boolean physical union; typed attribute projections; three tanh GCN updates with self skip; endpoint-state fallback for isolated pairs; one binary residual output initialized to zero. No author code was executed.',
        boundary='Delete current neural fold target positives before neighbor sets, induced edges, labels, and degree normalization. Heldout outcomes are never passed to graph construction.',
        novelty_scope='One query-conditioned subgraph mechanism family; not an additional count for each pooling or marking ablation. Its relation to the previously failed neighbor-union decoder is explicitly compared through no_message.',
        smoke=False, original_test_source_access=False, confirmed_innovations=0)
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for n in source_names:
        shutil.copy2(ROOT / n, out / 'source_snapshot' / n)
    common.base.write_json(out / 'protocol.json', p)
    print('FROZEN_BICENTRIC_CORE', out, flush=True)
