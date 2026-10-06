"""Original released iNcRD-HG classes with the shared, held-out-label-safe protocol."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GeneralConv, to_hetero
import torch_geometric.transforms as T
from bench_data import HERE, BASE, ROOT, load, sha
from graph_adapters import classes_from

SOURCE = HERE/'original_source/predict_association.py'
RELEASE_GRAPH = ROOT/'incrd-run/3.heter_data/MiDrug_train_data.pth'
DRUG_CACHE = BASE/'mphgnn_plmf_20260921/cache/drug_chemberta_zinc.npz'


def rowminmax(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return (x-x.min(1, keepdim=True).values)/(x.max(1, keepdim=True).values-x.min(1, keepdim=True).values+1e-8)


def knn_edges(x, k=10):
    x = F.normalize(x, dim=1)
    similarity = x@x.t()
    similarity.fill_diagonal_(-float('inf'))
    dst = similarity.topk(k, dim=1).indices.flatten()
    src = torch.arange(len(x)).repeat_interleave(k)
    return torch.unique(torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]), dim=1)


def prepare_inputs():
    """Freeze label-free inputs; fold-specific target edges are installed only at runtime."""
    manifest = {'source':json.loads((HERE/'original_source/SOURCE.json').read_text()), 'datasets':{}}
    for ds in ('MiDrug', 'MDR', 'MDS'):
        data = load(ds, 0)
        graph = HeteroData()
        if ds == 'MiDrug':
            released = torch.load(RELEASE_GRAPH, map_location='cpu')
            for node in ('lncRNA', 'miRNA', 'drug'):
                graph[node].x = released[node].x.clone()
            for edge in released.edge_types:
                if edge[0] in graph.node_types and edge[2] in graph.node_types and edge[1] not in ('MiDrug', 'rev_MiDrug'):
                    graph[edge].edge_index = released[edge].edge_index.clone()
            sources = [RELEASE_GRAPH]
            adaptation = 'Original expression/ChemBERTa features and lncRNA-miRNA-drug auxiliary relations; exclude later circRNA extension; shared unique target pairs.'
        else:
            cache = np.load(DRUG_CACHE)
            lookup = {str(v):i for i,v in enumerate(cache['drugbank_id'])}
            indices = [lookup[v] for v in data['drug_ids']]
            assert cache['smiles'][indices].tolist() == data['smiles'], 'Drug ID/SMILES alignment mismatch'
            # Use the 2- and 3-mer part of the frozen content bank (80 dimensions).
            graph['miRNA'].x = rowminmax(data['xr'][:, 4:])
            graph['drug'].x = rowminmax(cache['x'][indices])
            graph['miRNA', 'MiMi', 'miRNA'].edge_index = knn_edges(graph['miRNA'].x)
            graph['drug', 'DrugDrug', 'drug'].edge_index = knn_edges(graph['drug'].x)
            graph = T.AddSelfLoops()(graph)
            sources = [DRUG_CACHE, HERE/f'data/{ds}_content.npz', HERE/f'data/{ds}_entities.json']
            adaptation = 'No released expression/lncRNA auxiliary network: 80-d 2+3-mer miRNA, original ChemBERTa-zinc-base-v1 768-d drug cache, symmetric cosine kNN(10), one self-loop per node.'
        for node in graph.node_types:
            assert torch.isfinite(graph[node].x).all(), (ds, node)
        assert len(graph['miRNA'].x) == len(data['mirna_names'])
        assert len(graph['drug'].x) == len(data['drug_ids'])
        path = HERE/f'data/{ds}_incrd_original_inputs.pt'
        torch.save(graph, path)
        manifest['datasets'][ds] = {
            'adaptation':adaptation,
            'sources':[{'path':str(p), 'sha256':sha(p)} for p in sources],
            'input_path':str(path), 'input_sha256':sha(path),
            'entities_sha256':sha(HERE/f'data/{ds}_entities.json'),
            'features':{n:list(graph[n].x.shape) for n in graph.node_types},
            'side_edges':{'|'.join(e):int(graph[e].edge_index.shape[1]) for e in graph.edge_types},
        }
    (HERE/'data/incrd_original_manifest.json').write_text(json.dumps(manifest, indent=2))


def build_graph(data, dataset):
    path = HERE/f'data/{dataset}_incrd_original_inputs.pt'
    record = json.loads((HERE/'data/incrd_original_manifest.json').read_text())['datasets'][dataset]
    assert record['input_sha256'] == sha(path)
    assert record['entities_sha256'] == sha(HERE/f'data/{dataset}_entities.json')
    graph = torch.load(path, map_location='cpu')
    positive = data['train_pairs'][data['train_y'] == 1]
    edges = torch.tensor(positive.T.copy(), dtype=torch.long)
    graph['miRNA', 'MiDrug', 'drug'].edge_index = edges
    graph['drug', 'rev_MiDrug', 'miRNA'].edge_index = edges.flip(0)
    return graph


def original_classes():
    return classes_from(SOURCE, {'Model', 'GNNEncoder', 'EdgeDecoder'},
                        dict(torch=torch, nn=nn, F=F, Linear=nn.Linear,
                             GeneralConv=GeneralConv, to_hetero=to_hetero, RNA_type='miRNA'))


class OriginalINcRDAdapter:
    def __init__(self, data, args):
        self.data = data
        self.graph = build_graph(data, args.dataset).to(args.device)
        model = original_classes()
        self.model = model.Model(self.graph, hidden_channels=256, aggr_str='mean').to(args.device)
        self.pairs = {p:torch.tensor(data[p+'_pairs'].T, dtype=torch.long, device=args.device)
                      for p in ('train', 'val', 'test')}
        self.y = torch.tensor(data['train_y'], dtype=torch.float32, device=args.device)
        args.optimizer = 'RMSprop'
        args.optimizer_alpha = 0.99
        args.hidden = 256
        args.hetero_aggr = 'mean'
        args.batch = 0  # original full-batch training
        manifest = json.loads((HERE/'data/incrd_original_manifest.json').read_text())
        self.note = ('Original released iNcRD-HG GeneralConv x2 and MLP classes, RMSprop(lr=1e-4, alpha=.99), 300 full-batch epochs. '
                     + manifest['datasets'][args.dataset]['adaptation']
                     + ' Target association graph fixed to shared training positives for train/val/test; validation-only checkpoint selection.')

    def make_optimizer(self, args):
        return torch.optim.RMSprop(self.model.parameters(), lr=args.lr, alpha=.99, weight_decay=args.wd)

    def logits(self, part):
        return self.model(self.graph.x_dict, self.graph.edge_index_dict, self.pairs[part])[0]

    def train_epoch(self, optimizer, epoch):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(self.logits('train'), self.y)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def predict(self, part):
        return self.logits(part).sigmoid().cpu().numpy()


if __name__ == '__main__':
    prepare_inputs()
    print('Saved original iNcRD-HG input manifests for MiDrug/MDR/MDS.')
