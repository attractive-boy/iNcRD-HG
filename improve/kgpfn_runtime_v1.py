"""Pinned, frozen KGPFN execution with the author's actual RSPMM orientation.

This is a local runtime adaptation, not a reproduction of the paper's training.
The published CustomTabICL overwrites its first four feature positions with CLS
tokens. That behavior is deliberately preserved. Only inference precision and
the sum/DistMult sparse execution backend are made explicit here.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import types
import weakref

import numpy as np
import torch


EXTERNAL = Path('/data1/licheng/incrd-hg/.external')
SOURCE = EXTERNAL / 'repos/kgpfn/af415c33a950bdb975eeb1b6f4e93e86b6ec929c'
DEPENDENCIES = EXTERNAL / 'pydeps/kgpfn-torch26cu118'
CHECKPOINT = EXTERNAL / 'models/kgpfn/0594fbebe5f47007315be38273cd18363594e999/kgpfn_icl.pth'
CHECKPOINT_SHA256 = '366c3e3795738da81a05691c626497372bda58109ad7bf178bfc7fe21136a87a'
TRAIN_KEYS = {'edges', 'relation', 'target_codes', 'node_counts', 'pair_offset',
              'nd', 'folds', 'pairs_train', 'y_train'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class RelationalSparseSum:
    """sum_e w_e * r[type_e] * x[col_e], accumulated into row_e.

    Implements only the frozen model's sum/DistMult path. No gradients, other
    reducers, or message functions are silently approximated. Adjacencies are
    cached with tensor identity/version checks and weak references.
    """

    def __init__(self):
        self.cache = {}

    def clear(self):
        self.cache.clear()

    def __call__(self, edge_index, edge_type, edge_weight, relation, input,
                 sum='add', mul='mul'):
        if sum != 'add' or mul != 'mul':
            raise ValueError('This backend only implements sum/DistMult')
        if any(x.requires_grad for x in (edge_weight, relation, input)):
            raise ValueError('The local RSPMM backend is frozen inference only')
        if (input.ndim != 2 or relation.ndim != 2 or input.shape[1] != relation.shape[1]
                or edge_index.shape != (2, edge_type.numel())
                or edge_weight.shape != edge_type.shape):
            raise ValueError('Invalid relational sparse product shapes')
        n = input.shape[0]
        # Keep only live graphs; tensors can otherwise reuse pointers later.
        for key, (refs, _) in list(self.cache.items()):
            if any(ref() is None for ref in refs):
                del self.cache[key]
        tensors = (edge_index, edge_type, edge_weight)
        key = (tuple((id(t), t._version) for t in tensors), n,
               str(input.device), input.dtype, relation.shape[0])
        item = self.cache.get(key)
        if item is None or not all(ref() is t for ref, t in zip(item[0], tensors)):
            if edge_type.numel() and (int(edge_type.min()) < 0
                    or int(edge_type.max()) >= relation.shape[0]
                    or int(edge_index.min()) < 0 or int(edge_index.max()) >= n):
                raise ValueError('Out-of-range graph index')
            matrices = []
            for kind in range(relation.shape[0]):
                selected = edge_type == kind
                if not bool(selected.any()):
                    continue
                adjacency = torch.sparse_coo_tensor(
                    edge_index[:, selected], edge_weight[selected].to(input.dtype),
                    size=(n, n), device=input.device).coalesce().to_sparse_csr()
                matrices.append((kind, adjacency))
            item = (tuple(weakref.ref(t) for t in tensors), matrices)
            self.cache[key] = item
        output = torch.zeros_like(input)
        for kind, adjacency in item[1]:
            output.add_(torch.sparse.mm(adjacency, input) * relation[kind])
        return output


OPERATOR = RelationalSparseSum()


def install_runtime():
    for path in (DEPENDENCIES, SOURCE, SOURCE / 'model'):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in reversed((DEPENDENCIES, SOURCE, SOURCE / 'model')):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    name = 'model.ultra.rspmm'
    if name in sys.modules and getattr(sys.modules[name], 'generalized_rspmm', None) is not OPERATOR:
        raise RuntimeError('An incompatible RSPMM backend was already imported')
    module = types.ModuleType(name)
    module.generalized_rspmm = OPERATOR
    sys.modules[name] = module


def configure(device):
    if str(device) not in ('cpu', 'cuda:1'):
        raise ValueError('Use CPU for implementation checks or cuda:1 for the study')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if str(device) == 'cuda:1':
        torch.cuda.set_device(device)


def load_model(device='cuda:1', entity_chunk_size=8):
    install_runtime()
    configure(device)
    from model.ultra.encoder import StructureEncoderRelationAware
    from model.kgpfnsem import KGPFN
    from tabicl.model.custom_tabicl import CustomTabICL
    from tabicl.model.inference_config import InferenceConfig

    class ExplicitFP32TabICL(CustomTabICL):
        def forward(self, x, y, eval_pos, task_type='reg'):
            if self.thinking_rows is not None:
                raise ValueError('The pinned checkpoint has no thinking rows')
            cfg = InferenceConfig()
            for manager in (cfg.ROW_CONFIG, cfg.ICL_CONFIG):
                manager.update(dict(device=x.device, use_amp=False, use_fa3=False,
                                    offload=False, use_async=False))
            embedded = self.x_adapter(x)
            # The vendored RowInteraction overwrites the first four positions.
            # Do not add padding: that would alter the released model.
            rows = self.row_interactor(embedded, mgr_config=cfg.ROW_CONFIG)
            return self.icl_predictor(rows, y[:, :eval_pos], mgr_config=cfg.ICL_CONFIG)

    common = dict(input_dim=64, hidden_dims=[64] * 6, message_func='distmult',
                  aggregate_func='sum', short_cut=True, layer_norm=True)
    encoder = StructureEncoderRelationAware(
        rel_model_cfg={'class': 'RelNBFNet', **common},
        entity_model_cfg={'class': 'EntityNBFNet', **common},
        entity_chunk_size=entity_chunk_size)
    # The checkpoint lacks all 32 PFN LayerNorm bias tensors. This architectural
    # switch is identified by strict state matching, not performance selection.
    model = KGPFN(structure_encoder=encoder, feature_transformer=ExplicitFP32TabICL(bias_free_ln=True),
                  enhance_structure=True, structure_score_enhance=True,
                  context_label_correction=True, with_relation=True,
                  context_graph=3, context_tail=True, seq_chunk_size=40)
    if digest(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint SHA256 mismatch')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    status = model.load_state_dict(checkpoint['model'], strict=True)
    if status.missing_keys or status.unexpected_keys or len(model.state_dict()) != 368:
        raise ValueError('The full published state was not loaded')
    model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    return model


def graph_from_edges(edges, relation, node_count, num_relations, device='cpu'):
    """Create inverse arcs and relation graph after caller has masked facts."""
    install_runtime()
    from torch_geometric.data import Data
    from pfn.tasks import build_relation_graph
    edges, relation = np.asarray(edges, dtype=np.int64), np.asarray(relation, dtype=np.int64)
    if edges.shape != (2, len(relation)) or len(relation) == 0:
        raise ValueError('Require a nonempty aligned physical graph')
    graph = Data(edge_index=torch.as_tensor(np.concatenate((edges, edges[::-1]), axis=1)),
                 edge_type=torch.as_tensor(np.concatenate((relation, relation + num_relations))),
                 num_nodes=int(node_count), num_relations=int(2 * num_relations))
    return build_relation_graph(graph).to(device)


def masked_graph(a, excluded_rows, device='cuda:1'):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Graph construction accepts training inputs only')
    pairs = a['pairs_train'][:, np.asarray(excluded_rows, dtype=np.int64)]
    codes = pairs[0] * int(a['nd']) + pairs[1]
    keep = ~np.isin(a['target_codes'], codes)
    if np.isin(a['target_codes'][keep], codes).any():
        raise AssertionError('An excluded target fact survived')
    relation_count = int(a['relation'].max()) + 1
    graph = graph_from_edges(a['edges'][:, keep], a['relation'][keep],
                             int(a['node_counts'].sum()), relation_count, device)
    return graph


def triples(a, pairs, relation_id):
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError('Expected local RNA/drug pairs as [2,N]')
    shifted = pairs + a['pair_offset'][:, None]
    return np.column_stack((shifted.T, np.full(pairs.shape[1], relation_id, dtype=np.int64)))


@contextmanager
def without_label_correction(model):
    previous = model.context_label_correction
    model.context_label_correction = False
    try:
        yield
    finally:
        model.context_label_correction = previous


@torch.no_grad()
def score(model, graph, query, context, labels):
    device = next(model.parameters()).device
    q, c = [torch.as_tensor(x, device=device, dtype=torch.long) for x in (query, context)]
    y = torch.as_tensor(labels, device=device, dtype=torch.float32)
    result = model(graph, q[None], [c], [y], task_type='reg')[0]
    if result.shape != (len(query),) or not torch.isfinite(result).all():
        raise ValueError('Invalid KGPFN predictions')
    return result.cpu().numpy()


@torch.no_grad()
def encode_once(model, graph, input_triples, chunk_size=8):
    """Gather all needed layer outputs during one traversal per unique head.

    Shares exactly the published layer modules and graph boundary conditions.
    In contrast, the author convenience forward repeats the first k layers
    separately for each k and endpoint. Numerical equivalence is checked before
    the local path is used on any performance split.
    """
    device = next(model.parameters()).device
    values = torch.as_tensor(input_triples, dtype=torch.long, device=device)
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) == 0
            or not bool((values[:, 2] == values[0, 2]).all())):
        raise ValueError('Require nonempty triples sharing a relation')
    if model.context_graph != 3 or not model.context_tail or not model.with_relation:
        raise ValueError('Unexpected pinned runtime configuration')
    encoder = model.structure_encoder
    em = encoder.entity_model
    if em.concat_hidden or len(em.layers) != 6:
        raise ValueError('Unexpected entity architecture')
    rel = values[0, 2].reshape(1)
    representations = encoder.relation_model(graph.relation_graph, query=rel)
    heads, inverse = torch.unique(values[:, 0], sorted=True, return_inverse=True)
    h_final = torch.empty(len(values), 128, device=device)
    t_final = torch.empty_like(h_final)
    hops = torch.empty(len(values), 6, 64, device=device)
    weight = torch.ones(graph.num_edges, device=device)
    for start in range(0, len(heads), chunk_size):
        stop = min(start + chunk_size, len(heads))
        hs = heads[start:stop]
        rels = representations.expand(len(hs), -1, -1)
        query = rels[:, rel.item()]
        boundary = torch.zeros(len(hs), graph.num_nodes, 64, device=device)
        boundary.scatter_add_(1, hs[:, None, None].expand(-1, 1, 64), query[:, None])
        selected = ((inverse >= start) & (inverse < stop)).nonzero().flatten()
        local = inverse[selected] - start
        h_index, t_index = values[selected, 0], values[selected, 1]
        hidden = boundary
        for k, layer in enumerate(em.layers):
            layer.relation = rels
            out = layer(hidden, query, boundary, graph.edge_index, graph.edge_type,
                        (graph.num_nodes, graph.num_nodes), weight)
            if em.short_cut and out.shape == hidden.shape:
                out = out + hidden
            hidden = out
            if k < 3:
                hops[selected, 2 * k] = hidden[local, h_index]
                hops[selected, 2 * k + 1] = hidden[local, t_index]
        h_final[selected] = torch.cat((hidden[local, h_index], query[local]), dim=-1)
        t_final[selected] = torch.cat((hidden[local, t_index], query[local]), dim=-1)
    r_final = representations[0, rel.item()].expand(len(values), -1)
    raw_score = encoder.get_mlp_scores(t_final)
    h = model.entity_adapter(h_final)
    r = model.relation_adapter(r_final)
    t = model.entity_adapter(t_final)
    structure = model.structure_norm(torch.stack((h, r, t), dim=1))
    h, r, t = structure.unbind(dim=1)
    cos = torch.nn.functional.cosine_similarity(h + r, t, dim=-1, eps=1e-8)
    enhanced = model.structure_enhance_adapter(torch.cat(
        (h + r - t, h * r * t, cos[:, None].expand(-1, 64)), dim=-1))
    enhanced = model.structure_enhance_norm(enhanced.reshape(len(values), 3, 64))
    score_token = model.structure_score_norm(model.structure_score_adapter(raw_score[:, None]))
    hop_tokens = model.context_graph_norm(model.context_graph_adapter(hops))
    fused = torch.cat((hop_tokens, structure, enhanced, score_token[:, None]), dim=1)
    if fused.shape != (len(values), 13, 64) or not torch.isfinite(fused).all():
        raise ValueError('Invalid encoded features')
    return fused, raw_score


def inference_config(device):
    from tabicl.model.inference_config import InferenceConfig
    config = InferenceConfig()
    for manager in (config.ROW_CONFIG, config.ICL_CONFIG):
        manager.update(dict(device=device, use_amp=False, use_fa3=False,
                            offload=False, use_async=False))
    return config


@torch.no_grad()
def encode_rows(model, fused, chunk_size=512):
    transformer = model.feature_transformer
    cfg = inference_config(fused.device)
    result = []
    for start in range(0, len(fused), chunk_size):
        x = transformer.x_adapter(fused[start:start + chunk_size][None])
        result.append(transformer.row_interactor(x, mgr_config=cfg.ROW_CONFIG)[0])
    return torch.cat(result)


@torch.no_grad()
def score_rows(model, query_rows, context_rows, labels, batch_size=256):
    if len(context_rows) != len(labels) or not len(context_rows):
        raise ValueError('Require a nonempty aligned context')
    device = next(model.parameters()).device
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)[None]
    cfg = inference_config(device)
    result = []
    for start in range(0, len(query_rows), batch_size):
        rows = torch.cat((context_rows, query_rows[start:start + batch_size]))[None]
        pred = model.feature_transformer.icl_predictor(rows, y, mgr_config=cfg.ICL_CONFIG)
        result.append(pred[0, :, 0])
    result = torch.cat(result)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite PFN output')
    return result.cpu().numpy()
