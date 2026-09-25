"""Head-context-dependent logical-rule scoring inspired by SLogic.

All symbolic supports and rule statistics are reconstructed after deleting the
entire query fold. This implementation is an independent binary-link adaptation.
"""
import hashlib
import json

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from improve import overlap_block_models as graph_lib
from improve import sharpness_graph_v1 as shared
from improve.nested_anchor_inputs_v2 import TRAIN_KEYS as INPUT_KEYS

KINDS = ('full', 'global_rules')
identity_task = shared.identity_task


def parameters():
    return dict(width=64, depth=2, dropout=.1, folds=5,
        physical_schema=[[0,0],[0,2],[1,1],[1,2],[2,2],[0,1],[3,3],[3,2],[3,3]],
        max_rule_length=3, minimum_rule_support=2, candidate_rules=32,
        subgraph_hops=1, neighbor_cap=32, subgraph_seed=29016,
        coverage_penalty=.65, temperature=.5, grounding_tau=2.,
        margin=1., ranking_weight=.1, triplet_batch=256,
        lnc_policy='identity_strong_reference')


def schemas(p):
    physical = {i:tuple(pair) for i,pair in enumerate(p['physical_schema'])}
    nr = len(physical)
    return {**physical, **{r+nr:(right,left) for r,(left,right) in physical.items() if left != right}}


def rule_vocabulary(p):
    schema = schemas(p)
    result = []
    def visit(node,path):
        if len(path) >= 2 and node == 2:
            result.append(tuple(path))
        if len(path) == p['max_rule_length']:
            return
        for relation,(left,right) in sorted(schema.items()):
            if left == node:
                visit(right,path+[relation])
    if p['max_rule_length'] not in (2,3):
        raise ValueError('Exact simple-path implementation supports lengths two and three')
    visit(1,[])
    return tuple(sorted(result,key=lambda x:(len(x),x)))


def relation_matrices(a,p,fold):
    """Canonical binary predicates; same-type predicates are undirected."""
    if set(a)-(INPUT_KEYS|{'anchor_oof'}):
        raise ValueError('Symbolic support accepts training arrays only')
    counts = a['node_counts']
    offsets = np.r_[0,np.cumsum(counts)[:-1]]
    nr = len(p['physical_schema'])
    if int(a['relation'].max())+1 != nr:
        raise ValueError('Unexpected physical relation dictionary')
    omitted = np.flatnonzero(a['folds']==fold)
    edges,relation = graph_lib.masked_physical(a,omitted)
    matrices = {}
    for r,(left,right) in enumerate(p['physical_schema']):
        selected = edges[:,relation==r]-offsets[[left,right],None]
        if (selected < 0).any() or (selected >= counts[[left,right],None]).any():
            raise ValueError('Physical relation has unexpected node types')
        mat = sparse.csr_matrix((np.ones(selected.shape[1],dtype=np.float64),selected),
                                shape=(int(counts[left]),int(counts[right])))
        mat.sum_duplicates()
        mat.data[:] = 1.
        if left == right:
            mat = mat.maximum(mat.T).tocsr()
            mat.setdiag(0)
            mat.eliminate_zeros()
        else:
            matrices[r+nr] = mat.T.tocsr()
        matrices[r] = mat
    positive = (a['folds']!=fold)&(a['y_train']==1)
    expected = sparse.csr_matrix((np.ones(int(positive.sum())),a['pairs_train'][:,positive]),
                                 shape=(int(counts[1]),int(counts[2])))
    expected.data[:] = 1.
    if (matrices[3] != expected).nnz:
        raise ValueError('Target support differs from other-fold positive training pairs')
    return matrices,edges,relation


def simple_grounding(path,matrices,schema):
    """Count vertex-simple paths; subtract both possible three-edge repeats."""
    if len(path) not in (2,3):
        raise ValueError('Only length-two/three simple paths are supported')
    a,b = matrices[path[0]],matrices[path[1]]
    ab = a@b
    if len(path)==2:
        value = ab
    else:
        c = matrices[path[2]]
        value = ab@c
        first_repeat = schema[path[0]][0] == schema[path[1]][1]
        second_repeat = schema[path[0]][1] == schema[path[2]][1]
        if first_repeat:
            value = value-sparse.diags(ab.diagonal())@c
        if second_repeat:
            value = value-a@sparse.diags((b@c).diagonal())
        if first_repeat and second_repeat:
            value = value+a.multiply(b.T).multiply(c)
    result = value.toarray()
    if not np.isfinite(result).all() or result.min() < -1e-8:
        raise ValueError('Invalid simple-path count')
    return result.clip(0).astype(np.float32)


def wilson_lower(successes,trials,z=1.96):
    successes,trials = np.asarray(successes,dtype=float),np.asarray(trials,dtype=float)
    n = np.maximum(trials,1.)
    ratio = successes/n
    result = (ratio+z*z/(2*n)-z*np.sqrt(ratio*(1-ratio)/n+z*z/(4*n*n)))/(1+z*z/n)
    return np.where(trials>0,result.clip(0),0.)


def head_subgraphs(a,p,matrices,fold):
    if p['subgraph_hops'] != 1:
        raise ValueError('This version registers exactly one sampled BFS hop')
    offsets = np.r_[0,np.cumsum(a['node_counts'])[:-1]]
    schema = schemas(p)
    n,nr = int(a['node_counts'].sum()),int(a['node_counts'][1])
    adjacency = [[] for _ in range(n)]
    global_degree = np.zeros(n,dtype=np.int64)
    for relation,mat in sorted(matrices.items()):
        left,right = schema[relation]
        rows,cols = mat.nonzero()
        for first,second in zip(rows+offsets[left],cols+offsets[right]):
            adjacency[first].append((int(second),relation))
            global_degree[first] += 1
            global_degree[second] += 1
    features,src,dst,rels,batch,head_indices,subgraph_sizes = [],[],[],[],[],[],[]
    for local in range(nr):
        head = int(offsets[1]+local)
        neighbors = sorted({node for node,_ in adjacency[head]})
        if len(neighbors)>p['neighbor_cap']:
            rng = np.random.default_rng(p['subgraph_seed']+1009*fold+local)
            neighbors = sorted(rng.choice(neighbors,p['neighbor_cap'],replace=False).tolist())
        nodes = [head]+neighbors
        start = len(features)
        index = {node:start+i for i,node in enumerate(nodes)}
        head_indices.append(start)
        subgraph_sizes.append(len(nodes))
        for node in nodes:
            features.append([float(node==head),float(node!=head),float(node!=head),np.log1p(global_degree[node])])
            batch.append(local)
        # Edge-built one-hop neighborhoods, as in the author sampler; include
        # all relation labels on both directions of each sampled head edge.
        for node in nodes:
            for other,relation in adjacency[node]:
                if other in index and (node==head or other==head):
                    src.append(index[node]);dst.append(index[other]);rels.append(relation)
    return dict(node_features=np.asarray(features,dtype=np.float32),
        context_edges=np.asarray([src,dst],dtype=np.int64),context_relations=np.asarray(rels,dtype=np.int64),
        node_batch=np.asarray(batch,dtype=np.int64),head_indices=np.asarray(head_indices,dtype=np.int64),
        subgraph_sizes=np.asarray(subgraph_sizes,dtype=np.float32))


def build_support(a,p,fold,device='cpu'):
    matrices,physical,physical_relations = relation_matrices(a,p,fold)
    paths,schema = rule_vocabulary(p),schemas(p)
    raw = np.stack([simple_grounding(path,matrices,schema) for path in paths],axis=1)
    positive = (a['folds']!=fold)&(a['y_train']==1)
    pairs = a['pairs_train'][:,positive]
    support = (raw[pairs[0],:,pairs[1]]>0).sum(axis=0).astype(np.int64)
    body = (raw>0).sum(axis=(0,2)).astype(np.int64)
    confidence = support/np.maximum(body,1)
    lower = wilson_lower(support,body)
    stats = np.stack((confidence,support/max(1,int(positive.sum())),(support+1)/(body+2),lower),axis=1).astype(np.float32)
    tail_count = (raw>0).sum(axis=2)
    valid = (support>=p['minimum_rule_support'])&(body>0)
    candidates = np.zeros(tail_count.shape,dtype=bool)
    order = sorted(range(len(paths)),key=lambda i:(-float(lower[i]),paths[i]))
    for head in range(len(candidates)):
        selected = [i for i in order if valid[i] and tail_count[head,i]>0][:p['candidate_rules']]
        candidates[head,selected] = True
    nr = len(p['physical_schema'])*2
    path_ids = np.full((len(paths),p['max_rule_length']),nr,dtype=np.int64)
    lengths = np.asarray([len(path) for path in paths],dtype=np.int64)
    for i,path in enumerate(paths):
        path_ids[i,:len(path)] = path
    arrays = dict(**head_subgraphs(a,p,matrices,fold),
        grounding=np.tanh(raw/p['grounding_tau']).astype(np.float32),
        candidates=candidates,tail_count=tail_count.astype(np.float32),rule_stats=stats,
        rule_ids=path_ids,rule_lengths=lengths,pair_offset=a['pair_offset'].copy())
    fingerprint = {name:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for name,value in arrays.items()}
    audit = dict(fold=fold,rules=len(paths),valid_rules=int(valid.sum()),
                 heads_with_candidates=int(candidates.any(axis=1).sum()),
                 support_counts=support.tolist(),body_counts=body.tolist(),array_sha256=fingerprint,
                 paths=[list(path) for path in paths],physical_edges_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
                 physical_relations_sha256=hashlib.sha256(physical_relations.tobytes()).hexdigest())
    graph = {name:torch.as_tensor(value,device=device) for name,value in arrays.items()}
    graph['rule_lengths'] = graph['rule_lengths'].cpu()
    graph['audit'] = audit
    return graph


def rule_pool(a,graph,fold,device):
    grounding = graph['grounding'].detach().cpu().numpy()
    candidates = graph['candidates'].detach().cpu().numpy()
    queries,positive_rules,negative_rules = [],[],[]
    for index in np.flatnonzero((a['folds']==fold)&(a['y_train']==1)):
        head,tail = a['pairs_train'][:,index]
        hits = grounding[head,:,tail]>0
        positive = np.flatnonzero(candidates[head]&hits)
        negative = np.flatnonzero(candidates[head]&~hits)
        if len(positive) and len(negative):
            queries.append(index);positive_rules.append(positive);negative_rules.append(negative)
    if not queries:
        raise ValueError('No training-positive query has both positive and hard-negative rules')
    count = grounding.shape[1]
    pos,neg = np.full((len(queries),count),-1,dtype=np.int64),np.full((len(queries),count),-1,dtype=np.int64)
    for i,(first,second) in enumerate(zip(positive_rules,negative_rules)):
        pos[i,:len(first)],neg[i,:len(second)] = first,second
    arrays = dict(queries=np.asarray(queries,dtype=np.int64),positive=pos,negative=neg,
                  positive_count=np.asarray(list(map(len,positive_rules))),negative_count=np.asarray(list(map(len,negative_rules))))
    return {key:torch.as_tensor(value,device=device) for key,value in arrays.items()}


def sample_rules(pool,count,generator):
    device = pool['queries'].device
    ix = torch.randint(len(pool['queries']),(count,),device=device,generator=generator)
    p = (torch.rand(count,device=device,generator=generator)*pool['positive_count'][ix]).long()
    n = (torch.rand(count,device=device,generator=generator)*pool['negative_count'][ix]).long()
    return pool['queries'][ix],pool['positive'][ix,p],pool['negative'][ix,n]


class RelationalMean(nn.Module):
    def __init__(self,input_width,width,relations):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(relations,input_width,width))
        self.root = nn.Linear(input_width,width)
        nn.init.xavier_uniform_(self.weight)

    def forward(self,h,edges,relation):
        out = self.root(h)
        src,dst = edges
        for r,weight in enumerate(self.weight):
            take = relation==r
            if not bool(take.any()):
                continue
            source,target = src[take],dst[take]
            degree = torch.bincount(target,minlength=len(h)).to(h.dtype).clamp_min(1)
            message = (h[source]@weight)/degree[target,None]
            aggregate = h.new_zeros((len(h),weight.shape[1]))
            aggregate.index_add_(0,target,message)
            out = out+aggregate
        return F.relu(out)


class ContextualRules(nn.Module):
    def __init__(self,p,kind):
        super().__init__()
        self.p,self.kind = p,kind
        width,nr = p['width'],2*len(p['physical_schema'])
        self.embedding = nn.Embedding(nr+1,width,padding_idx=nr)
        self.gru = nn.GRU(width,width,batch_first=True)
        self.layers = nn.ModuleList([RelationalMean(4 if i==0 else width,width,nr) for i in range(p['depth'])])
        self.fc1,self.fc2 = nn.Linear(4*width+4,2*width),nn.Linear(2*width,1)
        self.dropout = nn.Dropout(p['dropout'])
        nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)
        self.rna_degree,self.drug_degree = nn.Identity(),nn.Identity()

    def rule_scores(self,graph):
        h = graph['node_features']
        for i,layer in enumerate(self.layers):
            h = layer(h,graph['context_edges'],graph['context_relations'])
            if i+1<len(self.layers):
                h = self.dropout(h)
        pooled = h.new_zeros((len(graph['head_indices']),h.shape[1]))
        pooled.index_add_(0,graph['node_batch'],h)
        pooled = pooled/graph['subgraph_sizes'][:,None]
        context = torch.cat((h[graph['head_indices']],pooled),dim=1)
        if self.kind == 'global_rules':
            context = context.mean(0,keepdim=True).expand_as(context)
        embedded = self.embedding(graph['rule_ids'])
        packed = pack_padded_sequence(embedded,graph['rule_lengths'],batch_first=True,enforce_sorted=False)
        _,hidden = self.gru(packed)
        rule = hidden[-1]
        query = self.embedding.weight[3].expand(len(rule),-1)
        shared_features = torch.cat((query,rule,graph['rule_stats']),dim=1)
        both = torch.cat((context[:,None,:].expand(-1,len(rule),-1),
                          shared_features[None,:,:].expand(len(context),-1,-1)),dim=-1)
        return self.fc2(self.dropout(F.relu(self.fc1(both)))).squeeze(-1)

    def weights(self,scores,graph):
        candidates = graph['candidates']
        adjusted = (scores.double()-self.p['coverage_penalty']*graph['tail_count'].double().clamp_min(1).log())/self.p['temperature']
        adjusted = adjusted.masked_fill(~candidates,-torch.inf)
        active = candidates.any(dim=1)
        adjusted = torch.where(active[:,None],adjusted,torch.zeros_like(adjusted))
        return adjusted.softmax(1)*candidates

    def residual(self,scores,graph,pairs):
        local = pairs-graph['pair_offset'][:,None]
        learned = self.weights(scores,graph)
        initial = self.weights(torch.zeros_like(scores),graph)
        evidence = graph['grounding'][local[0],:,local[1]].double()
        return ((learned-initial)[local[0]]*evidence).sum(dim=1)

    def forward(self,xs,graph,pairs):
        return self.residual(self.rule_scores(graph),graph,pairs)


def setup(a,p,kind,device):
    if set(a)-(INPUT_KEYS|{'anchor_oof'}):
        raise ValueError('Rule setup accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered contextual-rule arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds']==fold),device=device) for fold in range(p['folds'])]
    if identity_task(a,p):
        return shared.IdentityResidual().to(device),[],[None]*p['folds'],folds
    graphs = [build_support(a,p,fold,device) for fold in range(p['folds'])]
    return ContextualRules(p,kind).to(device),[],graphs,folds
