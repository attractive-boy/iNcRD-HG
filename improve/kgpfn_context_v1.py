"""Training-only demonstration selection for frozen relational in-context inference."""
import numpy as np
import torch

from improve import kgpfn_runtime_v1 as runtime


ARMS = ('full', 'shuffled_context', 'ultra_only')
CONTEXT_SEEDS = (666, 0, 1)


def context_rows(a, fold, seed):
    if set(a) - runtime.TRAIN_KEYS:
        raise ValueError('Context selection accepts training inputs only')
    support = np.flatnonzero(a['folds'] != fold)
    rng = np.random.default_rng(61753 + int(seed) * 97 + int(fold) * 1009)
    positive = support[a['y_train'][support] == 1]
    negative = support[a['y_train'][support] == 0]
    if len(positive) < 20 or len(negative) < 80:
        raise ValueError('Insufficient support for the registered context size')
    chosen = np.concatenate((rng.choice(positive, 20, replace=False),
                             rng.choice(negative, 80, replace=False)))
    return chosen[rng.permutation(len(chosen))]


def target_relation(a):
    kinds = np.unique(a['relation'][a['target_codes'] >= 0])
    if kinds.shape != (1,):
        raise ValueError('Target association requires one forward relation')
    return int(kinds[0])


def setup_fold(a, fold, device='cuda:1'):
    contexts = [context_rows(a, fold, seed) for seed in CONTEXT_SEEDS]
    excluded = np.unique(np.concatenate([np.flatnonzero(a['folds'] == fold), *contexts]))
    graph = runtime.masked_graph(a, excluded, device)
    return graph, contexts, excluded


@torch.no_grad()
def encode_contexts(model, graph, a, contexts):
    indices = np.unique(np.concatenate(contexts))
    triples = runtime.triples(a, a['pairs_train'][:, indices], target_relation(a))
    fused, ultra = runtime.encode_once(model, graph, triples)
    rows = runtime.encode_rows(model, fused)
    records = []
    for seed, context in zip(CONTEXT_SEEDS, contexts):
        select = torch.as_tensor(np.searchsorted(indices, context), device=rows.device)
        labels = torch.as_tensor(a['y_train'][context], device=rows.device)
        corrected = model._apply_label_correction(ultra[select][None], [labels])[0]
        rng = np.random.default_rng(32491 + seed * 71)
        permutation = torch.as_tensor(rng.permutation(len(corrected)), device=rows.device)
        records.append(dict(context=rows[select], labels=corrected,
                            shuffled=corrected[permutation], indices=context.copy()))
    return records


@torch.no_grad()
def predict(model, graph, a, query_pairs, contexts, batch_size=256):
    query = runtime.triples(a, query_pairs, target_relation(a))
    fused, ultra = runtime.encode_once(model, graph, query)
    rows = runtime.encode_rows(model, fused)
    result = {'ultra_only': ultra.cpu().numpy().astype(np.float64)}
    for arm, key in (('full', 'labels'), ('shuffled_context', 'shuffled')):
        scores = [runtime.score_rows(model, rows, item['context'], item[key], batch_size)
                  for item in contexts]
        result[arm] = np.mean(np.stack(scores).astype(np.float64), axis=0)
    return result
