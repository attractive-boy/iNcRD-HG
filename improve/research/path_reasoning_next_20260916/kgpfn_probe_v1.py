"""Performance-blind GPU and isolation probe for the frozen KGPFN adapter."""
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch

from improve import kgpfn_context_v1 as context
from improve import kgpfn_runtime_v1 as runtime
from improve import tabular_foundation as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs/MiDrug/arrays.npz'


def main():
    destination = HERE / 'KGPFN_PROBE_V1.json'
    if destination.exists():
        raise ValueError('Do not replace an existing probe')
    with np.load(SOURCE, allow_pickle=False) as z:
        a = {key: z[key].copy() for key in runtime.TRAIN_KEYS}
    model = runtime.load_model('cuda:1')
    start = time.monotonic()
    graph, contexts, excluded = context.setup_fold(a, 0)
    torch.cuda.reset_peak_memory_stats('cuda:1')
    query_indices = np.flatnonzero(a['folds'] == 0)[:32]
    query_pairs = a['pairs_train'][:, query_indices]
    cached = context.encode_contexts(model, graph, a, contexts)
    predictions = context.predict(model, graph, a, query_pairs, cached)
    backward = context.predict(model, graph, a, query_pairs[:, ::-1].copy(), cached, batch_size=7)
    errors = {}
    for arm in context.ARMS:
        np.testing.assert_allclose(predictions[arm], backward[arm][::-1], atol=3e-5, rtol=3e-5)
        errors[arm] = float(np.abs(predictions[arm] - backward[arm][::-1]).max())
    # One complete author convenience call checks the optimized path on the
    # real graph, still using training queries only and no performance metric.
    chosen = contexts[0]
    normal = runtime.score(model, graph,
        runtime.triples(a, query_pairs[:, :3], context.target_relation(a)),
        runtime.triples(a, a['pairs_train'][:, chosen], context.target_relation(a)),
        a['y_train'][chosen])
    fused, raw = runtime.encode_once(model, graph,
        runtime.triples(a, query_pairs[:, :3], context.target_relation(a)))
    optimized = runtime.score_rows(model, runtime.encode_rows(model, fused),
                                   cached[0]['context'], cached[0]['labels'])
    np.testing.assert_allclose(normal, optimized, atol=3e-5, rtol=3e-5)
    author_error = float(np.abs(normal - optimized).max())
    audits = []
    for fold in range(5):
        g, cs, omitted = context.setup_fold(a, fold, 'cpu')
        altered = {k: v.copy() for k, v in a.items()}
        rows = np.flatnonzero(a['folds'] == fold)
        altered['y_train'][rows] = 1 - altered['y_train'][rows]
        for seed, original in zip(context.CONTEXT_SEEDS, cs):
            np.testing.assert_array_equal(original, context.context_rows(altered, fold, seed))
        codes = a['pairs_train'][0, omitted] * 216 + a['pairs_train'][1, omitted]
        visible = a['target_codes'][~np.isin(a['target_codes'], codes)]
        assert not np.isin(visible, codes).any()
        audits.append(dict(fold=fold, excluded_rows=len(omitted), context_sizes=[len(x) for x in cs],
                           physical_arcs=g.num_edges, context_and_query_labels_isolated=True))
    sources = [Path(runtime.__file__), Path(context.__file__), Path(__file__),
               ROOT / 'test_kgpfn_runtime_v1.py']
    record = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        passed=True, source_sha256={str(p): base.digest(p) for p in sources},
        input_sha256={str(SOURCE): base.digest(SOURCE), str(runtime.CHECKPOINT): base.digest(runtime.CHECKPOINT)},
        loaded_state_items=368, parameter_elements=26668874, trainable_parameter_elements=0,
        neural_fits=0, performance_metrics_computed=False, original_test_source_access=False,
        query_count=32, contexts=len(cached), context_size=100,
        seconds=time.monotonic() - start, replay_errors=errors, author_forward_error=author_error,
        gpu_peak_memory_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
        corrected_context_label_counts=[dict(zip(*[x.tolist() for x in
            torch.unique(item['labels'], return_counts=True)])) for item in cached],
        fold_audits=audits))
    base.write_json(destination, record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
