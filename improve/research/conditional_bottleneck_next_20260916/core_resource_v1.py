"""Training-only memory/gradient feasibility; no parameter update or performance score."""
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import information_bottleneck_graph_v1 as core
from improve import anchor_priority_graph_v1 as common
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    target = HERE / 'CORE_RESOURCE_V1.json'
    if target.exists():
        raise ValueError('Preserve resource evidence')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.cuda.manual_seed(719)
    torch.random.default_generator.manual_seed(719)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = common.ROOT / 'results/anchor_residual_energy_v2_20260916/inputs/MiDrug/arrays.npz'
    a = common.nested.load_selected(source, common.TRAIN_KEYS)
    start = time.monotonic()
    model, xs, graphs, folds = core.setup(a, core.parameters(), 'full', 'cuda:1')
    construction_seconds = time.monotonic() - start
    records = []
    for count in (128, len(folds[0])):
        ids = folds[0][:count]
        pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device='cuda:1')[:, ids]
        label = torch.as_tensor(a['y_train'], device='cuda:1')[ids]
        offset = torch.as_tensor(a['anchor_oof'], device='cuda:1')[ids]
        torch.cuda.reset_peak_memory_stats('cuda:1')
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize('cuda:1')
        start = time.monotonic()
        value, kl, detail = model.forward_with_kl(xs, graphs[0], pairs,
                                                  torch.Generator(device='cuda:1').manual_seed(932))
        loss = F.binary_cross_entropy_with_logits(offset + value.double(), label) + core.parameters()['information_weight'] * kl.mean()
        loss.backward()
        torch.cuda.synchronize('cuda:1')
        if any(not torch.isfinite(w.grad).all() for w in model.parameters() if w.grad is not None):
            raise FloatingPointError('Nonfinite resource gradient')
        record = dict(training_queries=count, construction_seconds=construction_seconds,
            forward_backward_seconds=time.monotonic() - start,
            peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
            mean_kl=float(kl.detach().mean()), objective=float(loss.detach()), **detail,
            parameter_updates=0, heldout_performance_computed=False)
        records.append(record)
        print('CORE_RESOURCE', record, flush=True)
        del value, kl, loss
    report = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), records=records,
        graph_audits=[graph.audit for graph in graphs], parameters=core.parameters(),
        source_sha256={str(path): base.digest(path) for path in (Path(__file__), Path(core.__file__))},
        input_sha256={str(source): base.digest(source)}, device='cuda:1',
        parameter_updates=0, new_neural_fits=0, evaluation_labels_read=False,
        performance_metrics_computed=False, confirmed_final_innovations=0))
    base.write_json(target, report)
    print('CORE_RESOURCE_COMPLETE', report['sha256'], flush=True)


if __name__ == '__main__':
    main()
