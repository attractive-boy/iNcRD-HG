"""Real-size tensor contractions only: CUDA values, gradients, and runtime."""
from datetime import datetime, timezone
from pathlib import Path
import gc
import time

import torch

from improve import triangular_pair_graph_v1 as reference
from improve import triangular_pair_graph_v2 as candidate
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    destination = HERE / 'COMPILE_PROBE_V1.json'
    if destination.exists():
        raise ValueError('Preserve completed contraction probe')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for n in (821, 1538):
        torch.manual_seed(791)
        shapes = [(8, n, 2, 8), (n, n, 2, 8)] * 2
        inputs = [(torch.randn(s, device='cuda:1') * .2).requires_grad_() for s in shapes]
        weight = torch.randn(8, n, 2, 8, device='cuda:1')
        for kind in candidate.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            torch.cuda.synchronize('cuda:1'); start = time.monotonic()
            expected = reference.triangle_block(*inputs, kind)
            expected_grad = torch.autograd.grad((expected * weight).sum(), inputs)
            torch.cuda.synchronize('cuda:1'); reference_seconds = time.monotonic() - start
            torch.cuda.synchronize('cuda:1'); start = time.monotonic()
            actual = candidate.triangle_block(*inputs, kind)
            actual_grad = torch.autograd.grad((actual * weight).sum(), inputs)
            torch.cuda.synchronize('cuda:1'); compile_seconds = time.monotonic() - start
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            for x, y in zip(actual_grad, expected_grad):
                torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
            runtime = []
            for _ in range(3):
                torch.cuda.synchronize('cuda:1'); start = time.monotonic()
                value = candidate.triangle_block(*inputs, kind)
                gradient = torch.autograd.grad((value * weight).sum(), inputs)
                torch.cuda.synchronize('cuda:1'); runtime.append(time.monotonic() - start)
            row = dict(nodes=n, kind=kind, reference_seconds=reference_seconds,
                first_compiled_call_seconds=compile_seconds, warm_seconds=runtime,
                max_value_difference=float((actual-expected).abs().max()),
                max_gradient_difference=max(float((x-y).abs().max()) for x, y in zip(actual_grad, expected_grad)),
                peak_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30)
            rows.append(row); print('COMPILE_PROBE', row, flush=True)
            del expected, actual, expected_grad, actual_grad, value, gradient
            gc.collect(); torch.cuda.empty_cache()
        del inputs, weight
        gc.collect(); torch.cuda.empty_cache()
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
        source_sha256={str(path): base.digest(path) for path in
                      (Path(__file__), Path(reference.__file__), Path(candidate.__file__))},
        device='cuda:1', evaluation_labels_read=False, training_fits=0, formal_fits=0))
    base.write_json(destination, result)


if __name__ == '__main__':
    main()
