"""Engineering-only gradient/memory probe; no validation metrics or selection."""
import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from improve import tabular_foundation as base
from improve import tabicl_adaptation_modules as kernels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('LncDrug', 'MiDrug'), required=True)
    parser.add_argument('--mode', choices=('all', 'icl'), required=True)
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    out = Path(__file__).parent/f'probe_{args.task}_{args.mode}_{args.rows}.json'
    if out.exists():
        raise ValueError('Probe output already exists')
    result = dict(task=args.task, mode=args.mode, rows=args.rows, no_validation_metrics=True)
    try:
        p = base.protocol(base.DEFAULT_OUT)
        chosen = base.selection(base.DEFAULT_OUT, p)['chosen'][args.task]['config']
        a = base.load_features(base.DEFAULT_OUT, p, args.task, 'inner')
        prefix = 'raw' if chosen['input_view'] == 'raw' else 'x'
        indices = base.balanced_subset(a['y_train'], args.rows, 20260912)
        x, y = a[prefix+'_train'][indices], a['y_train'][indices]
        torch.cuda.set_device(args.device)
        clf = base.estimator(dict(n_estimators=2), 666, p, args.device)
        base.fit_context(clf, x, y)
        before = base.parameter_digest(clf)
        count = kernels.set_trainable(clf.model_, args.mode)
        optimizer = torch.optim.AdamW([v for v in clf.model_.parameters() if v.requires_grad], lr=1e-5, weight_decay=.01)
        batch, epoch = next(kernels.meta_batches(x, y, seed=666, chunk_size=args.rows))
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
        started = time.monotonic()
        details = kernels.update(clf.model_, batch, optimizer, args.device)
        torch.cuda.synchronize(args.device)
        result.update(passed=True, trainable_parameters=count, elapsed_seconds=time.monotonic()-started,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device),
                      parameter_digest_changed=before != base.parameter_digest(clf), **details)
    except Exception as error:
        result.update(passed=False, error=repr(error))
        base.write_json(out, result)
        raise
    base.write_json(out, result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
