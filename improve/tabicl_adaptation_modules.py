"""Task-episodic adaptation kernels using TabICL's official meta-batch builder.

The official training-data module is loaded directly so inference's frozen
environment needs neither a newer sklearn nor the optional transformers trainer.
Installed TabICL source is never patched.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F


def group_digests(model):
    result = {}
    for name in ('col_embedder', 'row_interactor', 'icl_predictor'):
        digest = hashlib.sha256()
        for key, value in getattr(model, name).state_dict().items():
            array = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode())
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        result[name] = digest.hexdigest()
    return result


@contextmanager
def inference_rng_guard(device):
    """Validation and persistence must not change subsequent training draws."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def official_meta_module():
    import tabicl
    name = '_incrd_tabicl_official_meta_data'
    if name not in sys.modules:
        path = Path(tabicl.__file__).parent/'_finetune'/'data.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def set_trainable(model, mode):
    if mode not in ('all', 'icl'):
        raise ValueError('Unknown adaptation mode')
    model.requires_grad_(True)
    if mode == 'icl':
        model.col_embedder.requires_grad_(False)
        model.row_interactor.requires_grad_(False)
    model.train()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def meta_batches(x, y, *, seed, chunk_size=2048, n_estimators=2):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.shape != (len(x),) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected binary fitting rows')
    if not np.isfinite(x).all() or chunk_size < 50:
        raise ValueError('Invalid features or chunk size')
    module = official_meta_module()
    epoch = 0
    while True:
        count = 0
        for batch in module.iter_epoch_meta_batches(
            x, y.astype(np.int64), classification=True, n_estimators=n_estimators,
            max_chunk_size=chunk_size, query_ratio=.2, epoch_seed=seed+epoch,
            preprocessing_seed=seed, norm_methods=None, feat_shuffle_method='latin',
            class_shuffle_method='shift', outlier_threshold=4.,
        ):
            count += 1
            yield batch, epoch
        if not count:
            raise ValueError('No valid meta-batches')
        epoch += 1


def learning_rate(step, total_steps, base_lr):
    """Zero-based step; outer refits retain the original full schedule horizon."""
    if not 0 <= step < total_steps or base_lr <= 0:
        raise ValueError('Invalid optimizer schedule')
    warmup = max(1, int(round(total_steps*.1)))
    scale = (step+1)/warmup if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total_steps-warmup)))
    return base_lr*scale


def update(model, batch, optimizer, device):
    batch = official_meta_module().move_meta_batch(batch, torch.device(device))
    if batch.y_train.shape[1] != batch.train_size or batch.X.shape[1] != batch.train_size+batch.y_query.shape[1]:
        raise ValueError('Context/query boundary differs')
    if set(torch.unique(batch.y_train).tolist()) != {0., 1.} or set(torch.unique(batch.y_query).tolist()) != {0, 1}:
        raise ValueError('Both classes must occur in context and query')
    model.train()  # Frozen encoders must also use the training forward path.
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.X, batch.y_train.float())
    if logits.shape[:2] != batch.y_query.shape:
        raise ValueError('Model returned the wrong query logits')
    loss = F.cross_entropy(logits[..., :2].reshape(-1, 2), batch.y_query.long().reshape(-1))
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite adaptation loss')
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    if not any(p.grad is not None for p in params) or not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
        raise FloatingPointError('Missing or nonfinite gradients')
    grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in params):
        raise FloatingPointError('Nonfinite adapted weights')
    return dict(loss=float(loss.detach()), grad_norm=float(grad_norm),
                context_rows=int(batch.train_size), query_rows=int(batch.y_query.shape[1]))
