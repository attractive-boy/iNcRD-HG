"""Fitting-only gradient probe, not a model-selection or performance experiment."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
from improve.tabular_foundation import DEFAULT_CHECKPOINT, digest
from improve.tabicl_adaptation_modules import group_digests


class InputProbe(nn.Module):
    def __init__(self, shared_features=31, rank=8):
        super().__init__()
        self.shared_features = shared_features
        self.down = nn.Linear(shared_features, rank, bias=False)
        self.up = nn.Linear(rank, shared_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        core, rest = x[..., :self.shared_features], x[..., self.shared_features:]
        return torch.cat([core + self.up(torch.tanh(self.down(core))), rest], dim=-1)


def main():
    torch.manual_seed(137)
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    reference = REPO / 'improve/results/numeric_embedding_20260912/features/inner'
    names = {task: json.loads((reference / task / 'manifest.json').read_text())['names']
             for task in ('LncDrug', 'MiDrug')}
    assert names['LncDrug'][:31] == names['MiDrug'][:31]
    loader = TabICLClassifier(model_path=DEFAULT_CHECKPOINT, allow_auto_download=False,
                             n_estimators=1, device=str(device), use_amp=False, use_fa3=False)
    loader._load_model()
    model = loader.model_.to(device).requires_grad_(False)
    model.train()  # The inference path may detach activations or use detached caches.
    before = group_digests(model)
    adapter = InputProbe().to(device)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Two small fitting-only tensor-interface checks; no optimizer steps, validation scoring, or test reads.',
        'checkpoint': DEFAULT_CHECKPOINT, 'checkpoint_sha256': digest(DEFAULT_CHECKPOINT),
        'probe_source_sha256': digest(__file__), 'device': str(device),
        'shared_prefix_features': names['LncDrug'][:31],
        'task_specific_suffix_names': {task: value[31:] for task, value in names.items()},
        'trainable_probe_parameters': sum(x.numel() for x in adapter.parameters()),
        'preprocessing': 'Use existing task input view; support-only z-score and fixed clipping for this probe. No feature shuffling.',
        'attention_backend': 'math', 'tasks': {},
    }
    for task in ('LncDrug', 'MiDrug'):
        source = reference / task / 'arrays.npz'
        feature_key = 'x_train' if task == 'LncDrug' else 'raw_train'
        with np.load(source, allow_pickle=False) as archive:
            features = archive[feature_key]
            labels = archive['y_train']
            fold_ids = archive['fold_ids']
        indices = np.flatnonzero(fold_ids == 0)
        rng = np.random.default_rng(137)
        indices = rng.permutation(indices)[:128]
        train_size = 96
        x_np, y_np = features[indices].astype(np.float32), labels[indices]
        assert len(np.unique(y_np[:train_size])) == 2
        assert len(np.unique(y_np[train_size:])) == 2
        mu = x_np[:train_size].mean(0)
        scale = np.maximum(x_np[:train_size].std(0), 1e-5)
        x_np = np.clip((x_np - mu) / scale, -8, 8)
        x = torch.tensor(x_np, device=device)[None].requires_grad_(True)
        ys = torch.tensor(y_np[:train_size], device=device).float()[None]
        yq = torch.tensor(y_np[train_size:], device=device).long()
        adapter.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
        with sdpa_kernel(SDPBackend.MATH):
            torch.manual_seed(137)
            baseline = model(x.detach(), ys)[..., :2].detach()
            torch.manual_seed(137)
            logits = model(adapter(x), ys)[..., :2]
            loss = F.cross_entropy(logits.reshape(-1, 2), yq)
            loss.backward()
        gradient = torch.cat([p.grad.reshape(-1) for p in adapter.parameters()])
        row = {
            'source': str(source.relative_to(REPO)), 'source_sha256': digest(source),
            'arrays_read': [feature_key, 'y_train', 'fold_ids'],
            'indices': indices.tolist(), 'graph_fold': 0, 'support_rows': train_size,
            'query_rows': len(yq), 'identity_max_logit_error': float((logits.detach() - baseline).abs().max()),
            'loss': float(loss.detach()), 'adapter_gradient_norm': float(gradient.norm()),
            'support_input_gradient_norm': float(x.grad[:, :train_size].norm()),
            'query_input_gradient_norm': float(x.grad[:, train_size:].norm()),
            'finite_gradient': bool(torch.isfinite(gradient).all()),
            'backbone_has_any_gradient': any(p.grad is not None for p in model.parameters()),
            'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
        }
        assert row['identity_max_logit_error'] == 0.
        assert row['finite_gradient'] and row['adapter_gradient_norm'] > 0
        assert not row['backbone_has_any_gradient']
        report['tasks'][task] = row
        print(task, {key: row[key] for key in ('identity_max_logit_error', 'adapter_gradient_norm', 'peak_allocated_mib')}, flush=True)
    after = group_digests(model)
    report['backbone_group_digests'] = before
    report['backbone_unchanged'] = before == after
    assert report['backbone_unchanged']
    report['probe_passed'] = True
    report['limitation'] = 'This establishes gradient access through the frozen backbone only; it does not establish AUC/AP gains or verify the full ensemble/cache integration.'
    (ROOT / 'input_adapter_probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
