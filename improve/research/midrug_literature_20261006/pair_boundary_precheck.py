"""Frozen diagnostic of existing pure-miRNA pair embeddings, without refitting."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / 'midrug_conditioned_expression_20261006'
sys.path.insert(0, str(EXPERIMENT))
from common import load_inner, seed_all, sha256, CLEAN
from model import make_adapter


@torch.no_grad()
def representations(adapter, split):
    captured = {}
    hook = adapter.model.lin3.register_forward_pre_hook(
        lambda module, args: captured.update(z=args[0].detach()))
    try:
        logits = adapter.logits(split)
    finally:
        hook.remove()
    return captured['z'].double(), logits.sigmoid().cpu().numpy()


def summarize_group(y, probability, mismatch, band):
    errors = (probability >= .5) != y
    band, mismatch = band.cpu().numpy(), mismatch.cpu().numpy()
    return dict(count=len(y), error_count=int(errors.sum()),
                error_rate=float(errors.mean()), band_count=int(band.sum()),
                band_fraction=float(band.mean()),
                mean_knn_label_disagreement=float(mismatch.mean()),
                majority_opposite_knn_count=int((mismatch > .5).sum()),
                band_majority_opposite_knn_count=int(((mismatch > .5) & band).sum()),
                band_error_count=int((errors & band).sum()),
                fraction_errors_in_band=float((errors & band).sum()/max(1, errors.sum())),
                outside_band_error_count=int((errors & ~band).sum()),
                band_error_rate=float(errors[band].mean()) if band.any() else None,
                outside_band_error_rate=float(errors[~band].mean()) if (~band).any() else None)


@torch.no_grad()
def examine(a, data):
    a.model.eval()
    x, train_prob = representations(a, 'train')
    z, val_prob = representations(a, 'val')
    y = torch.as_tensor(data['train_y'], device=x.device, dtype=torch.long)
    vy = torch.as_tensor(data['val_y'], device=x.device, dtype=torch.long)
    # All geometry is determined solely by this fold's training pair features.
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(.001)
    x, z = (x-mean)/scale, (z-mean)/scale
    centered = x-x.mean(0)
    covariance = centered.T@centered/(len(x)-1)
    ridge = .1*covariance.diag().mean().clamp_min(1e-8)
    covariance = covariance+ridge*torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    c0, c1 = x[y == 0].mean(0), x[y == 1].mean(0)
    direction = torch.linalg.solve(covariance, c1-c0)
    midpoint = .5*(c0+c1)
    train_margin, val_margin = (x-midpoint)@direction, (z-midpoint)@direction
    threshold = torch.quantile(train_margin.abs(), .25)
    train_band, val_band = train_margin.abs() <= threshold, val_margin.abs() <= threshold
    # Standardized Euclidean 5-NN; exclude each training record itself.
    dist = torch.cdist(x, x)
    dist.fill_diagonal_(torch.inf)
    neighbors = dist.topk(5, largest=False).indices
    train_mismatch = (y[neighbors] != y[:, None]).double().mean(1)
    val_neighbors = torch.cdist(z, x).topk(5, largest=False).indices
    val_mismatch = (y[val_neighbors] != vy[:, None]).double().mean(1)
    return dict(train=summarize_group(data['train_y'], train_prob, train_mismatch, train_band),
                inner_validation=summarize_group(data['val_y'], val_prob, val_mismatch, val_band),
                inner_auc=roc_auc_score(data['val_y'], val_prob),
                inner_ap=average_precision_score(data['val_y'], val_prob),
                boundary_positive_count=int(((train_mismatch > .5) & train_band & (y == 1)).sum()),
                boundary_zero_count=int(((train_mismatch > .5) & train_band & (y == 0)).sum()),
                ridge=float(ridge), band_abs_margin_threshold=float(threshold))


def main():
    prior = json.loads((EXPERIMENT/'results/plan.json').read_text())
    assert json.loads((EXPERIMENT/'archive_recheck.json').read_text())['status'] == 'passed'
    plan = dict(status='frozen_diagnostic', controls=prior['controls'],
                source_hashes={**prior['source_hashes'], str(Path(__file__).resolve()): sha256(__file__)},
                graph_sha256=sha256(CLEAN/'graph.pt'), data_sha256=sha256(CLEAN/'inner_data.npz'),
                seed=42, folds=[0, 1, 2], representation='original lin3 input after query deletion',
                geometry='training-pair feature standardization, ridge covariance, centered equal-prior LDA slab',
                ridge_fraction=.1, std_floor=.001, training_abs_margin_quantile=.25, knn=5,
                boundary='training slab AND majority opposite observed label among5 training neighbors',
                checkpoint_refitting=False, candidate_selection=False, no_lncrna=True,
                original_validation_evaluated=False, test_evaluated=False)
    plan_path = HERE/'pair_boundary_precheck_plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    else:
        plan_path.write_text(json.dumps(plan, indent=2))
    rows = []
    for fold in range(3):
        data = load_inner(fold)
        for name, refs in plan['controls'].items():
            record = refs[str(fold)]
            path = Path(record['path'])
            metadata = json.loads(path.read_text())
            assert sha256(path) == record['metadata_sha256']
            assert sha256(path.with_suffix('.pt')) == record['checkpoint_sha256']
            seed_all(42)
            a = make_adapter(data, name, 'cuda:0')
            a.model.load_state_dict(torch.load(path.with_suffix('.pt'), map_location='cuda:0', weights_only=True))
            assert set(a.graph.node_types) == {'miRNA', 'drug'}
            result = examine(a, data)
            assert abs(result['inner_auc']-metadata['inner_auc']) < 1e-12
            assert abs(result['inner_ap']-metadata['inner_ap']) < 1e-12
            rows.append(dict(config=name, fold=fold, checkpoint_sha256=record['checkpoint_sha256'], **result))
            print(name, fold, 'train_boundary', result['boundary_positive_count']+result['boundary_zero_count'],
                  'inner_error_share_in_band', result['inner_validation']['fraction_errors_in_band'], flush=True)
            del a
            torch.cuda.empty_cache()
    for path, digest in plan['source_hashes'].items():
        assert sha256(path) == digest
    result = dict(status='passed', plan_sha256=sha256(plan_path), records=rows,
                  checkpoint_refitting=False, candidate_selection=False,
                  no_lncrna=True, original_validation_evaluated=False, test_evaluated=False,
                  limitation='Diagnostic thresholds fixed before reading output; no claim of BES replication or identifiable biology. Internal diagnostic labels are not used to refit checkpoints or define training geometry.')
    (HERE/'pair_boundary_precheck.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
