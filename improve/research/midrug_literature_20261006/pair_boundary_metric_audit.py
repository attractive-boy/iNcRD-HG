"""Supplementary ranking-error attribution after the frozen boundary precheck.

This diagnostic was added after inspecting threshold-based error concentration;
it is not a preregistered training experiment or model selection criterion.
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from pair_boundary_precheck import HERE, load_inner, make_adapter, representations, seed_all, sha256


def attribution(y, p, band):
    y, p, band = np.asarray(y), np.asarray(p), np.asarray(band)
    pi, ni = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    wrong = (p[pi, None] < p[ni]).astype(float)+.5*(p[pi, None] == p[ni])
    groups = {}
    for positive_band in [False, True]:
        for negative_band in [False, True]:
            mask = (band[pi, None] == positive_band) & (band[ni] == negative_band)
            groups[f'positive_band{int(positive_band)}_negative_band{int(negative_band)}'] = dict(
                comparison_count=int(mask.sum()), inversion_mass=float(wrong[mask].sum()),
                global_auc_deficit=float(wrong[mask].sum()/wrong.size))
    auc_deficit = float(wrong.mean())
    assert abs(auc_deficit-(1-roc_auc_score(y, p))) < 1e-12
    any_band = band[pi, None] | band[ni]
    # sklearn AP uses precision at the end of each equal-score group.
    order = np.argsort(-p, kind='stable')
    sorted_y, sorted_p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(order)-1]
    precision = np.cumsum(sorted_y)[ends]/(ends+1)
    group_index = np.searchsorted(ends, np.arange(len(order)), side='left')
    positive_deficits = (1-precision[group_index])*(sorted_y == 1)/len(pi)
    ap_deficit = float(positive_deficits.sum())
    assert abs(ap_deficit-(1-average_precision_score(y, p))) < 1e-12
    return dict(auc_deficit=auc_deficit, auc_groups=groups,
                any_band_comparison_fraction=float(any_band.mean()),
                any_band_inversion_fraction=float(wrong[any_band].sum()/max(1e-12, wrong.sum())),
                ap_deficit=ap_deficit,
                band_positive_fraction=float(band[pi].mean()),
                band_positive_ap_deficit_fraction=float(positive_deficits[band[order]].sum()/max(1e-12, ap_deficit)))


@torch.no_grad()
def main():
    preplan = HERE/'pair_boundary_precheck_plan.json'
    precheck = HERE/'pair_boundary_precheck.json'
    plan, pre = json.loads(preplan.read_text()), json.loads(precheck.read_text())
    assert pre['plan_sha256'] == sha256(preplan)
    # A tie example validates grouping against both library metrics.
    attribution(np.array([1, 0, 1, 0, 1]), np.array([.8, .8, .5, .2, .2]),
                np.array([True, False, True, False, True]))
    rows = []
    for fold in range(3):
        data = load_inner(fold)
        for name, refs in plan['controls'].items():
            record = refs[str(fold)]
            checkpoint = Path(record['path']).with_suffix('.pt')
            assert sha256(checkpoint) == record['checkpoint_sha256']
            seed_all(42)
            a = make_adapter(data, name, 'cuda:0')
            a.model.load_state_dict(torch.load(checkpoint, map_location='cuda:0', weights_only=True))
            a.model.eval()
            x, _ = representations(a, 'train')
            z, probability = representations(a, 'val')
            mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(.001)
            x, z = (x-mean)/scale, (z-mean)/scale
            centered = x-x.mean(0)
            covariance = centered.T@centered/(len(x)-1)
            ridge = .1*covariance.diag().mean().clamp_min(1e-8)
            covariance = covariance+ridge*torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
            y = torch.as_tensor(data['train_y'], device=x.device)
            c0, c1 = x[y == 0].mean(0), x[y == 1].mean(0)
            direction = torch.linalg.solve(covariance, c1-c0)
            midpoint = .5*(c0+c1)
            threshold = torch.quantile(((x-midpoint)@direction).abs(), .25)
            band = (((z-midpoint)@direction).abs() <= threshold).cpu().numpy()
            ref = next(r for r in pre['records'] if r['config'] == name and r['fold'] == fold)
            assert int(band.sum()) == ref['inner_validation']['band_count']
            # The source encoder uses float32 CUDA sparse reductions. Validate
            # exact group membership against the saved cutoff, not bit-identical
            # derived float64 thresholds across fresh encoder evaluations.
            fixed_band = (((z-midpoint)@direction).abs() <= ref['band_abs_margin_threshold']).cpu().numpy()
            np.testing.assert_array_equal(band, fixed_band)
            assert int((((probability >= .5) != data['val_y']) & band).sum()) == ref['inner_validation']['band_error_count']
            assert abs(roc_auc_score(data['val_y'], probability)-ref['inner_auc']) < 1e-12
            assert abs(average_precision_score(data['val_y'], probability)-ref['inner_ap']) < 1e-12
            rows.append(dict(config=name, fold=fold,
                             recomputed_threshold_difference=float(threshold)-ref['band_abs_margin_threshold'],
                             saved_cutoff_membership_equal=True,
                             **attribution(data['val_y'], probability, band)))
            del a
            torch.cuda.empty_cache()
    for path, digest in plan['source_hashes'].items():
        assert sha256(path) == digest
    result = dict(status='passed', records=rows, source_sha256=sha256(__file__),
                  precheck_sha256=sha256(precheck), no_refitting=True, no_selection=True,
                  original_validation_evaluated=False, test_evaluated=False,
                  limitation='Error concentration is descriptive, not evidence that the errors are learnable or recoverable. AUC-any-band and AP-positive-band fractions use different denominators.')
    (HERE/'pair_boundary_metric_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
