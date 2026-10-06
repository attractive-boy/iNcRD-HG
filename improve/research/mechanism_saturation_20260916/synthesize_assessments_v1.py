"""Read-only synthesis of every sealed graph-mechanism assessment.

This script does not train a model, touch evaluation labels, or modify any
sealed artifact. It aggregates the already-saved MiDrug deltas that each study
wrote next to its frozen protocol, so the mechanism-level contribution can be
separated from the shared residual-branch contribution.
"""
from datetime import datetime, timezone
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCORE_GLOB = 'results/anchor_priority_signed_regulatory_v1_20260916/runs/MiDrug/full/lr0.001/s666/scores.npz'


def stage_of(path):
    name = path.upper()
    if 'PARTITION' in name or 'OUTER' in name:
        return 'partition_or_outer'
    if 'HISTORICAL' in name:
        return 'historical'
    if 'SCREEN' in name:
        return 'screen'
    return 'other'


def collect():
    rows = []
    for path in glob.glob(str(ROOT / 'results/**/*.json'), recursive=True):
        try:
            record = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or 'means' not in record or 'primary_kind' not in record:
            continue
        try:
            mi = record['means']['MiDrug']
            full = mi[record['primary_kind']]
            controls = {k: v for k, v in mi.items() if k != record['primary_kind']}
        except (KeyError, TypeError):
            continue
        row = dict(path=str(Path(path).relative_to(ROOT)), stage=stage_of(path),
            primary=record['primary_kind'],
            full_delta_auc=full['mean']['auc'], full_delta_ap=full['mean']['ap'],
            performance_gate=record.get('performance_gate'),
            mechanism_gate=record.get('primary_mechanism_gate'),
            declared_gate=record.get('declared_gate'))
        margins = []
        for name, value in controls.items():
            row['minus_' + name] = [full['mean']['auc'] - value['mean']['auc'],
                                    full['mean']['ap'] - value['mean']['ap']]
            margins.append((full['mean']['auc'] - value['mean']['auc'],
                            full['mean']['ap'] - value['mean']['ap']))
        if margins:
            row['worst_control_auc'] = min(m[0] for m in margins)
            row['worst_control_ap'] = min(m[1] for m in margins)
        rows.append(row)
    return rows


def fusion_optimum():
    path = ROOT / SCORE_GLOB
    if not path.exists():
        return None
    z = np.load(path)
    y, anchor, residual = z['y'], z['anchor'], z['standalone']
    out = dict(correlation=float(np.corrcoef(anchor, residual)[0, 1]),
               residual_auc=float(roc_auc_score(y, residual)),
               residual_ap=float(average_precision_score(y, residual)),
               anchor_auc=float(roc_auc_score(y, anchor)),
               anchor_ap=float(average_precision_score(y, anchor)), weights={})
    for weight in (-0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        value = anchor + weight * residual
        out['weights'][str(weight)] = dict(auc=float(roc_auc_score(y, value)),
                                           ap=float(average_precision_score(y, value)))
    return out


def main():
    rows = collect()
    with_margin = [r for r in rows if 'worst_control_auc' in r]
    perf = [r for r in rows if r['performance_gate']]
    mech = [r for r in rows if r['mechanism_gate']]
    gate = [r for r in rows if r['declared_gate']]
    full_auc = np.array([r['full_delta_auc'] for r in rows])
    full_ap = np.array([r['full_delta_ap'] for r in rows])
    margin_auc = np.array([r['worst_control_auc'] for r in with_margin])
    margin_ap = np.array([r['worst_control_ap'] for r in with_margin])
    result = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_globs='results/**/*.json with means+primary_kind',
        assessment_count=len(rows),
        performance_gate_true=int(len(perf)),
        mechanism_gate_true=int(len(mech)),
        declared_gate_true=int(len(gate)),
        full_residual_gain=dict(
            auc_mean=float(full_auc.mean()), auc_median=float(np.median(full_auc)),
            auc_min=float(full_auc.min()), auc_max=float(full_auc.max()),
            ap_mean=float(full_ap.mean()), ap_median=float(np.median(full_ap)),
            ap_min=float(full_ap.min()), ap_max=float(full_ap.max())),
        mechanism_margin=dict(
            count=len(with_margin),
            auc_mean=float(margin_auc.mean()), auc_positive_fraction=float((margin_auc > 0).mean()),
            ap_mean=float(margin_ap.mean()), ap_positive_fraction=float((margin_ap > 0).mean())),
        fusion_optimum=fusion_optimum(),
        rows=rows)
    Path(HERE / 'SYNTHESIS_V1.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
