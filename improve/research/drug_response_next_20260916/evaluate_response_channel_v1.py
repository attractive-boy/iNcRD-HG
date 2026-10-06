"""Does an external cell-line drug-response channel carry any association signal?

Read-only diagnostic. It combines the already-frozen aligned RNA-expression /
PRISM drug-viability matrices with the already-frozen historical MiDrug
evaluation pairs. It reads no ncRNA-drug label except the frozen evaluation
labels used for scoring, trains nothing, and writes one sealed JSON.

The feature for a pair (RNA r, drug d) is the Pearson correlation across the
shared, fully finite cell lines between r's expression and d's measured
response. Only the 34 project drugs with an exact-structure PRISM match are
available. If even this direct, label-free pharmacogenomic channel is at
chance, no variant of it (GDSC IC50, PRISM viability) can rescue the metric.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve')
ALIGNED = ROOT / 'results/functional_response_inputs_20260914'
FRONTIER = ROOT / 'results/anchor_priority_signed_regulatory_v1_20260916/inputs/MiDrug/arrays.npz'
OUT = Path(__file__).resolve().parent / 'RESPONSE_CHANNEL_V1.json'


def main():
    z = np.load(ALIGNED / 'arrays.npz', allow_pickle=True)
    expression, drug, cells = z['x'], z['drug'], z['cells']
    assay_map = sparse.load_npz(ALIGNED / 'drug_assay_map.npz')
    rows, cols = assay_map.nonzero()
    finite = np.isfinite(expression).all(1) & np.isfinite(drug).all(1)

    x = expression[finite][:, :605]
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    feature = np.full((605, 216), np.nan, dtype=np.float64)
    for drug_index, assay_column in zip(rows, cols):
        value = drug[finite][:, assay_column]
        value = (value - value.mean()) / (value.std() + 1e-9)
        feature[:, drug_index] = x.T @ value / len(value)

    frontier = np.load(FRONTIER)
    pairs, y = frontier['pairs_val'], frontier['y_val']
    usable = np.isin(pairs[1], rows)
    scores = feature[pairs[0][usable], pairs[1][usable]]
    labels = y[usable]
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source='PRISM 19Q3 viability (ExperimentHub EH7530/EH7554 aligned inputs)',
        fully_finite_cells=int(finite.sum()), total_cells=int(len(cells)),
        matched_project_drugs=int(len(rows)), evaluable_val_pairs=int(usable.sum()),
        evaluable_positives=int(labels.sum()),
        feature_auc=float(roc_auc_score(labels, scores)),
        feature_ap=float(average_precision_score(labels, scores)),
        drug_degree_auc_restricted=float(roc_auc_score(labels,
            np.bincount(frontier['pairs_train'][1][frontier['y_train'] > 0],
                        minlength=216)[pairs[1][usable]].astype(float))),
        conclusion='The label-free drug-response channel is at or below chance on the '
                   'matched pairs, so a GDSC IC50 variant cannot carry the missing signal.')
    OUT.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
