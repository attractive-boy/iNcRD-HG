"""Recover the two affine degrees of freedom removed by row min-max."""
import json
import numpy as np
import pandas as pd
import torch
from common import HERE, CLEAN, sha256

ORIGINAL = HERE.parents[3] / 'incrd-orig/2.construct_network'
def main():
    assert not (HERE / 'covariates.npz').exists()
    graph = torch.load(CLEAN / 'graph.pt', map_location='cpu')
    expected = json.loads((CLEAN / 'groups_provenance.json').read_text())['original_source_hashes']
    for file, h in expected.items():
        assert sha256(ORIGINAL / file) == h
    names = pd.read_csv(ORIGINAL / 'miRNA.csv').iloc[:, 0].str.upper().tolist()
    drug_table = pd.read_csv(ORIGINAL / 'Drug.csv')
    ids = drug_table.CID.astype(int).tolist()
    expression = pd.read_csv(ORIGINAL / 'MiExp.csv', index_col=0).T
    assert expression.index.str.upper().tolist() == names
    expression = expression[sorted(expression.columns, key=lambda s: int(s.split('-')[-1]))]
    chem = pd.read_csv(ORIGINAL / 'drug_features-chemBertA.csv', index_col=0)
    assert chem.index.astype(int).tolist() == ids
    arrays, records = {}, {}
    rng = np.random.RandomState(20261007)
    for name, frame in [('miRNA', expression), ('drug', chem)]:
        raw = frame.values.astype(np.float32)
        low, span = raw.min(1), np.ptp(raw, axis=1)
        shape = (raw - low[:, None]) / (span[:, None] + 1e-8)
        np.testing.assert_array_equal(shape, graph[name].x.numpy())
        stats = np.stack([low, span], 1)
        mean, scale = stats.mean(0), stats.std(0).clip(1e-6)
        standardized = (stats - mean) / scale
        recovered_stats = standardized * scale + mean
        recovered = shape * (recovered_stats[:, 1, None] + 1e-8) + recovered_stats[:, 0, None]
        np.testing.assert_allclose(recovered, raw, atol=1e-5, rtol=1e-5)
        permutation = rng.permutation(len(raw))
        arrays.update({name + '_stats': standardized, name + '_mean': mean,
                       name + '_scale': scale, name + '_permutation': permutation})
        records[name] = dict(shape=list(raw.shape), original_feature_replay_exact=True,
                             inverse_max_abs_error=float(abs(recovered - raw).max()),
                             minimum_range=float(span.min()), maximum_range=float(span.max()),
                             stats_mean=mean.tolist(), stats_scale=scale.tolist(),
                             permutation_fixed_points=int((permutation == np.arange(len(raw))).sum()))
    np.savez_compressed(HERE / 'covariates.npz', **arrays)
    result = dict(status='passed', records=records,
                  original_source_hashes={str(ORIGINAL / f): h for f, h in expected.items()},
                  covariates_sha256=sha256(HERE / 'covariates.npz'),
                  graph_sha256=sha256(CLEAN / 'graph.pt'),
                  data_sha256=sha256(CLEAN / 'inner_data.npz'),
                  prepare_sha256=sha256(__file__),
                  no_labels_used=True, no_lncrna=True,
                  drug_name_coverage=dict(unique_names=int(drug_table.Drug_Name.nunique()),
                                          used_for_training=False))
    (HERE / 'input_manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
if __name__ == '__main__': main()
