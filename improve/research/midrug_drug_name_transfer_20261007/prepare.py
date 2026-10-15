"""Use the previous frozen TF-IDF exactly; no vocabulary or similarity tuning."""
import json
import numpy as np
from common import HERE, CLEAN, sha256
PREVIOUS=HERE.parent/'midrug_drug_name_20261007'
def main():
    assert not (HERE/'covariates.npz').exists()
    previous=json.loads((PREVIOUS/'input_manifest.json').read_text())
    assert sha256(PREVIOUS/'covariates.npz')==previous['covariates_sha256']
    assert not json.loads((PREVIOUS/'selection.json').read_text())['promote_to_outer_validation']
    with np.load(PREVIOUS/'covariates.npz') as z:
        features=z['features'].astype(np.float64);permutation=z['permutation']
    kernel=features@features.T;np.fill_diagonal(kernel,0.)
    assert (kernel>=0).all() and np.isfinite(kernel).all()
    np.testing.assert_array_equal(kernel,kernel.T)
    permuted=kernel[permutation][:,permutation]
    np.testing.assert_allclose(permuted,features[permutation]@features[permutation].T-np.diag(np.diag(features[permutation]@features[permutation].T)),atol=1e-12)
    assert np.diag(kernel).sum()==0 and np.diag(permuted).sum()==0
    np.savez_compressed(HERE/'covariates.npz',aligned=kernel,permuted=permuted,zero=np.zeros_like(kernel))
    meta=dict(status='passed',covariates_sha256=sha256(HERE/'covariates.npz'),
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        original_source_hashes=previous['original_source_hashes'],
        frozen_previous_covariates_sha256=previous['covariates_sha256'],prepare_sha256=sha256(__file__),
        zero_diagonal_exact=True,symmetric=True,nonzero_directed_pairs=int((kernel>0).sum()),
        no_labels_used=True,no_lncrna=True,tfidf_refitted=False)
    (HERE/'input_manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
