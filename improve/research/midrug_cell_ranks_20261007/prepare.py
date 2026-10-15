"""Original MiExp only: fix axes, ties and percentile convention before fitting."""
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from scipy.special import ndtri
from common import HERE, CLEAN, sha256

ORIGINAL=HERE.parents[3]/'incrd-orig/2.construct_network'
def gaussian_rank(x, axis):
    n=x.shape[axis]
    rank=rankdata(x,axis=axis,method='average')
    return (.5+.5*ndtri((rank-.5)/n)/ndtri(1.-.5/n)).astype(np.float32)

def independent_rank(x,axis):
    # Direct comparisons, independently of scipy sorting/rankdata.
    moved=np.moveaxis(x,axis,-1)
    less=(moved[...,None,:]<moved[...,:,None]).sum(-1)
    equal=(moved[...,None,:]==moved[...,:,None]).sum(-1)
    r=1+less+.5*(equal-1)
    n=x.shape[axis]
    out=(.5+.5*ndtri((r-.5)/n)/ndtri(1.-.5/n)).astype(np.float32)
    return np.moveaxis(out,-1,axis)

def main():
    assert not (HERE/'covariates.npz').exists()
    expected=json.loads((CLEAN/'groups_provenance.json').read_text())['original_source_hashes']
    for file in ['miRNA.csv','MiExp.csv']: assert sha256(ORIGINAL/file)==expected[file]
    names=pd.read_csv(ORIGINAL/'miRNA.csv').iloc[:,0].str.upper().tolist()
    frame=pd.read_csv(ORIGINAL/'MiExp.csv',index_col=0).T
    assert frame.index.str.upper().tolist()==names
    frame=frame[sorted(frame.columns,key=lambda s:int(s.split('-')[-1]))]
    raw=frame.values.astype(np.float32)
    graph=torch.load(CLEAN/'graph.pt',map_location='cpu')
    shape=(raw-raw.min(1,keepdims=True))/(np.ptp(raw,axis=1,keepdims=True)+1e-8)
    np.testing.assert_array_equal(shape,graph['miRNA'].x.numpy())
    arrays=dict(cell_gaussian=gaussian_rank(raw,0),
                cell_minmax=(raw-raw.min(0,keepdims=True))/(np.ptp(raw,axis=0,keepdims=True)+1e-8),
                row_gaussian=gaussian_rank(raw,1))
    toy=np.array([[1.,2.,2.,4.],[5.,2.,3.,4.],[2.,9.,2.,1.]])
    for axis in [0,1]:
        np.testing.assert_array_equal(gaussian_rank(toy,axis),independent_rank(toy,axis))
    for name,axis in [('cell_gaussian',0),('row_gaussian',1)]:
        np.testing.assert_array_equal(arrays[name],gaussian_rank(raw.astype(np.float64)**3,axis))
    # Check all miRNAs in three cells with a direct pairwise rank oracle.
    np.testing.assert_array_equal(arrays['cell_gaussian'][:,[0,468,937]],independent_rank(raw[:,[0,468,937]],0))
    for values in arrays.values():
        assert values.shape==(605,938) and np.isfinite(values).all()
        assert values.min()>=-1e-6 and values.max()<=1+1e-6
    np.savez_compressed(HERE/'covariates.npz',**arrays)
    meta=dict(status='passed',original_source_hashes={str(ORIGINAL/f):expected[f] for f in ['miRNA.csv','MiExp.csv']},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        covariates_sha256=sha256(HERE/'covariates.npz'),prepare_sha256=sha256(__file__),
        original_normalized_feature_replay_exact=True,independent_rank_oracle_passed=True,
        monotone_transform_invariance=True,axes=dict(cell_gaussian='rank605 miRNAs within each of938 cells',
            row_gaussian='rank938 cells within each miRNA',cell_minmax='minmax605 miRNAs within each cell'),
        ranges={k:[float(x.min()),float(x.max()),float(x.mean()),float(x.std())] for k,x in arrays.items()},
        no_lncrna=True,no_labels_used=True,no_cell_atlas_or_count_normalization=True)
    (HERE/'input_manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
