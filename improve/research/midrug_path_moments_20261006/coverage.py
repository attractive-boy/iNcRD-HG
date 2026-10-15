"""Descriptive complete-path support; no labels used to choose a configuration."""
import json
import numpy as np
from common import HERE,load_inner,sha256,CLEAN


def main():
    rows=[]
    for fold in range(3):
        data=load_inner(fold);A=np.zeros((605,216),dtype=np.float64)
        ix=data['train_pairs'][data['train_y']==1];A[ix[:,0],ix[:,1]]=1
        for part in ['train','val']:
            r,d=data[part+'_pairs'].T;mr=A[:,d].T.copy();md=A[r].copy();ix=np.arange(len(r));mr[ix,r]=0;md[ix,d]=0
            wr=mr*(md@A.T);wd=md*(mr@A);n=wr.sum(1);np.testing.assert_array_equal(n,wd.sum(1))
            many_r=(wr>0).sum(1);many_d=(wd>0).sum(1)
            rows.append(dict(fold=fold,part=part,queries=len(r),with_path=int((n>0).sum()),
                distinct_intermediates_on_both_sides=int(((many_r>1)&(many_d>1)).sum()),
                path_quantiles=np.quantile(n,[0,.25,.5,.75,.9,1]).tolist()))
    (HERE/'coverage.json').write_text(json.dumps(dict(records=rows,inner_data_sha256=sha256(CLEAN/'inner_data.npz'),
        source_sha256=sha256(HERE/'coverage.py'),used_for_selection=False,no_lncrna=True,
        interpretation='Two distinct intermediates on both sides are necessary, not sufficient, for a nonzero path covariance.'),indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
