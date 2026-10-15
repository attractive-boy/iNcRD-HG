"""Measure retained local context without accessing outer held-out data."""
import json
import numpy as np
from common import HERE,load_inner
from extract import Extractor


def main():
    rows=[]
    for fold in range(3):
        data=load_inner(fold);e=Extractor(data);P3=e.Cr@e.A
        for k,h in ((16,1),(32,2)):
            with np.load(HERE/'cache'/f'f{fold}_h{h}_k{k}.npz') as z:
                for part in ('train','val'):
                    pairs=data[part+'_pairs'];r,d=pairs.T;q=e.A[r,d]
                    p3=P3[r,d]-q*(e.dr[r]+e.dd[d]-1)
                    A=z[part+'_pos'].astype(np.float32)
                    kept=np.einsum('bi,bji,bj->b',A[:,0,:],A,A[:,:,0])
                    assert np.all(kept<=p3)
                    positive=p3>0
                    dr,dd=e.dr[r]-q,e.dd[d]-q
                    fr=A[:,0,:].sum(1)/np.maximum(dr,1);fd=A[:,:,0].sum(1)/np.maximum(dd,1)
                    rows.append(dict(fold=fold,part=part,k=k,hops=h,pairs=len(pairs),nonzero_full_p3=int(positive.sum()),
                        mean_p3_retention=float((kept[positive]/p3[positive]).mean()),
                        p3_retention_quantiles=np.quantile(kept[positive]/p3[positive],[0,.25,.5,.75,1]).tolist(),
                        fraction_complete_root_r_neighbors=float((fr[dr>0]==1).mean()),
                        fraction_complete_root_d_neighbors=float((fd[dd>0]==1).mean())))
    (HERE/'context_coverage.json').write_text(json.dumps(dict(status='passed',records=rows,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    print(json.dumps(rows[:4]),flush=True)

if __name__=='__main__':main()
