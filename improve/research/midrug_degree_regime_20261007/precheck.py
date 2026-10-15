"""Read-only error decomposition by fixed training-degree strata."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, CLEAN, load_inner, sha256


def degree(data):
    positives=data['train_pairs'][data['train_y']==1]
    dr=np.bincount(positives[:,0],minlength=605);dd=np.bincount(positives[:,1],minlength=216)
    result={}
    keys=set(map(tuple,positives))
    for part in ['train','val']:
        p=data[part+'_pairs'];q=np.array([tuple(x) in keys for x in p],dtype=np.int64)
        result[part]=np.stack((dr[p[:,0]]-q,dd[p[:,1]]-q),axis=1)
    threshold=np.median(result['train'],axis=0)
    groups={part:(x[:,0]>threshold[0]).astype(int)*2+(x[:,1]>threshold[1]).astype(int) for part,x in result.items()}
    return threshold,result,groups


def decomposition(y,p,g):
    pos=y==1;neg=~pos;mass=np.zeros((4,4));denominator=int(pos.sum())*int(neg.sum())
    for i in range(4):
        a=p[pos&(g==i)]
        for j in range(4):
            b=np.sort(p[neg&(g==j)])
            lo=np.searchsorted(b,a,side='left');hi=np.searchsorted(b,a,side='right')
            mass[i,j]=(len(b)-hi+.5*(hi-lo)).sum()/denominator
    order=np.argsort(-p,kind='stable');ps=p[order];ys=y[order]
    ends=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(ps)-1]
    precision=np.cumsum(ys)[ends]/(ends+1)
    counts=np.diff(np.r_[-1,ends]);per_item=np.repeat(1-precision,counts)
    apmass=np.bincount(g[order][ys==1],weights=per_item[ys==1]/pos.sum(),minlength=4)
    assert abs(mass.sum()-(1-roc_auc_score(y,p)))<1e-12
    assert abs(apmass.sum()-(1-average_precision_score(y,p)))<1e-12
    return dict(auc_error_by_positive_group_negative_group=mass.tolist(),within_group_auc_error_fraction=float(np.trace(mass)/mass.sum()),
                ap_error_by_positive_group=apmass.tolist(),ap_error_group_fractions=(apmass/apmass.sum()).tolist())


def stats(y,p,g):
    rows=[]
    for i in range(4):
        take=g==i;yy=y[take];pp=p[take];positive=int(yy.sum())
        metric=dict(auc=None,ap=None)
        if 0<positive<len(yy):metric=dict(auc=float(roc_auc_score(yy,pp)),ap=float(average_precision_score(yy,pp)))
        clipped=np.clip(pp,1e-7,1-1e-7)
        rows.append(dict(group=i,count=len(yy),positive=positive,positive_rate=float(yy.mean()) if len(yy) else None,
                         bce=float(-(yy*np.log(clipped)+(1-yy)*np.log1p(-clipped)).mean()) if len(yy) else None,**metric))
    return rows


def main():
    # Includes tied probabilities, which must be treated as tied thresholds for AP.
    decomposition(np.array([1,0,1,0,1,0]),np.array([.9,.7,.7,.7,.4,.1]),np.array([0,1,2,3,0,1]))
    prior=HERE.parent/'midrug_training_size_20261007';v=json.loads((prior/'verification.json').read_text());assert v['status']=='passed'
    sources=[HERE/'PRECHECK_PLAN.md',HERE/'precheck.py',HERE/'common.py',CLEAN/'inner_data.npz',CLEAN/'graph.pt',prior/'verification.json',prior/'archive_recheck.json']
    records=[]
    for f in range(3):
        data=load_inner(f);threshold,degrees,groups=degree(data)
        for name in ['control','unmasked10']:
            record=next(r for r in v['records'] if r['config']==name and r['size']=='full' and r['fold']==f)
            meta=Path(record['metadata_path']);assert sha256(meta)==record['metadata_sha256']
            path=prior/'replay_predictions'/(record['tag']+'.npz');assert sha256(path)==record['replay_predictions_sha256']
            sources.extend([meta,path]);out=dict(config=name,fold=f,thresholds=threshold.tolist())
            with np.load(path) as z:
                for part in ['train','val']:
                    for k in ['pairs','y','indices']:np.testing.assert_array_equal(z[part+'_'+k],data[part+'_'+k])
                    y,p=data[part+'_y'],z[part+'_probability']
                    out[part]=stats(y,p,groups[part])
                    if part=='val':out['error_decomposition']=decomposition(y,p,groups[part])
            records.append(out)
    result=dict(status='passed',records=records,source_hashes={str(p):sha256(p) for p in sources},new_fits=0,new_checkpoint_replays=0,
        uses_verified_saved_predictions=True,original_validation_evaluated=False,test_evaluated=False,no_lncrna=True)
    (HERE/'precheck.json').write_text(json.dumps(result,indent=2));print(json.dumps(records,indent=2))


if __name__=='__main__':main()
