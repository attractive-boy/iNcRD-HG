"""Describe pure-miRNA ranking errors using only original-training inner folds."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,sha256

def metric(y,p):return dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)))

def summarize_errors(data,p):
    y=data['val_y'];pairs=data['val_pairs'];pos=data['train_pairs'][data['train_y']==1]
    dr=np.bincount(pos[:,0],minlength=605);dd=np.bincount(pos[:,1],minlength=216)
    ip,ineg=np.flatnonzero(y==1),np.flatnonzero(y==0)
    # Each tied positive-negative score contributes half an error.
    mistakes=(p[ip,None]<p[None,ineg]).astype(np.float64)+.5*(p[ip,None]==p[None,ineg])
    per_positive=mistakes.sum(1)/mistakes.size
    assert abs(per_positive.sum()-(1-roc_auc_score(y,p)))<1e-12
    cuts=[(-1,0),(0,1),(1,4),(4,15),(15,10**6)];buckets=[]
    for low,high in cuts:
        take=(dr[pairs[ip,0]]>low)&(dr[pairs[ip,0]]<=high)
        buckets.append(dict(miRNA_positive_training_degree=f'{low+1}-{high}',positive_queries=int(take.sum()),auc_error_mass=float(per_positive[take].sum())))
    conditional={}
    for axis,name in [(0,'within_miRNA'),(1,'within_drug')]:
        total_errors,total_comparisons,groups=0.,0,0
        for node in np.unique(pairs[:,axis]):
            pi=ip[pairs[ip,axis]==node];ni=ineg[pairs[ineg,axis]==node]
            if len(pi) and len(ni):
                total_errors+=(p[pi,None]<p[None,ni]).sum()+.5*(p[pi,None]==p[None,ni]).sum()
                total_comparisons+=len(pi)*len(ni);groups+=1
        conditional[name]=dict(auc=1-total_errors/total_comparisons,positive_negative_comparisons=total_comparisons,groups=groups,
            share_of_all_comparisons=total_comparisons/mistakes.size)
    return dict(metrics=metric(y,p),positive_degree_error_buckets=buckets,conditional=conditional,
        drug_positive_degree_only=metric(y,dd[pairs[:,1]]),miRNA_positive_degree_only=metric(y,dr[pairs[:,0]]))

def main():
    rows=[]
    for fold in range(3):
        data=load_inner(fold)
        for name,root in [('control',CLEAN),('masked1',HERE)]:
            path=root/'results'/f'{name}_f{fold}_s42_inner.npz'
            with np.load(path) as z:
                np.testing.assert_array_equal(z['pairs'],data['val_pairs']);np.testing.assert_array_equal(z['y'],data['val_y'])
                r=summarize_errors(data,z['probability'])
            rows.append(dict(fold=fold,config=name,prediction_sha256=sha256(path),**r))
    result=dict(scope='descriptive ranking diagnostics on original-training inner held-outs only; not causal attribution or final test analysis',records=rows,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'error_audit.json').write_text(json.dumps(result,indent=2))
    for name in ['control','masked1']:
        rs=[r for r in rows if r['config']==name]
        print(name,{k:float(np.mean([r['conditional'][k]['auc'] for r in rs])) for k in ['within_miRNA','within_drug']})
        print('degree error mass',[(rs[0]['positive_degree_error_buckets'][i]['miRNA_positive_training_degree'],float(np.mean([r['positive_degree_error_buckets'][i]['auc_error_mass'] for r in rs]))) for i in range(5)])

if __name__=='__main__':main()
