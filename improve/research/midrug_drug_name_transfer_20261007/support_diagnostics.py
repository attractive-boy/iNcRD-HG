"""Fixed-kernel support coverage only; no fitting, scoring or model selection."""
import json
import numpy as np
from common import HERE,CLEAN,load_inner,sha256

def main():
 assert not (HERE/'support_diagnostics.json').exists()
 meta=json.loads((HERE/'input_manifest.json').read_text())
 assert sha256(HERE/'covariates.npz')==meta['covariates_sha256']
 records=[]
 with np.load(HERE/'covariates.npz') as z:
  for fold in range(3):
   data=load_inner(fold)
   A=np.zeros((605,216));B=np.zeros_like(A)
   pos=data['train_pairs'][data['train_y']==1];neg=data['train_pairs'][data['train_y']==0]
   A[pos[:,0],pos[:,1]]=1;B[neg[:,0],neg[:,1]]=1
   assert not np.any(A*B)
   r,d=data['val_pairs'].T
   for name in ['aligned','permuted']:
    kernel=z[name];assert not np.count_nonzero(np.diag(kernel))
    P=(A@kernel)[r,d];N=(B@kernel)[r,d]
    count=((A+B)@(kernel>0))[r,d]
    records.append(dict(fold=fold,kernel=name,queries=len(r),fraction_any_support=float(np.mean(count>0)),
     fraction_positive_support=float(np.mean(P>0)),fraction_negative_support=float(np.mean(N>0)),
     fraction_both=float(np.mean((P>0)&(N>0))),median_support_count=float(np.median(count)),
     mean_support_count=float(np.mean(count)),mean_positive_mass=float(P.mean()),mean_negative_mass=float(N.mean())))
 result=dict(records=records,no_fitting=True,no_auc_ap_reselection=True,no_heldout_labels_used=True,
  sources={str(p):sha256(p) for p in [HERE/'support_diagnostics.py',HERE/'covariates.npz',CLEAN/'inner_data.npz']},
  no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 (HERE/'support_diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(records,indent=2))
if __name__=='__main__':main()
