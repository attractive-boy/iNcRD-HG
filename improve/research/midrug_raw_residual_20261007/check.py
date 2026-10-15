"""Base-margin identity, independent tree traversal, and full query deletion checks."""
import json
import numpy as np
import torch
import xgboost as xgb
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,PARAMS,make_adapter,pca_coordinates,features,predict_margin,independent_tree_margin

def toy():
 x=np.array([[-2.,0],[-1,1],[0,0],[1,1],[2,0],[3,1]],dtype=np.float32);y=np.array([0,0,1,0,1,1],dtype=np.float32);base=np.array([-1.,.5,.1,-.2,.8,-.6],dtype=np.float32)
 dm=xgb.DMatrix(x,label=y,base_margin=base,nthread=2);params={**PARAMS,'min_child_weight':0.,'subsample':1.,'colsample_bytree':1.}
 b=xgb.Booster(params,[dm]);p0=b.predict(dm,output_margin=True)
 np.testing.assert_array_equal(p0,base)
 for i in range(5):b.update(dm,i)
 actual=b.predict(dm,output_margin=True);ref=independent_tree_margin(b,x,base)
 np.testing.assert_allclose(actual,ref,atol=2e-6,rtol=2e-6)
 shifted=xgb.DMatrix(x,base_margin=base+.7,nthread=2);delta=b.predict(shifted,output_margin=True)-actual
 np.testing.assert_allclose(delta,.7,atol=2e-6,rtol=2e-6)
 # Binary logistic gradient/Hessian used by booster agree with independent scalar derivatives.
 h=1e-4;z=base.astype(float);sigmoid=lambda q:1/(1+np.exp(-q))
 loss=lambda q:np.logaddexp(0,q)-y*q
 grad=(loss(z+h)-loss(z-h))/(2*h);hess=(sigmoid(z+h)-sigmoid(z-h))/(2*h)
 np.testing.assert_allclose(grad,sigmoid(z)-y,atol=1e-8,rtol=1e-8)
 np.testing.assert_allclose(hess,sigmoid(z)*(1-sigmoid(z)),atol=1e-8,rtol=1e-8)
 return dict(stage0_exact=True,tree_oracle_error=float(abs(actual-ref).max()),base_margin_shift_error=float(abs(delta-.7).max()),logistic_derivatives=True)

def main():
 t=toy();records=[]
 for fold in range(3):
  data=load_inner(fold);seed_all(42);a=make_adapter(data,'unmasked10','cuda:0',fold=fold)
  with np.load(HERE/'inputs'/f'common_f{fold}.npz') as c:
   pc=[c['pc_rna'],c['pc_drug']];base=c['train_margin'].copy()
  models={};xs={}
  for name in CONFIGS:
   with np.load(HERE/'inputs'/f'{name}_f{fold}.npz') as c:xs[name]=c['train'].copy()
   np.testing.assert_array_equal(xs[name],features(a,data['train_pairs'],name,pc))
   dm=xgb.DMatrix(xs[name],label=data['train_y'],base_margin=base,nthread=2);b=xgb.Booster(PARAMS,[dm])
   np.testing.assert_array_equal(b.predict(dm,output_margin=True),base)
   for i in range(5):b.update(dm,i)
   models[name]=b
  indices=np.r_[np.flatnonzero(data['train_y']==1)[[0,1,30,100]],np.flatnonzero(data['train_y']==0)[:2]]
  rows=[]
  for idx in indices:
   keep=np.arange(len(data['train_y']))!=idx;other={**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
   b=make_adapter(other,'unmasked10','cuda:0');b.model.load_state_dict(a.model.state_dict());b.mu=a.mu;b.sd=a.sd;b.model.eval()
   pairs=data['train_pairs'][idx:idx+1];bm=predict_margin(b,pairs)
   np.testing.assert_allclose(bm,base[idx:idx+1],atol=2e-4,rtol=5e-5)
   for name in CONFIGS:
    xx=features(b,pairs,name,pc);np.testing.assert_allclose(xx,xs[name][idx:idx+1],atol=2e-4,rtol=5e-5)
    expected=models[name].predict(xgb.DMatrix(xs[name][idx:idx+1],base_margin=base[idx:idx+1],nthread=2),output_margin=True)
    got=models[name].predict(xgb.DMatrix(xx,base_margin=bm,nthread=2),output_margin=True)
    np.testing.assert_allclose(got,expected,atol=2e-4,rtol=5e-5)
    rows.append(dict(config=name,pair=pairs.tolist()[0],label=int(data['train_y'][idx]),feature_error=float(abs(xx-xs[name][idx:idx+1]).max()),full_logit_error=float(abs(got-expected).max())))
   del b;torch.cuda.empty_cache()
  changed=make_adapter({**data,'val_y':1-data['val_y']},'unmasked10','cuda:0',fold=fold)
  for name in CONFIGS:np.testing.assert_array_equal(features(a,data['val_pairs'],name,pc),features(changed,data['val_pairs'],name,pc))
  np.testing.assert_allclose(predict_margin(a,data['val_pairs']),predict_margin(changed,data['val_pairs']),atol=2e-4,rtol=5e-5)
  records.append(dict(fold=fold,physical_deletions=rows,heldout_label_invariance=True));print('PASS',fold,flush=True)
  del a,changed;torch.cuda.empty_cache()
 (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',toy=t,records=records,source_sha256=sha256(__file__),no_new_inner_metric=True,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
if __name__=='__main__':main()
