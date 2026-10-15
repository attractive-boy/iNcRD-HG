"""Verify offset-only matched arms and full300+tree query deletion with exact tree inputs."""
import json
import numpy as np
import torch
import xgboost as xgb
from common import HERE,PREV,load_inner,seed_all,read_module,sha256
from model import PARAMS,CONFIGS,make_adapter,features,predict_margin
oracle=read_module('canonical_tree_oracle',PREV/'audit_correction_v1/oracle.py')

def main():
 records=[]
 for fold in range(3):
  data=load_inner(fold);seed_all(42);a=make_adapter(data,'unmasked10','cuda:0',fold=fold)
  with np.load(HERE/'inputs'/f'common_f{fold}.npz') as z:c={k:z[k] for k in z.files}
  pc=[c['pc_rna'],c['pc_drug']];models={};xs={};pred0={}
  assert np.isfinite(c['oof']).all() and (c['coverage']==1).all() and abs(c['oof']-c['in_sample']).mean()>0
  for name,cfg in CONFIGS.items():
   with np.load(PREV/'inputs'/f'{cfg["features"]}_f{fold}.npz') as z:xs[name]={p:z[p] for p in ['train','val']}
   for part in ['train','val']:np.testing.assert_array_equal(xs[name][part],features(a,data[part+'_pairs'],cfg['features'],pc))
   dm=xgb.DMatrix(xs[name]['train'],label=data['train_y'],base_margin=c[cfg['offset']],nthread=2)
   dv=xgb.DMatrix(xs[name]['val'],base_margin=c['val_margin'],nthread=2);b=xgb.Booster(PARAMS,[dm]);pred0[name]=b.predict(dv,output_margin=True)
   np.testing.assert_array_equal(pred0[name],c['val_margin'])
   # The prescribed source margin is actually the training base score, without duplication or clipping.
   np.testing.assert_array_equal(b.predict(dm,output_margin=True),c[cfg['offset']])
   for i in range(5):b.update(dm,i)
   expected=b.predict(dv,output_margin=True);ref=oracle.independent_tree_margin(b,xs[name]['val'][:16],c['val_margin'][:16])
   np.testing.assert_allclose(ref,expected[:16],atol=5e-5,rtol=2e-5);oracle.boundary_check(b,xs[name]['val'],c['val_margin']);models[name]=b
  np.testing.assert_array_equal(xs['oof_raw']['train'],xs['insample_raw']['train'])
  rows=[];indices=np.r_[np.flatnonzero(data['train_y']==1)[[0,1,30,100]],np.flatnonzero(data['train_y']==0)[:2]]
  for ix in indices:
   keep=np.arange(len(data['train_y']))!=ix;other={**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
   b=make_adapter(other,'unmasked10','cuda:0');b.model.load_state_dict(a.model.state_dict());b.mu=a.mu;b.sd=a.sd;b.model.eval();pairs=data['train_pairs'][ix:ix+1];bm=predict_margin(b,pairs)
   for name,cfg in CONFIGS.items():
    xx=features(b,pairs,cfg['features'],pc);np.testing.assert_array_equal(xx,xs[name]['train'][ix:ix+1])
    expected=models[name].predict(xgb.DMatrix(xs[name]['train'][ix:ix+1],base_margin=c['in_sample'][ix:ix+1],nthread=2),output_margin=True)
    got=models[name].predict(xgb.DMatrix(xx,base_margin=bm,nthread=2),output_margin=True);np.testing.assert_allclose(got,expected,atol=2e-4,rtol=5e-5)
    rows.append(dict(config=name,pair=pairs.tolist()[0],label=int(data['train_y'][ix]),tree_inputs_bit_exact=True,full_logit_error=float(abs(got-expected).max())))
   del b;torch.cuda.empty_cache()
  b=make_adapter({**data,'val_y':1-data['val_y']},'unmasked10','cuda:0',fold=fold)
  np.testing.assert_allclose(predict_margin(b,data['val_pairs']),c['val_margin'],atol=2e-4,rtol=5e-5)
  for name,cfg in CONFIGS.items():np.testing.assert_array_equal(features(b,data['val_pairs'],cfg['features'],pc),xs[name]['val'])
  records.append(dict(fold=fold,stage0_exact=True,train_margin_exact=True,raw_feature_arms_identical=True,heldout_label_invariance=True,physical_deletions=rows))
  print('PASS',fold,flush=True);del a,b;torch.cuda.empty_cache()
 (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,oracle_source_sha256=sha256(PREV/'audit_correction_v1/oracle.py'),no_new_inner_metric=True,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
if __name__=='__main__':main()
