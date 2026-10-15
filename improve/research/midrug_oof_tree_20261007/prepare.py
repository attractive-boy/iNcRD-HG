"""Current hash/coverage audit and 18 fresh inference replays of previously trained bases."""
import json
from pathlib import Path
import numpy as np
import torch
from common import HERE,PREV,BASES,CLEAN,load_inner,nested_data,seed_all,sha256
from model import make_adapter,source,predict_margin

def bce(y,z):return float(np.mean(np.logaddexp(0,z.astype(float))-y*z))
def main():
 old=json.loads((BASES/'bases/plan.json').read_text());verification=json.loads((BASES/'base_verification.json').read_text());assert verification['status']=='passed'
 for p,h in {**old['source_hashes'],**old['evidence_hashes']}.items():assert sha256(p)==h,p
 checks={};records=[];gaps=[];out=HERE/'inputs';out.mkdir()
 checks[str(BASES/'bases/plan.json')]=sha256(BASES/'bases/plan.json');checks[str(BASES/'base_verification.json')]=sha256(BASES/'base_verification.json')
 for fold in range(3):
  outer=load_inner(fold);coverage=np.zeros(len(outer['train_y']),dtype=int);oof=np.full(len(coverage),np.nan,dtype=np.float32)
  oldoff=verification['offsets'][fold];op=Path(oldoff['path']);assert sha256(op)==oldoff['sha256'];checks[str(op)]=oldoff['sha256']
  for k in range(-1,5):
   p=source(fold,k);r=json.loads(p.read_text());data,rows=nested_data(fold,k)
   assert r['epochs']==300 and not r['prediction_fold_labels_used_for_selection'] and r['plan_sha256']==sha256(BASES/'bases/plan.json')
   for suffix,key in [('.pt','checkpoint_sha256'),('.npz','cache_sha256')]:assert sha256(p.with_suffix(suffix))==r[key];checks[str(p.with_suffix(suffix))]=r[key]
   checks[str(p)]=sha256(p)
   with np.load(p.with_suffix('.npz')) as z:
    for key in ['train_pairs','val_pairs','train_indices','val_indices']:np.testing.assert_array_equal(z[key],data[key])
    np.testing.assert_array_equal(z['prediction_rows'],rows);tr=z['train_logit'].copy();va=z['val_logit'].copy()
   assert np.intersect1d(data['train_indices'],data['val_indices']).size==0
   seed_all(42);a=make_adapter(data,'unmasked10','cuda:0',fold=fold,nested=k)
   assert int(a.model.eb.num_batches_tracked)==300 and set(a.graph.node_types)=={'miRNA','drug'}
   rt,rv=predict_margin(a,data['train_pairs']),predict_margin(a,data['val_pairs'])
   np.testing.assert_allclose(rt,tr,atol=2e-4,rtol=5e-5);np.testing.assert_allclose(rv,va,atol=2e-4,rtol=5e-5)
   if k==-1:insample,fullval=tr,va
   else:
    assert not np.intersect1d(data['train_indices'],outer['val_indices']).size and not np.intersect1d(data['val_indices'],outer['val_indices']).size
    coverage[rows]+=1;oof[rows]=va
   records.append(dict(fold=fold,nested=k,current_replay=True,train_error=float(abs(rt-tr).max()),prediction_error=float(abs(rv-va).max()),epochs=300,train_prediction_indices_disjoint=True))
   print('REPLAY',fold,k,flush=True);del a;torch.cuda.empty_cache()
  assert (coverage==1).all() and np.isfinite(oof).all()
  with np.load(op) as z:
   for n,x in [('oof',oof),('in_sample',insample),('val',fullval),('coverage',coverage),('train_indices',outer['train_indices']),('train_pairs',outer['train_pairs'])]:np.testing.assert_array_equal(z[n],x)
  with np.load(PREV/'inputs'/f'common_f{fold}.npz') as z:
   for key in ['train_pairs','train_y','val_pairs','val_y','train_indices','val_indices']:np.testing.assert_array_equal(z[key],outer[key])
   pc_r,pc_d=z['pc_rna'].copy(),z['pc_drug'].copy()
  np.savez_compressed(out/f'common_f{fold}.npz',oof=oof,in_sample=insample,val_margin=fullval,coverage=coverage,pc_rna=pc_r,pc_drug=pc_d,
   **{key:outer[key] for key in ['train_pairs','train_y','val_pairs','val_y','train_indices','val_indices']})
  gaps.append(dict(fold=fold,in_sample_bce=bce(outer['train_y'],insample),oof_bce=bce(outer['train_y'],oof),full_validation_bce=bce(outer['val_y'],fullval),
   mean_abs_oof_insample_logit_difference=float(abs(oof-insample).mean()),mean_oof_logit=float(oof.mean()),mean_insample_logit=float(insample.mean())))
  for name in ['raw','pca32']:
   p=PREV/'inputs'/f'{name}_f{fold}.npz';checks[str(p)]=sha256(p)
  checks[str(PREV/'inputs'/f'common_f{fold}.npz')]=sha256(PREV/'inputs'/f'common_f{fold}.npz')
 checks.update({str(CLEAN/'graph.pt'):sha256(CLEAN/'graph.pt'),str(CLEAN/'inner_data.npz'):sha256(CLEAN/'inner_data.npz')})
 r=dict(status='passed',current_base_checkpoint_replays=18,new_base_fits=0,records=records,gaps=gaps,source_hashes=checks,input_hashes={str(p):sha256(p) for p in out.iterdir()},
  oof_exactly_once=True,prediction_labels_used_for_base_selection=False,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 (HERE/'input_manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps(gaps,indent=2))
if __name__=='__main__':main()
