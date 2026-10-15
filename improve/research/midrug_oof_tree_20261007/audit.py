"""Replay selected trees, original paired controls and full300+tree physical query deletion."""
import json
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,PREV,BASES,load_inner,seed_all,read_module,sha256
from model import PARAMS,CONFIGS,make_adapter,features,predict_margin
oracle=read_module('oof_tree_audit_oracle',PREV/'audit_correction_v1/oracle.py')
def metrics(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p))
def main():
 out=HERE/'results';plan=json.loads((out/'plan.json').read_text())
 for p,h in {**plan['source_hashes'],**plan['evidence_hashes'],**plan['input_hashes']}.items():assert sha256(p)==h,p
 records=[]
 for fold in range(3):
  data=load_inner(fold);seed_all(42);a=make_adapter(data,'unmasked10','cuda:0',fold=fold)
  with np.load(HERE/'inputs'/f'common_f{fold}.npz') as z:c={k:z[k] for k in z.files}
  for k in ['train_pairs','train_y','val_pairs','val_y','train_indices','val_indices']:np.testing.assert_array_equal(c[k],data[k])
  pc=[c['pc_rna'],c['pc_drug']]
  np.testing.assert_allclose(predict_margin(a,data['train_pairs']),c['in_sample'],atol=2e-4,rtol=5e-5)
  np.testing.assert_allclose(predict_margin(a,data['val_pairs']),c['val_margin'],atol=2e-4,rtol=5e-5)
  for name,cfg in CONFIGS.items():
   tag=f'{name}_f{fold}_s42';r=json.loads((out/f'{tag}.json').read_text());assert r['plan_sha256']==sha256(out/'plan.json') and r['params']==PARAMS and r['settings']==cfg
   for p,h in r['artifact_hashes'].items():assert sha256(out/p)==h
   with np.load(PREV/'inputs'/f'{cfg["features"]}_f{fold}.npz') as z:xs={k:z[k] for k in ['train','val']}
   for part in xs:np.testing.assert_array_equal(xs[part],features(a,data[part+'_pairs'],cfg['features'],pc))
   b=xgb.Booster();b.load_model(out/f'{tag}_model.json');bf=xgb.Booster();bf.load_model(out/f'{tag}_final_model.json')
   assert b.num_boosted_rounds()==r['best_round'] and bf.num_boosted_rounds()==r['rounds_run']
   dm=xgb.DMatrix(xs['val'],base_margin=c['val_margin'],nthread=2);margin=b.predict(dm,output_margin=True);prob=expit(margin)
   with np.load(out/f'{tag}_inner.npz') as z:
    for k in ['pairs','y','indices']:np.testing.assert_array_equal(z[k],data['val_'+k])
    np.testing.assert_array_equal(prob,z['probability']);np.testing.assert_array_equal(margin,z['margin'])
   for k,v in metrics(data['val_y'],prob).items():assert abs(v-r['inner_'+k])<1e-12
   curves=json.loads((out/f'{tag}_curves.json').read_text());assert len(curves)==r['rounds_run']+1 and max(curves,key=lambda x:x['selection_score'])['round']==r['best_round']
   for i,t in enumerate(curves):assert t['round']==i and abs(t['selection_score']-.5*(t['inner_auc']+t['inner_ap']))<1e-12
   assert r['rounds_run']==1600 or (r['rounds_run']>=300 and r['rounds_run']-r['best_round']>=150)
   for m,val in metrics(data['val_y'],expit(c['val_margin'])).items():assert abs(val-curves[0]['inner_'+m])<1e-12
   ref=oracle.independent_tree_margin(b,xs['val'][:16],c['val_margin'][:16]);np.testing.assert_allclose(ref,margin[:16],atol=5e-5,rtol=2e-5)
   boundary=oracle.boundary_check(b,xs['val'],c['val_margin'])
   finalref=oracle.independent_tree_margin(bf,xs['val'][:3],c['val_margin'][:3]);assert np.isfinite(finalref).all()
   if r['best_round']==0:np.testing.assert_array_equal(margin,c['val_margin'])
   indices=np.r_[np.flatnonzero(data['train_y']==1)[[0,1,30,100]],np.flatnonzero(data['train_y']==0)[:2]];deletions=[]
   for ix in indices:
    keep=np.arange(len(data['train_y']))!=ix;other={**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
    aa=make_adapter(other,'unmasked10','cuda:0');aa.model.load_state_dict(a.model.state_dict());aa.mu=a.mu;aa.sd=a.sd;aa.model.eval()
    pairs=data['train_pairs'][ix:ix+1];xx=features(aa,pairs,cfg['features'],pc);np.testing.assert_array_equal(xx,xs['train'][ix:ix+1]);bm=predict_margin(aa,pairs)
    got=b.predict(xgb.DMatrix(xx,base_margin=bm,nthread=2),output_margin=True);original=b.predict(xgb.DMatrix(xs['train'][ix:ix+1],base_margin=c['in_sample'][ix:ix+1],nthread=2),output_margin=True)
    np.testing.assert_allclose(got,original,atol=2e-4,rtol=5e-5);deletions.append(dict(pair=pairs.tolist()[0],label=int(data['train_y'][ix]),features_bit_exact=True,full_logit_error=float(abs(got-original).max())))
    del aa;torch.cuda.empty_cache()
   records.append(dict(config=name,fold=fold,status='passed',selected_round=r['best_round'],final_round=r['rounds_run'],saved_predictions_bit_exact=True,independent_tree_error=float(abs(ref-margin[:16]).max()),threshold_boundary_reference=boundary,physical_deletions=deletions))
   print('PASS',tag,flush=True)
  del a;torch.cuda.empty_cache()
 controls=[]
 for name,cs in plan['controls'].items():
  for fold,s in cs.items():
   p=Path(s['path']);assert sha256(p)==s['metadata_sha256'];assert sha256(p.with_suffix('.pt'))==s['checkpoint_sha256'];assert sha256(p.with_name(p.stem+'_inner.npz'))==s['probability_sha256'];data=load_inner(int(fold));r=json.loads(p.read_text())
   with np.load(p.with_name(p.stem+'_inner.npz')) as z:
    for k in ['pairs','y','indices']:np.testing.assert_array_equal(z[k],data['val_'+k])
    for k,v in metrics(z['y'],z['probability']).items():assert abs(v-r['inner_'+k])<1e-12
   controls.append(dict(config=name,fold=int(fold),reused=True))
 for name,cs in plan['extra_controls'].items():
  for fold,s in cs.items():
   p=Path(s['path']);assert sha256(p)==s['sha256'];r=json.loads(p.read_text());data=load_inner(int(fold))
   if name=='full300':
    with np.load(BASES/'offsets'/f'fold{fold}.npz') as z:prob=torch.sigmoid(torch.from_numpy(z['val'])).numpy()
   else:
    for f,h in r['artifact_hashes'].items():assert sha256(PREV/'results'/f)==h
    with np.load(PREV/'results'/f'raw_f{fold}_s42_inner.npz') as z:
     for k in ['pairs','y','indices']:np.testing.assert_array_equal(z[k],data['val_'+k])
     prob=z['probability'].copy()
   for k,val in metrics(data['val_y'],prob).items():assert abs(val-r['inner_'+k])<1e-12
   controls.append(dict(config=name,fold=int(fold),reused=True))
 (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,controls=controls,new_model_replays=9,final_booster_inspections=9,base_replays_in_preparation=18,new_base_fits=0,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
if __name__=='__main__':main()
