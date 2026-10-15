"""Fixed-base logistic tree residuals, stage0 and dual inner metric checkpoint selection."""
import argparse,json,time
import numpy as np
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,sha256
from model import PARAMS,CONFIGS

def metrics(y,p):return dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',choices=CONFIGS,required=True);p.add_argument('--fold',type=int,choices=range(3),required=True);args=p.parse_args()
 out=HERE/'results';plan=json.loads((out/'plan.json').read_text())
 for f,h in {**plan['source_hashes'],**plan['evidence_hashes'],**plan['input_hashes']}.items():assert sha256(f)==h
 tag=f'{args.config}_f{args.fold}_s42';assert not (out/f'{tag}.json').exists()
 with np.load(HERE/'inputs'/f'common_f{args.fold}.npz') as z:data={k:z[k] for k in z.files}
 with np.load(HERE/'inputs'/f'{args.config}_f{args.fold}.npz') as z:xs={k:z[k] for k in ['train','val']}
 dm={p:xgb.DMatrix(xs[p],label=data[p+'_y'],base_margin=data[p+'_margin'],nthread=2) for p in ['train','val']}
 b=xgb.Booster(PARAMS,[dm['train']]);best=-1.;best_ep=0;curves=[];start=time.time()
 for ep in range(1601):
  if ep:b.update(dm['train'],ep-1)
  margin=b.predict(dm['val'],output_margin=True);prob=expit(margin);s=metrics(data['val_y'],prob);score=.5*(s['auc']+s['ap'])
  curves.append(dict(round=ep,inner_auc=s['auc'],inner_ap=s['ap'],selection_score=score))
  if score>best:best,best_ep,best_p,best_margin=score,ep,prob.copy(),margin.copy();selected=b.copy()
  if ep%100==0:print(tag,ep,best,best_ep,time.time()-start,flush=True)
  if ep>=300 and ep-best_ep>=150:break
 selected.save_model(out/f'{tag}_model.json');b.save_model(out/f'{tag}_final_model.json')
 np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p,margin=best_margin)
 (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
 r=dict(config=args.config,fold=args.fold,seed=42,tag=tag,best_round=best_ep,rounds_run=ep,inner_auc=metrics(data['val_y'],best_p)['auc'],inner_ap=metrics(data['val_y'],best_p)['ap'],selection_score=best,
  params=PARAMS,xgboost_version=xgb.__version__,n_features=xs['train'].shape[1],plan_sha256=sha256(out/'plan.json'),
  artifact_hashes={f'{tag}{s}':sha256(out/f'{tag}{s}') for s in ['_model.json','_final_model.json','_inner.npz','_curves.json']},elapsed_seconds=time.time()-start,
  no_lncrna=True,original_m123_frozen=True,outer_validation_evaluated=False,test_evaluated=False)
 (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print('DONE',tag,flush=True)
if __name__=='__main__':main()
