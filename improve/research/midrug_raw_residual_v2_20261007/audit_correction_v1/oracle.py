"""Independent indexed-tree traversal of canonical saved arrays, with native float32 thresholds."""
import json
import numpy as np
import xgboost as xgb

def trees(booster):return json.loads(booster.save_raw(raw_format='json'))['learner']['gradient_booster']['model']['trees']
def independent_tree_margin(booster,x,base):
 out=np.array(base,dtype=np.float64).copy()
 for t in trees(booster):
  left=t['left_children'];right=t['right_children'];feature=t['split_indices'];condition=np.asarray(t['split_conditions'],dtype=np.float32);default=t['default_left']
  assert np.isfinite(condition).all()
  for row in range(len(x)):
   i=0
   while left[i]>=0:
    val=np.float32(x[row,feature[i]])
    i=(left[i] if default[i] else right[i]) if np.isnan(val) else left[i] if val<condition[i] else right[i]
   out[row]+=float(condition[i])
 return out

def boundary_check(booster,x,base):
 configs=[]
 for t in trees(booster):
  for i,left in enumerate(t['left_children']):
   if left>=0:configs.append((t['split_indices'][i],np.float32(t['split_conditions'][i])))
  if len(configs)>=6:break
 configs=configs[:6]
 if not configs:return dict(boundaries=0,error=0.)
 rows=[];margins=[]
 for f,threshold in configs:
  for value in [np.nextafter(threshold,np.float32(-np.inf)),threshold,np.nextafter(threshold,np.float32(np.inf))]:
   a=x[0].copy();a[f]=value;rows.append(a);margins.append(base[0])
 xx=np.asarray(rows,dtype=np.float32);bb=np.asarray(margins,dtype=np.float32)
 expected=booster.predict(xgb.DMatrix(xx,base_margin=bb,nthread=2),output_margin=True);actual=independent_tree_margin(booster,xx,bb)
 np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=2e-5)
 return dict(boundaries=len(configs),points=len(rows),error=float(abs(actual-expected).max()))
