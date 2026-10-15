import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
"""Replay nine selected models, inspect final states, optimizer/RNG and paired immutable controls."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,make_optimizer,mask_generator
from check import deletion_check,branch_check,batch_check

def scores(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p))
def saved(p,data):
 with np.load(p.with_name(p.stem+'_inner.npz')) as z:
  for k in ['pairs','y','indices']:np.testing.assert_array_equal(z[k],data['val_'+k])
  return z['probability'].copy()
def main():
 out=HERE/'results';plan=json.loads((out/'plan.json').read_text())
 for p,h in {**plan['source_hashes'],**plan['evidence_hashes'],**plan['input_hashes']}.items():assert sha256(p)==h,p
 records=[]
 for name,cfg in CONFIGS.items():
  for fold in range(3):
   p=out/f'{name}_f{fold}_s42.json';r=json.loads(p.read_text());data=load_inner(fold)
   for f,h in r['hashes'].items():assert sha256(out/f)==h
   assert r['plan_sha256']==sha256(out/'plan.json') and r['settings']==cfg
   assert not r['outer_validation_evaluated'] and not r['test_evaluated']
   stored=saved(p,data)
   for k,v in scores(data['val_y'],stored).items():assert abs(v-r['inner_'+k])<1e-12
   seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model
   m.load_state_dict(torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True));m.eval()
   assert int(m.eb.num_batches_tracked)==r['best_epoch'] and not m.lin1._forward_hooks
   assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in m.state_dict())
   with torch.no_grad():prob=a.predict('val')
   np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
   branch=branch_check(a);batch=batch_check(a);deleted=deletion_check(a,data,name,count=4)
   with torch.no_grad():repeat=a.predict('val')
   np.testing.assert_allclose(prob,repeat,atol=2e-5,rtol=2e-5)
   reconstruction=m.reconstruction;del m.reconstruction
   with torch.no_grad():bare=a.predict('val')
   m.reconstruction=reconstruction
   np.testing.assert_allclose(prob,bare,atol=2e-5,rtol=2e-5)
   curves=json.loads(p.with_name(p.stem+'_curves.json').read_text())
   assert len(curves)==r['epochs_run'] and max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
   assert r['epochs_run']==800 or (r['epochs_run']>=300 and r['epochs_run']-r['best_epoch']>=200)
   for c in curves:
    assert c['weight']==10.*min(1.,c['epoch']/50.)
    assert abs(c['loss']-(c['bce']+c['weight']*c['reconstruction']))<2e-6
    assert abs(c['selection_score']-.5*(c['inner_auc']+c['inner_ap']))<1e-12
   opts=torch.load(p.with_name(p.stem+'_optimizers.pt'),map_location='cpu',weights_only=False)
   finals=torch.load(p.with_name(p.stem+'_final.pt'),map_location='cpu',weights_only=True)
   assert int(finals['eb.num_batches_tracked'])==r['epochs_run']
   assert all(not v.is_floating_point() or torch.isfinite(v).all() for v in finals.values())
   template=make_optimizer(m,cfg);counts={};param_names={id(v):k for k,v in m.named_parameters()}
   for stage,ep in [('selected',r['best_epoch']),('final',r['epochs_run'])]:
    o=opts[stage];groups=o['param_groups'];assert len(groups)==3
    assert [g['lr'] for g in groups]==[1e-4,3e-4,1e-4] and [g['weight_decay'] for g in groups]==[0,1e-4,1e-4]
    assert all(g['alpha']==.99 and g['eps']==1e-8 and g['momentum']==0 and not g['centered'] for g in groups)
    names={idx:param_names[id(param)] for s,t in zip(groups,template.param_groups) for idx,param in zip(s['params'],t['params'])}
    assert len(names)==len(set(names.values()))==len(param_names)
    for idx,s in o['state'].items():
     assert idx in names and int(s['step'])==ep
     for v in s.values():
      if torch.is_tensor(v):assert torch.isfinite(v).all()
    # All new branch tensors must have an update each epoch, including zero-grad first epoch.
    branch_ids=[idx for idx,n in names.items() if n.startswith(('token_context.','token_projection.'))]
    assert all(idx in o['state'] for idx in branch_ids)
    counts[stage]=dict(all=len(o['state']),branch=len(branch_ids))
   gen=mask_generator('cuda:0',20261006+fold)
   for ep in range(1,r['epochs_run']+1):
    m.reconstruction.masks(a.graph,.5,gen)
    if ep==r['best_epoch']:torch.testing.assert_close(gen.get_state().cpu(),opts['selected_mask_rng'],atol=0,rtol=0)
   torch.testing.assert_close(gen.get_state().cpu(),opts['final_mask_rng'],atol=0,rtol=0)
   records.append(dict(tag=p.stem,config=name,fold=fold,prediction_error=float(abs(prob-stored).max()),decoder_removal_error=float(abs(prob-bare).max()),
    unchanged_repeat_error=float(abs(prob-repeat).max()),branch=branch,batch_error=batch,physical_deletions=deleted,final_state_finite=True,optimizer_counts=counts,mask_rng_replayed=True))
   print('PASS',p.stem,flush=True);del a,m,opts,finals,template,reconstruction;torch.cuda.empty_cache()
 controls=[]
 for name,cs in plan['controls'].items():
  for fold,item in cs.items():
   p=Path(item['path']);assert sha256(p)==item['metadata_sha256']
   assert sha256(p.with_suffix('.pt'))==item['checkpoint_sha256'];assert sha256(p.with_name(p.stem+'_inner.npz'))==item['probability_sha256']
   data=load_inner(int(fold));prob=saved(p,data);r=json.loads(p.read_text());s=scores(data['val_y'],prob)
   for k,v in s.items():assert abs(v-r['inner_'+k])<1e-12
   controls.append(dict(name=name,fold=int(fold),metrics=s,reused=True))
 assert len(records)==9 and len(controls)==6
 (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,controls=controls,new_selected_replays=9,final_state_inspections=9,
  selected_final_optimizer_audits=9,audit_amendment=str(HERE/'audit_correction_v1/AMENDMENT.md'),audit_source_sha256=sha256(__file__),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
if __name__=='__main__':main()
