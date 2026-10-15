"""Replay selections and verify disjoint per-block optimizer/RNG histories."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,make_optimizers,mask_generator
from check import deletion_check
def scores(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p))
def saved(p,data):
    with np.load(p.with_name(p.stem+'_inner.npz')) as z:
        for k in ['pairs','y','indices']:np.testing.assert_array_equal(z[k],data['val_'+k])
        return z['probability'].copy()
def main():
    out=HERE/'results';plan=json.loads((out/'plan.json').read_text())
    for f,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(f)==h
    assert sha256(CLEAN/'graph.pt')==plan['graph_sha256'];assert sha256(CLEAN/'inner_data.npz')==plan['data_sha256']
    records=[]
    for name,cfg in CONFIGS.items():
        for fold in range(3):
            p=out/f'{name}_f{fold}_s42.json';r=json.loads(p.read_text());data=load_inner(fold)
            for f,h in r['hashes'].items():assert sha256(out/f)==h
            assert r['plan_sha256']==sha256(out/'plan.json');stored=saved(p,data)
            for k,v in scores(data['val_y'],stored).items():assert abs(v-r['inner_'+k])<1e-12
            seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model
            state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True);m.load_state_dict(state);m.eval()
            assert int(m.eb.num_batches_tracked)==r['best_epoch'] and m.eb.track_running_stats
            with torch.no_grad():prob=a.predict('val');trainp=a.predict('train')
            np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5);deleted=deletion_check(a,data,name,count=4)
            bare=make_adapter(data,'control','cuda:0');bare.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')});bare.model.eval()
            with torch.no_grad():plain=bare.predict('val')
            np.testing.assert_allclose(prob,plain,atol=2e-5,rtol=2e-5)
            curves=json.loads(p.with_name(p.stem+'_curves.json').read_text())
            assert len(curves)==r['epochs_run'] and max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            for c in curves:
                assert c['reconstruction_weight']==10.*min(1.,c['epoch']/50.)
                for part in ['first','second']:
                    assert abs(c[part+'_objective']-(c[part+'_bce']+c['reconstruction_weight']*c[part+'_reconstruction']))<2e-6
                assert abs(c['first_update_bce_change']-(c['second_bce']-c['first_bce']))<1e-12
                assert abs(c['selection_score']-.5*(c['inner_auc']+c['inner_ap']))<1e-12
                if name=='joint':assert abs(c['first_objective']-c['second_objective'])<2e-6
                if name=='consumer_first':assert abs(c['first_reconstruction']-c['second_reconstruction'])<2e-6
            opts=torch.load(p.with_name(p.stem+'_optimizers.pt'),map_location='cpu',weights_only=False)
            finals=torch.load(p.with_name(p.stem+'_final.pt'),map_location='cpu',weights_only=True)
            assert int(finals['eb.num_batches_tracked'])==r['epochs_run']
            assert all(not v.is_floating_point() or torch.isfinite(v).all() for v in finals.values())
            templates=make_optimizers(m);parameter_names={id(v):k for k,v in m.named_parameters()};counts={}
            for stage,ep in [('selected',r['best_epoch']),('final',r['epochs_run'])]:
                all_names=[];counts[stage]={}
                for block in ['C','R']:
                    o=opts[stage][block];template=templates[block]
                    assert len(o['param_groups'])==(1 if block=='C' else 2)
                    assert all(g['lr']==1e-4 and g['alpha']==.99 and g['eps']==1e-8 and g['momentum']==0 and not g['centered'] for g in o['param_groups'])
                    assert [g['weight_decay'] for g in o['param_groups']]==([0.] if block=='C' else [0.,1e-4])
                    names={idx:parameter_names[id(param)] for ss,tt in zip(o['param_groups'],template.param_groups) for idx,param in zip(ss['params'],tt['params'])}
                    all_names+=list(names.values());assert all(idx in names for idx in o['state'])
                    for idx,s in o['state'].items():
                        assert int(s['step'])==ep
                        for vv in s.values():
                            if torch.is_tensor(vv):assert torch.isfinite(vv).all()
                    if block=='C':assert len(o['state'])==6 and all(n.startswith(('lin1.','lin2.','lin3.')) for n in names.values())
                    counts[stage][block]=len(o['state'])
                assert len(all_names)==len(set(all_names))==len(parameter_names)
            # Exactly one reconstruction mask generation per epoch, despite two passes.
            gen=mask_generator('cuda:0',20261006+fold)
            for ep in range(1,r['epochs_run']+1):
                m.reconstruction.masks(a.graph,.5,gen)
                if ep==r['best_epoch']:torch.testing.assert_close(gen.get_state().cpu(),opts['selected_mask_rng'],atol=0,rtol=0)
            torch.testing.assert_close(gen.get_state().cpu(),opts['final_mask_rng'],atol=0,rtol=0)
            records.append(dict(tag=p.stem,config=name,fold=fold,prediction_error=float(abs(prob-stored).max()),bare_m123_error=float(abs(prob-plain).max()),
                selected_train_metrics=scores(data['train_y'],trainp),optimizer_state_counts=counts,final_state_finite=True,mask_rng_replayed=True,
                selected_epoch=r['best_epoch'],epochs_run=r['epochs_run'],physical_deletions=deleted,
                mean_first_update_bce_change=float(np.mean([c['first_update_bce_change'] for c in curves]))))
            print('PASS',p.stem,flush=True);del a,bare,m,state,opts,finals,templates;torch.cuda.empty_cache()
    controls=[]
    for name,cs in plan['controls'].items():
        for fold,item in cs.items():
            p=Path(item['path']);assert sha256(p)==item['metadata_sha256'];assert sha256(p.with_suffix('.pt'))==item['checkpoint_sha256']
            assert sha256(p.with_name(p.stem+'_inner.npz'))==item['probability_sha256'];d=load_inner(int(fold));prob=saved(p,d);rr=json.loads(p.read_text());s=scores(d['val_y'],prob)
            for k,val in s.items():assert abs(val-rr['inner_'+k])<1e-12
            controls.append(dict(name=name,fold=int(fold),metrics=s,reused=True))
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,controls=controls,new_selected_replays=9,final_state_inspections=9,
        selected_final_optimizer_audits=9,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
if __name__=='__main__':main()

