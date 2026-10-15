"""Replay selections, inspect final/optimizer states, and verify physical deletion."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,make_optimizer,mixing_plan,capture_features,decode
from check import deletion_check

def scores(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p))

def verify_saved(p,data):
    with np.load(p.with_name(p.stem+'_inner.npz')) as z:
        for k,dk in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[k],data[dk])
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
            assert r['plan_sha256']==sha256(out/'plan.json')
            stored=verify_saved(p,data);actual=scores(data['val_y'],stored)
            for k,v in actual.items():assert abs(v-r['inner_'+k])<1e-12
            seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model
            state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True);m.load_state_dict(state);m.eval()
            assert not m.lin1._forward_pre_hooks and int(m.eb.num_batches_tracked)==r['best_epoch']
            with torch.no_grad():
                prob=a.predict('val');logits,h=capture_features(a,'val');decoded=decode(m,h)
                trainp=a.predict('train')
            np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5);torch.testing.assert_close(logits,decoded,atol=2e-5,rtol=2e-5)
            deleted=deletion_check(a,data,name,count=4)
            bare=make_adapter(data,'control','cuda:0');bare.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')});bare.model.eval()
            with torch.no_grad():plain=bare.predict('val')
            np.testing.assert_allclose(prob,plain,atol=2e-5,rtol=2e-5)
            curves=json.loads(p.with_name(p.stem+'_curves.json').read_text())
            assert len(curves)==r['epochs_run'] and max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            for c in curves:
                ep=c['epoch'];pi,lam,active=mixing_plan(data['train_pairs'],cfg['partner'],fold,ep)
                assert c['mixing_lambda']==lam and c['eligible_count']==len(active)
                assert c['alpha']==.5*min(1.,max(0.,(ep-30)/30.))
                assert c['reconstruction_weight']==10.*min(1.,ep/50.)
                assert c['label_mean_error']<1e-6
                assert c['same_drug_fraction']==float(np.mean(data['train_pairs'][:,1]==data['train_pairs'][pi,1]))
                assert c['label_disagreement_fraction']==float(np.mean(data['train_y']!=data['train_y'][pi]))
                assert abs(c['association_loss']-((1-c['alpha'])*c['clean_bce']+c['alpha']*c['mixed_bce']))<2e-6
                assert abs(c['loss']-(c['association_loss']+c['reconstruction_weight']*c['reconstruction']))<2e-6
                assert abs(c['selection_score']-.5*(c['inner_auc']+c['inner_ap']))<1e-12
            # Hash-verified optimizer files contain ordinary float param-group values.
            opts=torch.load(p.with_name(p.stem+'_optimizers.pt'),map_location='cpu',weights_only=False)
            finals=torch.load(p.with_name(p.stem+'_final.pt'),map_location='cpu',weights_only=True)
            assert int(finals['eb.num_batches_tracked'])==r['epochs_run']
            for v in finals.values():
                if v.is_floating_point():assert torch.isfinite(v).all()
            counts={}
            template=make_optimizer(m);parameter_names={id(v):k for k,v in m.named_parameters()}
            for stage,ep in [('selected',r['best_epoch']),('final',r['epochs_run'])]:
                o=opts[stage];assert len(o['param_groups'])==2
                assert all(g['lr']==1e-4 and g['alpha']==.99 for g in o['param_groups'])
                assert [g['weight_decay'] for g in o['param_groups']]==[0.,1e-4]
                names={idx:parameter_names[id(param)] for saved,current in zip(o['param_groups'],template.param_groups) for idx,param in zip(saved['params'],current['params'])}
                assert all(idx in names for idx in o['state'])
                for idx,s in o['state'].items():
                    assert int(s['step'])==ep
                    for v in s.values():
                        if torch.is_tensor(v) and v.is_floating_point():assert torch.isfinite(v).all()
                for prefix in ['lin1.','lin2.','lin3.']:
                    assert all(idx in o['state'] for idx,n in names.items() if n.startswith(prefix))
                counts[stage]=len(o['state'])
            records.append(dict(tag=p.stem,config=name,fold=fold,prediction_error=float(abs(prob-stored).max()),decode_error=float(abs(logits-decoded).max()),
                bare_m123_error=float(abs(prob-plain).max()),selected_train_metrics=scores(data['train_y'],trainp),optimizer_state_counts=counts,final_state_finite=True,
                selected_epoch=r['best_epoch'],epochs_run=r['epochs_run'],physical_deletions=deleted))
            print('PASS',p.stem,flush=True);del a,bare,m,state,opts,finals,template;torch.cuda.empty_cache()
    controls=[]
    for name,cs in plan['controls'].items():
        for fold,item in cs.items():
            p=Path(item['path']);assert sha256(p)==item['metadata_sha256'];assert sha256(p.with_suffix('.pt'))==item['checkpoint_sha256']
            assert sha256(p.with_name(p.stem+'_inner.npz'))==item['probability_sha256']
            data=load_inner(int(fold));prob=verify_saved(p,data);r=json.loads(p.read_text());s=scores(data['val_y'],prob)
            for k,v in s.items():assert abs(v-r['inner_'+k])<1e-12
            controls.append(dict(name=name,fold=int(fold),metrics=s,reused=True))
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,controls=controls,new_selected_replays=6,final_state_inspections=6,
        selected_final_optimizer_audits=6,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
