"""Replay six ensemble fusion models and six immutable controls."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,member_logits,mixture_logit
from check import member_deletion_check,batch_check,QUERY_CHECK_SOURCE
PREVIOUS=HERE.parent/'midrug_cell_qknorm_20261007'

def saved_scores(p,data,r):
    with np.load(p.with_name(p.stem+'_inner.npz')) as z:
        for k,dk in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[k],data[dk])
        assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
        assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
        return z['probability'].copy()

@torch.no_grad()
def head_check(a):
    m=a.model;features=m.fusion_features(a.pairs['val'][:,:3],a.ps,a.qis['val'][:3],a.ease['val'][:3],a.dense)
    z=m.head_logits(features);rows=[]
    for i in range(m.members):
        h=features
        for layer in [m.lin1,m.lin2]:
            matrix=layer.s[i,:,None]*layer.weight*layer.r[i,None,:]
            h=h@matrix.T+layer.bias[i]
            h=torch.where(h>=0,h,.1*h)
        rows.append(h@m.lin3.weight[i]+m.lin3.bias[i])
    ref=torch.stack(rows,-1)
    torch.testing.assert_close(z,ref,atol=2e-4,rtol=5e-5)
    torch.testing.assert_close(mixture_logit(z).sigmoid(),z.sigmoid().mean(1),atol=2e-6,rtol=2e-6)
    return dict(member_oracle_error=float(abs(z-ref).max()),mean_probability_error=float(abs(mixture_logit(z).sigmoid()-z.sigmoid().mean(1)).max()))


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    for p,h in plan['evidence_hashes'].items():assert sha256(p)==h
    prior=json.loads((PREVIOUS/'verification.json').read_text());assert prior['status']=='passed'
    for p,h in json.loads((PREVIOUS/'results/plan.json').read_text())['source_hashes'].items():assert sha256(p)==h
    tags={r['tag']:r for r in prior['records']};records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name,sources in plan['controls'].items():
            src=sources[str(fold)];p=Path(src['path']);r=json.loads(p.read_text())
            assert sha256(p)==src['metadata_sha256']
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']==src['checkpoint_sha256']
            assert sha256(p.with_name(p.stem+'_inner.npz'))==src['probability_sha256']
            assert tags[p.stem]['status']=='passed';saved_scores(p,data,r)
            records.append(dict(tag=p.stem,status='passed',reused_control=True,verification='immutable prior replay, current saved probabilities/metrics and pair/label/index checks'))
        for name,cfg in CONFIGS.items():
            p=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(p.read_text());assert r['settings']==cfg and r['source_hashes']==plan['source_hashes']
            assert r['graph_sha256']==plan['graph_sha256']==sha256(CLEAN/'graph.pt')
            assert r['data_sha256']==plan['data_sha256']==sha256(CLEAN/'inner_data.npz')
            assert not r['outer_validation_evaluated'] and not r['test_evaluated']
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
            seed_all(42);a=make_adapter(data,name,'cuda:0');state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            assert int(a.model.eb.num_batches_tracked)==r['best_epoch'] and not a.model.lin1._forward_hooks
            assert a.model.members==cfg['members']
            with torch.no_grad():prob=a.predict('val')
            stored=saved_scores(p,data,r);np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            curves=json.loads(p.with_name(p.stem+'_curves.json').read_text());assert len(curves)==r['epochs_run']
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            for x in curves:
                assert abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12
                assert x['bce']==x['member_bce' if cfg['objective']=='member_bce' else 'mixture_bce']
                assert x['member_bce']>=x['mixture_bce']-1e-6
                assert x['weight']==cfg['reconstruction_weight']*min(1.,x['epoch']/50.)
                assert x['loss']==float(np.float32(x['bce'])+np.float32(x['weight'])*np.float32(x['reconstruction']))
            rec=dict(tag=p.stem,status='passed',reused_control=False,max_prediction_error=float(abs(prob-stored).max()),
                head=head_check(a),explicit_deletions=member_deletion_check(a,data,name,count=4),batch_max_logit_error=batch_check(a))
            bare=make_adapter(data,name.replace('_rec10',''),'cuda:0')
            bare.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')});bare.model.eval()
            with torch.no_grad():plain=bare.predict('val')
            np.testing.assert_allclose(prob,plain,atol=2e-5,rtol=2e-5);rec['without_reconstruction_error']=float(abs(prob-plain).max())
            records.append(rec);print('PASS',p.stem,flush=True);del a,bare,state;torch.cuda.empty_cache()
    assert len(records)==12 and sum(not r['reused_control'] for r in records)==6
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,new_checkpoint_replays=6,immutable_reused_control_checks=6,
        audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',QUERY_CHECK_SOURCE]},no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
