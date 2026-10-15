"""Replay new and reused checkpoints; audit sampling, loss composition and query deletion."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256,SameDrugRanking,rank_scale
from model import CONFIGS,make_adapter
import check_modules

def replay(data,name,path,reused):
    build='control' if name=='control' else ('rec10_bpr010' if name.startswith('unmasked') else name)
    seed_all(42);a=make_adapter(data,build,'cuda:0')
    state=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True);a.model.load_state_dict(state);a.model.eval()
    assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
    with torch.no_grad():p=a.predict('val')
    r=json.loads(path.read_text());assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']
    with np.load(path.with_name(path.stem+'_inner.npz')) as z:
        for key,dkey in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[dkey])
        np.testing.assert_allclose(p,z['probability'],atol=2e-5,rtol=2e-5);delta=float(abs(p-z['probability']).max())
        assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
        assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
    record=dict(tag=path.stem,reused_control=reused,status='passed',max_prediction_error=delta)
    if not reused:
        cfg=CONFIGS[name];curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
        assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
        for x in curves:
            assert abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12
            assert x['rank_weight']==rank_scale(x['epoch'],cfg['rank_weight'])
            assert x['reconstruction_weight']==cfg['reconstruction_weight']*min(1.,x['epoch']/50.)
            expected=float((np.float32(x['bce'])+np.float32(x['rank_weight'])*np.float32(x['bpr']))+np.float32(x['reconstruction_weight'])*np.float32(x['reconstruction']))
            assert x['loss']==expected
        rank=SameDrugRanking(data['train_pairs'],data['train_y'],'cuda:0',42,'mixed_middle')
        with torch.no_grad():negative=rank.sample(a.logits('train'))
        pairs=torch.as_tensor(data['train_pairs'],device='cuda:0')
        assert a.y[rank.anchors].eq(1).all() and a.y[negative].eq(0).all()
        assert (pairs[negative,1]==pairs[rank.anchors,1,None]).all()
        assert r['ranking_coverage']==rank.coverage
        record['sampling_valid']=True;record['full_logit_deletion']=check_modules.check(a,count=6)
        if cfg['reconstruction_weight']:
            seed_all(42);plain=make_adapter(data,'control','cuda:0')
            plain.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')});plain.model.eval()
            with torch.no_grad():bare=plain.predict('val')
            np.testing.assert_allclose(p,bare,atol=2e-5,rtol=2e-5);record['no_decoder_max_error']=float(abs(p-bare).max())
    return record

def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for path,h in plan['source_hashes'].items():assert sha256(path)==h
    records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name,control in plan['controls'].items():
            from pathlib import Path
            path=Path(control[str(fold)]['path']);assert sha256(path.with_suffix('.pt'))==control[str(fold)]['checkpoint_sha256']
            records.append(replay(data,name,path,True))
        for name in CONFIGS:
            path=HERE/'results'/f'{name}_f{fold}_s42.json'
            r=json.loads(path.read_text());assert r['no_lncrna'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
            for f,h in r['hashes'].items():assert sha256(HERE/f)==h
            records.append(replay(data,name,path,False));print('PASS',path.stem,flush=True);torch.cuda.empty_cache()
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
