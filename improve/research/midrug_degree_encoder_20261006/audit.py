"""Replay every checkpoint and redo explicit graph deletion on the trained model."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256,Reconstruction
from model import CONFIGS,make_adapter
from check import deletion_check


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    records=[]
    for fold in range(3):
        data=load_inner(fold)
        assert not set(data['train_indices'])&set(data['val_indices'])
        sources=[(name,Path(c[str(fold)]['path']),True) for name,c in plan['controls'].items()]
        sources += [(name,HERE/'results'/f'{name}_f{fold}_s42.json',False) for name in CONFIGS]
        for name,path,reused in sources:
            r=json.loads(path.read_text());seed_all(42)
            a=make_adapter(data,'control' if reused else name,'cuda:0')
            if name=='unmasked10':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to('cuda:0'))
            state=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']
            a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            with torch.no_grad():prob=a.predict('val')
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:
                for key,dkey in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:
                    np.testing.assert_array_equal(z[key],data[dkey])
                np.testing.assert_allclose(prob,z['probability'],atol=2e-5,rtol=2e-5)
                delta=float(abs(prob-z['probability']).max())
                assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
                assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
            record=dict(tag=path.stem,reused_control=reused,status='passed',max_prediction_error=delta)
            if not reused:
                assert r['settings']==CONFIGS[name] and not r['outer_validation_evaluated'] and not r['test_evaluated']
                assert r['source_hashes']==plan['source_hashes']
                curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
                assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
                for x in curves:
                    assert abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12
                    assert x['weight']==CONFIGS[name]['reconstruction_weight']*min(1.,x['epoch']/50.)
                    expected=float(np.float32(x['bce'])+np.float32(x['weight'])*np.float32(x['reconstruction']))
                    assert expected==x['loss']
                record['explicit_deletions']=deletion_check(a,data,name)
                if name.endswith('_rec10'):
                    plain=make_adapter(data,name.replace('_rec10',''),'cuda:0')
                    plain.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')});plain.model.eval()
                    with torch.no_grad():bare=plain.predict('val')
                    np.testing.assert_allclose(prob,bare,atol=2e-5,rtol=2e-5)
                    record['without_decoder_max_error']=float(abs(prob-bare).max())
            records.append(record);print('PASS',path.stem,flush=True);torch.cuda.empty_cache()
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))


if __name__=='__main__':main()
