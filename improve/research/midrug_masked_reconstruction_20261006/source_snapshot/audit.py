"""Replay checkpoints, query deletion, decoder-free inference and reconstruction diagnostics."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,mask_generator
import check_modules

def diagnostic(adapter,fold):
    rec=adapter.model.reconstruction;indices=rec.masks(adapter.graph,.5,mask_generator('cuda:0',91000+fold));out={}
    with torch.no_grad():
        for masked in [False,True]:
            loss,parts=rec.loss(adapter.model.encoder,adapter.graph,indices,masked)
            out['masked' if masked else 'unmasked']={name:float(value) for name,value in parts.items()}
    return out

def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for path,h in plan['source_hashes'].items():assert sha256(path)==h
    records=[]
    for fold in range(3):
        data=load_inner(fold);seed_all(42);a=make_adapter(data,'control','cuda:0')
        path=CLEAN/'results'/f'control_f{fold}_s42.json';assert sha256(path.with_suffix('.pt'))==plan['controls'][str(fold)]['checkpoint_sha256']
        a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval()
        with torch.no_grad():p=a.predict('val')
        with np.load(path.with_name(path.stem+'_inner.npz')) as z:np.testing.assert_allclose(p,z['probability'],atol=2e-5,rtol=2e-5)
        records.append(dict(tag=path.stem,reused_control=True,status='passed'));del a
        for name,cfg in CONFIGS.items():
            path=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(path.read_text());assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']
            for file,h in r['hashes'].items():assert sha256(HERE/file)==h
            assert r['no_lncrna'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.eval();initial=diagnostic(a,fold)
            a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval();trained=diagnostic(a,fold)
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in a.model.state_dict())
            with torch.no_grad():p=a.predict('val')
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:
                for key,dkey in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[dkey])
                np.testing.assert_allclose(p,z['probability'],atol=2e-5,rtol=2e-5);delta=float(abs(p-z['probability']).max())
                assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
                assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
            curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            for x in curves:
                assert abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12
                assert abs(x['weight']-cfg['weight']*min(1,x['epoch']/50))<1e-12
                expected=float(np.float32(x['bce'])+np.float32(x['weight'])*np.float32(x['reconstruction']))
                assert x['loss']==expected
            checks=check_modules.check(a,count=6)
            # Removing all auxiliary inference parameters must preserve association predictions.
            seed_all(42);plain=make_adapter(data,'control','cuda:0')
            plain.model.load_state_dict({k:v for k,v in a.model.state_dict().items() if not k.startswith('reconstruction.')});plain.model.eval()
            with torch.no_grad():no_decoder=plain.predict('val')
            np.testing.assert_allclose(p,no_decoder,atol=2e-5,rtol=2e-5)
            records.append(dict(tag=r['tag'],reused_control=False,status='passed',max_prediction_error=delta,
                no_decoder_max_error=float(abs(p-no_decoder).max()),full_logit_deletion=checks,
                reconstruction_initial=initial,reconstruction_selected=trained))
            print('PASS',r['tag'],flush=True);del a,plain;torch.cuda.empty_cache()
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
