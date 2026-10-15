"""Replay checkpoints, exact observed-record deletion, and extra-message diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter
from check import deletion_check,QUERY_CHECK_SOURCE


@torch.no_grad()
def intervention(a,data,tag):
    m=a.model;m.eval();original=a.predict('val')
    params=[(n,p) for n,p in m.named_parameters() if n.startswith('encoder.extra')]
    saved={n:p.clone() for n,p in params}
    try:
        for n,p in params:p.zero_()
        disabled=a.predict('val')
    finally:
        for n,p in params:p.copy_(saved[n])
    restored=a.predict('val');np.testing.assert_allclose(original,restored,atol=2e-5,rtol=2e-5)
    path=HERE/'results'/f'{tag}_without_extra.npz'
    np.savez_compressed(path,pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=disabled)
    return dict(parameter_norms={n:float(p.norm()) for n,p in params},max_prediction_change=float(abs(disabled-original).max()),
        auc=float(roc_auc_score(data['val_y'],disabled)),ap=float(average_precision_score(data['val_y'],disabled)),
        prediction_sha256=sha256(path),interpretation='disable extra messages on the selected checkpoint; not a separately trained control or selection criterion')


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    records=[]
    for fold in range(3):
        data=load_inner(fold);assert not set(data['train_indices'])&set(data['val_indices']) and not len(data['test_pairs'])
        paths=[(name,Path(c[str(fold)]['path']),True) for name,c in plan['controls'].items()]
        paths += [(name,HERE/'results'/f'{name}_f{fold}_s42.json',False) for name in CONFIGS]
        for name,path,reused in paths:
            r=json.loads(path.read_text());seed_all(42);a=make_adapter(data,name,'cuda:0')
            state=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']
            if reused:assert r['checkpoint_sha256']==plan['controls'][name][str(fold)]['checkpoint_sha256']
            a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            with torch.no_grad():prob=a.predict('val')
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:
                for key,dkey in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[dkey])
                np.testing.assert_allclose(prob,z['probability'],atol=2e-5,rtol=2e-5);error=float(abs(prob-z['probability']).max())
                assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
                assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
            record=dict(tag=path.stem,reused_control=reused,status='passed',max_prediction_error=error)
            if not reused:
                cfg=CONFIGS[name];assert r['settings']==cfg and r['source_hashes']==plan['source_hashes']
                assert r['no_lncrna'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
                context=a.model.encoder.context;expected=torch.zeros_like(context)
                ix=torch.as_tensor(data['train_pairs'][data['train_y']==cfg['support']],device=context.device)
                expected[ix[:,0],ix[:,1]]=1;torch.testing.assert_close(context,expected,atol=0,rtol=0)
                v=a.pairs['val'];assert not context[v[0],v[1]].any() and not a.ps.A[v[0],v[1]].any()
                assert not any(k.endswith('context') for k in state)
                curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
                assert max(curves,key=lambda c:c['selection_score'])['epoch']==r['best_epoch']
                for c in curves:
                    assert abs(c['selection_score']-.5*(c['inner_auc']+c['inner_ap']))<1e-12
                    assert c['weight']==cfg['reconstruction_weight']*min(1.,c['epoch']/50.)
                    assert c['loss']==float(np.float32(c['bce'])+np.float32(c['weight'])*np.float32(c['reconstruction']))
                record['explicit_deletions']=deletion_check(a,data,name)
                record['extra_intervention']=intervention(a,data,path.stem)
                record['context_edges']=int(context.sum())
                if cfg['reconstruction_weight']:
                    rec=a.model.reconstruction;del a.model.reconstruction
                    with torch.no_grad():bare=a.predict('val')
                    a.model.add_module('reconstruction',rec)
                    np.testing.assert_allclose(prob,bare,atol=2e-5,rtol=2e-5);record['without_decoder_error']=float(abs(prob-bare).max())
            records.append(record);print('PASS',name,fold,flush=True);torch.cuda.empty_cache()
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,
        audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))


if __name__=='__main__':main()
