"""Checkpoint, exact query deletion and post-selection uniform-gate diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter
from check import deletion_check,QUERY_CHECK_SOURCE

degree_reference=read_module('audited_uniform_degree_model',DEGREE/'model.py')


@torch.no_grad()
def gate_diagnostics(a,data,tag):
    encoder=a.model.encoder;model=a.model;model.eval()
    p1,c1=encoder.full_layer(encoder.original.conv1,a.graph.x_dict,1)
    h1={n:F.leaky_relu(v,.1) for n,v in p1.items()}
    _,c2=encoder.full_layer(encoder.original.conv2,h1,2)
    stats={}
    for number,cache in [(1,c1),(2,c2)]:
        for edge,c in cache.items():
            g=c['gate'];stats[f'{number}:{edge}']=dict(min=float(g.min()),max=float(g.max()),
                mean=float(g.mean()),std=float(g.std()),mean_absolute_deviation_from_uniform=float(abs(g-.5).mean()))
    original=a.predict('val')
    gates=[g for layer in [encoder.gates1,encoder.gates2] for g in layer.values()]
    flags=[g.enabled for g in gates]
    try:
        for g in gates:g.enabled=False
        uniform=a.predict('val')
    finally:
        for g,flag in zip(gates,flags):g.enabled=flag
    restored=a.predict('val');np.testing.assert_allclose(restored,original,atol=2e-5,rtol=2e-5)
    path=HERE/'results'/f'{tag}_uniform_intervention.npz'
    np.savez_compressed(path,pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=uniform)
    return dict(gate_statistics=stats,uniform_intervention_auc=float(roc_auc_score(data['val_y'],uniform)),
        uniform_intervention_ap=float(average_precision_score(data['val_y'],uniform)),
        max_prediction_change=float(abs(original-uniform).max()),intervention_sha256=sha256(path),
        interpretation='replace gates after checkpoint selection; not a separately trained uniform control')


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    records=[]
    for fold in range(3):
        data=load_inner(fold);assert not set(data['train_indices'])&set(data['val_indices']) and not len(data['test_pairs'])
        paths=[(name,Path(c[str(fold)]['path']),True) for name,c in plan['controls'].items()]
        paths += [(name,HERE/'results'/f'{name}_f{fold}_s42.json',False) for name in CONFIGS]
        for name,path,reused in paths:
            r=json.loads(path.read_text());seed_all(42)
            a=degree_reference.make_adapter(data,name[len('degree_'):],'cuda:0') if name.startswith('degree_') else make_adapter(data,name,'cuda:0')
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
            record=dict(tag=path.stem,control_name=name if reused else None,reused_control=reused,status='passed',max_prediction_error=error)
            if not reused:
                assert r['settings']==CONFIGS[name] and r['source_hashes']==plan['source_hashes']
                assert r['no_lncrna'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
                curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
                assert max(curves,key=lambda c:c['selection_score'])['epoch']==r['best_epoch']
                for c in curves:
                    assert abs(c['selection_score']-.5*(c['inner_auc']+c['inner_ap']))<1e-12
                    assert c['weight']==CONFIGS[name]['reconstruction_weight']*min(1.,c['epoch']/50.)
                    assert c['loss']==float(np.float32(c['bce'])+np.float32(c['weight'])*np.float32(c['reconstruction']))
                record['explicit_deletions']=deletion_check(a,data,name)
                record['gate_diagnostics']=gate_diagnostics(a,data,path.stem)
                if CONFIGS[name]['reconstruction_weight']:
                    rec=a.model.reconstruction;del a.model.reconstruction
                    with torch.no_grad():bare=a.predict('val')
                    a.model.add_module('reconstruction',rec)
                    np.testing.assert_allclose(prob,bare,atol=2e-5,rtol=2e-5)
                    record['without_decoder_max_error']=float(abs(prob-bare).max())
            records.append(record);print('PASS',name,fold,flush=True);torch.cuda.empty_cache()
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,
        audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))


if __name__=='__main__':main()
