"""Replay subgraph models and compare independently deleted query inputs."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from model import HERE,CLEAN,CONFIGS,load_inner,make_adapter,seed_all,sha256
from extract import Extractor
from common import Model
import check_modules


def check_local(a,data,cfg):
    ix=np.r_[np.flatnonzero(data['train_y']==1)[[0,77,511]],np.flatnonzero(data['train_y']==0)[[0,77,511]]]
    rows=[]
    for i in ix:
        keep=np.arange(len(data['train_y']))!=i
        deleted=Extractor({**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]})
        pair=a.pairs['train'][:,i:i+1]
        expected=a.model.cache.select(*pair)
        actual=deleted.one(int(pair[0,0]),int(pair[1,0]),cfg['k'],cfg['hops'])
        actual={key:torch.as_tensor(value[None],device=pair.device,dtype=expected[key].dtype) for key,value in actual.items()}
        for key in expected:torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
        with torch.no_grad():
            s1=a.model.subgraph(expected,a.dense[0],a.dense[1]);s2=a.model.subgraph(actual,a.dense[0],a.dense[1])
        torch.testing.assert_close(s1,s2,rtol=0,atol=0)
        rows.append(dict(index=int(i),label=int(data['train_y'][i]),subgraph_score_error=float((s1-s2).abs().max())))
    return rows


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for file,hashed in plan['source_hashes'].items():assert sha256(HERE/file)==hashed
    for file,hashed in plan['cache_hashes'].items():assert sha256(HERE/file)==hashed
    rows=[]
    for fold in range(3):
        data=load_inner(fold)
        # Reused control must replay through exactly the same clean loader.
        control=plan['controls'][str(fold)];path=CLEAN/'results'/f'control_f{fold}_s42.json'
        assert sha256(path)==control['result_sha256'] and sha256(path.with_suffix('.pt'))==control['checkpoint_sha256']
        seed_all(42);a=make_adapter(data,'control',fold,'cuda:0')
        a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval()
        with torch.no_grad():p=a.predict('val')
        saved=np.load(path.with_name(path.stem+'_inner.npz'))
        np.testing.assert_allclose(p,saved['probability'],rtol=2e-5,atol=2e-5)
        rows.append(dict(tag=f'control_f{fold}_s42',status='passed',reused_control=True,max_prediction_error=float(np.max(np.abs(p-saved['probability'])))))
        del a
        for name,cfg in CONFIGS.items():
            path=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(path.read_text())
            assert r['checkpoint_sha256']==sha256(path.with_suffix('.pt'))
            assert r['source_sha256']==sha256(HERE/'train.py') and r['model_sha256']==sha256(HERE/'model.py')
            assert not r['lncrna_used'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
            seed_all(42);a=make_adapter(data,name,fold,'cuda:0')
            a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in key.lower() for key in a.model.state_dict())
            with torch.no_grad():p=a.predict('val')
            z=np.load(path.with_name(path.stem+'_inner.npz'))
            for key,data_key in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[data_key])
            np.testing.assert_allclose(p,z['probability'],rtol=2e-5,atol=2e-5)
            assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
            assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
            curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
            for row in curves:assert abs(row['selection_score']-.5*(row['inner_auc']+row['inner_ap']))<1e-12
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            # Decompose the frozen checkpoint on inner-held-out queries only.
            with torch.no_grad():
                base_logits=Model.forward(a.model,a.pairs['val'],a.ps,a.qis['val'],a.ease['val'],a.dense)
                local_logits=a.model.subgraph(a.model.cache.select(*a.pairs['val']),a.dense[0],a.dense[1])
            bp,sp=base_logits.sigmoid().cpu().numpy(),local_logits.sigmoid().cpu().numpy()
            component_metrics=dict(base_component_auc=roc_auc_score(z['y'],bp),base_component_ap=average_precision_score(z['y'],bp),
                subgraph_component_auc=roc_auc_score(z['y'],sp),subgraph_component_ap=average_precision_score(z['y'],sp))
            np.savez_compressed(path.with_name(path.stem+'_components.npz'),pairs=z['pairs'],y=z['y'],
                base_logit=base_logits.cpu().numpy(),subgraph_logit=local_logits.cpu().numpy())
            local=check_local(a,data,cfg);full=check_modules.check(a,count=6)
            rows.append(dict(tag=r['tag'],status='passed',max_prediction_error=float(np.max(np.abs(p-z['probability']))),
                local_record_deletion=local,final_logit_deletion=full,component_metrics=component_metrics,reused_control=False))
            print('PASS',r['tag'],flush=True);del a;torch.cuda.empty_cache()
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=rows,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
