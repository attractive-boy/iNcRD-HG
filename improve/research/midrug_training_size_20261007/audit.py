"""Replay twelve new and six existing checkpoints on their exact fitting subsets."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_subset,seed_all,sha256
from model import CONFIGS,SIZES,make_adapter
from check import deletion_check


def metrics(y,p):
    return dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)))


@torch.no_grad()
def batch_check(a):
    all_logits=a.logits('val')[:3]
    singleton=torch.cat([a.model(a.pairs['val'][:,i:i+1],a.ps,a.qis['val'][i:i+1],a.ease['val'][i:i+1],a.dense) for i in range(3)])
    torch.testing.assert_close(all_logits,singleton,atol=2e-4,rtol=5e-5)
    return float(abs(all_logits-singleton).max())


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for path,digest in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(path)==digest
    assert sha256(HERE/'subsets.npz')==plan['subsets_sha256']
    records=[];predictions=HERE/'replay_predictions';predictions.mkdir(exist_ok=True)
    for fold in range(3):
        for size,fraction in {**SIZES,'full':1.}.items():
            data=load_subset(fold,size)
            for name in CONFIGS:
                tag=f'{name}_{size}_f{fold}_s42'
                if size=='full':
                    ref=plan['controls'][name][str(fold)];path=Path(ref['path'])
                    assert sha256(path)==ref['metadata_sha256']
                    assert sha256(path.with_name(path.stem+'_inner.npz'))==ref['probability_sha256']
                else:path=HERE/'results'/f'{tag}.json'
                r=json.loads(path.read_text())
                assert r['data_sha256']==plan['data_sha256']==sha256(CLEAN/'inner_data.npz')
                assert r['graph_sha256']==plan['graph_sha256']==sha256(CLEAN/'graph.pt')
                assert not r['outer_validation_evaluated'] and not r['test_evaluated']
                if size!='full':
                    assert r['source_hashes']==plan['source_hashes'] and r['subsets_sha256']==plan['subsets_sha256']
                    assert r['train_count']==len(data['train_y']) and r['size']==size
                assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']
                with np.load(path.with_name(path.stem+'_inner.npz')) as z:
                    for key,dk in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[dk])
                    stored=z['probability'].copy()
                expected=metrics(data['val_y'],stored)
                assert abs(expected['auc']-r['inner_auc'])<1e-12 and abs(expected['ap']-r['inner_ap'])<1e-12
                seed_all(42);a=make_adapter(data,name,'cuda:0');state=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
                a.model.load_state_dict(state);a.model.eval()
                assert int(a.model.eb.num_batches_tracked)==r['best_epoch']
                assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
                with torch.no_grad():
                    prob=a.predict('val');train_logits=a.logits('train');train_prob=train_logits.sigmoid().cpu().numpy()
                    train_bce=float(F.binary_cross_entropy_with_logits(train_logits,a.y))
                np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
                curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
                assert len(curves)==r['epochs_run'] and max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
                for x in curves:
                    assert abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12
                    if size!='full':
                        assert x['weight']==CONFIGS[name]['reconstruction_weight']*min(1.,x['epoch']/50.)
                        assert x['loss']==float(np.float32(x['bce'])+np.float32(x['weight'])*np.float32(x['reconstruction']))
                training=metrics(data['train_y'],train_prob)
                deletions=deletion_check(a,data,name,count=4);batch_error=batch_check(a)
                bare=make_adapter(data,'control','cuda:0');bare.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')});bare.model.eval()
                with torch.no_grad():plain=bare.predict('val')
                np.testing.assert_allclose(prob,plain,atol=2e-5,rtol=2e-5)
                saved=predictions/f'{tag}.npz'
                np.savez_compressed(saved,train_pairs=data['train_pairs'],train_y=data['train_y'],train_indices=data['train_indices'],train_probability=train_prob,
                    val_pairs=data['val_pairs'],val_y=data['val_y'],val_indices=data['val_indices'],val_probability=prob)
                records.append(dict(status='passed',tag=tag,config=name,size=size,fraction=fraction,fold=fold,existing_full_size_model=size=='full',
                    metadata_path=str(path),metadata_sha256=sha256(path),checkpoint_sha256=r['checkpoint_sha256'],replay_predictions_sha256=sha256(saved),
                    train_count=len(data['train_y']),positive_count=int(data['train_y'].sum()),val_count=len(data['val_y']),
                    train_metrics=training,train_bce=train_bce,val_metrics=expected,train_minus_val={k:training[k]-expected[k] for k in ['auc','ap']},
                    best_epoch=r['best_epoch'],epochs_run=r['epochs_run'],maximum_prediction_error=float(abs(prob-stored).max()),
                    explicit_deletions=deletions,batch_max_logit_error=batch_error,without_reconstruction_error=float(abs(prob-plain).max())))
                print('PASS',tag,flush=True);del a,bare,state,train_logits;torch.cuda.empty_cache()
    assert len(records)==18 and sum(not r['existing_full_size_model'] for r in records)==12
    result=dict(status='passed',records=records,new_fits=12,new_checkpoint_replays=12,existing_checkpoint_replays=6,total_checkpoint_replays=18,
                no_lncrna=True,original_validation_evaluated=False,test_evaluated=False)
    (HERE/'verification.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
