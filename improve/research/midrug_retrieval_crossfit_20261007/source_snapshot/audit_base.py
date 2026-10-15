"""Replay 18 bases, prove index coverage and materialize aligned offsets."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,nested_data,make_base,seed_all,sha256,read_module
QUERY_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
query=read_module('crossfit_base_deletion',QUERY_SOURCE);query.make_adapter=lambda data,name,device:make_base(data,device)

def main():
    plan=json.loads((HERE/'bases/plan.json').read_text())
    for p,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(p)==h
    records=[];manifests=[];out=HERE/'offsets';out.mkdir(exist_ok=True)
    for fold in range(3):
        outer=load_inner(fold);oof=np.full(len(outer['train_y']),np.nan,dtype=np.float32);coverage=np.zeros(len(oof),dtype=int)
        in_sample=val=None;sources=[]
        for k in range(-1,5):
            data,rows=nested_data(fold,k);p=HERE/'bases'/f'base_f{fold}_k{k}.json';r=json.loads(p.read_text())
            assert r['fold']==fold and r['nested']==k and r['seed']==42 and r['epochs']==300
            assert r['source_hashes']==plan['source_hashes'] and r['plan_sha256']==sha256(HERE/'bases/plan.json')
            assert not r['prediction_fold_labels_used_for_selection'] and r['no_lncrna'] and not r['test_evaluated'] and not r['outer_validation_evaluated']
            assert r['graph_sha256']==plan['graph_sha256']==sha256(CLEAN/'graph.pt') and r['data_sha256']==plan['data_sha256']==sha256(CLEAN/'inner_data.npz')
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256'] and sha256(p.with_suffix('.npz'))==r['cache_sha256']
            assert len(r['curves'])==300
            for ep,row in enumerate(r['curves'],1):
                assert row['epoch']==ep and row['weight']==10*min(1.,ep/50.)
                assert all(np.isfinite(row[key]) and row[key]>=0 for key in ['bce','reconstruction','loss'])
                assert row['loss']==float(np.float32(row['bce'])+np.float32(row['weight'])*np.float32(row['reconstruction']))
            with np.load(p.with_suffix('.npz')) as z:
                for key in ['train_pairs','val_pairs','train_indices','val_indices']:np.testing.assert_array_equal(z[key],data[key])
                np.testing.assert_array_equal(z['prediction_rows'],rows);train=z['train_logit'].copy();predict=z['val_logit'].copy()
            seed_all(42);a=make_base(data,'cuda:0');state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True);a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in n.lower() for n in state)
            assert int(a.model.eb.num_batches_tracked)==300
            with torch.no_grad():tr=a.logits('train').cpu().numpy();va=a.logits('val').cpu().numpy()
            np.testing.assert_allclose(tr,train,atol=2e-4,rtol=5e-5);np.testing.assert_allclose(va,predict,atol=2e-4,rtol=5e-5)
            deletion=query.deletion_check(a,data,'unmasked10',count=4)
            sources.append(dict(path=str(p),metadata_sha256=sha256(p),checkpoint_sha256=r['checkpoint_sha256'],cache_sha256=r['cache_sha256']))
            if k==-1:in_sample,val=train,predict
            else:
                assert not np.intersect1d(data['train_indices'],outer['val_indices']).size and not np.intersect1d(data['val_indices'],outer['val_indices']).size
                oof[rows]=predict;coverage[rows]+=1
            records.append(dict(fold=fold,nested=k,status='passed',max_train_logit_error=float(abs(tr-train).max()),max_prediction_logit_error=float(abs(va-predict).max()),explicit_deletions=deletion))
            print('PASS',fold,k,flush=True);del a,state;torch.cuda.empty_cache()
        assert (coverage==1).all() and np.isfinite(oof).all()
        file=out/f'fold{fold}.npz'
        np.savez_compressed(file,oof=oof,in_sample=in_sample,val=val,train_pairs=outer['train_pairs'],train_indices=outer['train_indices'],coverage=coverage)
        prob=torch.sigmoid(torch.from_numpy(val)).numpy()
        full=dict(config='full300',fold=fold,inner_auc=roc_auc_score(outer['val_y'],prob),inner_ap=average_precision_score(outer['val_y'],prob),
            best_epoch=300,epochs_run=300,source=sources[0],offsets_sha256=sha256(file),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
        (out/f'full300_f{fold}.json').write_text(json.dumps(full,indent=2));manifests.append(dict(fold=fold,path=str(file),sha256=sha256(file),sources=sources))
    (HERE/'base_verification.json').write_text(json.dumps(dict(status='passed',records=records,new_fits=18,new_checkpoint_replays=18,oof_exactly_once=True,
        offsets=manifests,source_hashes={str(p):sha256(p) for p in [HERE/'audit_base.py',QUERY_SOURCE]},no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
