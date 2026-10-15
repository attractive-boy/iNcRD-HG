"""Replay selected models, inspect final states and validate actual optimizer steps."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,make_optimizer,initialize,source_record
from check import branch_check,chunk_gradient_check,deletion_check,QUERY_CHECK_SOURCE
PREVIOUS=HERE.parent/'midrug_training_size_20261007'

def saved_scores(path,data,r):
    with np.load(path.with_name(path.stem+'_inner.npz')) as z:
        for k,d in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[k],data[d])
        p=z['probability'].copy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert abs(roc_auc_score(data['val_y'],p)-r['inner_auc'])<1e-12
    assert abs(average_precision_score(data['val_y'],p)-r['inner_ap'])<1e-12
    return p

def optimizer_check(a,states,r):
    m=a.model;m.eval();parameters={n:p for n,p in m.named_parameters() if p.requires_grad}
    q=a.pairs['train'][:,:70]
    loss=m.feedback_projection(m.feedback(*a.dense[:2],q)).square().mean()
    grads=torch.autograd.grad(loss,list(parameters.values()),allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    opt=make_optimizer(m);template=opt.state_dict();lookup={id(p):n for n,p in parameters.items()}
    names={pid:lookup[id(p)] for actual,saved in zip(opt.param_groups,template['param_groups']) for p,pid in zip(actual['params'],saved['params'])}
    checks={}
    for part,count in [('selected',r['best_epoch']),('final',r['epochs_run'])]:
        state=states[part];assert state['param_groups']==template['param_groups']
        assert set(state['state'])==(set(names) if count else set())
        for pid,v in state['state'].items():
            assert set(v)=={'step','square_avg'} and int(v['step'])==count
            assert v['square_avg'].shape==parameters[names[pid]].shape
            assert torch.isfinite(v['square_avg']).all() and (v['square_avg']>=0).all()
        checks[part]=dict(feedback_updates=count,main_updates=0,state_parameter_count=len(state['state']))
    return checks

def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for p,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(p)==h
    prior=json.loads((PREVIOUS/'verification.json').read_text());assert prior['status']=='passed'
    bypath={r['metadata_path']:r for r in prior['records']};records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name,sources in plan['controls'].items():
            src=sources[str(fold)];path=Path(src['path']);r=json.loads(path.read_text());old=bypath[str(path)]
            assert old['status']=='passed' and old['existing_full_size_model']
            assert sha256(path)==src['metadata_sha256']==old['metadata_sha256']
            assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']==src['checkpoint_sha256']==old['checkpoint_sha256']
            assert sha256(path.with_name(path.stem+'_inner.npz'))==src['probability_sha256']
            assert sha256(PREVIOUS/'replay_predictions'/(old['tag']+'.npz'))==old['replay_predictions_sha256']
            assert r['data_sha256']==plan['data_sha256'] and r['graph_sha256']==plan['graph_sha256']
            assert r.get('no_lncrna') is True or r.get('lncrna_used') is False
            saved_scores(path,data,r)
            records.append(dict(tag=path.stem,metadata_path=str(path),status='passed',reused_control=True,
                prior_replay_tag=old['tag'],verification='Immutable prior GPU replay; current hashes, pairs, labels, indices and saved metric checks'))
        for name in CONFIGS:
            path=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(path.read_text());source,origin=source_record(fold)
            assert r['config']==name and r['fold']==fold and r['seed']==42
            assert r['source_hashes']==plan['source_hashes'] and r['plan_sha256']==sha256(HERE/'results/plan.json')
            assert r['source_metadata']==str(source) and r['source_metadata_sha256']==sha256(source)
            assert r['source_checkpoint_sha256']==origin['checkpoint_sha256'] and r['source_best_epoch']==origin['best_epoch']
            assert r['graph_sha256']==plan['graph_sha256']==sha256(CLEAN/'graph.pt')
            assert r['data_sha256']==plan['data_sha256']==sha256(CLEAN/'inner_data.npz')
            assert r['no_lncrna'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
            curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
            assert len(curves)==r['epochs_run']+1 and 100<=r['epochs_run']<=300
            assert max(curves,key=lambda x:x['selection_score'])['stage_epoch']==r['best_epoch']
            assert curves[r['best_epoch']]['selection_score']==r['selection_score']
            assert r['epochs_run']==300 or r['epochs_run']-r['best_epoch']>=100
            for ep,row in enumerate(curves):
                assert row['stage_epoch']==ep and row['optimizer_update']==(ep>0)
                assert abs(row['selection_score']-.5*(row['inner_auc']+row['inner_ap']))<1e-12
                assert all(np.isfinite(row[k]) and 0<=row[k]<=1 for k in ['inner_auc','inner_ap'])
                if ep==0:
                    assert row['inner_auc']==origin['inner_auc'] and row['inner_ap']==origin['inner_ap'] and 'loss' not in row
                else:assert np.isfinite(row['loss']) and row['loss']>=0
            seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model
            assert all(p.requires_grad==n.startswith(('feedback.','feedback_projection.')) for n,p in m.named_parameters())
            assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']
            state=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True);m.load_state_dict(state);m.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            assert all(torch.isfinite(v).all() for v in state.values())
            final=path.with_name(path.stem+'_final.pt');assert sha256(final)==r['final_checkpoint_sha256']
            last=torch.load(final,map_location='cuda:0',weights_only=True)
            assert set(last)==set(state) and all(torch.isfinite(v).all() for v in last.values())
            original=torch.load(source.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            assert set(original)=={k for k in state if not k.startswith(('feedback.','feedback_projection.'))}
            for k,v in original.items():
                torch.testing.assert_close(v,state[k],atol=0,rtol=0);torch.testing.assert_close(v,last[k],atol=0,rtol=0)
            assert int(state['eb.num_batches_tracked'])==int(last['eb.num_batches_tracked'])==origin['best_epoch']
            if r['best_epoch']==0:
                seed_all(42);initial,_,_=initialize(data,name,fold,'cuda:0')
                for k,v in initial.model.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
                del initial
            del last,original
            stored=saved_scores(path,data,r)
            with torch.no_grad():prob=a.predict('val');train=a.predict('train')
            np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            rp=HERE/'replay_predictions';rp.mkdir(exist_ok=True);np.savez_compressed(rp/(path.stem+'.npz'),train=train,val=prob)
            op=path.with_name(path.stem+'_optimizers.pt');assert sha256(op)==r['optimizers_sha256']
            # This local hash-verified optimizer file needs the Torch 1.13 full
            # unpickler for floating point param-group values.
            states=torch.load(op,map_location='cpu',weights_only=False);assert set(states)=={'selected','final'}
            optimizers=optimizer_check(a,states,r)
            rec=dict(tag=path.stem,status='passed',reused_control=False,metadata_path=str(path),best_stage_epoch=r['best_epoch'],epochs_run=r['epochs_run'],
                max_prediction_error=float(abs(prob-stored).max()),optimizers=optimizers,source_selected_and_final_equal=True,
                branch=branch_check(a),chunk=chunk_gradient_check(a),explicit_deletions=deletion_check(a,data,name,count=4),
                replay_predictions_sha256=sha256(rp/(path.stem+'.npz')))
            reconstruction=m.reconstruction;del m.reconstruction
            with torch.no_grad():plain=a.predict('val')
            np.testing.assert_allclose(prob,plain,atol=2e-5,rtol=2e-5);m.add_module('reconstruction',reconstruction)
            rec['without_reconstruction_error']=float(abs(prob-plain).max())
            assert not m.lin1._forward_hooks
            records.append(rec);print('PASS',path.stem,flush=True)
            del a,m,state,states;torch.cuda.empty_cache()
    assert len(records)==15
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,new_fits=9,new_checkpoint_replays=9,
        final_state_inspections=9,immutable_reused_control_checks=6,audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
