"""Replay immutable features and independently verify fitted convex quantities."""
import json
import numpy as np
import torch
from scipy.special import expit
from common import HERE,load_inner,initialize,make_adapter,features,verify_plan,sha256,seed_all
from numerics import design,predict
from train import metrics

@torch.no_grad()
def query_deletions(a,data,z):
    pos=np.flatnonzero(data['train_y']==1);neg=np.flatnonzero(data['train_y']==0)
    degree=np.bincount(data['train_pairs'][pos,0],minlength=605)
    choices=np.unique([pos[np.argmin(degree[data['train_pairs'][pos,0]])],pos[len(pos)//2],neg[0]])
    records=[]
    for ix in choices:
        keep=np.arange(len(data['train_y']))!=ix
        other={**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
        b=make_adapter(other,'unmasked10',a.y.device);b.model.load_state_dict(a.model.state_dict());b.model.eval()
        r,d=a.pairs['train'][:,ix:ix+1];query=torch.stack((r,d))
        qi2=(b.ps.qis(r,d)-a.mu)/a.sd;ease2=b.ps.ease(r,d)
        hs=[]
        for adapter,qi,ease in [(a,a.qis['train'][ix:ix+1],a.ease['train'][ix:ix+1]),(b,qi2,ease2)]:
            hook=adapter.model.lin3.register_forward_pre_hook(lambda m,args:hs.append(args[0].detach().cpu().double().numpy()))
            try:adapter.model(query,adapter.ps,qi,ease,adapter.dense)
            finally:hook.remove()
        np.testing.assert_allclose(hs[0],hs[1],atol=2e-4,rtol=5e-5)
        pp=[]
        for h in hs:
            xx,_=design(h,z['scale']);p,_=predict(xx,z['theta'],z,float(z['constant_variance']));pp.append(p)
        errors={k:float(np.max(abs(pp[0][k]-pp[1][k]))) for k in pp[0]}
        assert max(errors.values())<2e-5
        records.append(dict(pair=data['train_pairs'][ix].tolist(),label=int(data['train_y'][ix]),
            feature_error=float(np.max(abs(hs[0]-hs[1]))),readout_errors=errors))
        del b
    return records

def main():
    plan=verify_plan();records=[];out=HERE/'results'
    for fold in range(3):
        tag=f'f{fold}_s42';r=json.loads((out/f'{tag}.json').read_text())
        assert sha256(out/f'{tag}.npz')==r['artifacts_sha256']
        assert sha256(out/'plan.json')==r['plan_sha256']
        z=dict(np.load(out/f'{tag}.npz'));data=load_inner(fold);seed_all(42)
        a,source,sr=initialize(data,fold,'cuda:0');state={k:v.clone() for k,v in a.model.state_dict().items()}
        errors={}
        for part in ['train','val']:
            h,logits=features(a,part)
            np.testing.assert_array_equal(data[part+'_pairs'],z[part+'_pairs'])
            np.testing.assert_array_equal(data[part+'_y'],z[part+'_y'])
            np.testing.assert_array_equal(data[part+'_indices'],z[part+'_indices'])
            np.testing.assert_allclose(h,z[part+'_features'],atol=1e-5,rtol=1e-5)
            np.testing.assert_allclose(logits,z[part+'_source_logits'],atol=1e-5,rtol=1e-5)
            errors[part]=float(np.max(abs(h-z[part+'_features'])))
        deleted=query_deletions(a,data,z)
        for k,v in state.items():torch.testing.assert_close(v,a.model.state_dict()[k],atol=0,rtol=0)
        del a,state;torch.cuda.empty_cache()
        x,scale=design(z['train_features']);xv,_=design(z['val_features'],scale)
        np.testing.assert_array_equal(scale,z['scale']);theta=z['theta'];y=z['train_y'];n=len(y)
        # Independent formulas and solves; do not call covariance()/fit().
        prob=expit(x@theta);gradient=(x.T@(prob-y)+theta)/n
        assert np.max(abs(gradient))<=1e-9
        precision=x.T@(prob[:,None]*(1-prob[:,None])*x)+np.eye(len(theta))
        inverse=np.linalg.solve(precision,np.eye(len(theta)))
        score=x*(prob-y)[:,None];centered=score-score.mean(0)
        sandwich=n/(n+1)*inverse@(centered.T@centered)@inverse
        np.testing.assert_allclose(z['hessian'],precision/n,atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(z['laplace'],inverse,atol=1e-10,rtol=1e-8)
        np.testing.assert_allclose(z['sandwich'],sandwich,atol=1e-10,rtol=1e-8)
        np.testing.assert_allclose(z['score_covariance'],centered.T@centered/n,atol=1e-12,rtol=1e-10)
        constant=np.sum((x@sandwich)*x)/n
        np.testing.assert_allclose(constant,z['constant_variance'],atol=1e-10,rtol=1e-8)
        cov=dict(laplace=inverse,sandwich=sandwich)
        predictions,variance=predict(xv,theta,cov,constant)
        prediction_errors={}
        for name,p in predictions.items():
            np.testing.assert_allclose(p,z['prob_'+name],atol=1e-10,rtol=1e-8)
            prediction_errors[name]=float(np.max(abs(p-z['prob_'+name])))
            actual=metrics(y=z['val_y'],p=z['prob_'+name])
            for k,v in actual.items():assert abs(v-r['inner_metrics'][name][k])<1e-12
        # Query order and batching must not determine uncertainty or ranking.
        reversed_p,_=predict(xv[::-1],theta,cov,constant)
        for name in predictions:np.testing.assert_allclose(predictions[name],reversed_p[name][::-1],atol=1e-14,rtol=1e-14)
        batched={name:[] for name in predictions}
        for xx in np.array_split(xv,7):
            pp,_=predict(xx,theta,cov,constant)
            for name in pp:batched[name].append(pp[name])
        for name in predictions:np.testing.assert_allclose(predictions[name],np.concatenate(batched[name]),atol=1e-14,rtol=1e-14)
        objectives=[v['objective'] for v in r['convergence']]
        assert np.max(np.diff(objectives))<=1e-14
        assert r['convergence'][-1]['gradient_inf']<=1e-9
        point=r['inner_metrics']['point'];const=r['inner_metrics']['constant']
        for k in ['auc','ap']:assert abs(point[k]-const[k])<1e-12
        records.append(dict(fold=fold,feature_replay_errors=errors,prediction_replay_errors=prediction_errors,
            stationarity_gradient_inf=float(np.max(abs(gradient))),hessian_condition_number=float(np.linalg.cond(precision)),
            source_frozen_exact=True,order_batch_invariant=True,constant_variance_preserves_auc_ap=True,physical_query_deletions=deleted))
        print('PASS',fold,flush=True)
    control_records=[]
    for name,cs in plan['controls'].items():
        for fold,item in cs.items():
            from pathlib import Path
            p=Path(item['path']);assert sha256(p)==item['metadata_sha256']
            assert sha256(p.with_suffix('.pt'))==item['checkpoint_sha256']
            f=p.with_name(p.stem+'_inner.npz');assert sha256(f)==item['probability_sha256']
            d=load_inner(int(fold));v=dict(np.load(f));np.testing.assert_array_equal(v['y'],d['val_y']);np.testing.assert_array_equal(v['pairs'],d['val_pairs'])
            control_records.append(dict(name=name,fold=int(fold),metrics=metrics(v['y'],v['probability']),reused=True))
    report=dict(status='passed',records=records,reused_controls=control_records,new_fitted_state_replays=3,
        new_readout_replays=12,plan_sha256=sha256(out/'plan.json'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'verification.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
