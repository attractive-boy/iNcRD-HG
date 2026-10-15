"""One converged conditional fit per fold, with four preregistered readouts."""
import argparse,json,time
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from common import HERE,load_inner,initialize,features,sha256,seed_all,verify_plan
from numerics import design,fit,objective,covariance,predict

def metrics(y,p):
    return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p),bce=log_loss(y,p),brier=brier_score_loss(y,p))

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(3),required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    plan=verify_plan();out=HERE/'results';tag=f'f{args.fold}_s42';assert not (out/f'{tag}.npz').exists()
    start=time.time();seed_all(42);torch.cuda.set_device(args.device);data=load_inner(args.fold)
    a,source,record=initialize(data,args.fold,args.device)
    state={k:v.clone() for k,v in a.model.state_dict().items()}
    ht,zt=features(a,'train');hv,zv=features(a,'val')
    for k,v in state.items():torch.testing.assert_close(v,a.model.state_dict()[k],atol=0,rtol=0)
    with np.load(source.with_name(source.stem+'_inner.npz')) as saved:
        np.testing.assert_allclose(expit(zv),saved['probability'],atol=2e-5,rtol=2e-5)
    x,scale=design(ht);xv,_=design(hv,scale)
    initial=np.r_[a.model.lin3.weight.detach().cpu().numpy().ravel()*scale,float(a.model.lin3.bias.detach().cpu()[0])]
    original_gradient=float(np.max(abs(objective(x,data['train_y'],initial)[1])))
    del a,state;torch.cuda.empty_cache()
    theta,history=fit(x,data['train_y']);cov=covariance(x,data['train_y'],theta)
    constant=float(np.einsum('ij,jk,ik->i',x,cov['sandwich'],x).mean())
    train_pred,train_var=predict(x,theta,cov,constant)
    pred,var=predict(xv,theta,cov,constant)
    more,_=predict(xv,theta,cov,constant,points=128)
    quadrature_error={k:float(np.max(abs(pred[k]-more[k]))) for k in pred}
    assert max(quadrature_error.values())<1e-6,quadrature_error
    np.savez_compressed(out/f'{tag}.npz',theta=theta,scale=scale,constant_variance=np.array(constant),
        train_features=ht,val_features=hv,train_source_logits=zt,val_source_logits=zv,
        train_pairs=data['train_pairs'],val_pairs=data['val_pairs'],train_y=data['train_y'],val_y=data['val_y'],
        train_indices=data['train_indices'],val_indices=data['val_indices'],
        **cov,**{f'prob_{k}':v for k,v in pred.items()},**{f'variance_{k}':v for k,v in var.items()})
    r=dict(fold=args.fold,seed=42,source_metadata=str(source),source_record=record,
        train_metrics={k:metrics(data['train_y'],v) for k,v in train_pred.items()},
        inner_metrics={k:metrics(data['val_y'],v) for k,v in pred.items()},convergence=history,
        original_gradient_inf=original_gradient,quadrature_error=quadrature_error,
        variance_summary={k:dict(mean=float(v.mean()),std=float(v.std()),max=float(v.max())) for k,v in var.items()},
        constant_variance=constant,artifacts_sha256=sha256(out/f'{tag}.npz'),plan_sha256=sha256(out/'plan.json'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-start)
    (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':main()
