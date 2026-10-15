"""Numerical independent checks and strict frozen-feature boundary checks."""
import ast,json
import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm
from common import HERE,load_inner,initialize,make_adapter,features,sha256,seed_all
from numerics import design,fit,objective,covariance,marginal

def mathematics():
    rng=np.random.default_rng(1031);x=np.column_stack((rng.normal(size=(24,6)),np.ones(24)))
    y=rng.integers(0,2,len(x));theta=rng.normal(size=x.shape[1])
    t=torch.tensor(theta,requires_grad=True);xx=torch.tensor(x);yy=torch.tensor(y,dtype=torch.float64)
    def loss(v):return torch.nn.functional.binary_cross_entropy_with_logits(xx@v,yy)+(v@v)/(2*len(x))
    actual=objective(x,y,theta)
    autodiff_g=torch.autograd.grad(loss(t),t)[0].detach().numpy()
    autodiff_h=torch.autograd.functional.hessian(loss,t).detach().numpy()
    np.testing.assert_allclose(actual[1],autodiff_g,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(actual[2],autodiff_h,atol=1e-12,rtol=1e-12)
    theta,hist=fit(x,y);c=covariance(x,y,theta)
    scores=(expit(x@theta)-y)[:,None]*x
    direction=rng.normal(size=len(x));direction-=direction.mean();direction/=np.max(abs(direction))
    influence=-np.linalg.solve(c['hessian'],scores.T@direction/len(x))
    errors=[]
    for eps in [1e-2,5e-3]:
        weighted,_=fit(x,y,weights=(1+eps*direction)/len(x),initial=theta)
        errors.append(float(np.linalg.norm(weighted-theta-eps*influence)))
    assert errors[1]<.3*errors[0]
    # Dirichlet Monte Carlo checks the finite-n coefficient and centered gradients.
    w=rng.dirichlet(np.ones(len(x)),size=100000)
    perturb=-(w-1/len(x))@scores@np.linalg.inv(c['hessian'])
    mc=np.cov(perturb,rowvar=False,bias=True)
    relative=float(np.linalg.norm(mc-c['sandwich'])/np.linalg.norm(c['sandwich']))
    assert relative<.025
    mean=np.array([-10.,-2.,0.,2.,10.]);var=np.array([0.,.2,1.,4.,16.])
    gh=marginal(mean,var)
    quadrature=np.array([quad(lambda z:expit(m+np.sqrt(v)*z)*norm.pdf(z),-12,12,epsabs=1e-12)[0] for m,v in zip(mean,var)])
    np.testing.assert_allclose(gh,quadrature,atol=1e-6,rtol=0)
    zero=marginal(mean,np.zeros(len(mean)));np.testing.assert_allclose(zero,expit(mean),atol=2e-16)
    return dict(autodiff_gradient_error=float(np.max(abs(actual[1]-autodiff_g))),
        autodiff_hessian_error=float(np.max(abs(actual[2]-autodiff_h))),stationarity=hist[-1],
        influence_refit_errors=errors,dirichlet_covariance_relative_error=relative,
        gaussian_quadrature_error=float(np.max(abs(gh-quadrature))))

def main():
    result=dict(mathematics=mathematics());data=load_inner(0);seed_all(42)
    a,path,r=initialize(data,0,'cuda:0');state={k:v.clone() for k,v in a.model.state_dict().items()}
    h,z=features(a,'train');hv,zv=features(a,'val')
    w=a.model.lin3.weight.detach().cpu().double().numpy().ravel()
    b=float(a.model.lin3.bias.detach().cpu()[0]);linear=h@w+b
    np.testing.assert_allclose(linear,z,atol=1e-5,rtol=1e-5)
    with np.load(path.with_name(path.stem+'_inner.npz')) as saved:
        np.testing.assert_allclose(expit(zv),saved['probability'],atol=2e-5,rtol=2e-5)
    seed_all(42);other,_,_=initialize({**data,'val_y':1-data['val_y']},0,'cuda:0')
    for k,v in state.items():torch.testing.assert_close(v,other.model.state_dict()[k],atol=0,rtol=0)
    for part in a.qis:
        torch.testing.assert_close(a.qis[part],other.qis[part],atol=0,rtol=0)
        torch.testing.assert_close(a.ease[part],other.ease[part],atol=0,rtol=0)
    oh,oz=features(other,'val')
    np.testing.assert_allclose(hv,oh,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(zv,oz,atol=1e-5,rtol=1e-5)
    del other
    # Reuse only the independent physical deletion function, not mixed-task code.
    source=HERE.parent/'midrug_degree_encoder_20261006/check.py'
    tree=ast.parse(source.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deletion_check']
    ns=dict(torch=torch,np=np,make_adapter=make_adapter)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ns)
    deletion=ns['deletion_check'](a,data,'unmasked10',count=4)
    for k,v in state.items():torch.testing.assert_close(v,a.model.state_dict()[k],atol=0,rtol=0)
    assert all(not p.requires_grad and p.grad is None for p in a.model.parameters())
    result.update(status='passed',feature_shape=list(h.shape),source_linear_error=float(np.max(abs(linear-z))),
        heldout_label_flip_error=float(np.max(abs(oh-hv))),frozen_state_exact=True,explicit_deletions=deletion,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
