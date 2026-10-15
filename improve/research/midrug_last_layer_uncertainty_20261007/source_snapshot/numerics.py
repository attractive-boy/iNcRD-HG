"""Conditional convex fitting and exact local covariance, all float64."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.special import expit,roots_hermitenorm

READOUTS=('point','laplace','sandwich','constant')

def design(h,scale=None):
    h=np.asarray(h,dtype=np.float64)
    if scale is None:scale=np.maximum(np.sqrt(np.mean(h*h,axis=0)),1e-6)
    return np.column_stack((h/scale,np.ones(len(h)))),scale

def objective(x,y,theta,weights=None):
    n=len(x)
    if weights is None:weights=np.full(n,1/n)
    z=x@theta;p=expit(z)
    loss=np.dot(weights,np.logaddexp(0.,z)-y*z)+np.dot(theta,theta)/(2*n)
    grad=x.T@(weights*(p-y))+theta/n
    hess=(x.T*(weights*p*(1-p)))@x+np.eye(x.shape[1])/n
    return loss,grad,hess

def fit(x,y,weights=None,initial=None):
    theta=np.zeros(x.shape[1]) if initial is None else initial.copy()
    history=[]
    for iteration in range(101):
        loss,g,h=objective(x,y,theta,weights)
        norm=float(np.max(abs(g)))
        history.append(dict(iteration=iteration,objective=float(loss),gradient_inf=norm))
        if norm<=1e-9:return theta,history
        assert iteration<100,'stationarity failed'
        step=cho_solve(cho_factor(h,lower=True),g)
        decrement=float(g@step);assert decrement>0 and np.isfinite(decrement)
        rate=1.
        for _ in range(50):
            trial=theta-rate*step
            next_loss=objective(x,y,trial,weights)[0]
            if next_loss<=loss-1e-4*rate*decrement:break
            rate*=.5
        else:raise RuntimeError('Armijo line search failed')
        history[-1].update(step_size=rate,newton_decrement=decrement)
        theta=trial
    raise AssertionError('unreachable')

def covariance(x,y,theta):
    n=len(x);_,g,h=objective(x,y,theta)
    inv=cho_solve(cho_factor(h,lower=True),np.eye(len(theta)))
    scores=(expit(x@theta)-y)[:,None]*x
    centered=scores-scores.mean(0)
    b=centered.T@centered/n
    lap=inv/n;sand=inv@b@inv/(n+1)
    lap=(lap+lap.T)/2;sand=(sand+sand.T)/2
    assert np.linalg.eigvalsh(lap).min()>0
    assert np.linalg.eigvalsh(sand).min()>-1e-10
    return dict(hessian=h,gradient=g,score_covariance=b,laplace=lap,sandwich=sand)

def marginal(mean,variance,points=64):
    assert np.isfinite(mean).all() and np.isfinite(variance).all()
    assert np.min(variance)>=-1e-10
    nodes,weights=roots_hermitenorm(points)
    return expit(mean[:,None]+np.sqrt(np.maximum(variance,0))[:,None]*nodes)@(weights/np.sqrt(2*np.pi))

def predict(x,theta,cov,constant,points=64):
    mean=x@theta
    vs={k:np.einsum('ij,jk,ik->i',x,cov[k],x) for k in ['laplace','sandwich']}
    pred=dict(point=expit(mean),**{k:marginal(mean,v,points) for k,v in vs.items()},
              constant=marginal(mean,np.full(len(mean),constant),points))
    return pred,vs
