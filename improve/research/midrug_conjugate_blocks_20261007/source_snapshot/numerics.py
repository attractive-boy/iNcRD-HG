"""Float64 conjugate mean-field CAVI over observed binary entries only."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import digamma,gammaln,betaln,softmax,xlogy,expit,logit

KEYS=('qr','qd','a','b','gamma_r','gamma_d')
def observations(pairs,y,shape):
    pairs=np.asarray(pairs);y=np.asarray(y)
    assert pairs.shape==(len(y),2) and pairs.dtype.kind in 'iu'
    assert np.isin(y,[0,1]).all()
    assert (pairs>=0).all() and (pairs[:,0]<shape[0]).all() and (pairs[:,1]<shape[1]).all()
    assert len(np.unique(pairs,axis=0))==len(pairs)
    matrices=[]
    for label in [1,0]:
        pp=pairs[y==label];matrices.append(csr_matrix((np.ones(len(pp)),(pp[:,0],pp[:,1])),shape=shape))
    return tuple(matrices)
def initialization(row_labels,col_labels,soft=True):
    qr=np.eye(16)[row_labels];qd=np.eye(8)[col_labels]
    if soft:qr=.95*qr+.05/16;qd=.95*qd+.05/8
    return qr,qd
def globals_update(qr,qd,P,N):
    return dict(qr=qr,qd=qd,a=.5+qr.T@P.dot(qd),b=.5+qr.T@N.dot(qd),
                gamma_r=1+qr.sum(0),gamma_d=1+qd.sum(0))
def expected_logs(s):
    a,b=s['a'],s['b'];total=digamma(a+b)
    return digamma(a)-total,digamma(b)-total
def sweep(s,P,N):
    e1,e0=expected_logs(s);gr,gd=s['gamma_r'],s['gamma_d']
    qr=softmax(digamma(gr)-digamma(gr.sum())+P.dot(s['qd'])@e1.T+N.dot(s['qd'])@e0.T,axis=1)
    qd=softmax(digamma(gd)-digamma(gd.sum())+P.T.dot(qr)@e1+N.T.dot(qr)@e0,axis=1)
    return globals_update(qr,qd,P,N)
def dir_kl(g):
    k=len(g);return gammaln(g.sum())-gammaln(g).sum()-gammaln(k)+np.sum((g-1)*(digamma(g)-digamma(g.sum())))
def elbo(s,P,N):
    qr,qd=s['qr'],s['qd'];e1,e0=expected_logs(s)
    ll=np.sum((qr.T@P.dot(qd))*e1+(qr.T@N.dot(qd))*e0)
    beta_kl=np.sum(betaln(.5,.5)-betaln(s['a'],s['b'])+(s['a']-.5)*e1+(s['b']-.5)*e0)
    out=ll-beta_kl
    for q,g in [(qr,s['gamma_r']),(qd,s['gamma_d'])]:
        out+=np.sum(q.sum(0)*(digamma(g)-digamma(g.sum())))-np.sum(xlogy(q,q))-dir_kl(g)
    return float(out)
def fit(P,N,qr,qd,max_sweeps=2000):
    s=globals_update(qr.copy(),qd.copy(),P,N);old=elbo(s,P,N);stable=0
    history=[dict(sweep=0,elbo=old,membership_change=None,relative_elbo_change=None)]
    for step in range(1,max_sweeps+1):
        nxt=sweep(s,P,N);value=elbo(nxt,P,N)
        change=max(float(np.max(abs(nxt[k]-s[k]))) for k in ['qr','qd'])
        relative=abs(value-old)/(1+abs(old))
        assert value>=old-(1e-8+1e-12*abs(old)),(step,value,old)
        assert np.isfinite(value) and all(np.isfinite(v).all() for v in nxt.values())
        history.append(dict(sweep=step,elbo=value,membership_change=change,relative_elbo_change=relative))
        stable=stable+1 if change<=1e-8 and relative<=1e-10 else 0
        s,old=nxt,value
        if stable>=3:break
    return s,history,stable>=3
def predict(s,pairs):
    r=s['qr'][pairs[:,0]];d=s['qd'][pairs[:,1]]
    return np.einsum('ik,kl,il->i',r,s['a']/(s['a']+s['b']),d,optimize=False)
def global_mean(y):return float((np.sum(y)+.5)/(len(y)+1))
def fuse(base,block,prior,weight=.1):
    base=np.asarray(base,dtype=np.float64)
    assert np.isfinite(base).all() and ((base>=0)&(base<=1)).all()
    offset=weight*(logit(np.clip(block,1e-12,1-1e-12))-logit(prior))
    out=np.empty_like(base);inside=(base>0)&(base<1)
    out[inside]=expit(logit(base[inside])+offset[inside]);out[~inside]=base[~inside]
    out[offset==0]=base[offset==0]
    return out
