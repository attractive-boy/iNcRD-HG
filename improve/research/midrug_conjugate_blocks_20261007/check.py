"""Independent small-model calculations and training-only integration checks."""
import itertools
import numpy as np
from scipy.special import gammaln,betaln,digamma,logsumexp,softmax
from scipy.stats import beta,dirichlet
from scipy.integrate import quad
from common import HERE,CLEAN,load_inner,groups,write
from numerics import observations,globals_update,elbo,sweep,fit,predict,initialization,fuse,global_mean

def scalar_counts(qr,qd,pairs,y):
    c1=np.zeros((qr.shape[1],qd.shape[1]));c0=c1.copy()
    for (i,j),v in zip(pairs,y):
        for k in range(qr.shape[1]):
            for l in range(qd.shape[1]):
                (c1 if v else c0)[k,l]+=qr[i,k]*qd[j,l]
    return c1,c0

def independent_elbo(s,pairs,y):
    e1=digamma(s['a'])-digamma(s['a']+s['b']);e0=digamma(s['b'])-digamma(s['a']+s['b'])
    c1,c0=scalar_counts(s['qr'],s['qd'],pairs,y)
    value=np.sum(c1*e1+c0*e0)
    # Weighted quadrature handles the algebraic/logarithmic endpoint singularity;
    # direct/default quadrature loses accuracy for Beta shapes below one.
    ent=np.empty_like(s['a'])
    for k,l in np.ndindex(ent.shape):
        a,b=s['a'][k,l],s['b'][k,l];lb=betaln(a,b)
        i1=quad(lambda x:1.,0,1,weight='alg-loga',wvar=(a-1,b-1),epsabs=1e-12,epsrel=1e-12)[0]
        i0=quad(lambda x:1.,0,1,weight='alg-logb',wvar=(a-1,b-1),epsabs=1e-12,epsrel=1e-12)[0]
        ent[k,l]=lb-((a-1)*i1+(b-1)*i0)/np.exp(lb)
    value+=np.sum(-betaln(.5,.5)-.5*e1-.5*e0+ent)
    for q,g in [(s['qr'],s['gamma_r']),(s['qd'],s['gamma_d'])]:
        value+=gammaln(len(g))+dirichlet.entropy(g)
        for i in range(len(q)):
            for k in range(q.shape[1]):
                if q[i,k]>0:value+=q[i,k]*(digamma(g[k])-digamma(g.sum())-np.log(q[i,k]))
    return float(value)

def exact_evidence(pairs,y,nr,nd,kr,kd):
    values=[]
    for rr in itertools.product(range(kr),repeat=nr):
        r=np.eye(kr)[list(rr)];rc=np.bincount(rr,minlength=kr)
        rp=gammaln(kr)-gammaln(kr+nr)+gammaln(1+rc).sum()
        for dd in itertools.product(range(kd),repeat=nd):
            d=np.eye(kd)[list(dd)];dc=np.bincount(dd,minlength=kd)
            dp=gammaln(kd)-gammaln(kd+nd)+gammaln(1+dc).sum()
            c1,c0=scalar_counts(r,d,pairs,y)
            values.append(rp+dp+np.sum(betaln(c1+.5,c0+.5)-betaln(.5,.5)))
    return float(logsumexp(values))

def main():
    pairs=np.array([[0,0],[0,2],[1,1],[1,2]],dtype=np.int64);y=np.array([1,0,1,0])
    P,N=observations(pairs,y,(2,3));assert P.nnz+N.nnz==4 and N.nnz==2
    assert P[0,1]==N[0,1]==0 and P[1,0]==N[1,0]==0
    qr=np.array([[.8,.2],[.3,.7]]);qd=np.array([[.7,.3],[.2,.8],[.4,.6]])
    s=globals_update(qr,qd,P,N);c1,c0=scalar_counts(qr,qd,pairs,y)
    np.testing.assert_allclose(s['a'],c1+.5,atol=1e-15,rtol=0)
    np.testing.assert_allclose(s['b'],c0+.5,atol=1e-15,rtol=0)
    initial_elbo_error=abs(elbo(s,P,N)-independent_elbo(s,pairs,y))
    assert initial_elbo_error<1e-11
    # Independently enumerate the local row/column membership scores.
    e1=digamma(s['a'])-digamma(s['a']+s['b']);e0=digamma(s['b'])-digamma(s['a']+s['b'])
    er=digamma(s['gamma_r'])-digamma(s['gamma_r'].sum());ed=digamma(s['gamma_d'])-digamma(s['gamma_d'].sum())
    sr=np.tile(er,(2,1))
    for (i,j),v in zip(pairs,y):
        for k in range(2):
            for l in range(2):sr[i,k]+=qd[j,l]*(e1 if v else e0)[k,l]
    rnew=softmax(sr,axis=1);sd=np.tile(ed,(3,1))
    for (i,j),v in zip(pairs,y):
        for k in range(2):
            for l in range(2):sd[j,l]+=rnew[i,k]*(e1 if v else e0)[k,l]
    dnew=softmax(sd,axis=1);nextstate=sweep(s,P,N)
    np.testing.assert_allclose(nextstate['qr'],rnew,atol=1e-15,rtol=0)
    np.testing.assert_allclose(nextstate['qd'],dnew,atol=1e-15,rtol=0)
    fitted,trace,converged=fit(P,N,qr,qd);assert converged
    evidence=exact_evidence(pairs,y,2,3,2,2);bound=elbo(fitted,P,N);assert bound<=evidence+1e-12
    final_elbo_error=abs(bound-independent_elbo(fitted,pairs,y));assert final_elbo_error<1e-11
    # Exact collapsed evidence without observations must be one.
    assert abs(exact_evidence(np.empty((0,2),dtype=int),np.empty(0,dtype=int),2,3,2,2))<1e-12
    # Fixed groups: independently derive posterior means and query probabilities.
    hardr=np.eye(2)[[0,1]];hardd=np.eye(2)[[0,1,0]]
    hard=globals_update(hardr,hardd,P,N);cp,cn=scalar_counts(hardr,hardd,pairs,y)
    expected=np.array([(cp[k,l]+.5)/(cp[k,l]+cn[k,l]+1) for k,l in [[0,0],[1,1],[0,1]]])
    np.testing.assert_allclose(predict(hard,np.array([[0,0],[1,1],[0,1]])),expected,atol=1e-15,rtol=0)
    base=np.array([0,.01,.5,.99,1.]);prior=global_mean(y)
    np.testing.assert_array_equal(fuse(base,np.full(5,prior),prior),base)
    changed=fuse(base,np.array([.8,.8,.8,.8,.8]),prior);assert changed[0]==0 and changed[-1]==1
    assert (changed[1:-1]>base[1:-1]).all()
    # Real training-only five sweeps: no heldout prediction or performance evaluation.
    g=groups();real=[]
    for fold in range(3):
        data=load_inner(fold);P,N=observations(data['train_pairs'],data['train_y'],(605,216))
        q0,q1=initialization(g['miRNA'],g['drug']);a,h,_=fit(P,N,q0,q1,max_sweeps=5)
        flipped={**data,'val_y':1-data['val_y']}
        PP,NN=observations(flipped['train_pairs'],flipped['train_y'],(605,216));b,_,_=fit(PP,NN,q0,q1,max_sweeps=5)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        assert abs((a['a']-.5).sum()-data['train_y'].sum())<1e-9
        assert abs((a['b']-.5).sum()-(len(data['train_y'])-data['train_y'].sum()))<1e-9
        trainset=set(map(tuple,data['train_pairs']));assert not trainset.intersection(map(tuple,data['val_pairs']))
        real.append(dict(fold=fold,observed_positive=P.nnz,observed_negative=N.nnz,unknown_ignored=605*216-P.nnz-N.nnz,
                         heldout_label_flip_state_identical=True,five_sweep_elbo=h[-1]['elbo']))
    import torch
    graph=torch.load(CLEAN/'graph.pt',map_location='cpu');assert set(graph.node_types)=={'miRNA','drug'}
    assert all('lnc' not in str(e).lower() for e in graph.edge_types)
    write(HERE/'initial_checks.json',dict(status='passed',scalar_counts=True,independent_elbo=True,coordinate_scores=True,
        tiny_log_evidence=evidence,tiny_elbo=bound,tiny_sweeps=len(trace)-1,empty_evidence_normalized=True,
        weighted_quadrature_elbo_errors=[initial_elbo_error,final_elbo_error],
        exact_fixed_group_beta=True,unknown_not_negative=True,fusion_identity_and_endpoints=True,real_training_only=real,
        no_lncrna=True,heldout_predictions_evaluated=False))
    print('Prechecks passed; no heldout metric evaluated.',flush=True)

if __name__=='__main__':main()
