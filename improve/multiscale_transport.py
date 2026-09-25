"""FALCON-inspired fused, rectangular cross-RNA hypergraph transport.

Adaptation of arXiv:2608.29635v1, not its official reproduction. Drug
neighborhoods are hyperedges. A shared rectangular coupling aligns the two
RNA populations across cumulative degree filtrations. Observed common-drug
profiles and known LncMi edges provide the fused linear cost. There is no
asserted one-to-one biological correspondence and no Hungarian decoding.
"""
import numpy as np
import torch


def association_tables(a):
    counts=a['node_counts']; offsets=np.concatenate(([0],np.cumsum(counts)))
    ti=0 if int(a['pair_offset'][0])==0 else 1
    if int(a['pair_offset'][0])!=int(offsets[ti]) or int(a['pair_offset'][1])!=int(offsets[2]):
        raise ValueError('Unexpected typed node offsets')
    oi=1-ti; n,m,d=int(counts[ti]),int(counts[oi]),int(a['nd'])
    auxiliary=np.zeros((m,d),dtype=np.float32)
    edges=a['edges'][:,a['relation']==(1 if oi==0 else 3)]
    auxiliary[edges[0]-offsets[oi],edges[1]-offsets[2]]=1.
    bridge=np.zeros((int(counts[0]),int(counts[1])),dtype=np.float32)
    edges=a['edges'][:,a['relation']==5]
    bridge[edges[0]-offsets[0],edges[1]-offsets[1]]=1.
    return n,auxiliary,bridge if ti==0 else bridge.T


def support_matrix(a, query_indices):
    n,_,_=association_tables(a)
    keep=np.ones(len(a['y_train']),bool);keep[np.asarray(query_indices,dtype=int)]=False
    pairs=a['pairs_train'][:,keep & (a['y_train']==1)]
    result=np.zeros((n,int(a['nd'])),dtype=np.float32)
    result[pairs[0],pairs[1]]=1.
    query=a['pairs_train'][:,query_indices]
    assert not result[query[0],query[1]].any()
    return result


def filtration_pair(a,b,levels=4):
    """Pooled score-quantile buckets; zero evidence on either side gets weight 0.

    This synchronized bucket rule differs from FALCON's scan-and-coarsen
    implementation. Binary co-occurrence costs and balanced edge-count
    weights follow its equations (1)-(6). Isolated nodes retain all costs 1
    except the diagonal; their positive marginal smoothing is explicit.
    """
    def scores(x):
        score=x.T@x.sum(1)
        score=score/max(float(score.max()),1.)
        return score,x.sum(0)>0
    sa,ka=scores(a);sb,kb=scores(b)
    values=np.concatenate([sa[ka],sb[kb]])
    if not len(values):
        raise ValueError('No association hyperedges')
    thresholds=np.unique(np.quantile(values,np.arange(1,levels+1)/levels))
    aa,bb,weights=[],[],[]
    previous=-np.inf
    def costs(x,active):
        co=x[:,active]@x[:,active].T
        c=(co==0).astype(np.float32);np.fill_diagonal(c,0.)
        return c
    for threshold in thresholds:
        na=ka & (sa>previous) & (sa<=threshold)
        nb=kb & (sb>previous) & (sb<=threshold)
        weight=np.sqrt(na.sum()*nb.sum())
        if weight>0:
            aa.append(costs(a,ka & (sa<=threshold)))
            bb.append(costs(b,kb & (sb<=threshold)))
            weights.append(weight)
        previous=threshold
    if not weights:
        aa=[costs(a,ka)];bb=[costs(b,kb)];weights=[1.]
    weights=np.asarray(weights,dtype=np.float32);weights/=weights.sum()
    return np.stack(aa),np.stack(bb),weights


def fused_feature_cost(a,b,bridge):
    # Both populations share real drug IDs; independent PCA coordinates are
    # intentionally not compared across RNA types.
    idf=1./np.sqrt(1.+a.sum(0)+b.sum(0))
    xa,xb=a*idf,b*idf
    xa=xa/np.maximum(np.linalg.norm(xa,axis=1,keepdims=True),1e-8)
    xb=xb/np.maximum(np.linalg.norm(xb,axis=1,keepdims=True),1e-8)
    return .75*(1.-np.clip(xa@xb.T,0.,1.))+.25*(1.-bridge)


def gw_tensor(cs,ct,plan,p,q):
    return (cs.square()@p)[:,None]+(ct.square()@q)[None,:]-2.*(cs@plan@ct.T)


def sinkhorn(cost,p,q,epsilon=.1,steps=200,tolerance=2e-6):
    if epsilon<=0 or (p<=0).any() or (q<=0).any():
        raise ValueError('Positive temperature and marginals are required')
    logk=-cost/epsilon
    u=torch.zeros_like(p);v=torch.zeros_like(q)
    for iteration in range(steps):
        u=p.log()-torch.logsumexp(logk+v[None,:],dim=1)
        v=q.log()-torch.logsumexp(logk+u[:,None],dim=0)
        if iteration%10==9 or iteration+1==steps:
            plan=(logk+u[:,None]+v[None,:]).exp()
            error=max(float((plan.sum(1)-p).abs().sum()),float((plan.sum(0)-q).abs().sum()))
            if error<tolerance:
                break
    if error>2e-5:
        raise ArithmeticError('Sinkhorn marginal residual exceeds numerical gate: '+str(error))
    return plan,error,iteration+1


@torch.no_grad()
def transport(a,b,bridge,mode='full',device='cpu',levels=4,epsilon=.1,alpha=.5,outer_steps=40):
    if mode not in ('full','collapsed','feature_only','no_capacity','independent'):
        raise ValueError('Unknown transport control')
    dtype=torch.float32
    convert=lambda x:torch.as_tensor(x,dtype=dtype,device=device)
    p=convert(a.sum(1)+1.);p/=p.sum()
    q=convert(b.sum(1)+1.);q/=q.sum()
    linear=convert(fused_feature_cost(a,b,bridge))
    plan=p[:,None]*q[None,:]
    trace=[]
    if mode=='no_capacity':
        plan=p[:,None]*torch.softmax(q.log()[None,:]-(1-alpha)*linear/epsilon,dim=1)
    elif mode=='feature_only':
        plan,error,inner=sinkhorn((1-alpha)*linear,p,q,epsilon)
        trace.append(dict(iteration=1,marginal_l1=error,sinkhorn_steps=inner))
    elif mode in ('full','collapsed'):
        if mode=='collapsed':
            csa,csa_other,weights=filtration_pair(a,b,1)
        else:
            csa,csa_other,weights=filtration_pair(a,b,levels)
        cs,ct=convert(csa),convert(csa_other);weights=convert(weights)
        for iteration in range(outer_steps):
            g=sum(w*gw_tensor(x,y,plan,p,q) for w,x,y in zip(weights,cs,ct))
            proposal,error,inner=sinkhorn((1-alpha)*linear+2*alpha*g,p,q,epsilon)
            # Damping is explicit; the objective is nonconvex and this finite
            # iteration operator does not promise a global optimum.
            updated=.5*plan+.5*proposal
            change=float((updated-plan).abs().sum())
            plan=updated
            distortion=sum(w*(gw_tensor(x,y,plan,p,q)*plan).sum() for w,x,y in zip(weights,cs,ct))
            objective=(1-alpha)*(linear*plan).sum()+alpha*distortion+epsilon*(plan*plan.clamp_min(1e-30).log()).sum()
            trace.append(dict(iteration=iteration+1,change_l1=change,marginal_l1=error,
                              sinkhorn_steps=inner,objective=float(objective),gw=float(distortion)))
            if iteration>=9 and change<1e-6:
                break
    row_error=float((plan.sum(1)-p).abs().sum())
    column_error=float((plan.sum(0)-q).abs().sum())
    if row_error>2e-5 or (mode!='no_capacity' and column_error>2e-5):
        raise ArithmeticError('Transport conservation failed')
    transfer=(plan.double()@convert(b).double())/p.double()[:,None]
    return transfer.cpu().numpy(),plan.cpu().numpy(),dict(
        row_marginal_l1=row_error,column_marginal_l1=column_error,
        converged=mode not in ('full','collapsed') or trace[-1]['change_l1']<1e-6,
        trace=trace)


def calibration_features(predicted,support,pairs):
    r,d=pairs
    probability=np.clip(predicted[r,d],1e-7,1-1e-7)
    return np.column_stack((np.log(probability)-np.log1p(-probability),
                            np.log1p(support.sum(1)[r]),np.log1p(support.sum(0)[d])))
