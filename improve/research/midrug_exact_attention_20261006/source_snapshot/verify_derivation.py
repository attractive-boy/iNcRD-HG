"""Independent NumPy proof-by-examples of the two-layer incident-update algebra.

This does not implement or verify the future PyTorch model or its gradients.
"""
import hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
TYPES={'mm':('m','m'),'dd':('d','d'),'md':('m','d'),'dm':('d','m')}
NODES={'m':5,'d':4}


def aggregate(degree,mass,numerator,alpha):
    out=np.zeros_like(numerator);ok=degree>0
    out[ok]=degree[ok,None]**(1-alpha)*numerator[ok]/mass[ok,None]
    return out


def gate(source,target,p):
    score=np.tanh(source@p['a']+target@p['b'])
    return 1/(1+np.exp(-score))


def layer(x,edges,params,alpha):
    out={n:np.zeros((count,5)) for n,count in NODES.items()};cache={}
    for name,(st,dt) in TYPES.items():
        p=params[name];s,d=edges[name].T
        g=gate(x[st][s],x[dt][d],p);values=x[st]@p['W']+p['bias']
        degree=np.bincount(d,minlength=NODES[dt]);mass=np.bincount(d,weights=g,minlength=NODES[dt])
        numerator=np.zeros((NODES[dt],5));np.add.at(numerator,d,g[:,None]*values[s])
        message=aggregate(degree,mass,numerator,alpha)
        out[dt] += (message+x[dt]@p['R']+p['root_bias'])/2
        cache[name]=dict(g=g,values=values,degree=degree,mass=mass,numerator=numerator,message=message)
    return out,cache


def activate(x):return {n:np.where(v>=0,v,.1*v) for n,v in x.items()}


def remove(edges,r,d):
    out={k:e.copy() for k,e in edges.items()}
    for name,pair in [('md',(r,d)),('dm',(d,r))]:
        out[name]=out[name][~np.all(out[name]==pair,axis=1)]
    return out


def updated(x,edges,params1,params2,r,d,alpha):
    endpoint={'m':r,'d':d};p1,c1=layer(x,edges,params1,alpha);h1=activate(p1)
    base,c2=layer(h1,edges,params2,alpha)
    changed_p1={n:v.copy() for n,v in p1.items()}
    removed={}
    for name,(st,dt) in TYPES.items():
        es,ed=edges[name].T
        mask=(es==endpoint[st])&(ed==endpoint[dt])&(st!=dt)
        removed[name]=mask
        if not mask.any():continue
        c=c1[name];v=endpoint[dt]
        degree=c['degree'][v:v+1]-int(mask.sum())
        mass=c['mass'][v:v+1]-c['g'][mask].sum()
        numerator=c['numerator'][v:v+1]-(c['g'][mask,None]*c['values'][es[mask]]).sum(0,keepdims=True)
        after=aggregate(degree,mass,numerator,alpha)
        changed_p1[dt][v]+=(after[0]-c['message'][v])/2
    changed_h1=activate(changed_p1)
    result={n:v.copy() for n,v in base.items()}
    event_count=0
    for name,(st,dt) in TYPES.items():
        es,ed=edges[name].T;p=params2[name];c=c2[name]
        incident=(es==endpoint[st])|(ed==endpoint[dt]);ix=np.flatnonzero(incident)
        event_count+=len(ix)
        gnew=gate(changed_h1[st][es[ix]],changed_h1[dt][ed[ix]],p)
        gnew[removed[name][ix]]=0
        vnew=changed_h1[st][es[ix]]@p['W']+p['bias']
        mass=c['mass'].copy();numerator=c['numerator'].copy();degree=c['degree'].copy()
        np.add.at(mass,ed[ix],gnew-c['g'][ix])
        np.add.at(numerator,ed[ix],gnew[:,None]*vnew-c['g'][ix,None]*c['values'][es[ix]])
        np.add.at(degree,ed[removed[name]],-1)
        message=aggregate(degree,mass,numerator,alpha)
        result[dt]+=(message-c['message']+(changed_h1[dt]-h1[dt])@p['R'])/2
    return result,changed_h1,event_count


def parameters(rng,width):
    return {name:dict(a=rng.normal(size=width),b=rng.normal(size=width),
        W=rng.normal(size=(width,5)),R=rng.normal(size=(width,5)),
        bias=rng.normal(size=5),root_bias=rng.normal(size=5)) for name in TYPES}


def main():
    # Repeated same-type self-loops; one repeated target pair; degree-one and empty targets.
    md=np.array([[0,0],[0,1],[1,0],[2,2],[2,2],[3,1]],dtype=int)
    edges=dict(mm=np.array([[0,0],[0,0],[1,1],[2,2],[3,3],[4,4],[0,1],[1,0],[2,4]]),
               dd=np.array([[0,0],[1,1],[2,2],[3,3],[2,2],[0,1],[1,0]]),md=md,dm=md[:,::-1])
    queries=[(0,0),(0,1),(1,0),(2,2),(3,1),(4,3),(1,3),(4,0)]
    count=0;maximum=0.;max_h1=0.;events=[]
    for seed in range(10):
        rng=np.random.default_rng(seed);x={n:rng.normal(size=(size,4)) for n,size in NODES.items()}
        p1,p2=parameters(rng,4),parameters(rng,5)
        for alpha in (0.,.5,1.):
            for r,d in queries:
                actual,h1,n=updated(x,edges,p1,p2,r,d,alpha)
                deleted=remove(edges,r,d);explicit_h1=activate(layer(x,deleted,p1,alpha)[0])
                explicit=layer(explicit_h1,deleted,p2,alpha)[0]
                for name in NODES:
                    np.testing.assert_allclose(h1[name],explicit_h1[name],atol=2e-11,rtol=2e-11)
                    np.testing.assert_allclose(actual[name],explicit[name],atol=2e-11,rtol=2e-11)
                    maximum=max(maximum,float(abs(actual[name]-explicit[name]).max()))
                    max_h1=max(max_h1,float(abs(h1[name]-explicit_h1[name]).max()))
                count+=1;events.append(n)
    result=dict(status='numpy_forward_verified',cases=count,max_layer1_error=max_h1,
        max_layer2_error=maximum,zero_remaining_degree=True,repeated_edges=True,nonedges=True,
        normalizations=[0.,.5,1.],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pytorch_model_implemented=False,autograd_verified=False,training_performed=False,
        original_validation_evaluated=False,test_evaluated=False)
    (HERE/'DERIVATION_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
