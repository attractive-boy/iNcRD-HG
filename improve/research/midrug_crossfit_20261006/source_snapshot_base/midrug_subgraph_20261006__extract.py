"""Query-deleted enclosing bipartite subgraphs from original training labels."""
import argparse,json,time
import numpy as np
from common import HERE,CLEAN,load_inner,sha256


class Extractor:
    def __init__(self,data):
        self.A=np.zeros((605,216),dtype=np.int32);self.N=np.zeros_like(self.A)
        p=data['train_pairs'][data['train_y']==1];n=data['train_pairs'][data['train_y']==0]
        self.A[p[:,0],p[:,1]]=1;self.N[n[:,0],n[:,1]]=1
        self.dr,self.dd=self.A.sum(1),self.A.sum(0)
        self.Cr,self.Cd=self.A@self.A.T,self.A.T@self.A

    @staticmethod
    def choose(first,second,score,degree,root,cap,hops):
        def rank(nodes):
            nodes=np.asarray(sorted(set(nodes)-{root}),dtype=np.int64)
            if not len(nodes):return nodes
            # Stable covariate-free ranking on the graph with query removed.
            value=score[nodes]/np.sqrt(np.maximum(degree[nodes],1))
            return nodes[np.lexsort((nodes,-value))]
        first=rank(first)
        if hops==1:return first[:cap]
        chosen=first[:(cap+1)//2]
        rest=rank(set(second)-set(chosen))
        chosen=np.r_[chosen,rest[:cap-len(chosen)]]
        if len(chosen)<cap:
            rest=rank(set(first)-set(chosen));chosen=np.r_[chosen,rest[:cap-len(chosen)]]
        return chosen

    def one(self,r,d,k,hops):
        A,N=self.A,self.N;q=A[r,d]
        ar=A[r].copy();ad=A[:,d].copy();ar[d]=0;ad[r]=0
        # Exact co-count downdate for candidates other than the query roots.
        sr=self.Cr[r]-q*A[:,d];sd=self.Cd[d]-q*A[r]
        sr[r]=0;sd[d]=0
        ri=self.choose(np.flatnonzero(ad),np.flatnonzero(sr),sr,self.dr,r,k-1,hops)
        di=self.choose(np.flatnonzero(ar),np.flatnonzero(sd),sd,self.dd,d,k-1,hops)
        ri=np.r_[r,ri];di=np.r_[d,di]
        rid=np.full(k,605,dtype=np.int32);did=np.full(k,216,dtype=np.int32)
        rid[:len(ri)]=ri;did[:len(di)]=di
        pos=np.zeros((k,k),dtype=np.uint8);neg=np.zeros_like(pos)
        pos[:len(ri),:len(di)]=A[np.ix_(ri,di)];neg[:len(ri),:len(di)]=N[np.ix_(ri,di)]
        pos[0,0]=0;neg[0,0]=0
        masks=np.r_[rid<605,did<216]
        edge=np.block([[np.zeros_like(pos),pos],[pos.T,np.zeros_like(pos)]])
        def distance(root):
            dist=np.full(2*k,6,dtype=np.int32);dist[root]=0;front=np.zeros(2*k,dtype=bool);front[root]=True
            for level in range(1,6):
                reach=(edge[front].sum(0)>0)&(dist==6)&masks
                dist[reach]=level;front=reach
                if not front.any():break
            return dist
        dr,dd=distance(0),distance(k)
        onehot=np.eye(7,dtype=np.float32)
        role=np.zeros((2*k,2),dtype=np.float32);role[0,0]=1;role[k,1]=1
        types=np.zeros((2*k,2),dtype=np.float32);types[:k,0]=1;types[k:,1]=1
        global_degree=np.zeros(2*k,dtype=np.float32)
        global_degree[:len(ri)]=self.dr[ri];global_degree[k:k+len(di)]=self.dd[di]
        global_degree[0]-=q;global_degree[k]-=q
        feat=np.c_[onehot[dr],onehot[dd],role,types,np.log1p(global_degree),np.log1p(edge.sum(1))].astype(np.float32)
        feat[~masks]=0
        return dict(rid=rid,did=did,pos=pos,neg=neg,features=feat)

    def many(self,pairs,k,hops):
        rows=[self.one(int(r),int(d),k,hops) for r,d in pairs]
        return {key:np.stack([row[key] for row in rows]) for key in rows[0]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',type=int,choices=range(3),required=True)
    args=p.parse_args();data=load_inner(args.fold);ext=Extractor(data);started=time.time()
    out=HERE/'cache';out.mkdir(exist_ok=True)
    for k,hops in ((16,1),(32,2)):
        path=out/f'f{args.fold}_h{hops}_k{k}.npz'
        if path.exists():raise FileExistsError(path)
        arrays={}
        for part in ('train','val'):
            for key,value in ext.many(data[part+'_pairs'],k,hops).items():arrays[part+'_'+key]=value
        np.savez_compressed(path,**arrays)
        rec=dict(fold=args.fold,k=k,hops=hops,source_sha256=sha256(__file__),cache_sha256=sha256(path),
            clean_graph_sha256=sha256(CLEAN/'graph.pt'),inner_data_sha256=sha256(CLEAN/'inner_data.npz'),
            lncrna_used=False,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-started)
        path.with_suffix('.json').write_text(json.dumps(rec,indent=2));print('DONE',path.name,time.time()-started,flush=True)

if __name__=='__main__':main()
