"""Structural incident-edge events, cached by query VALUES on an immutable graph."""
import numpy as np
import torch


class EventIndex:
    def __init__(self,graph):
        self.graph=graph;self.cpu_edges={};self.incident={};self.cache={}
        for e in graph.edge_types:
            index=graph[e].edge_index.detach().cpu().numpy();self.cpu_edges[e]=index
            source=[[] for _ in range(graph[e[0]].num_nodes)]
            target=[[] for _ in range(graph[e[2]].num_nodes)]
            for j,(s,d) in enumerate(index.T):source[int(s)].append(j);target[int(d)].append(j)
            self.incident[e]=(source,target)

    def get(self,r,d,q):
        rr,dd=r.detach().cpu().numpy(),d.detach().cpu().numpy()
        key=(rr.tobytes(),dd.tobytes())
        if key in self.cache:return self.cache[key]
        positive=np.flatnonzero(q.detach().cpu().numpy()>0)
        pairs={'miRNA':rr,'drug':dd};result={}
        for e in self.graph.edge_types:
            source,target=self.incident[e];ss,tt=self.cpu_edges[e]
            qs=[];ids=[]
            for query in positive:
                s,t=int(pairs[e[0]][query]),int(pairs[e[2]][query])
                ix=np.union1d(source[s],target[t]).astype(np.int64)
                ids.extend(ix.tolist());qs.extend([int(query)]*len(ix))
            qs=np.asarray(qs,dtype=np.int64);ids=np.asarray(ids,dtype=np.int64)
            si,ti=ss[ids],tt[ids]
            src_query=si==pairs[e[0]][qs];dst_query=ti==pairs[e[2]][qs]
            grouping=np.stack([qs,ti],axis=1)
            groups,inverse=np.unique(grouping,axis=0,return_inverse=True)
            arrays=dict(query=qs,edge=ids,source=si,target=ti,group=inverse,
                group_query=groups[:,0],group_target=groups[:,1],
                source_is_query=src_query,target_is_query=dst_query,
                removed=src_query&dst_query&(e[0]!=e[2]))
            result[e]={name:torch.as_tensor(value.copy(),device=r.device,
                           dtype=torch.bool if value.dtype==bool else torch.long) for name,value in arrays.items()}
        self.cache[key]=result
        return result
