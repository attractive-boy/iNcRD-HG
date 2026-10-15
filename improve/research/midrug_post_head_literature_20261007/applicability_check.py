"""No-fit structural applicability checks and a fixed-kernel counterexample."""
import importlib.util,json,itertools
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'midrug_subgraph_20261006/common.py'
spec=importlib.util.spec_from_file_location('strict_applicability',SOURCE);strict=importlib.util.module_from_spec(spec);spec.loader.exec_module(strict)

def fixed_kernel_example():
    # K3,3 and triangular prism: distinct connected 3-regular graphs on six nodes.
    a=np.zeros((6,6));b=np.zeros_like(a)
    for i in range(3):
        for j in range(3,6):a[i,j]=a[j,i]=1
    for i,j in [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]:b[i,j]=b[j,i]=1
    assert np.array_equal(a.sum(1),b.sum(1)) and np.trace(a@a@a)!=np.trace(b@b@b)
    rng=np.random.default_rng(42);x=np.tile(rng.normal(size=(1,3)),(6,1));records=[]
    for size in [2,3,4]:
        h=rng.uniform(size=(size,size));h=(h+h.T)/2;z=rng.normal(size=(size,3));s=(z@x.T).reshape(-1,order='F')
        for length in range(1,5):
            ka=float(s@np.linalg.matrix_power(np.kron(a,h),length)@s)
            kb=float(s@np.linalg.matrix_power(np.kron(b,h),length)@s)
            np.testing.assert_allclose(ka,kb,atol=1e-8,rtol=1e-12)
            records.append(dict(size=size,length=length,first=ka,second=kb,error=abs(ka-kb)))
    return dict(status='passed',records=records,triangle_counts=[float(np.trace(c@c@c)/6) for c in [a,b]],
                scope='Fixed input subgraphs with constant node attributes only; not a counterexample to every extraction policy or the full MoSE model.')

def main():
    records=[]
    for fold in range(3):
        data=strict.load_inner(fold);g=strict.make_graph(data);nm=g['miRNA'].num_nodes;nd=g['drug'].num_nodes;n=nm+nd
        assert set(g.node_types)=={'miRNA','drug'}
        offset={'miRNA':0,'drug':nm};relations={}
        for src,rel,dst in g.edge_types:
            e=g[src,rel,dst].edge_index.numpy();a=np.zeros((n,n),dtype=np.int64)
            a[e[0]+offset[src],e[1]+offset[dst]]=1;relations[rel]=a
        names=list(relations);nr=np.stack([a.any(1) for a in relations.values()],1)
        signatures,counts=np.unique(nr.astype(np.int8),axis=0,return_counts=True)
        sets=[dict(relations=[name for name,present in zip(names,row) if present],count=int(count)) for row,count in zip(signatures,counts)]
        target=relations['MiDrug'];tj=names.index('MiDrug');context=[]
        # Formula (2)/(3), outgoing relation presence, no threshold tuning.
        for j,name in enumerate(names):
            if j==tj:continue
            both=(nr[:,j]&nr[:,tj]).sum();den=nr[:,j].sum()
            context.append(dict(neighbor_relation=name,precision=float(both/den) if den else None,recall=float(both/nr[:,tj].sum())))
        rules=[]
        for r1,a in relations.items():
            for r2,b in relations.items():
                s=(a@b)[:nm,nm:]
                if not s.any():continue
                den=int(s.sum());num=int((s*target[:nm,nm:]).sum())
                rules.append(dict(relations=[r1,r2],two_hop_events=den,positive_closed_events=num,empirical_ratio=num/den))
        # Support overlap across different declared relation labels is absent here.
        overlap=int(sum(np.count_nonzero(relations[x]&relations[y]) for x,y in itertools.combinations(names,2)))
        records.append(dict(fold=fold,relation_names=names,nodes=n,outgoing_relation_signatures=sets,signature_count=len(sets),
            nontrivial_same_pair_relation_overlap=overlap,context_singleton_scores=context,target_two_hop_compositions=rules,
            same_type_symmetry={name:bool(np.array_equal(relations[name],relations[name].T)) for name in ['MiMi','DrugDrug']}))
    out=dict(status='passed',records=records,fixed_rwk_counterexample=fixed_kernel_example(),no_fits=True,no_validation_metrics=True,no_test_metrics=True,
        no_lncrna=True,source_hashes={str(p):strict.sha256(p) for p in [SOURCE,HERE/'applicability_check.py']},
        graph_sha256=strict.sha256(strict.CLEAN/'graph.pt'),data_sha256=strict.sha256(strict.CLEAN/'inner_data.npz'))
    (HERE/'applicability_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
