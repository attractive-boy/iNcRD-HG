"""Prepare training/validation evidence and verify explicit-query deletion."""
import json
import numpy as np
import torch
from modules import HERE, RELEASE, Evidence, DegreePairStruct, signed_ease, masked_walks, load_fold, build_graph, structure_data, sha256
from incrd_hg_m123.config import ModelConfig


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = "cuda:0"
    # Small graphs also cover negative entries and zero-degree nodes.
    X = torch.randint(0,4,(9,7),device=device).double()
    X = torch.where(X==1,torch.ones_like(X),torch.where(X==2,-.3*torch.ones_like(X),torch.zeros_like(X)))
    X[0]=0
    pairs = torch.cartesian_prod(torch.arange(9,device=device),torch.arange(7,device=device))
    P = torch.linalg.inv(X.T@X+2*torch.eye(7,device=device,dtype=X.dtype))
    fast_ease = signed_ease(X,P,*pairs.T)
    fast_walk = masked_walks(X,pairs,.5,1.)
    err_ease,err_walk = 0.,0.
    for j,(r,d) in enumerate(pairs.tolist()):
        Z = X.clone()
        Z[r,d]=0
        Q = torch.linalg.inv(Z.T@Z+2*torch.eye(7,device=device,dtype=X.dtype))
        col = -Q[:,d]/Q[d,d]
        col[d]=0
        brute = Z[r]@col
        err_ease=max(err_ease,float(abs(brute-fast_ease[j])))
        # Independent full-matrix walk multiplication on the deleted graph.
        R = torch.diag(Z.abs().sum(1).clamp(min=1).pow(-.5))
        D = torch.diag(Z.abs().sum(0).clamp(min=1).pow(-1))
        W = Z.clone()
        for k in range(3):
            W = W@D@Z.T@R@Z
            err_walk=max(err_walk,float(abs(W[r,d]-fast_walk[j,k])))
    assert err_ease<1e-9 and err_walk<1e-9
    data=load_fold("MiDrug",0)
    inputs={k:data[k] for k in ("train_pairs","train_y","val_pairs")}
    graph=build_graph(inputs,"MiDrug").to(device)
    evidence=Evidence(inputs["train_pairs"],inputs["train_y"],
                      (graph["miRNA"].num_nodes,graph["drug"].num_nodes),device)
    queries=np.concatenate([inputs["train_pairs"],inputs["val_pairs"]])
    features,names=evidence.features(queries)
    assert torch.isfinite(features).all()
    n=len(inputs["train_y"])
    selected=np.r_[np.flatnonzero(inputs["train_y"]==1)[[0,103,717]],np.flatnonzero(inputs["train_y"]==0)[[0,117,912]]]
    deletion=[]
    for i in selected:
        keep=np.arange(n)!=i
        ev=Evidence(inputs["train_pairs"][keep],inputs["train_y"][keep],evidence.A.shape,device)
        brute,names2=ev.features(queries[i:i+1])
        assert names==names2
        torch.testing.assert_close(features[i:i+1],brute,atol=2e-5,rtol=2e-5)
        deletion.append(float((features[i:i+1]-brute).abs().max()))
    np.savez_compressed(HERE/"features.npz",train=features[:n].cpu().numpy(),val=features[n:].cpu().numpy(),
                        names=names,train_pairs=inputs["train_pairs"],val_pairs=inputs["val_pairs"])
    report=dict(status="passed",synthetic_signed_ease_max_error=err_ease,synthetic_masked_walk_max_error=err_walk,
                real_query_deletion_max_errors=deletion,feature_count=len(names),
                features_sha256=sha256(HERE/"features.npz"),source_sha256=sha256(HERE/"modules.py"),test_features_constructed=False)
    (HERE/"feature_checks.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
