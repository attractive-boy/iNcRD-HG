"""Use only the original frozen lncRNA auxiliary graph in the M3 prior."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit,logit
from xgboost import XGBClassifier
from modules import HERE,signed_ease,load_fold,build_graph,sha256
from probe import metrics
from incrd_hg_m123.data import structure_data
from incrd_hg_m123.config import ModelConfig
from incrd_hg_m123.pair_channels import PairStruct


def auxiliary_features(graph,train_pairs,train_y,queries,device="cuda:0"):
    graph=graph.to(device)
    def adjacency(edge):
        X=torch.zeros(graph[edge[0]].num_nodes,graph[edge[2]].num_nodes,dtype=torch.float64,device=device)
        ix=graph[edge].edge_index
        X.index_put_((ix[0],ix[1]),torch.ones(ix.shape[1],device=device,dtype=X.dtype),accumulate=True)
        return X
    L=adjacency(("lncRNA","LncDrug","drug"))
    C=adjacency(("lncRNA","LncMi","miRNA")).T
    A=torch.zeros(graph["miRNA"].num_nodes,graph["drug"].num_nodes,dtype=L.dtype,device=device)
    p=torch.as_tensor(train_pairs[train_y==1],device=device)
    A[p[:,0],p[:,1]]=1
    r,d=torch.as_tensor(queries.T,device=device)
    def expr(x):
        x=x.double()
        return F.normalize(x-x.mean(1,keepdim=True),dim=1)
    corr=expr(graph["miRNA"].x) @ expr(graph["lncRNA"].x).T
    candidates={"original":C}
    for name,S in (("abs",corr.abs()),("pos",corr.clamp(min=0))):
        for k in (5,15,40,100):
            v,ix=S.topk(k,dim=1)
            candidates[f"{name}{k}"]=torch.zeros_like(S).scatter(1,ix,v)
    columns,names=[],[]
    def add(name,v):
        columns.append(torch.sign(v)*torch.log1p(v.abs()*10))
        names.append(name)
    for name,S in candidates.items():
        R=S/S.sum(1,keepdim=True).clamp(min=1e-8)
        add(f"aux_bridge_{name}",(R@L)[r,d])
        add(f"aux_bridge_ra_{name}",(R@(L/L.sum(1,keepdim=True).clamp(min=1)))[r,d])
        # Transfer a denoised lncRNA drug profile through the fixed cross-RNA relation.
        for lam in (1.,10.,50.):
            P=torch.linalg.inv(L.T@L+lam*torch.eye(L.shape[1],device=device,dtype=L.dtype))
            B=-P/P.diagonal()[None,:]
            B.fill_diagonal_(0)
            add(f"aux_transfer_{name}_{lam}",(R@L@B)[r,d])
    # Drug-side MC-EASE pools covariance from the same-dataset auxiliary task.
    for weight in (.1,1.,10.):
        for lam in (1.,10.,50.):
            P=torch.linalg.inv(A.T@A+weight*(L.T@L)+lam*torch.eye(A.shape[1],device=device,dtype=L.dtype))
            add(f"aux_ease_{weight}_{lam}",signed_ease(A,P,r,d))
    return torch.stack(columns,1).float(),names


def main():
    torch.set_num_threads(2)
    data=load_fold("MiDrug",0)
    graph=build_graph(data,"MiDrug")
    n=len(data["train_y"])
    queries=np.concatenate([data["train_pairs"],data["val_pairs"]])
    X,names=auxiliary_features(graph,data["train_pairs"],data["train_y"],queries)
    errors=[]
    for ix in np.flatnonzero(data["train_y"]==1)[[0,117,808]]:
        keep=np.arange(n)!=ix
        brute,_=auxiliary_features(graph,data["train_pairs"][keep],data["train_y"][keep],queries[ix:ix+1])
        torch.testing.assert_close(X[ix:ix+1],brute,rtol=2e-5,atol=2e-5)
        errors.append(float((X[ix:ix+1]-brute).abs().max()))
    np.savez_compressed(HERE/"aux_features.npz",train=X[:n].cpu().numpy(),val=X[n:].cpu().numpy(),names=names)
    checks=dict(status="passed",source_sha256=sha256(__file__),features_sha256=sha256(HERE/"aux_features.npz"),
                deletion_max_errors=errors,feature_count=len(names),source="Frozen MiDrug graph only",test_evaluated=False)
    (HERE/"aux_checks.json").write_text(json.dumps(checks,indent=2))
    print("FEATURES",checks,flush=True)
    g=graph.to("cpu")
    ps=PairStruct(structure_data(g),g["miRNA","MiDrug","drug"].edge_index.T,ModelConfig())
    pairs=torch.as_tensor(queries.T)
    baseX=torch.cat([ps.qis(*pairs),ps.ease(*pairs)],1).numpy()
    SX=np.concatenate([baseX,X.cpu().numpy()],1)
    out=HERE/"aux_probe"
    out.mkdir(exist_ok=True)
    (out/"plan.json").write_text(json.dumps(dict(source_sha256=sha256(__file__),depths=[2,4,6],test_evaluated=False),indent=2))
    reference=HERE.parent/"midrug_internal_20261006/fullgraph_drop/drop0.2_s42_val.npz"
    base=np.load(reference)["probability"]
    records=[]
    for depth in (2,4,6):
        model=XGBClassifier(n_estimators=1800,max_depth=depth,min_child_weight=10,reg_lambda=10,learning_rate=.03,
            subsample=.85,colsample_bytree=.85,tree_method="hist",n_jobs=3,random_state=42,eval_metric="auc",early_stopping_rounds=150)
        model.fit(SX[:n],data["train_y"],eval_set=[(SX[n:],data["val_y"])],verbose=False)
        p=model.predict_proba(SX[n:])[:,1]
        blends=[]
        for w in (0.,.25,.5,.75,1.):
            pp=expit((1-w)*logit(np.clip(base,1e-6,1-1e-6))+w*logit(np.clip(p,1e-6,1-1e-6)))
            blends.append(dict(weight=w,**metrics(data["val_y"],pp)))
        rec=dict(depth=depth,metrics=metrics(data["val_y"],p),blends=blends,best_iteration=model.best_iteration,test_evaluated=False)
        records.append(rec)
        model.save_model(out/f"d{depth}.json")
        np.savez_compressed(out/f"d{depth}_val.npz",pairs=data["val_pairs"],y=data["val_y"],probability=p)
        (out/"results.json").write_text(json.dumps(records,indent=2))
        print(json.dumps(rec),flush=True)


if __name__=="__main__":main()
