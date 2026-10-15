"""Diagnose information in signed/multistep channels, validation only."""
import json
import numpy as np
import torch
from scipy.special import expit,logit
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from modules import HERE,load_fold,build_graph,structure_data,sha256
from incrd_hg_m123.pair_channels import PairStruct
from incrd_hg_m123.config import ModelConfig


def metrics(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p))


def main():
    torch.set_num_threads(2)
    out=HERE/"probe"
    out.mkdir(exist_ok=True)
    data=load_fold("MiDrug",0)
    graph=build_graph(data,"MiDrug")
    ps=PairStruct(structure_data(graph),graph["miRNA","MiDrug","drug"].edge_index.T,ModelConfig())
    extra=np.load(HERE/"features.npz")
    names=extra["names"]
    X={}
    for part in ("train","val"):
        pairs=torch.as_tensor(data[part+"_pairs"].T)
        X[part]=torch.cat([ps.qis(*pairs),ps.ease(*pairs),torch.as_tensor(extra[part])],1).numpy()
    plan=dict(feature_sets=["baseline","ease","walk","all"],depths=[2,4,6],
              iterations=1800,early_stopping_rounds=150,learning_rate=.03,source_sha256=sha256(__file__),test_evaluated=False)
    (out/"plan.json").write_text(json.dumps(plan,indent=2))
    reference=HERE.parent/"midrug_internal_20261006/fullgraph_drop/drop0.2_s42_val.npz"
    b=np.load(reference)
    np.testing.assert_array_equal(b["pairs"],data["val_pairs"])
    base=b["probability"]
    records=[]
    for fs in plan["feature_sets"]:
        mask=np.r_[np.ones(21,dtype=bool),[fs=="all" or n.startswith(fs+"_") for n in names]]
        for depth in plan["depths"]:
            tag=f"{fs}_d{depth}"
            model=XGBClassifier(n_estimators=1800,max_depth=depth,min_child_weight=10,reg_lambda=10,
                learning_rate=.03,subsample=.85,colsample_bytree=.85,tree_method="hist",n_jobs=3,random_state=42,
                eval_metric="auc",early_stopping_rounds=150)
            model.fit(X["train"][:,mask],data["train_y"],eval_set=[(X["val"][:,mask],data["val_y"])],verbose=False)
            pred=model.predict_proba(X["val"][:,mask])[:,1]
            blends=[]
            for w in (0.,.25,.5,.75,1.):
                p=expit((1-w)*logit(np.clip(base,1e-6,1-1e-6))+w*logit(np.clip(pred,1e-6,1-1e-6)))
                blends.append(dict(weight=w,**metrics(data["val_y"],p)))
            r=dict(tag=tag,metrics=metrics(data["val_y"],pred),best_iteration=model.best_iteration,
                   blends=blends,test_evaluated=False)
            records.append(r)
            model.save_model(out/f"{tag}.json")
            np.savez_compressed(out/f"{tag}_val.npz",pairs=data["val_pairs"],y=data["val_y"],probability=pred)
            print(json.dumps(r),flush=True)
            (out/"results.json").write_text(json.dumps(records,indent=2))


if __name__=="__main__":main()
