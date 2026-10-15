"""M123 on original covariates with alternative intrinsic drug-similarity graphs."""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from model import HERE,load_fold,build_graph,seed_all,sha256,Adapter
from train import anchor_metrics
from sklearn.metrics import roc_auc_score,average_precision_score


def make_chemical_adapter(data,kernel,k,device):
    inputs={key:data[key] for key in ("train_pairs","train_y","val_pairs")}
    inputs["test_pairs"]=np.empty((0,2),dtype=np.int64)
    g=build_graph(inputs,"MiDrug")
    if kernel!="original":
        info=json.loads((HERE/"chemical_provenance.json").read_text())
        assert info["kernel_sha256"]==sha256(HERE/"chemical_kernels.npz")
        K=np.load(HERE/"chemical_kernels.npz")[kernel].copy()
        np.fill_diagonal(K,-1)
        ix=np.argsort(-K,axis=1,kind="stable")[:,:k]
        src=np.repeat(np.arange(len(K)),k);dst=ix.ravel()
        edges=np.unique(np.concatenate([np.stack([src,dst]),np.stack([dst,src])],axis=1),axis=1)
        old=g["drug","DrugDrug","drug"].edge_index.numpy()
        loops=old[:,old[0]==old[1]]
        g["drug","DrugDrug","drug"].edge_index=torch.as_tensor(np.concatenate([edges,loops],axis=1),dtype=torch.long)
    return Adapter(inputs,g,device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel",choices=["original","morgan2","morgan3","maccs","pharm2"],required=True)
    p.add_argument("--k",type=int,choices=[3,10],default=3)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args();seed_all(42)
    out=HERE/"chemical_results";out.mkdir(exist_ok=True)
    tag=f"{args.kernel}_k{args.k}"
    if (out/f"{tag}.json").exists():raise FileExistsError(tag)
    data=load_fold("MiDrug",0)
    ba,bp=anchor_metrics(data)
    adapter=make_chemical_adapter(data,args.kernel,args.k,args.device)
    model=adapter.model
    opt=torch.optim.RMSprop(model.parameters(),lr=1e-4,alpha=.99)
    best,best_ep,state,best_p=-1,0,None,None
    curves=[];started=time.time()
    for ep in range(1,701):
        model.train();opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(adapter.logits("train"),adapter.y)
        assert torch.isfinite(loss)
        loss.backward();opt.step();model.eval()
        with torch.no_grad():pred=adapter.predict("val")
        auc,ap=roc_auc_score(data["val_y"],pred),average_precision_score(data["val_y"],pred)
        score=min(auc-ba,ap-bp)
        curves.append(dict(epoch=ep,loss=float(loss),val_auc=auc,val_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,pred.copy()
            state={key:v.detach().cpu().clone() for key,v in model.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-started,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f"{tag}.pt")
    np.savez_compressed(out/f"{tag}_val.npz",pairs=data["val_pairs"],y=data["val_y"],probability=best_p)
    (out/f"{tag}_curves.json").write_text(json.dumps(curves,indent=2))
    r=dict(tag=tag,kernel=args.kernel,k=args.k,val_auc=roc_auc_score(data["val_y"],best_p),
           val_ap=average_precision_score(data["val_y"],best_p),best_epoch=best_ep,epochs_run=ep,selection_score=best,
           source_sha256=sha256(__file__),checkpoint_sha256=sha256(out/f"{tag}.pt"),
           chemical_sha256=sha256(HERE/"chemical_kernels.npz"),test_evaluated=False,elapsed_seconds=time.time()-started)
    (out/f"{tag}.json").write_text(json.dumps(r,indent=2));print("DONE",json.dumps(r),flush=True)


if __name__=="__main__":main()
