"""Full original graph, M1/M2/M3 evidence variants; validation-only selection."""
import argparse
import json
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from modules import HERE, RELEASE, DegreePairStruct, load_fold, build_graph, sha256
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.engine import seed_all
from sklearn.metrics import roc_auc_score,average_precision_score

CONFIGS={
    "baseline":dict(power=0.,ease="none",walk="none"),
    "degree05":dict(power=.5,ease="none",walk="none"),
    "degree1":dict(power=1.,ease="none",walk="none"),
    "signed01":dict(power=0.,ease="signed01",walk="none"),
    "signed03":dict(power=0.,ease="signed03",walk="none"),
    "signed1":dict(power=0.,ease="signed1",walk="none"),
    "signedall":dict(power=0.,ease="all",walk="none"),
    "walk_positive":dict(power=0.,ease="none",walk="positive"),
    "walk_signed":dict(power=0.,ease="none",walk="signed03"),
    "all":dict(power=.5,ease="all",walk="all"),
}


def extend(linear,n):
    if n==0:return linear
    new=nn.Linear(linear.in_features+n,linear.out_features).to(linear.weight.device)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:,:linear.in_features].copy_(linear.weight)
        new.bias.copy_(linear.bias)
    return new


def make_adapter(data,config,device):
    inputs={k:data[k] for k in ("train_pairs","train_y","val_pairs")}
    inputs["test_pairs"]=np.empty((0,2),dtype=np.int64)
    adapter=Adapter(inputs,build_graph(inputs,"MiDrug"),device)
    features=np.load(HERE/"features.npz")
    checks=json.loads((HERE/"feature_checks.json").read_text())
    assert sha256(HERE/"features.npz")==checks["features_sha256"]
    assert sha256(HERE/"modules.py")==checks["source_sha256"]
    for part in ("train","val"):
        np.testing.assert_array_equal(features[part+"_pairs"],inputs[part+"_pairs"])
    names=features["names"]
    def select(family,choice):
        return np.array([choice!="none" and n.startswith(family+"_") and
                         (choice=="all" or n.startswith(family+"_"+choice+"_")) for n in names])
    emask,wmask=select("ease",config["ease"]),select("walk",config["walk"])
    extra={p:torch.as_tensor(features[p],device=device) for p in ("train","val")}
    train_w=extra["train"][:,wmask]
    mu,sd=train_w.mean(0),train_w.std(0)+1e-6
    for part in ("train","val"):
        adapter.qis[part]=torch.cat([adapter.qis[part],(extra[part][:,wmask]-mu)/sd],1)
        adapter.ease[part]=torch.cat([adapter.ease[part],extra[part][:,emask]],1)
    model=adapter.model
    model.pc[0]=extend(model.pc[0],int(wmask.sum()))
    model.em[0]=extend(model.em[0],int(emask.sum()))
    if emask.any():
        old=model.eb
        new=nn.BatchNorm1d(old.num_features+int(emask.sum())).to(device)
        with torch.no_grad():
            for name in ("weight","bias","running_mean","running_var"):
                getattr(new,name)[:old.num_features].copy_(getattr(old,name))
            new.num_batches_tracked.copy_(old.num_batches_tracked)
        model.eb=new
    adapter.ps=DegreePairStruct(adapter.D,adapter.graph["miRNA","MiDrug","drug"].edge_index.T,
                                 adapter.cfg,power=config["power"])
    adapter.extra_masks=(emask,wmask)
    adapter.extra_norm=(mu,sd)
    return adapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",choices=CONFIGS,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    seed_all(args.seed)
    out=HERE/"results"
    out.mkdir(exist_ok=True)
    tag=f"{args.config}_s{args.seed}"
    if (out/f"{tag}.json").exists():raise FileExistsError(tag)
    data=load_fold("MiDrug",0)
    adapter=make_adapter(data,CONFIGS[args.config],args.device)
    model=adapter.model
    opt=torch.optim.RMSprop(model.parameters(),lr=1e-4,alpha=.99)
    best,best_ep,state,best_p=-1,0,None,None
    curves=[]
    start=time.time()
    for ep in range(1,601):
        model.train();opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(adapter.logits("train"),adapter.y)
        assert torch.isfinite(loss)
        loss.backward();opt.step()
        model.eval()
        with torch.no_grad():prob=adapter.predict("val")
        auc=roc_auc_score(data["val_y"],prob)
        ap=average_precision_score(data["val_y"],prob)
        curves.append(dict(epoch=ep,loss=float(loss),val_auc=auc,val_ap=ap))
        if auc>best:
            best,best_ep,best_p=auc,ep,prob.copy()
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-start,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f"{tag}.pt")
    np.savez_compressed(out/f"{tag}_val.npz",pairs=data["val_pairs"],y=data["val_y"],probability=best_p)
    (out/f"{tag}_curves.json").write_text(json.dumps(curves,indent=2))
    result=dict(tag=tag,config=CONFIGS[args.config],seed=args.seed,val_auc=best,
                val_ap=average_precision_score(data["val_y"],best_p),best_epoch=best_ep,epochs_run=ep,
                checkpoint_sha256=sha256(out/f"{tag}.pt"),source_sha256=sha256(__file__),
                modules_sha256=sha256(HERE/"modules.py"),test_evaluated=False,elapsed_seconds=time.time()-start)
    (out/f"{tag}.json").write_text(json.dumps(result,indent=2))
    print("DONE",json.dumps(result),flush=True)


if __name__=="__main__":main()
