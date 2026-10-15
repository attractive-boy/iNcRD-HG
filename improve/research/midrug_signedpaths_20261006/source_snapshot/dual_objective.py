"""Train M123 using global AUC/AP surrogates, select by both validation gains."""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from train import HERE,make_adapter,CONFIGS,load_fold,seed_all,sha256
from sklearn.metrics import roc_auc_score,average_precision_score

OBJECTIVES={
    "bce":dict(auc=0.,ap=0.,tau=1.),
    "auc03":dict(auc=.3,ap=0.,tau=1.),
    "ap03":dict(auc=0.,ap=.3,tau=1.),
    "ap10":dict(auc=0.,ap=1.,tau=1.),
    "dual03":dict(auc=.3,ap=.3,tau=1.),
    "ap03_t02":dict(auc=0.,ap=.3,tau=.2),
}


def ranking(logits,labels,cfg):
    pos,neg=logits[labels==1],logits[labels==0]
    auc,ap=logits.new_zeros(()),logits.new_zeros(())
    if not cfg["auc"] and not cfg["ap"]:
        return auc,ap
    for anchor in pos.split(512):
        delta=(neg[None,:]-anchor[:,None])/cfg["tau"]
        fraction=len(anchor)/len(pos)
        if cfg["auc"]:
            auc=auc+fraction*F.softplus(delta).mean()
        if cfg["ap"]:
            higher_positive=torch.sigmoid((pos[None,:]-anchor[:,None])/cfg["tau"]).sum(1)
            higher_negative=torch.sigmoid(delta).sum(1)
            ap=ap-fraction*torch.log(higher_positive/(higher_positive+higher_negative)).mean()
    return auc,ap


def anchor_metrics(data):
    source=HERE.parent/"mirna_baselines_20260928/results"
    vals=[]
    for seed in (42,43,44):
        f=source/f"incrd_original_MiDrug_f0_s{seed}_predictions.npz"
        p=np.load(f)
        np.testing.assert_array_equal(p["val_pairs"],data["val_pairs"])
        np.testing.assert_array_equal(p["val_y"],data["val_y"])
        vals.append((roc_auc_score(data["val_y"],p["val_probability"]),average_precision_score(data["val_y"],p["val_probability"])))
    return np.mean(vals,axis=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",choices=OBJECTIVES,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    seed_all(args.seed)
    out=HERE/"dual_objective"
    out.mkdir(exist_ok=True)
    tag=f"{args.config}_s{args.seed}"
    if (out/f"{tag}.json").exists():raise FileExistsError(tag)
    data=load_fold("MiDrug",0)
    baseline_auc,baseline_ap=anchor_metrics(data)
    adapter=make_adapter(data,CONFIGS["degree05"],args.device)
    model=adapter.model
    cfg=OBJECTIVES[args.config]
    opt=torch.optim.RMSprop(model.parameters(),lr=1e-4,alpha=.99)
    best,best_ep,state,best_p=-1,0,None,None
    curves=[]
    start=time.time()
    for ep in range(1,601):
        model.train();opt.zero_grad()
        logits=adapter.logits("train")
        bce=F.binary_cross_entropy_with_logits(logits,adapter.y)
        ramp=min(1.,max(0.,(ep-30)/30))
        ra,rp=ranking(logits,adapter.y,cfg) if ramp else (logits.new_zeros(()),logits.new_zeros(()))
        loss=bce+ramp*(cfg["auc"]*ra+cfg["ap"]*rp)
        assert torch.isfinite(loss)
        loss.backward();opt.step()
        model.eval()
        with torch.no_grad():pred=adapter.predict("val")
        auc,ap=roc_auc_score(data["val_y"],pred),average_precision_score(data["val_y"],pred)
        score=min(auc-baseline_auc,ap-baseline_ap)
        curves.append(dict(epoch=ep,loss=float(loss),bce=float(bce),auc_loss=float(ra),ap_loss=float(rp),val_auc=auc,val_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,pred.copy()
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-start,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f"{tag}.pt")
    np.savez_compressed(out/f"{tag}_val.npz",pairs=data["val_pairs"],y=data["val_y"],probability=best_p)
    (out/f"{tag}_curves.json").write_text(json.dumps(curves,indent=2))
    result=dict(tag=tag,objective=cfg,model_config=CONFIGS["degree05"],seed=args.seed,
                val_auc=roc_auc_score(data["val_y"],best_p),val_ap=average_precision_score(data["val_y"],best_p),
                validation_anchor=dict(auc=baseline_auc,ap=baseline_ap),selection="min(validation AUC gain, validation AP gain)",
                selection_score=best,best_epoch=best_ep,epochs_run=ep,source_sha256=sha256(__file__),
                checkpoint_sha256=sha256(out/f"{tag}.pt"),test_evaluated=False,elapsed_seconds=time.time()-start)
    (out/f"{tag}.json").write_text(json.dumps(result,indent=2))
    print("DONE",json.dumps(result),flush=True)


if __name__=="__main__":main()
