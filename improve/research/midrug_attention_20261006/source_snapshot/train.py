"""Validation-only, both-metric selection of M2 path-attention variants."""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from model import HERE,CONFIGS,make_adapter,load_fold,seed_all,sha256


def anchor_metrics(data):
    values=[]
    for seed in (42,43,44):
        p=np.load(HERE.parent/"mirna_baselines_20260928/results"/f"incrd_original_MiDrug_f0_s{seed}_predictions.npz")
        np.testing.assert_array_equal(p["val_pairs"],data["val_pairs"])
        np.testing.assert_array_equal(p["val_y"],data["val_y"])
        values.append((roc_auc_score(data["val_y"],p["val_probability"]),average_precision_score(data["val_y"],p["val_probability"])))
    return np.mean(values,axis=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",choices=CONFIGS,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    seed_all(args.seed)
    out=HERE/"results";out.mkdir(exist_ok=True)
    tag=f"{args.config}_s{args.seed}"
    if (out/f"{tag}.json").exists():raise FileExistsError(tag)
    data=load_fold("MiDrug",0)
    anchor_auc,anchor_ap=anchor_metrics(data)
    cfg=CONFIGS[args.config]
    adapter=make_adapter(data,cfg,args.device)
    model=adapter.model
    opt=torch.optim.RMSprop(model.parameters(),lr=1e-4,alpha=.99)
    best,best_ep,state,best_p=-1,0,None,None
    curves=[];started=time.time()
    for ep in range(1,701):
        model.train();opt.zero_grad()
        logits=adapter.logits("train")
        bce=F.binary_cross_entropy_with_logits(logits,adapter.y)
        loss=bce+cfg["reg"]*model.attention_penalty
        assert torch.isfinite(loss)
        loss.backward();opt.step()
        model.eval()
        with torch.no_grad():pred=adapter.predict("val")
        auc,ap=roc_auc_score(data["val_y"],pred),average_precision_score(data["val_y"],pred)
        score=min(auc-anchor_auc,ap-anchor_ap)
        curves.append(dict(epoch=ep,loss=float(loss),val_auc=auc,val_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,pred.copy()
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-started,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f"{tag}.pt")
    np.savez_compressed(out/f"{tag}_val.npz",pairs=data["val_pairs"],y=data["val_y"],probability=best_p)
    (out/f"{tag}_curves.json").write_text(json.dumps(curves,indent=2))
    result=dict(tag=tag,config=cfg,seed=args.seed,val_auc=roc_auc_score(data["val_y"],best_p),
                val_ap=average_precision_score(data["val_y"],best_p),selection_score=best,best_epoch=best_ep,epochs_run=ep,
                checkpoint_sha256=sha256(out/f"{tag}.pt"),source_sha256=sha256(__file__),model_sha256=sha256(HERE/"model.py"),
                selection="min(validation AUC gain, validation AP gain)",test_evaluated=False,elapsed_seconds=time.time()-started)
    (out/f"{tag}.json").write_text(json.dumps(result,indent=2));print("DONE",json.dumps(result),flush=True)


if __name__=="__main__":main()
