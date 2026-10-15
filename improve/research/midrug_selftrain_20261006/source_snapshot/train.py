"""M123 on a frozen consensus-completed training graph; dual validation gate."""
import argparse
import json
import time
import sys
import numpy as np
import torch
import torch.nn.functional as F
from prepare import HERE,make_adapter,load_fold,seed_all,sha256
sys.path.insert(0,str(HERE.parent/"midrug_signedpaths_20261006"))
from dual_objective import anchor_metrics
from sklearn.metrics import roc_auc_score,average_precision_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quota",type=int,choices=[0,200,600,1200],required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    seed_all(args.seed)
    data=load_fold("MiDrug",0)
    anchor_auc,anchor_ap=anchor_metrics(data)
    plan=json.loads((HERE/"plan.json").read_text())
    assert sha256(HERE/"pseudo_edges.npz")==plan["pseudo_sha256"]
    pseudo=np.load(HERE/"pseudo_edges.npz")["pairs"][:args.quota]
    adapter=make_adapter(data,args.device,pseudo)
    model=adapter.model
    opt=torch.optim.RMSprop(model.parameters(),lr=1e-4,alpha=.99)
    out=HERE/"results"
    out.mkdir(exist_ok=True)
    tag=f"q{args.quota}_s{args.seed}"
    if (out/f"{tag}.json").exists():raise FileExistsError(tag)
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
        auc,ap=roc_auc_score(data["val_y"],prob),average_precision_score(data["val_y"],prob)
        score=min(auc-anchor_auc,ap-anchor_ap)
        curves.append(dict(epoch=ep,loss=float(loss),val_auc=auc,val_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,prob.copy()
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-start,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f"{tag}.pt")
    np.savez_compressed(out/f"{tag}_val.npz",pairs=data["val_pairs"],y=data["val_y"],probability=best_p)
    (out/f"{tag}_curves.json").write_text(json.dumps(curves,indent=2))
    record=dict(tag=tag,quota=args.quota,seed=args.seed,val_auc=roc_auc_score(data["val_y"],best_p),
                val_ap=average_precision_score(data["val_y"],best_p),best_epoch=best_ep,epochs_run=ep,
                selection="min(validation AUC gain, validation AP gain)",selection_score=best,
                checkpoint_sha256=sha256(out/f"{tag}.pt"),source_sha256=sha256(__file__),
                plan_sha256=sha256(HERE/"plan.json"),test_evaluated=False,elapsed_seconds=time.time()-start)
    (out/f"{tag}.json").write_text(json.dumps(record,indent=2))
    print("DONE",json.dumps(record),flush=True)


if __name__=="__main__":main()
