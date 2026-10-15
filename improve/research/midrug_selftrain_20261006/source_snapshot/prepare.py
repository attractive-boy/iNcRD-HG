"""Freeze consensus pseudo edges from three existing train-only M123 models."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import logit

HERE=Path(__file__).resolve().parent
RELEASE=HERE.parents[2]/"release/iNcRD-HG-M1M2M3"
sys.path.insert(0,str(RELEASE))
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.data import load_fold,build_graph,split_path,sha256
from incrd_hg_m123.engine import seed_all


def make_adapter(data,device,pseudo=None):
    inputs={k:data[k] for k in ("train_pairs","train_y")}
    inputs["val_pairs"]=data.get("val_pairs",np.empty((0,2),dtype=np.int64))
    inputs["test_pairs"]=np.empty((0,2),dtype=np.int64)
    graph=build_graph(inputs,"MiDrug")
    if pseudo is not None and len(pseudo):
        edges=torch.as_tensor(pseudo.T,dtype=torch.long)
        old=graph["miRNA","MiDrug","drug"].edge_index
        graph["miRNA","MiDrug","drug"].edge_index=torch.cat([old,edges],1)
        graph["drug","rev_MiDrug","miRNA"].edge_index=graph["miRNA","MiDrug","drug"].edge_index.flip(0)
    return Adapter(inputs,graph,device)


def main():
    data=load_fold("MiDrug",0)
    inputs={k:data[k] for k in ("train_pairs","train_y")}
    all_pairs=np.stack(np.meshgrid(np.arange(605),np.arange(216),indexing="ij"),axis=-1).reshape(-1,2)
    used=np.zeros((605,216),dtype=bool)
    used[inputs["train_pairs"][:,0],inputs["train_pairs"][:,1]]=True
    candidates=all_pairs[~used[all_pairs[:,0],all_pairs[:,1]]]
    teacher_dir=HERE.parent/"baselines_m123_20261004/results"
    probabilities,teachers=[],[]
    for seed in (42,43,44):
        seed_all(seed)
        adapter=make_adapter(inputs,"cuda:0")
        path=teacher_dir/f"incrd_m123_MiDrug_f0_s{seed}.pt"
        record=json.loads(path.with_suffix(".json").read_text())
        assert sha256(path)==record["checkpoint_sha256"] and sha256(split_path("MiDrug",0))==record["split_sha256"]
        adapter.model.load_state_dict(torch.load(path,map_location="cuda:0"))
        adapter.model.eval()
        chunks=[]
        with torch.no_grad():
            for start in range(0,len(candidates),4096):
                pair=torch.as_tensor(candidates[start:start+4096].T,device="cuda:0")
                qis=(adapter.ps.qis(*pair)-adapter.mu)/adapter.sd
                ease=adapter.ps.ease(*pair)
                p=adapter.model(pair,adapter.ps,qis,ease,adapter.dense).sigmoid().cpu().numpy()
                chunks.append(p)
        probabilities.append(np.concatenate(chunks))
        teachers.append(dict(seed=seed,checkpoint_sha256=sha256(path)))
        print("TEACHER DONE",seed,len(candidates),flush=True)
    probabilities=np.stack(probabilities)
    confidence=probabilities.min(0)
    # Same fixed ranking for all quotas. No held-out labels enter this choice.
    order=np.lexsort((candidates[:,1],candidates[:,0],-confidence))
    selected=candidates[order[:1200]]
    assert not used[selected[:,0],selected[:,1]].any()
    np.savez_compressed(HERE/"pseudo_edges.npz",pairs=selected,confidence=confidence[order[:1200]],
                        all_candidates=candidates,teacher_probability=probabilities)
    plan=dict(teachers=teachers,quotas=[0,200,600,1200],ranking="minimum probability across three fixed teacher seeds",
              training_labels="Original train labels only; pseudo edges provide message context, not new loss targets",
              candidate_pool="All original-node pairs except supplied training positives and negatives",
              split_sha256=sha256(split_path("MiDrug",0)),source_sha256=sha256(__file__),
              pseudo_sha256=sha256(HERE/"pseudo_edges.npz"),test_labels_used=False,external_data_used=False,
              threshold_by_quota={str(k):float(confidence[order[k-1]]) for k in (200,600,1200)})
    (HERE/"plan.json").write_text(json.dumps(plan,indent=2))
    print(json.dumps(plan,indent=2),flush=True)


if __name__=="__main__":main()
