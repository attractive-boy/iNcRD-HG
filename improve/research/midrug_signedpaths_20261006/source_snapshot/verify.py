"""Replay checkpoints and compare final logits against explicit record deletion."""
import json
import numpy as np
import torch
from modules import HERE,Evidence,DegreePairStruct,load_fold,structure_data,sha256
from train import CONFIGS,make_adapter,seed_all
from sklearn.metrics import roc_auc_score,average_precision_score


def main():
    seed_all(42)
    device="cuda:1"
    data=load_fold("MiDrug",0)
    cache=np.load(HERE/"features.npz")
    ix=np.r_[np.flatnonzero(data["train_y"]==1)[[0,117,808]],np.flatnonzero(data["train_y"]==0)[[0,217,911]]]
    deleted={}
    for i in ix:
        keep=np.arange(len(data["train_y"]))!=i
        ev=Evidence(data["train_pairs"][keep],data["train_y"][keep],(605,216),device)
        deleted[int(i)]=ev.features(data["train_pairs"][i:i+1])[0]
    rows=[]
    for config in CONFIGS:
        path=HERE/"results"/f"{config}_s42.json"
        rec=json.loads(path.read_text())
        assert rec["checkpoint_sha256"]==sha256(path.with_suffix(".pt"))
        assert rec["source_sha256"]==sha256(HERE/"train.py") and rec["modules_sha256"]==sha256(HERE/"modules.py")
        seed_all(42)
        adapter=make_adapter(data,CONFIGS[config],device)
        model=adapter.model
        model.load_state_dict(torch.load(path.with_suffix(".pt"),map_location=device))
        model.eval()
        with torch.no_grad():p=adapter.predict("val")
        saved=np.load(path.with_name(path.stem+"_val.npz"))
        np.testing.assert_array_equal(saved["pairs"],data["val_pairs"])
        np.testing.assert_array_equal(saved["y"],data["val_y"])
        np.testing.assert_allclose(p,saved["probability"],atol=2e-5,rtol=2e-5)
        assert abs(roc_auc_score(saved["y"],saved["probability"])-rec["val_auc"])<1e-12
        assert abs(average_precision_score(saved["y"],saved["probability"])-rec["val_ap"])<1e-12
        curves=json.loads(path.with_name(path.stem+"_curves.json").read_text())
        assert max(curves,key=lambda r:r["val_auc"])["epoch"]==rec["best_epoch"]
        errors=[]
        for i in ix:
            pair=adapter.pairs["train"][:,i:i+1]
            r,d=map(int,pair[:,0])
            with torch.no_grad():f1=model(pair,adapter.ps,adapter.qis["train"][i:i+1],adapter.ease["train"][i:i+1],adapter.dense)
            g=adapter.graph.clone()
            adj=dict(model.encoder.adj)
            for edge,u,v in ((("miRNA","MiDrug","drug"),r,d),(("drug","rev_MiDrug","miRNA"),d,r)):
                e=g[edge].edge_index
                e=e[:,~((e[0]==u)&(e[1]==v))]
                g[edge].edge_index=e
                adj[edge]=torch.sparse_coo_tensor(e.flip(0),torch.ones(e.shape[1],device=device),adj[edge].shape).coalesce()
            D=structure_data(g)
            ps=DegreePairStruct(D,g["miRNA","MiDrug","drug"].edge_index.T,adapter.cfg,power=CONFIGS[config]["power"])
            emask,wmask=adapter.extra_masks
            mu,sd=adapter.extra_norm
            extra=deleted[int(i)]
            qis=torch.cat([(ps.qis(*pair)-adapter.mu)/adapter.sd,(extra[:,wmask]-mu)/sd],1)
            ease=torch.cat([ps.ease(*pair),extra[:,emask]],1)
            old_graph,old_adj=model.encoder.graph,model.encoder.adj
            model.encoder.graph,model.encoder.adj=g,adj
            with torch.no_grad():f2=model(pair,ps,qis,ease,(*adapter.dense[:4],ps.A))
            model.encoder.graph,model.encoder.adj=old_graph,old_adj
            torch.testing.assert_close(f1,f2,atol=3e-4,rtol=3e-4)
            errors.append(float((f1-f2).abs().max()))
        rows.append(dict(tag=rec["tag"],status="passed",final_logit_deletion_errors=errors))
        print("PASS",rec["tag"],max(errors),flush=True)
    (HERE/"verification.json").write_text(json.dumps(dict(status="passed",records=rows,test_evaluated=False),indent=2))


if __name__=="__main__":main()
