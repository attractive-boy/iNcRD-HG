"""Verify pseudo-edge provenance and all four trained checkpoints."""
import json
import numpy as np
import torch
from prepare import HERE,make_adapter,load_fold,seed_all,sha256
from check_modules import check
from sklearn.metrics import roc_auc_score,average_precision_score


def main():
    plan=json.loads((HERE/"plan.json").read_text())
    assert sha256(HERE/"prepare.py")==plan["source_sha256"]
    assert sha256(HERE/"pseudo_edges.npz")==plan["pseudo_sha256"]
    z=np.load(HERE/"pseudo_edges.npz")
    candidates,prob=z["all_candidates"],z["teacher_probability"]
    confidence=prob.min(0)
    order=np.lexsort((candidates[:,1],candidates[:,0],-confidence))
    np.testing.assert_array_equal(z["pairs"],candidates[order[:1200]])
    data=load_fold("MiDrug",0)
    train_set=set(map(tuple,data["train_pairs"].tolist()))
    assert all(tuple(pair) not in train_set for pair in z["pairs"].tolist())
    rows=[]
    for quota in plan["quotas"]:
        path=HERE/"results"/f"q{quota}_s42.json"
        r=json.loads(path.read_text())
        assert sha256(path.with_suffix(".pt"))==r["checkpoint_sha256"]
        assert sha256(HERE/"train.py")==r["source_sha256"]
        assert sha256(HERE/"plan.json")==r["plan_sha256"]
        seed_all(42)
        adapter=make_adapter(data,"cuda:0",z["pairs"][:quota])
        assert len(adapter.y)==len(data["train_y"])
        assert adapter.ps.A.sum().item()==int(data["train_y"].sum())+quota
        adapter.model.load_state_dict(torch.load(path.with_suffix(".pt"),map_location="cuda:0"))
        adapter.model.eval()
        with torch.no_grad():p=adapter.predict("val")
        saved=np.load(path.with_name(path.stem+"_val.npz"))
        np.testing.assert_array_equal(saved["pairs"],data["val_pairs"])
        np.testing.assert_array_equal(saved["y"],data["val_y"])
        np.testing.assert_allclose(p,saved["probability"],atol=2e-5,rtol=2e-5)
        assert abs(roc_auc_score(saved["y"],saved["probability"])-r["val_auc"])<1e-12
        assert abs(average_precision_score(saved["y"],saved["probability"])-r["val_ap"])<1e-12
        curves=json.loads(path.with_name(path.stem+"_curves.json").read_text())
        assert max(curves,key=lambda x:x["selection_score"])["epoch"]==r["best_epoch"]
        checks=check(adapter,count=12)
        rows.append(dict(tag=r["tag"],status="passed",explicit_deletion=checks))
        print("PASS",r["tag"],flush=True)
    (HERE/"verification.json").write_text(json.dumps(dict(status="passed",records=rows,test_evaluated=False),indent=2))


if __name__=="__main__":main()
