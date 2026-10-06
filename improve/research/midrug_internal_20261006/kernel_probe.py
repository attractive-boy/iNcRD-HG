"""Convex pair-interaction diagnostic on released covariates, no test access."""
import json
import time
import numpy as np
import torch
from scipy.special import expit, logit
from probe import HERE, load_fold, load_graph, score, sha256


def kernels(x):
    x = x.float()
    z = torch.nn.functional.normalize(x, dim=1)
    dist = (2 - 2 * z @ z.T).clamp(min=0)
    scale = dist[dist > 1e-5].median()
    I = torch.eye(len(x), device=x.device)
    return {**{f"rbf{g}": torch.exp(-g*dist/scale) for g in (.5, 2., 8.)},
            "identity": I, "mixed": .5*I + .5*torch.exp(-2*dist/scale)}


def main():
    torch.set_num_threads(2)
    out = HERE / "kernel"
    out.mkdir(exist_ok=True)
    plan = dict(source_sha256=sha256(__file__), test_evaluated=False,
                kernels=["rbf0.5", "rbf2.0", "rbf8.0", "mixed", "identity"],
                ridge=[.1, 1., 10., 100.],
                model="Kpair = Kr*Kd + 2*(Kr+Kd); ridge on centered train labels",
                selection="Validation AUC diagnostic")
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    data = load_fold("MiDrug", 0)
    g = load_graph("MiDrug")
    kr = kernels(g["miRNA"].x.to("cuda:0"))
    kd = kernels(g["drug"].x.to("cuda:0"))
    tr, td = torch.as_tensor(data["train_pairs"].T, device="cuda:0")
    vr, vd = torch.as_tensor(data["val_pairs"].T, device="cuda:0")
    y = torch.as_tensor(data["train_y"], dtype=torch.float32, device="cuda:0")
    I = torch.eye(len(y), device=y.device)
    base_dir = HERE.parent / "midrug_ranking_20261006/results/validation"
    base = np.mean([np.load(base_dir / f"bpr010_mixed_s{s}_predictions.npz")["probability"] for s in (42,43,44)], axis=0)
    records = []
    for name in plan["kernels"]:
        R, D = kr[name], kd[name]
        rt, dt = R[tr[:, None],tr], D[td[:, None],td]
        rv, dv = R[vr[:, None],tr], D[vd[:, None],td]
        K, Kv = rt*dt+2*(rt+dt), rv*dv+2*(rv+dv)
        del rt, dt, rv, dv
        for ridge in plan["ridge"]:
            t = time.time()
            L = torch.linalg.cholesky(K + ridge*I)
            alpha = torch.cholesky_solve((y-y.mean())[:, None], L)
            pred = (Kv @ alpha).flatten() + y.mean()
            p = pred.cpu().numpy()
            rec = dict(kernel=name, ridge=ridge, metrics=score(data["val_y"], p),
                       elapsed_seconds=time.time()-t, test_evaluated=False)
            blends = []
            for w in (.25, .5, .75):
                pp = expit((1-w)*logit(np.clip(base, 1e-6, 1-1e-6)) + w*logit(np.clip(p, 1e-4, 1-1e-4)))
                blends.append(dict(weight=w, **score(data["val_y"], pp)))
            rec["m123_blends"] = blends
            records.append(rec)
            np.savez_compressed(out / f"{name}_l{ridge}_val.npz", probability=p, pairs=data["val_pairs"], y=data["val_y"])
            print(json.dumps(rec), flush=True)
            (out / "results.json").write_text(json.dumps(records, indent=2))
            del L, alpha
        del K, Kv


if __name__ == "__main__":
    main()
