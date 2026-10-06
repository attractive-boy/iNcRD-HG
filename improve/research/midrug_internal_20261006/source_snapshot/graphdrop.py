"""M123 training with stochastic train-positive context masking."""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from fusion import HERE, make_adapter, load_fold, seed_all, sha256
from incrd_hg_m123.pair_channels import PairStruct
from sklearn.metrics import roc_auc_score, average_precision_score


def context(adapter, edges):
    ps = PairStruct(adapter.D, edges.T, adapter.cfg)
    graph = adapter.graph.clone()
    graph["miRNA", "MiDrug", "drug"].edge_index = edges
    graph["drug", "rev_MiDrug", "miRNA"].edge_index = edges.flip(0)
    adj = dict(adapter.model.encoder.adj)
    for e in (("miRNA", "MiDrug", "drug"), ("drug", "rev_MiDrug", "miRNA")):
        ix = graph[e].edge_index
        adj[e] = torch.sparse_coo_tensor(ix.flip(0), torch.ones(ix.shape[1], device=ix.device), adj[e].shape).coalesce()
    pairs = adapter.pairs["train"]
    return dict(ps=ps, graph=graph, adj=adj,
                qis=(ps.qis(*pairs)-adapter.mu)/adapter.sd,
                ease=ps.ease(*pairs), dense=(*adapter.dense[:4], ps.A))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drop", type=float, required=True, choices=(0., .1, .2, .4))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_all(args.seed)
    out = HERE / "graphdrop"
    out.mkdir(exist_ok=True)
    tag = f"drop{args.drop}_s{args.seed}"
    if (out / f"{tag}.json").exists():
        raise FileExistsError(tag)
    data = load_fold("MiDrug", 0)
    adapter = make_adapter(data, args.device)
    model = adapter.model
    enc = model.encoder
    full_graph, full_adj = enc.graph, dict(enc.adj)
    edges = adapter.graph["miRNA", "MiDrug", "drug"].edge_index
    rng = np.random.RandomState(args.seed+73021)
    contexts = []
    for _ in range(10 if args.drop else 1):
        keep = rng.permutation(edges.shape[1])[:int((1-args.drop)*edges.shape[1])]
        contexts.append(context(adapter, edges[:,keep]))
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=.99)
    best, best_ep, best_state, best_p = -1, 0, None, None
    curves = []
    start = time.time()
    for epoch in range(1, 601):
        model.train()
        opt.zero_grad()
        c = contexts[rng.randint(len(contexts))]
        enc.graph, enc.adj = c["graph"], c["adj"]
        logits = model(adapter.pairs["train"], c["ps"], c["qis"], c["ease"], c["dense"])
        loss = F.binary_cross_entropy_with_logits(logits, adapter.y)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        enc.graph, enc.adj = full_graph, full_adj
        model.eval()
        with torch.no_grad():
            prob = adapter.predict("val")
        auc = roc_auc_score(data["val_y"], prob)
        ap = average_precision_score(data["val_y"], prob)
        curves.append(dict(epoch=epoch, loss=float(loss), val_auc=auc, val_ap=ap))
        if auc > best:
            best, best_ep, best_p = auc, epoch, prob.copy()
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch % 100 == 0:
            print(tag, epoch, best, best_ep, time.time()-start, flush=True)
        if epoch >= 300 and epoch-best_ep >= 200:
            break
    torch.save(best_state, out / f"{tag}.pt")
    np.savez_compressed(out / f"{tag}_val.npz", pairs=data["val_pairs"], y=data["val_y"], probability=best_p)
    (out / f"{tag}_curves.json").write_text(json.dumps(curves, indent=2))
    result = dict(tag=tag, drop=args.drop, seed=args.seed, contexts=len(contexts),
                  best_epoch=best_ep, epochs_run=epoch, val_auc=best,
                  val_ap=average_precision_score(data["val_y"], best_p), test_evaluated=False,
                  source_sha256=sha256(__file__), checkpoint_sha256=sha256(out / f"{tag}.pt"),
                  elapsed_seconds=time.time()-start)
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    print("DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
