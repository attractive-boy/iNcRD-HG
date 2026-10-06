"""Train M1/M2/M3 with explicit endpoint interactions, validation only."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[2] / "release/iNcRD-HG-M1M2M3"
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(HERE.parent / "midrug_ranking_20261006"))
from incrd_hg_m123.model import Model
from incrd_hg_m123.data import load_fold, sha256
from incrd_hg_m123.engine import seed_all
from run_ranking import make_adapter

CONFIGS = {
    "long_bce": dict(mode="none", rank=16, reg=0., lr=1e-4),
    "tensor16": dict(mode="tensor", rank=16, reg=.001, lr=1e-4),
    "tensor64": dict(mode="tensor", rank=64, reg=.001, lr=1e-4),
    "raw16": dict(mode="raw", rank=16, reg=.001, lr=1e-4),
    "raw64": dict(mode="raw", rank=64, reg=.001, lr=1e-4),
    "latent8": dict(mode="latent", rank=8, reg=.001, lr=1e-4),
    "latent32": dict(mode="latent", rank=32, reg=.001, lr=1e-4),
    "latent32_r01": dict(mode="latent", rank=32, reg=.01, lr=1e-4),
}


class InteractionModel(Model):
    """M1/M2/M3 unchanged; add a bilinear term to their fusion logit.

    tensor: query-hollow encoder endpoint states.
    raw: existing released expression/ChemBERTa covariates.
    latent: node embeddings fitted only by training supervision.
    """
    def __init__(self, graph, ps, n_qis, n_ease, cfg, experiment):
        super().__init__(graph, ps, n_qis, n_ease, cfg)
        self.mode = experiment["mode"]
        rank = experiment["rank"]
        self.rank = rank
        if self.mode == "tensor":
            self.rproj = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, rank))
            self.dproj = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, rank))
        elif self.mode == "raw":
            self.rproj = nn.Sequential(nn.LayerNorm(graph["miRNA"].x.shape[1]),
                                       nn.Linear(graph["miRNA"].x.shape[1], rank))
            self.dproj = nn.Sequential(nn.LayerNorm(graph["drug"].x.shape[1]),
                                       nn.Linear(graph["drug"].x.shape[1], rank))
        elif self.mode == "latent":
            self.rproj = nn.Embedding(graph["miRNA"].num_nodes, rank)
            self.dproj = nn.Embedding(graph["drug"].num_nodes, rank)
            nn.init.normal_(self.rproj.weight, std=.1)
            nn.init.normal_(self.dproj.weight, std=.1)
        if self.mode != "none":
            self.rbias = nn.Embedding(graph["miRNA"].num_nodes, 1)
            self.dbias = nn.Embedding(graph["drug"].num_nodes, 1)
            nn.init.zeros_(self.rbias.weight)
            nn.init.zeros_(self.dbias.weight)
        self.penalty = None

    def forward(self, eli, ps, qis, ease, dense):
        r, d = eli
        xr, xd, Sr1, Sd1, A = dense
        zr, zd, zm, zdd, corr = self.encoder.forward_dense(xr, xd, Sr1, Sd1, A, r, d, ps.qvals(r, d))
        parts = [zm, zdd]
        parts += [lin(e) for lin, e in zip(self.ncn_lin, ps.qncn(r, d, {"miRNA": zr, "drug": zd}, corr))]
        parts += [self.pc(qis), self.em(self.eb(ease))]
        h = F.leaky_relu(self.lin1(torch.cat(parts, -1)), .1)
        out = self.lin3(F.leaky_relu(self.lin2(h), .1)).view(-1)
        self.penalty = out.new_zeros(())
        if self.mode != "none":
            if self.mode == "tensor":
                er, ed = self.rproj(zm), self.dproj(zdd)
            elif self.mode == "raw":
                er, ed = self.rproj(xr[r]), self.dproj(xd[d])
            else:
                er, ed = self.rproj(r), self.dproj(d)
            br, bd = self.rbias(r).view(-1), self.dbias(d).view(-1)
            out = out + (er * ed).sum(1) / math.sqrt(self.rank) + br + bd
            self.penalty = er.square().mean() + ed.square().mean() + br.square().mean() + bd.square().mean()
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=CONFIGS, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    out = HERE / "fusion"
    out.mkdir(exist_ok=True)
    tag = f"{args.config}_s{args.seed}"
    if (out / f"{tag}.json").exists():
        raise FileExistsError(tag)
    seed_all(args.seed)
    data = load_fold("MiDrug", 0)
    adapter = make_adapter(data, args.device)
    config = CONFIGS[args.config]
    # Preserve common base initialization across all variants.
    state = adapter.model.state_dict()
    model = InteractionModel(adapter.graph, adapter.ps, adapter.qis["train"].shape[1],
                             adapter.ease["train"].shape[1], adapter.cfg, config).to(args.device)
    model.load_state_dict(state, strict=False)
    with torch.no_grad():
        model.encoder(adapter.graph.x_dict, adapter.graph.edge_index_dict)
    adapter.model = model
    extra = [p for n, p in model.named_parameters() if n.startswith(("rproj", "dproj", "rbias", "dbias"))]
    base = [p for n, p in model.named_parameters() if not n.startswith(("rproj", "dproj", "rbias", "dbias"))]
    opt = torch.optim.RMSprop([dict(params=base, lr=config["lr"]),
                              dict(params=extra, lr=.003 if config["mode"] == "latent" else config["lr"])], alpha=.99)
    best, best_ep, best_state, best_p = -1, 0, None, None
    curves = []
    start = time.time()
    for epoch in range(1, 801):
        model.train()
        opt.zero_grad()
        logits = adapter.logits("train")
        loss = F.binary_cross_entropy_with_logits(logits, adapter.y) + config["reg"] * model.penalty
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()
        model.eval()
        with torch.no_grad():
            prob = adapter.predict("val")
        auc = roc_auc_score(data["val_y"], prob)
        ap = average_precision_score(data["val_y"], prob)
        curves.append(dict(epoch=epoch, loss=float(loss), val_auc=auc, val_ap=ap))
        if auc > best:
            best, best_ep, best_p = auc, epoch, prob.copy()
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch % 100 == 0:
            print(tag, epoch, "best", best, best_ep, "elapsed", time.time()-start, flush=True)
        if epoch >= 300 and epoch-best_ep >= 200:
            break
    torch.save(best_state, out / f"{tag}.pt")
    np.savez_compressed(out / f"{tag}_val.npz", pairs=data["val_pairs"], y=data["val_y"], probability=best_p)
    (out / f"{tag}_curves.json").write_text(json.dumps(curves, indent=2))
    result = dict(tag=tag, config=config, seed=args.seed, best_epoch=best_ep, epochs_run=epoch,
                  val_auc=best, val_ap=average_precision_score(data["val_y"], best_p),
                  test_evaluated=False, checkpoint_sha256=sha256(out / f"{tag}.pt"),
                  source_sha256=sha256(__file__), elapsed_seconds=time.time()-start)
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    print("DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
