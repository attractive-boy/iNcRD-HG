#!/usr/bin/env python
"""Train and evaluate iNcRD-HG + M1 + M2 + M3 on one frozen split.

Examples:
  python run.py --dataset MiDrug --fold 0 --seed 42 --device cuda:0
  python run.py --dataset MDR --fold 3 --seed 42 --device cuda:1
"""
import argparse
from pathlib import Path

import torch

from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.data import build_graph, load_fold, sha256, split_path
from incrd_hg_m123.engine import fit, seed_all


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("MiDrug", "MDR", "MDS"), required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--tag", default="incrd_m123")
    p.add_argument("--output", default=str(Path(__file__).resolve().parent / "results"))
    args = p.parse_args()
    args.split_sha256 = sha256(split_path(args.dataset, args.fold))

    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    seed_all(args.seed)
    data = load_fold(args.dataset, args.fold)
    graph = build_graph(data, args.dataset)
    adapter = Adapter(data, graph, args.device)
    fit(adapter, data, args)


if __name__ == "__main__":
    main()
