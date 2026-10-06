"""Frozen inputs and train-only graph construction for the three datasets.

The files under ``data/frozen`` are the inputs used for the reported runs:

- ``{dataset}_graph.pt``      hetero graph with the released / adapted node
  features and all auxiliary relations; the target relation is left empty.
- ``{dataset}_fold{f}.npz``   train/val/test pairs and labels of one frozen
  split (MiDrug: fold 0; MDR/MDS: folds 0-4).

At run time only the target relation is (re)installed with the shared training
positives.  Validation and test positives never enter the message graph, and
the validation AUC is the only model-selection signal.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "frozen"
DATASETS = ("MiDrug", "MDR", "MDS")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_manifest():
    return json.loads((FROZEN / "manifest.json").read_text())


def split_path(dataset, fold):
    return FROZEN / f"{dataset}_fold{fold}.npz"


def load_fold(dataset, fold):
    """Train/val/test pairs (N, 2) and labels (N,) of one frozen split."""
    path = split_path(dataset, fold)
    record = frozen_manifest()["datasets"][dataset]["folds"]
    assert sha256(path) == record[str(fold)]["sha256"], f"split file changed: {path}"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_graph(dataset):
    path = FROZEN / f"{dataset}_graph.pt"
    assert sha256(path) == frozen_manifest()["datasets"][dataset]["graph_sha256"], f"graph file changed: {path}"
    return torch.load(path, map_location="cpu")


def build_graph(data, dataset):
    """Frozen graph with the shared training positives as the only target edges."""
    graph = load_graph(dataset)
    positive = data["train_pairs"][data["train_y"] == 1]
    edges = torch.as_tensor(positive.T.copy(), dtype=torch.long)
    graph["miRNA", "MiDrug", "drug"].edge_index = edges
    graph["drug", "rev_MiDrug", "miRNA"].edge_index = edges.flip(0)
    return graph


@dataclass
class Structure:
    xr: torch.Tensor
    xd: torch.Tensor
    xk: torch.Tensor
    n_mir: int
    n_drug: int
    Sr: torch.Tensor
    Sd: torch.Tensor
    Sr1: torch.Tensor
    Sd1: torch.Tensor


def structure_data(graph):
    """Dense same-type relation operators of the message graph.

    ``Sr1``/``Sd1`` keep the released self-loop multiplicity (needed by the
    exact M1 correction); ``Sr``/``Sd`` are the zero-diagonal channels used by
    the M2 content and M3 EASE statistics.
    """
    def adjacency(edge):
        idx = graph[edge].edge_index
        out = torch.zeros(graph[edge[0]].num_nodes, graph[edge[2]].num_nodes, device=idx.device)
        out.index_put_((idx[0], idx[1]), torch.ones(idx.shape[1], device=idx.device), accumulate=True)
        return out

    sr1 = adjacency(("miRNA", "MiMi", "miRNA"))
    sd1 = adjacency(("drug", "DrugDrug", "drug"))
    sr, sd = sr1.clone(), sd1.clone()
    sr.fill_diagonal_(0)
    sd.fill_diagonal_(0)
    return Structure(xr=graph["miRNA"].x, xd=graph["drug"].x, xk=graph["miRNA"].x,
                     n_mir=graph["miRNA"].num_nodes, n_drug=graph["drug"].num_nodes,
                     Sr=sr, Sd=sd, Sr1=sr1, Sd1=sd1)
