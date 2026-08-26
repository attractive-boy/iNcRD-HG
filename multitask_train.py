#!/usr/bin/env python
"""
Multi-task extension of iNcRD-HG.

The released ``predict_association.py`` trains lncRNA-drug and miRNA-drug as
two *completely separate* single-task models inside a for-loop.  Each model
sees the other task's full association edges during message passing (a
cross-task leakage noted in PAPER_REVIEW_AND_IMPROVEMENT.md §2.2) but never
shares learned parameters with the other task.

This module introduces a genuine multi-task design, directly addressing the
author-identified limitation that "naive multi-task design may cause task
interference" (improvement direction #2).  It adds:

  1. A **shared heterogeneous GNN encoder** with **task-specific decoders**,
     so lncRNA and miRNA tasks share representation learning while keeping
     task-specific scoring heads.
  2. **Cross-task consistency regularisation** via the existing lncRNA-miRNA
     interaction edges (LncMi): interacting lncRNA-miRNA pairs are pushed
     into a shared drug-association subspace, bridging the two tasks with
     known biology instead of blind parameter sharing.
  3. A **task-aware contrastive objective** (InfoNCE within ncRNA type):
     pulls positive ncRNA-drug pairs together and pushes random negative
     drugs apart, but only contrasts within the same ncRNA type so
     task-specific biological signals are preserved.
  4. **Uncertainty-weighted adaptive loss reweighting** (Kendall et al.,
     2018): each task carries a learnable homoscedastic uncertainty that
     automatically balances the aggregate loss, down-weighting the noisier
     task without manual tuning.
  5. **Validation-based model selection**: the epoch is chosen on the
     validation fold, not the test fold — fixing the test-set selection
     bias diagnosed in §2.3 of the paper review.

Usage (original splits)::

    python multitask_train.py --data-dir ./3.heter_data

Usage (popularity-controlled benchmark)::

    python multitask_train.py --data-dir ./9.bench/random
    python multitask_train.py --data-dir ./9.bench/matched
    python multitask_train.py --data-dir ./9.bench/cold_drug

Author notes
------------
All five design choices above are grounded in PAPER_REVIEW_AND_IMPROVEMENT.md:
  * §2.2 cross-task edge leakage  -> shared encoder uses *training* edges only
  * §2.3 test-set model selection -> val-based early stopping
  * §4.3 multi-task uncertainty weighting / GradNorm / PCGrad
  * §4.3 contrastive loss only for same type / same context
  * §4.3 cross-task consistency via known ncRNA interactions
"""

import argparse
import copy
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GeneralConv, to_hetero

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 666
TASKS = [("lncRNA", "LncDrug"), ("miRNA", "MiDrug")]
# Edge types that are *shared* across tasks (similarity + interaction + drug).
# These are identical in every task-specific split, so we take them once.
SHARED_EDGE_TYPES = [
    ("lncRNA", "LncLnc", "lncRNA"),
    ("miRNA", "MiMi", "miRNA"),
    ("drug", "DrugDrug", "drug"),
    ("lncRNA", "LncMi", "miRNA"),
    ("miRNA", "rev_LncMi", "lncRNA"),
]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Data loading & multi-task graph construction
# ---------------------------------------------------------------------------

def _load(data_dir, edge_type, phase):
    """Load one task-phase HeteroData file."""
    path = os.path.join(data_dir, f"{edge_type}_{phase}_data.pth")
    return torch.load(path, map_location="cpu")


def merge_multitask_graph(lnc_train, mi_train):
    """
    Build a single message-passing HeteroData from two task-specific train
    splits.

    Key correctness property: we take *training* association edges from each
    task only.  The released RandomLinkSplit leaves the *other* task's edges
    fully intact inside each split (e.g. LncDrug_train contains all 5394
    MiDrug edges including val/test ones).  Using those would leak held-out
    positives of the other task into message passing.  Here we instead take:

      * LncDrug + rev_LncDrug training edges  <- from LncDrug_train
      * MiDrug + rev_MiDrug training edges    <- from MiDrug_train
      * all shared edges (sim, LncMi, DrugDrug) <- from either (identical)
    """
    merged = HeteroData()

    # Node features (identical in both)
    for nt in lnc_train.node_types:
        merged[nt].x = lnc_train[nt].x

    # Shared edges — take from lnc_train (same in mi_train)
    for et in SHARED_EDGE_TYPES:
        if et in lnc_train.edge_types:
            merged[et].edge_index = lnc_train[et].edge_index

    # Task-specific association edges — training split only
    task_sources = {
        "LncDrug": lnc_train,
        "MiDrug": mi_train,
    }
    for et, src in task_sources.items():
        fwd = [k for k in src.edge_types if k[1] == et and k[1] != f"rev_{et}"]
        rev = [k for k in src.edge_types if k[1] == f"rev_{et}"]
        for k in fwd:
            merged[k].edge_index = src[k].edge_index
        for k in rev:
            merged[k].edge_index = src[k].edge_index

    # Attach supervision edges for both tasks
    for rna_type, et in TASKS:
        key = (rna_type, et, "drug")
        src = lnc_train if et == "LncDrug" else mi_train
        merged[key].edge_label_index = src[key].edge_label_index
        merged[key].edge_label = src[key].edge_label

    return merged


def make_eval_graph(train_graph, lnc_eval, mi_eval):
    """
    Build an evaluation HeteroData: same message-passing edges as the training
    graph, but with val/test supervision edges from both tasks.

    During evaluation we keep the *training* message-passing graph fixed and
    only swap the supervision edges — the standard transductive setting.
    """
    eval_data = copy.deepcopy(train_graph)
    # Remove train supervision labels to avoid confusion
    for rna_type, et in TASKS:
        key = (rna_type, et, "drug")
        src = lnc_eval if et == "LncDrug" else mi_eval
        eval_data[key].edge_label_index = src[key].edge_label_index
        eval_data[key].edge_label = src[key].edge_label
    return eval_data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """Two-layer GeneralConv, wrapped by to_hetero in MultiTaskModel."""

    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GeneralConv((-1, -1), hidden_channels)
        self.conv2 = GeneralConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x, edge_index)
        return x


class TaskDecoder(nn.Module):
    """
    Task-specific scoring head (same 3-layer MLP as the original EdgeDecoder)
    plus a projection into a shared drug-association subspace for cross-task
    consistency.
    """

    def __init__(self, hidden_channels, rna_type):
        super().__init__()
        self.rna_type = rna_type
        self.lin1 = Linear(2 * hidden_channels, 2 * hidden_channels)
        self.lin2 = Linear(2 * hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, 1)
        # Projection for cross-task consistency (shared space, separate params
        # per ncRNA type so the projection learns type-appropriate geometry)
        self.proj = Linear(hidden_channels, hidden_channels)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z_rna = z_dict[self.rna_type][row]
        z_drug = z_dict["drug"][col]
        z = torch.cat([z_rna, z_drug], dim=-1)
        z = F.leaky_relu(self.lin1(z), negative_slope=0.1)
        z = F.leaky_relu(self.lin2(z), negative_slope=0.1)
        logits = self.lin3(z).view(-1)
        return logits, z_rna, z_drug


class MultiTaskModel(nn.Module):
    """
    Shared heterogeneous encoder + per-task decoders + learnable task
    uncertainties for adaptive loss reweighting.
    """

    def __init__(self, data, hidden_channels, aggr="mean"):
        super().__init__()
        encoder = GNNEncoder(hidden_channels)
        self.encoder = to_hetero(encoder, data.metadata(), aggr=aggr)
        self.decoders = nn.ModuleDict({
            et: TaskDecoder(hidden_channels, rt) for rt, et in TASKS
        })
        # Homoscedastic task uncertainty (Kendall et al., 2018).
        #   loss_i = exp(-s_i) * L_i + s_i   (s_i = log sigma_i^2)
        self.log_vars = nn.ParameterDict({
            et: nn.Parameter(torch.zeros(1)) for _, et in TASKS
        })

    def forward(self, x_dict, edge_index_dict, edge_label_index_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        outputs = {}
        for _, et in TASKS:
            logits, z_rna, z_drug = self.decoders[et](
                z_dict, edge_label_index_dict[et]
            )
            outputs[et] = dict(logits=logits, z_rna=z_rna, z_drug=z_drug,
                                z_dict=z_dict)
        return outputs, z_dict


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def bce_loss(logits, labels):
    """Per-task binary cross-entropy."""
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def task_aware_contrastive(z_rna, z_drug, labels, tau=0.1):
    """
    InfoNCE within one ncRNA type.

    For each *positive* (rna, drug) pair we sample a random negative drug and
    push it apart.  The contrastive signal is restricted to the same ncRNA
    type (task-aware) so that lncRNA-specific and miRNA-specific biological
    signals are not washed out by cross-type mixing.
    """
    pos_mask = labels == 1
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=z_rna.device)

    z_rna_pos = F.normalize(z_rna[pos_mask], dim=-1)
    z_drug_pos = F.normalize(z_drug[pos_mask], dim=-1)
    n_pos = z_rna_pos.size(0)

    # Sample one negative drug per positive by shuffling the drug dimension
    perm = torch.randperm(n_pos, device=z_rna_pos.device)
    # Avoid accidental identity (unlikely with random perm, but guard anyway)
    same = (perm == torch.arange(n_pos, device=z_rna_pos.device))
    if same.any():
        perm = (perm + 1) % n_pos
    z_drug_neg = z_drug_pos[perm]

    pos_sim = (z_rna_pos * z_drug_pos).sum(dim=-1) / tau
    neg_sim = (z_rna_pos * z_drug_neg).sum(dim=-1) / tau
    # InfoNCE: log( exp(pos) / (exp(pos) + exp(neg)) )
    loss = -F.log_softmax(torch.stack([pos_sim, neg_sim], dim=-1), dim=-1)[:, 0]
    return loss.mean()


def cross_task_consistency(z_dict, decoders, lnc_mi_edge_index, max_pairs=4096):
    """
    Regularise interacting lncRNA-miRNA pairs into a shared drug-association
    subspace.

    The heterogeneous graph already carries LncMi interaction edges
    (lncRNA-miRNA).  For each such edge (l, m) we minimise the distance
    between the *projected* lncRNA_l and miRNA_m embeddings.  This bridges
    the two tasks through known biology rather than blind parameter sharing,
    and is the "cross-task consistency regularisation using known cross-ncRNA
    interactions" strategy from improvement direction #2.
    """
    if lnc_mi_edge_index is None or lnc_mi_edge_index.size(1) == 0:
        return torch.tensor(0.0, device=next(iter(z_dict.values())).device)

    src, dst = lnc_mi_edge_index[0], lnc_mi_edge_index[1]
    # Subsample if the interaction network is large
    if src.size(0) > max_pairs:
        idx = torch.randperm(src.size(0), device=src.device)[:max_pairs]
        src, dst = src[idx], dst[idx]

    proj_lnc = decoders["LncDrug"].proj(z_dict["lncRNA"][src])
    proj_mi = decoders["MiDrug"].proj(z_dict["miRNA"][dst])
    proj_lnc = F.normalize(proj_lnc, dim=-1)
    proj_mi = F.normalize(proj_mi, dim=-1)
    return F.mse_loss(proj_lnc, proj_mi)


def uncertainty_weighted_loss(task_losses, log_vars):
    """
    Kendall et al. (2018) homoscedastic uncertainty weighting.

        total = sum_i [ exp(-s_i) * L_i + s_i ]

    where s_i = log(sigma_i^2).  The second term prevents the model from
    trivially setting s_i -> inf to zero out all losses.  Tasks with higher
    loss/noise get automatically down-weighted.
    """
    total = 0.0
    for et, loss in task_losses.items():
        s = log_vars[et]
        total = total + torch.exp(-s) * loss + s
    return total


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def forward_supervision(model, data):
    """Run the model on the supervision edges of both tasks."""
    edge_label_index_dict = {}
    for _, et in TASKS:
        key = next(k for k in data.edge_types if k[1] == et and "rev" not in k[1])
        edge_label_index_dict[et] = data[key].edge_label_index
    return model(data.x_dict, data.edge_index_dict, edge_label_index_dict)


def get_labels(data, edge_type):
    key = next(k for k in data.edge_types if k[1] == edge_type and "rev" not in k[1])
    return data[key].edge_label


def get_edge_label_index(data, edge_type):
    key = next(k for k in data.edge_types if k[1] == edge_type and "rev" not in k[1])
    return data[key].edge_label_index


def train_epoch(model, train_data, optimizer, lnc_mi_edge_index,
               w_contrast, w_consist):
    model.train()
    optimizer.zero_grad()

    outputs, z_dict = forward_supervision(model, train_data)

    task_losses = {}
    contrast_losses = {}
    for _, et in TASKS:
        labels = get_labels(train_data, et)
        logits = outputs[et]["logits"]
        task_losses[et] = bce_loss(logits, labels)
        contrast_losses[et] = task_aware_contrastive(
            outputs[et]["z_rna"], outputs[et]["z_drug"], labels
        )

    # Uncertainty-weighted combination of BCE losses
    total_bce = uncertainty_weighted_loss(task_losses, model.log_vars)

    # Add contrastive and consistency regularisers (fixed weights)
    total_contrast = sum(contrast_losses.values()) / len(contrast_losses)
    total_consist = cross_task_consistency(
        z_dict, model.decoders, lnc_mi_edge_index
    )

    loss = (total_bce
            + w_contrast * total_contrast
            + w_consist * total_consist)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss),
        "bce": float(total_bce),
        "contrast": float(total_contrast),
        "consist": float(total_consist),
        "log_var_lnc": float(model.log_vars["LncDrug"].item()),
        "log_var_mi": float(model.log_vars["MiDrug"].item()),
    }


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    outputs, _ = forward_supervision(model, data)
    results = {}
    for _, et in TASKS:
        labels = get_labels(data, et).cpu().numpy()
        scores = outputs[et]["logits"].cpu().numpy()
        auc = _safe_auc(labels, scores)
        aupr = average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
        # Per-drug macro AUC (the metric that survives popularity debiasing)
        eli = get_edge_label_index(data, et).cpu().numpy()
        per_drug_auc = _per_drug_auc(labels, scores, eli[1])
        results[et] = dict(auc=auc, aupr=aupr, per_drug_auc=per_drug_auc)
    return results


def _safe_auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def _per_drug_auc(labels, scores, drug_indices):
    """Macro-averaged AUC computed within each drug."""
    aucs = []
    for d in np.unique(drug_indices):
        mask = drug_indices == d
        y_d = labels[mask]
        if len(np.unique(y_d)) < 2:
            continue
        aucs.append(roc_auc_score(y_d, scores[mask]))
    return float(np.mean(aucs)) if aucs else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-task iNcRD-HG: shared encoder + task decoders + "
                    "cross-task consistency + contrastive + uncertainty weighting"
    )
    parser.add_argument("--data-dir", default="./3.heter_data",
                        help="Directory with {LncDrug,MiDrug}_{train,val,test}_data.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128,
                        help="Shared hidden dimension (paper used 64 for lncRNA, "
                             "256 for miRNA separately; shared default is 128)")
    parser.add_argument("--aggr", default="mean", choices=["mean", "min", "max", "sum"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--w-contrast", type=float, default=0.1,
                        help="Weight for task-aware contrastive loss")
    parser.add_argument("--w-consist", type=float, default=0.1,
                        help="Weight for cross-task consistency regularisation")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early-stopping patience on validation AUC")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-scores", action="store_true",
                        help="Save per-task prediction scores to ./4.results/")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load & merge data
    # ------------------------------------------------------------------
    print(f"Loading data from {args.data_dir}")
    lnc_train = _load(args.data_dir, "LncDrug", "train")
    lnc_val = _load(args.data_dir, "LncDrug", "val")
    lnc_test = _load(args.data_dir, "LncDrug", "test")
    mi_train = _load(args.data_dir, "MiDrug", "train")
    mi_val = _load(args.data_dir, "MiDrug", "val")
    mi_test = _load(args.data_dir, "MiDrug", "test")

    train_data = merge_multitask_graph(lnc_train, mi_train).to(device)
    val_data = make_eval_graph(train_data.cpu(), lnc_val, mi_val).to(device)
    test_data = make_eval_graph(train_data.cpu(), lnc_test, mi_test).to(device)

    # LncMi interaction edges for cross-task consistency
    lnc_mi_key = ("lncRNA", "LncMi", "miRNA")
    if lnc_mi_key in train_data.edge_types:
        lnc_mi_edge_index = train_data[lnc_mi_key].edge_index
    else:
        lnc_mi_edge_index = None
        print("WARNING: no LncMi edges found — cross-task consistency disabled")

    print(f"  lncRNA nodes: {train_data['lncRNA'].x.shape[0]}")
    print(f"  miRNA nodes:  {train_data['miRNA'].x.shape[0]}")
    print(f"  drug nodes:   {train_data['drug'].x.shape[0]}")
    for _, et in TASKS:
        n = get_edge_label_index(train_data, et).size(1)
        print(f"  {et} supervision edges: {n}")

    # ------------------------------------------------------------------
    # Model & optimiser
    # ------------------------------------------------------------------
    model = MultiTaskModel(train_data, args.hidden, args.aggr).to(device)

    # GeneralConv((-1, -1), ...) uses lazy initialisation, so the parameters
    # are not materialised until the first forward pass.  Run a tiny dummy
    # forward so that the optimiser captures real tensors, not placeholders.
    with torch.no_grad():
        dummy_eli = {}
        for _, et in TASKS:
            key = next(k for k in train_data.edge_types
                       if k[1] == et and "rev" not in k[1])
            dummy_eli[et] = train_data[key].edge_label_index[:, :1]
        _ = model(train_data.x_dict, train_data.edge_index_dict, dummy_eli)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Training loop with val-based early stopping
    # ------------------------------------------------------------------
    best_val_auc = -1.0  # tracks per-drug AUC
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            model, train_data, optimizer, lnc_mi_edge_index,
            args.w_contrast, args.w_consist
        )

        val_results = evaluate(model, val_data, device)
        # Use per-drug AUC on validation - the metric that survives popularity
        # debiasing.  Global AUC is ~0.5 on matched splits by construction.
        val_per_drug = np.mean([val_results[et]["per_drug_auc"] for _, et in TASKS
                                if not np.isnan(val_results[et]["per_drug_auc"])])

        if val_per_drug > best_val_auc:
            best_val_auc = val_per_drug
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={metrics['loss']:.4f} bce={metrics['bce']:.4f} "
                f"ctr={metrics['contrast']:.4f} cns={metrics['consist']:.4f} | "
                f"s_lnc={metrics['log_var_lnc']:.3f} s_mi={metrics['log_var_mi']:.3f} | "
                f"val_perDrugAUC={val_per_drug:.4f} (best={best_val_auc:.4f})"
            )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ------------------------------------------------------------------
    # Final test evaluation with best val model
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    test_results = evaluate(model, test_data, device)

    print("\n" + "=" * 60)
    print("TEST RESULTS (model selected on validation)")
    print("=" * 60)
    for rna_type, et in TASKS:
        r = test_results[et]
        print(f"  {et:8s}  AUC={r['auc']:.4f}  AUPR={r['aupr']:.4f}  "
              f"perDrugAUC={r['per_drug_auc']:.4f}")
    print(f"  {'mean':8s}  "
          f"AUC={np.mean([test_results[et]['auc'] for _, et in TASKS]):.4f}  "
          f"perDrugAUC={np.mean([test_results[et]['per_drug_auc'] for _, et in TASKS]):.4f}")

    if args.save_scores:
        os.makedirs("./4.results", exist_ok=True)
        for _, et in TASKS:
            outputs, _ = forward_supervision(model, test_data)
            labels = get_labels(test_data, et).cpu().numpy()
            scores = outputs[et]["logits"].cpu().numpy()
            pd.DataFrame({"y_scores": scores, "y_true": labels}).to_csv(
                f"./4.results/multitask_{et}_score.csv", index=False
            )
            print(f"  saved ./4.results/multitask_{et}_score.csv")


if __name__ == "__main__":
    main()
