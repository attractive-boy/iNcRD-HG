# DLMVF baseline reproduction (Wang et al., JCIM 2026)

Repo: `/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/repos/DLMVF_model-master`
Python: `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python` (torch 1.13.1+cu117, torch_geometric 2.6.1)
GPU: `CUDA_VISIBLE_DEVICES=0` (NVIDIA A40)

## Status: SUCCESS (full 5-fold CV completed)

## Exact commands

```bash
REPO=/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/repos/DLMVF_model-master

# Step 1: reconstruct the missing data/MD_adjaceny.txt (see Fix 3)
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python Predataprocess/generate_MD_adjacency.py

# Step 2: run the official 5-fold training
cd $REPO && CUDA_VISIBLE_DEVICES=0 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -u training.py \
  > /data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/logs/dlmvf_train.log 2>&1
```

## Final 5-fold results (this run)

Per-fold (last epoch of each fold):

| Fold | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
|------|----------|-----------|--------|------|---------|--------|
| 1 | 0.8987 | 0.9048 | 0.8933 | 0.8990 | 0.9624 | 0.9549 |
| 2 | 0.8918 | 0.8840 | 0.9042 | 0.8940 | 0.9596 | 0.9549 |
| 3 | 0.8921 | 0.8789 | 0.9117 | 0.8950 | 0.9519 | 0.9388 |
| 4 | 0.8981 | 0.8959 | 0.9031 | 0.8995 | 0.9604 | 0.9544 |
| 5 | 0.9045 | 0.8902 | 0.9249 | 0.9072 | 0.9612 | 0.9541 |

Mean:

| Metric | This run | Reported | Gap |
|--------|----------|----------|-----|
| Accuracy | 0.8970 | -- | -- |
| Precision | 0.8908 | -- | -- |
| Recall | 0.9075 | -- | -- |
| F1 | 0.8989 | -- | -- |
| **ROC AUC (AUROC)** | **0.9591** | **0.9611** | **-0.0020** |
| **PR AUC (AUPRC)** | **0.9514** | **0.9543** | **-0.0029** |

The gap is ~0.2 percentage points on AUROC and ~0.3 points on AUPRC. The official
code fixes no random seed (`numpy` seed 0 is used only for the bipartite node
features; torch init/data shuffling are unseeded), so fold-level and run-level
results are stochastic; Fold 3 (PR AUC 0.9388) is the main source of the residual
gap. The match is well within expected run-to-run variance.

## Fixes / deviations from the official code (all in `patches/dlmvf.patch`)

1. **`training.py` device hardcoded to `cuda:7`** -> `cuda_name = os.environ.get("DLMVF_CUDA", "cuda:0")`
   (default `cuda:0`, still overridable). The 8-GPU-only assumption no longer crashes on a 2-GPU box.

2. **`training.py` `NUM_RUNS = 1`** -> `NUM_RUNS = 5` so all 5 folds produced by
   `dataprocess.kfoldprepare()` are trained/evaluated and averaged.

3. **Missing `data/MD_adjaceny.txt` reconstructed.** The repo references the
   integrated heterogeneous adjacency `[[M_MM, M_MD], [M_MD.T, M_DD]]` but ships
   neither it nor the `data/m-m.txt` / `data/d-d.txt` that `Predataprocess/miRTarbase.py::MD_Matrix()`
   reads. Added `Predataprocess/generate_MD_adjacency.py`, which reproduces the layout of
   `miRTarbase.py::MD_Matrix` and substitutes the two similarity blocks with the
   **Gaussian kernel similarity** from `Predataprocess/calculating_similarity.py`:
   `lambda = 1 / ((1/n) * sum_i ||A_i||^2)`, `sim_ij = exp(-lambda * ||A_i - A_j||^2)`.
   - `M_MD` = `data/miRNA_drug_matrix.xlsx` values, shape (1578, 156)
   - `M_MM` = Gaussian(A) on M_MD, shape (1578, 1578)
   - `M_DD` = Gaussian(A.T) on M_MD.T, shape (156, 156)
   - Output `data/MD_adjaceny.txt`, 1734x1734, `np.savetxt(..., fmt='%f')` (space-delimited,
     matching `pd.read_csv(..., delimiter=' ')` in `dataprocess.py`).
   This is a documented substitution, not the authors' original (unavailable) similarity files.

4. **`model.py` `drug_offset=1563` -> `1578`.** There are 1578 miRNAs, so drug nodes
   start at index 1578 in the 1734-node bipartite graph; `drug_md_feat = md_node_feats[col + offset]`
   would otherwise index miRNA nodes. Verified against `adj.shape[0] == 1734`.

5. **`dataprocess.py` `from fontTools.ttx import process`** (unused) wrapped in
   `try/except ImportError` so the pipeline still runs when fontTools is absent.
   (fontTools 4.56.0 was actually present in the env; no isolated venv needed.)

6. **Extra bug fixed (required for genuine 5-fold CV): `datasetindex()` dict aliasing.**
   The original reused one mutable `train_indices` / `test_indices` dict, appending the
   *same object* 5 times, so all `all_train_indices[0..4]` and `all_test_indices[0..4]`
   aliased the last fold — every fold trained/tested on identical data. Replaced with fresh
   dicts per fold. Without this, "5-fold CV" degenerates to a single fold repeated 5x.

7. **Kept** `Predataprocess/miRNA_embeddings.txt` (1578x512, tab-delimited) as the per-miRNA
   node feature (`gene`, variable name kept; `gene_encoder` in_dim=512).

## Notes / observations (not changed)

- `utils.py::TestbedDataset.process` stores `gene` via `torch.LongTensor([gene])`, which
  truncates the 512-d float embeddings to integers before `model.py` casts back to `.float()`.
  This is the official code's behavior and was left untouched to preserve fidelity.
- `Predataprocess/DeepWalk.py` is not needed for training; `gensim` is not installed and not required.
- Outputs `DLMVF_mean_fpr_tpr.csv` and `DLMVF_mean_recall_precision.csv` are written to the repo root;
  `csv_roc_auc=0.9581`, `csv_pr_auc=0.9506` match the predicted means closely.

## Artifacts

- Log: `/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/logs/dlmvf_train.log`
- Patch: `/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/patches/dlmvf.patch`
- Reconstructed data: `repos/DLMVF_model-master/data/MD_adjaceny.txt`
- Dump of original repo for diffing: `repos/DLMVF_model-master.orig/`
