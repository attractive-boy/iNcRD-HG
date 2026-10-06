# MSFFMDI — faithful REIMPLEMENTATION (from the abstract only)

**Baseline under reimplementation:** Lei et al., *"Predicting miRNA–Drug Interactions
Based on Multi-source Feature Fusion of Heterogeneous Network"*, Interdisciplinary
Sciences: Computational Life Sciences, 2025. DOI `10.1007/s12539-025-00775-7`.

> ## ⚠️ Status and honesty statement
> - **This is NOT the authors' code and NOT a reproduction of the paper's reported
>   numbers.** There is **no public code**, and the **full text is paywalled**. Every
>   design choice below was inferred from the published abstract / method description.
> - **Every number in this report comes from an actual run** whose full log is saved at
>   `logs/msffmdi_train.log` (stdout duplicate `logs/msffmdi_stdout.log`). Nothing is
>   fabricated.
> - The values are **only comparable to our own DLST-MDA / DLMVF / PLMF-MDA runs on the
>   shared DLST-MDA benchmark** (same association matrix, same folds). They are **not**
>   comparable to the numbers printed in the MSFFMDI paper.

---

## 1. Environment and exact run

| Item | Value |
|------|-------|
| Python | `/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python` |
| torch | 1.13.1+cu117 |
| torch_geometric | 2.6.1 |
| rdkit | 2024.03.5 |
| numpy | 1.24.4 |
| gensim | 4.3.3 (isolated `pip install --target venvs/msffmdi_site`, shared env untouched) |
| GPU | `CUDA_VISIBLE_DEVICES=0` (NVIDIA A40) |
| Wall time | ~5 min 40 s for all 5 folds, 80 epochs each |

```bash
BASE=/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919
cd $BASE/msffmdi
CUDA_VISIBLE_DEVICES=0 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -u msffmdi.py \
  --epochs 80 > $BASE/logs/msffmdi_stdout.log 2>&1
# the run also appends everything to $BASE/logs/msffmdi_train.log
```

> **Path note.** The task specified `BASE=/data1/licheng/incrd-hg/improve/research/new5_repro_20260919`.
> That path does not exist; the actual workspace (where the DLST-MDA data and the existing
> DLMVF/PLMF repros live) is
> `/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919`. All deliverables
> were written there. This is the same root used by `REPRO_DLMVF.md`.

---

## 2. Data and fold protocol (EXACT DLST-MDA protocol)

- `repos/DLST-MDA-main/data/miRNA_drug_matrix.xlsx`: **1578 miRNAs × 156 drugs**, 8 720 known
  associations.
- `miRNA_sequences.xlsx` (1578 sequences, alphabet `ACGU`, length 16–27, mean 21.6),
  `drug_id_smiles.xlsx` (156 valid SMILES, 2–96 heavy atoms). Row/column order matches the
  matrix exactly.
- `Positive.txt` / `Negative.txt` are **flat integer lists**, one per fold, of **indices into
  `np.where(A == 1)`** and **`np.where(A != 1)`** respectively (this is precisely how
  `DLST-MDA-main/code/process_data.py:123-131` consumes them — *not* `[miRNA_idx, drug_idx]`
  pairs as the brief assumed).
- For fold *i*: **test = fold-*i* positives + fold-*i* negatives**; **train = the other four
  folds' positives + their negatives**. Result: 6 976 pos / 6 976 neg train, 1 744 pos /
  1 744 neg test per fold — balanced, identical to our DLMVF/PLMF runs.

---

## 3. Design decisions and substitutions

### Channel 1 — attribute features

**miRNA.** The abstract concatenates 1-/2-/3-mer frequency vectors and learns a word2vec
embedding. Implemented as:
1. Concatenated 1-mer (4) + 2-mer (16) + 3-mer (64) **relative-frequency vector**, L2-normalised
   → 84-d (`kmer_frequency_vector`, `msffmdi.py`). Full 84/84 k-mer vocabulary is covered.
2. **gensim Word2Vec** (skip-gram, `sg=1`, 64-d, window 5, `min_count=1`, 50 epochs, seed 42)
   trained on one "document" per miRNA = its concatenated 1-/2-/3-mer token sequence; the
   miRNA word2vec embedding is the **mean of its token vectors** (64-d).
   → miRNA attribute = `[84-d frequency ‖ 64-d word2vec]` = **148-d**.
   *Choice documented:* the abstract is ambiguous about how freq-vectors feed word2vec; we
   use the actual k-mer token sequences as the corpus (the frequency vector is retained as a
   parallel fixed feature), which is the defensible reading.

**Drug (a) — GIN.** `torch_geometric.nn.GINConv` (3 layers, hidden 64, `global_add_pool`,
output 64-d) over the RDKit molecular graph. Atom features: 44-way symbol one-hot + degree
(11) + total-H (11) + implicit-valence (11) + aromaticity = **78-d**, normalized per atom
(same encoding as DLST-MDA). Bond one-hots are built as `edge_attr` but GIN does not consume
them (documented simplification). **The GIN is trainable and trained end-to-end per fold.**

**Drug (b) — mol2vec substitute.** The original uses a **pretrained mol2vec** model that is
**unavailable offline**. Substitution: a **gensim Word2Vec trained on SMILES token sequences**
(regex SMILES atom/substructure tokenizer, same skip-gram/64-d/50-epoch/seed settings), mean
pooled → 64-d. With only 156 drugs this embedding is necessarily weak — this is the single
largest fidelity gap, and it is deliberate and documented.

### Channel 2 — topological features

- **Integrated similarities.** Gaussian-kernel / GIP similarity on the association matrix:
  `gamma = n / Σᵢ‖Aᵢ‖²`, `sim_ij = exp(-gamma·‖Aᵢ-Aⱼ‖²)` — the same GIP form used in the DLMVF
  repro (`REPRO_DLMVF.md`, Fix 3).
  - miRNA–miRNA from rows of `A` (1578×1578), **top-10** neighbours per node.
  - drug–drug from columns of `A` (156×156), **top-10** neighbours per node.
  - An undirected **miRNA–drug association** edge for every known interaction (8 720 → 17 440
    directed edges).
  - Total graph: **1 734 nodes**, **52 120 directed edges** (31 560 mi–mi, 3 120 dr–dr, 17 440 mi–dr).
- The graph is built from the **full association matrix**, i.e. a **transductive** setting
  (see caveats §6). This deliberately matches the DLMVF repro convention (`MD_adjaceny.txt` =
  full associations + Gaussian similarity) so the comparison is apples-to-apples.
- Node initial features = channel-1 attribute embeddings (miRNA 148-d, drug 64-d word2vec
  ‖ 64-d GIN), projected to a common **64-d** space by node-type-specific linear layers.
- **GATConv** ×2 (4 heads, `concat=True`, dropout 0.5) updates node embeddings.
- **Multi-scale 1-D CNN** over each node's 64-d feature map: `Conv1d(1→64, k)` for
  `k ∈ {3,5,7}`, ReLU, global max-pool → **192-d** topological representation per node.

### Fusion, PCA, classifier

- Per node, **concatenate attribute (64-d) + topological (192-d)** → 256-d.
  *Interpretation documented:* attribute = the projected channel-1 features (input to the GAT);
  topological = the CNN output over the GAT-updated embeddings.
- **PCA** (`sklearn.decomposition.PCA`, per fold, fitted **only on training-fold nodes**):
  miRNA 148→64 (`explained_variance_ratio ≈ 0.997`), drug 64→32 (`≈ 1.000`). It is applied to
  the **fixed channel-1 attribute channel**: inserting a data-dependent PCA between the
  trainable GAT and the MLP is not differentiable end-to-end, so we reduce the high-dimensional
  fixed attribute features instead of the joint fused vector. **This placement is a documented
  deviation** from the abstract's literal wording.
- **MLP classifier** on the concatenated miRNA+drug pair representation (512→256→128→1),
  ReLU + dropout 0.5, **sigmoid** output.
- **Training.** BCE loss (`BCEWithLogitsLoss`, equivalent to sigmoid+BCE), **Adam lr 5e-4,
  weight decay 5e-4, dropout 0.5, 80 epochs/fold, batch 512**, seeds fixed per fold
  (`seed + fold`). No test-based early stopping or model selection (test metrics are logged
  only after the final epoch), so the reported numbers are not test-tuned.

---

## 4. Results (this reimplementation, 5-fold CV)

Per-fold (last/only evaluated epoch):

| Fold | AUC | AUPR | Accuracy | Precision | Recall | F1 | MCC |
|------|------|------|----------|-----------|--------|------|------|
| 0 | 0.9496 | 0.9405 | 0.8690 | 0.9138 | 0.8148 | 0.8615 | 0.7423 |
| 1 | 0.9538 | 0.9480 | 0.8667 | 0.9161 | 0.8073 | 0.8583 | 0.7386 |
| 2 | 0.9416 | 0.9285 | 0.8320 | 0.9159 | 0.7311 | 0.8131 | 0.6779 |
| 3 | 0.9478 | 0.9411 | 0.8151 | 0.9372 | 0.6755 | 0.7851 | 0.6563 |
| 4 | 0.9485 | 0.9376 | 0.8142 | 0.9342 | 0.6760 | 0.7844 | 0.6539 |
| **Mean ± std** | **0.9483 ± 0.0044** | **0.9391 ± 0.0071** | **0.8394 ± 0.0269** | **0.9234 ± 0.0113** | **0.7409 ± 0.0679** | **0.8205 ± 0.0378** | **0.6938 ± 0.0436** |

- **Mean AUC = 0.9483** (range 0.9416–0.9538); **mean AUPR = 0.9391** (range 0.9285–0.9480).
- Accuracy/F1/MCC are computed at the fixed 0.5 threshold. Recall is comparatively low,
  which is the usual threshold-calibration behaviour of a BCE-trained model; ranking metrics
  (AUC/AUPR) are unaffected.
- Full log: `logs/msffmdi_train.log`; machine-readable: `msffmdi/results.json`.

### Context from the same benchmark directory (for reference only)

| Method | AUC | AUPR | Source |
|--------|------|------|--------|
| MSFFMDI (this reimplementation) | **0.9483** | **0.9391** | `logs/msffmdi_train.log` |
| DLMVF (our repro) | 0.9591 | 0.9514 | `REPRO_DLMVF.md` |
| PLMF-MDA (our repro) | 0.9227 | 0.9084 | `logs/plmf_mdr.log` |

(A DLST-MDA reproduction was not present as a report in this directory at run time; the
DLST-MDA data/fold protocol is reused verbatim, so the MSFFMDI numbers above are on the
identical split.)

---

## 5. Deliverables

| Artifact | Path |
|----------|------|
| Runnable code | `msffmdi/msffmdi.py` |
| Cached features/graphs | `msffmdi/cache/` (miRNA attrs, drug mol2vec substitute, drug graphs, graph edges) |
| Results JSON | `msffmdi/results.json` |
| Full training log | `logs/msffmdi_train.log` (+ `logs/msffmdi_stdout.log`) |
| gensim install | `venvs/msffmdi_site/` |

Reproduce from scratch (delete `msffmdi/cache/` first to retrain word2vec/GIN graphs):

```bash
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/msffmdi
rm -rf cache
CUDA_VISIBLE_DEVICES=0 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -u msffmdi.py --epochs 80
```
Smoke test: `--smoke` (2 epochs, 1 fold).

---

## 6. Caveats (explicit)

1. **Not the paper's numbers.** This is a from-abstract reimplementation; no official code
   exists and the full text is paywalled. The reported MSFFMDI values in the paper cannot be
   compared to anything here.
2. **Comparability is limited to our own runs** (DLST-MDA / DLMVF / PLMF-MDA) on the DLST-MDA
   benchmark with the same folds and the same metric code.
3. **mol2vec substitution.** The pretrained mol2vec model is unavailable, so a SMILES-token
   word2vec (trained on only 156 drugs) substitutes it. This materially weakens the drug
   attribute channel and is the largest fidelity risk.
4. **Similarity-network reconstruction.** The paper's exact similarity construction and
   sparsification are unspecified here; we use GIP similarity + top-k, matching the DLMVF
   repro. Node similarity thresholding (e.g. `sim > τ` instead of top-k) was not explored.
5. **PCA placement** is a documented deviation (applied to the fixed attribute channel rather
   than the joint fused vector, for differentiability).
6. **Transductive setting / leakage.** Because the heterogeneous graph uses the **full**
   association matrix (including test-fold positive edges) for similarities and miRNA–drug
   edges — the standard convention in this MDA benchmark family and in our DLMVF repro — the
   GAT node embeddings see test associations. AUC/AUPR therefore characterise a transductive
   protocol and may overestimate inductive performance. This is consistent with the DLMVF
   comparison but should be read as a caveat, not a clean inductive result.
7. **Hyperparameters** (epochs=80, batch=512, top-k=10, PCA dims, GIN/CNN widths) are
   reasonable settings from the abstract, not tuned to match the paper; some, like the
   threshold-0.5 recall, are clearly not calibrated.
