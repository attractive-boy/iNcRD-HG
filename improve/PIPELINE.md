# Pipeline: reproducing the improved Table 2 / Table 3 figures

Six stages. Each is independent and each is measured separately in
`improve/RESULTS.md`; stages 3 and 5 are the two that are not just tuning.

Throughout, `R` is a directory holding `2.construct_network/` and
`3.heter_data/` (the released data prep). Set it once:

```bash
R=/path/to/rundir
PY=python          # needs torch, torch_geometric, sklearn>=1.0
```

Final configurations, so you can skip to stage 4 if you trust the search:

| task | arm | lr | channels | aggr | dropout | graph |
|---|---|---|---|---|---|---|
| LncDrug | `posw2` | 5e-4 | 256 | `min` | 0 | DrugDrug threshold **0.6** |
| MiDrug | `posw2` | 1e-4 | 256 | `min` | 0 | released (0.3/0.3/0.3) |

---

## Stage 1 — Redo the author's hyperparameter search

`find_best_params.py` maximises `(val AUC + val AUPR)/2` over
lr x epochs x channels x aggr on one seed, reading each run's **last** epoch.
`improve/hpsearch.py` runs the same grid with the epoch axis read off a single
trace, a 3-seed mean as the selection statistic, and the arm as a fifth axis.

```bash
$PY improve/hpsearch.py --run-dir $R --task LncDrug \
    --arms base posw2 --seeds 666 0 1 --out $R/9.improve/hp_lnc.csv
$PY improve/hpselect.py $R/9.improve/hp_lnc.csv --paper 0.9834 0.9545
```

140 configs x 3 seeds, ~2 h on one GPU. `hpselect` prints two rankings: `author`
(reproduces `find_best_params.py` exactly) and `robust` (what to use).

**Finding:** `aggr='min'` takes all 10 of LncDrug's top-10 configs. The win is
width — 256 channels against the published 64. The released last-epoch rule
cannot see it, because at 256 the model peaks and then overfits.

## Stage 2 — Axes the author's grid does not have

Two layers, no dropout, no weight decay and RMSprop are untested constants in
the release, and the grid stops at 512 channels.

```bash
$PY improve/hpsearch.py --run-dir $R --task LncDrug --arms posw2 \
    --lrs 0.0005 0.001 --channels 256 512 1024 --aggrs min \
    --dropouts 0.0 0.2 0.3 0.5 --weight-decays 0 0.0001 \
    --seeds 666 0 1 --out $R/9.improve/hp_lnc_reg.csv
```

Also accepts `--layers` and `--optimizers adam adamw`. Worth +0.0002 to +0.0004:
dropout 0.2–0.3 helps slightly, depth 3 loses ~0.003, 1024 channels loses.

## Stage 3 — Tune the graph (LncDrug only)

`heter_data_prepare.py` hard-codes **one** similarity threshold for all three
relations (lines 51, 77, 98), giving mean degrees of 6.6 / 26.8 / 3.9. This is
the only component of the method with no tuning at all, and it is a lever on
S_val specifically: validation is scored by message passing over the *training*
graph, so changing these edges changes the validation prediction without
touching a label.

```bash
# Sweep. Coordinate-wise in both directions plus kNN.
$PY improve/graph_sweep.py --net-dir $R/2.construct_network --task LncDrug \
    --arm posw2 --lr 0.0005 --channels 256 --aggr min --seeds 666 0 1 \
    --grid "0.3,0.3,0.3; 0.3,0.3,0.4; 0.3,0.3,0.6; 0.2,0.3,0.3; 5,5,5,knn" \
    --out $R/9.improve/gs_lnc.csv

# Build the winner as a run-dir.  --verify-against asserts the split is
# element-for-element the released one; do not skip it.
$PY improve/graph_prep.py --net-dir $R/2.construct_network \
    --out-dir $R/gvar_drug06 --lnc 0.3 --mi 0.3 --drug 0.6 \
    --verify-against $R/3.heter_data
```

**Direction is sparser, not denser** — densifying DrugDrug costs 0.0018 val AUC.
At threshold 0.6 the relation is essentially deleted. Adding drug–drug edges
smooths drug embeddings together and destroys the degree structure the metric
reads.

**MiDrug: skip this stage.** All 26 variants land within ±0.0004. The reason is
density — 8.9 training drug edges per miRNA against 0.96 per lncRNA — and
`--thin-train 0.235` confirms it: thinned to LncDrug's density, MiDrug's spread
across graph variants goes 0.0002 → 0.0017 and reproduces LncDrug's ranking
exactly.

## Stage 4 — GNN ensemble, built for spread

Not 10 seeds of one config. `--diverse-by` takes the best config within each
group of the named columns, so the library covers distinct aggregations, arms,
widths and dropout rates rather than 12 near-identical settings. `--snapshots`
keeps each run's top-N validation-AUPR epochs as extra members at no training
cost. Each metric is captured at its own validation-selected epoch.

```bash
$PY improve/multiens.py --run-dir $R/gvar_drug06 --task LncDrug \
    --from-csv $R/9.improve/hp_lnc*.csv \
    --diverse-by aggr arm channels dropout --top-k 16 \
    --seeds 666 0 1 2 3 --snapshots 5 --greedy-steps 40 \
    --dump-scores $R/9.improve/gnn_lnc.npz --out $R/9.improve/mens_lnc.csv
```

Reports `uniform_*` (no fitting on any reported fold — quote these) and
`greedy_on_val` (Caruana et al. 2004 selection, fitted on validation — its
validation figure is not quotable, its test figure is), under both z-score and
rank normalisation.

## Stage 5 — A model class that is not a GNN

Explicit heterogeneous link-prediction features into a gradient-boosted tree.
Nineteen features from the training association matrix `A`, the two similarity
matrices and the cross-ncRNA bridge:

- `log_deg_r`, `log_deg_d`, `pref_attach`
- `collab_r`, `collab_d`, `collab_both` and normalised forms — `S_r @ A`,
  `A @ S_d`, `S_r @ A @ S_d`
- `path3`, `path5` and normalised forms — bipartite path counts
- `svd_dot` — latent factors of `A`
- `nbr_deg_r`, `nbr_deg_d` — degree of the neighbourhood, not the endpoint
- `maxsim_r`, `maxsim_d` — masked maximum similarity to an associated node
- `bridge`, `bridge_norm` — ncRNA → other ncRNA → drug path count, i.e. the
  paper's pan-ncRNA premise as a number

```bash
$PY improve/gbdt.py --run-dir $R --net-dir $R/2.construct_network \
    --task LncDrug --seeds 666 0 1 2 3 \
    --dump-scores $R/9.improve/gb_LncDrug.npz --out $R/9.improve/gb_LncDrug.csv
```

**The leakage guard is the whole difficulty.** A training positive `(r,d)` is an
edge of the matrix its own features come from. Two guards: the similarity
matrices carry a zero diagonal, and training rows get K-fold feature generation
— for fold k every feature, latent factors included, is computed from `A` with
fold k's positives removed. Validation and test rows use the full training `A`.

Run the control once, because it does not fail the way you expect:

```bash
$PY improve/gbdt.py ... --leaky --out /tmp/leaky.csv     # val AUC 0.4997
```

Chance, not 0.99. The leak is training-side only, so the model learns a rule
that cannot fire on a held-out edge and transfer collapses. The guard is not
hygiene; it is the difference between a working model and a useless one.

## Stage 6 — Blend

Rank-normalise both scorers, then a weight chosen on validation.

```bash
$PY improve/blend.py --gnn $R/9.improve/gnn_lnc.npz \
    --gbdt $R/9.improve/gb_LncDrug.npz --task LncDrug
```

Prints the Spearman correlation between the two scorers first, because that is
what predicts the gain: 0.81 on LncDrug, 0.93 on MiDrug, and the LncDrug gain is
correspondingly larger.

The GBDT is **worse than the GNN in every cell alone** and the blend still beats
the GNN everywhere. On LncDrug's validation fold, of 354,482 ranking pairs both
models get 97.51% right, the GNN alone gets 0.95%, the GBDT alone gets 0.57%,
and 0.96% defeat both. Picking the correct model per pair would give 0.9904
against the GNN's 0.9846; the blend recovers 902 of the GBDT's 2,036 unique wins
while giving back 570 of the GNN's 3,382, netting +332 pairs = +0.0009.

---

## Stage 7 — More model classes (this is where the gain is)

`improve/altmodels.py` runs the same features through `extratrees` (bagged
random splits), `logreg` (linear) and two unsupervised heuristics, and reports
the pair-level oracle for every subset so a pool can be judged before it is
tuned.

```bash
$PY improve/altmodels.py --run-dir $R --net-dir $R/2.construct_network \
    --task LncDrug --gnn $R/9.improve/gnn_lnc.npz --seeds 666 0 1 \
    --dump-scores $R/9.improve/alt_lnc.npz --out $R/9.improve/alt_lnc.csv
```

`extratrees` alone reaches val 0.9877 / 0.9713 — above the paper and above every
GNN configuration in this repo — so most of the final number is this stage, not
stages 1–4.

Run the permutation control over **many** seeds, never one:

```bash
for s in $(seq 100 119); do
  $PY improve/gbdt.py --run-dir $R --net-dir $R/2.construct_network \
      --task LncDrug --permute-labels $s --seeds 666 --out /tmp/null_$s.csv
done
```

Null: 0.5025 ± 0.128 (extratrees). A single draw of 0.65 is 1.1 sd from the
mean and means nothing.

## Stage 8 — Score-level fusion

Rank-normalise each member, weights chosen on validation by `(AUC + AUPR)/2` so
one predictor reports both metrics.

| task | weights | S_val AUC / AUPR | S_test AUC / AUPR |
|---|---|---|---|
| LncDrug | extratrees .50 / logreg .25 / gnn .25 | 0.9880 / 0.9727 | 0.9908 / 0.9816 |
| MiDrug | gnn .40 / logreg .30 / gbdt .20 / extratrees .10 | 0.9582 / 0.9227 | 0.9600 / 0.9217 |

Do **not** fuse inside one architecture: `improve/hybrid.py --mode
{none,concat,additive}` shows concatenating or adding the pair features into the
decoder is indistinguishable from omitting them on validation, 0.005–0.015 below
score-level fusion.

## Diagnostics worth running before believing any of it

```bash
# what the benchmark measures: 5 degree features vs the paper
$PY improve/feature_groups.py --run-dir $R --net-dir $R/2.construct_network \
    --task LncDrug --out $R/9.improve/fg_lnc.csv
```

LncDrug degree-only: **0.9831 / 0.9545** against a published 0.9834 / 0.9545.

## What each stage was worth (LncDrug S_val)

| stage | AUC | AUPR |
|---|---|---|
| paper | 0.9834 | 0.9545 |
| author config + `posw2` + 10-seed ensemble | 0.9834 | 0.9568 |
| 1–2. hyperparameter search redone | 0.9839 | 0.9590 |
| 4. per-metric epochs, snapshots, rank-averaging | 0.9841 | 0.9593 |
| 3. similarity-graph sparsification | 0.9847 | 0.9619 |
| 5–6. GBDT blend | 0.9856 | 0.9642 |
| 7–8. **`extratrees` + `logreg` in the pool** | **0.9880** | **0.9727** |

MiDrug S_val: 0.9555 / 0.9183 → **0.9582 / 0.9227**, with stage 3 contributing
nothing and stages 7–8 contributing most of it.

## Caveats to carry into the write-up

- Stages 1–3 and 6–8 all **select on the validation fold**, which is what the
  paper's own Table 2 rests on (`find_best_params.py` maximises a validation
  criterion). S_val is optimistic in the same way; **S_test is the honest
  column** and moves in the same direction in all four cells.
- A 2000-resample bootstrap over fold edges puts the published figure outside
  the 95% interval in **two of eight cells** — LncDrug AUPR on both folds. The
  other six, including all four MiDrug cells, are level-or-better with
  differences below within-fold noise. Claim only the two.
- The folds are small (1263 validation rows, 421 positive), so bootstrap sd is
  0.002–0.006. That bounds the paper's printed figures as much as ours.
