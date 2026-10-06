# Neighbourhood Gaussian transport: prospective adaptation

2026-09-16, recorded before this candidate's performance screening.

The hypothesis is that **second-order dependence within the incoming message
distribution, compared by Gaussian optimal transport, contains useful MiDrug
ranking information beyond endpoint vectors and the existing strong reference**.
It is not a new feature bank, an alternative tabular foundation model, or another
ensemble recipe. This remains one distributional-geometry mechanism family;
the three distance arms and any later revision cannot be counted separately.

## Sources and what was actually obtained

| Source | Verified evidence | Relevance and limits |
|---|---|---|
| Rui Wang et al., *Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry*, CVPR 2025, DOI [10.1109/CVPR52734.2025.00776](https://doi.org/10.1109/CVPR52734.2025.00776), arXiv:2504.00660v1 | Exact-title Crossref record; full arXiv PDF and extracted text retained in `../bures_normalization_next_20260916`; §2 and §3 read | Gaussian/Bures matrix geometry and handling ill-conditioned covariance matrices. We do **not** implement its batch-normalization algorithm, learned metric parameter M, barycentre, parallel transport, or power-deformation scheme. |
| Ziheng Chen et al., *Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Geometry*, ICLR 2025, arXiv:2407.10484v3 | Full PDF; [official conference listing](https://iclr.cc/virtual/2025/poster/28270) in the retained ICLR page | Rationale for covariance representations, and the warning that matrix-square-root Euclidean geometry is its own metric. It must not be labelled exact Bures transport. No image experiment is reproduced. |
| Pengfei Zhu et al., *CCP-GNN: Competitive Covariance Pooling for Improving Graph Neural Networks*, IEEE TNNLS, April 2025 issue, DOI [10.1109/TNNLS.2024.3390249](https://doi.org/10.1109/TNNLS.2024.3390249) | Exact Crossref title, authors, DOI and issue verified; no matching arXiv entry; GitHub search retained | Related graph covariance pooling. Full text and author identity of the repository have **not** been verified, so no competitive-pooling formula or performance claim is attributed to or copied from it. |
| Michael Scholkemper et al., *A Wasserstein Graph Distance Based on Distributions of Probabilistic Node Embeddings*, arXiv:2401.03913v2 | Full PDF and text, abstract and graph-distance formulation read | Earlier graph/distribution context, not a new 2026 algorithm and not our supervised neighbourhood construction. Its GMM graph matching is not implemented. |

Public APIs were used because no literature MCP is mounted. The initial Crossref
covariance query returned HTTP 429; the failure and the successful exact-title
follow-up are both retained. PubMed preflight timed out. Source metadata is not
treated as evidence of performance on this project. No claim of global novelty,
complete paper reproduction, or permission to redistribute publisher PDFs is made.

## Fixed local model and controls

The common two-layer, width-64 typed sparse attention encoder uses the original
attributes and the query-masked physical graph. It has no new external data or
learned node-ID table. For all 605 miRNA and 216 drug endpoints, independently
of the requested query list, the incoming messages (including self loops and
distinct relation observations) are projected into four common eight-dimensional
spaces. Each space has a population mean and a centred population covariance.
Means and covariance sums are computed in FP64. These are empirical statistics
of latent messages, **not** measured biological covariance or calibrated uncertainty.

Each endpoint role has a learned 8×8 linear map, initially the identity. A fixed
0.01 I ridge is added after congruence, preserving positive definiteness even if
the role map becomes singular. The four Gaussian comparisons each return the
mean squared distance and a covariance distance. Log1p of these values divided
by eight, together with the two covariance traces, goes into a common residual
decoder alongside the usual endpoint vectors. The decoder is unconstrained:
drug response is not assumed to be monotonic in geometric similarity.

- `full`, the fixed primary: `tr(Cr + Cd) - 2 tr((Cr^(1/2) Cd Cr^(1/2))^(1/2))`.
- `diagonal`, mandatory control: discard cross-coordinate covariance after the
  role maps, using the squared difference between standard deviations. Means,
  traces, encoder, projections and all parameter tensors remain present.
- `root_euclidean`, mandatory control: `||Cr^(1/2) - Cd^(1/2)||_F^2`, retaining
  complete covariance while removing the joint transport square root. The
  distances agree for commuting covariances but can differ otherwise.

Every arm has identical initial stored/trainable parameters and random-number
consumption. Forward and backward operation counts differ across matrix-function
arms; equal FLOPs or speedups are not claimed. This is a learned distributional
decoder, not merely inserting a normalization layer. Related older transport,
hyperbolic and multiset-pooling failures remain part of the research history.

## Data, numerical and performance rules

The five query folds are removed from physical target edges before reverse arcs
or counts are made. The same audited nested OOF strong reference supplies the
training offset; no evaluation outcomes are passed to setup or fixed-step fits.
The new decoder learns `BCE(anchor_oof + residual, y)`; evaluation adds the fixed
0.1 residual to the unchanged reference. Sampled zero labels are unobserved
associations, not established biological negatives.

LncDrug is prospectively the exact reference identity, with zero new parameters
or training. Its identity scoring records are not neural fits. This task-specific
deployment policy is explicit rather than selected after observing its metrics.

The SPD square root is exact up to floating-point eigendecomposition. Its custom
first derivative solves the Sylvester equation with denominators sqrt(λi)+sqrt(λj),
which remains finite at repeated eigenvalues. Second derivatives are not used or
claimed. Nonfinite or nonpositive SPD inputs fail rather than being silently
repaired; only tiny negative Bures distances from cancellation are clamped to
zero within the frozen relative 1e-8 tolerance. Model, optimizer and completed
step count are retained on a training failure.

After seven CPU mathematical/execution tests, three five-step MiDrug GPU probes,
three LncDrug identities and all-fold input isolation audits, freeze the formal
historical protocol: learning rates 0.001 and 0.003, 400 AdamW updates, weight
decay 0.001, checkpoints every 20 steps, seed 666; choose by average global
ROC AUC and sklearn average precision. Both primary metrics must improve over
the strong reference and both mandatory controls. Only then run seeds 0 and 1;
the same three-initialization gate, three fixed development partitions and
separately frozen historical outer recheck remain required. No ridge, dimension,
head-count, depth, input or fusion-weight sweep is allowed in this version.

The development partitions and old outer have previous research exposure and
are not independent cohorts. No original retained test file is opened by this
study. A successful screen alone does not count as a validated new technique.
