# Analytic graph-attention covariance: prospective association-ranking study

Source: Ayday, Yang and Ghoshdastidar, *Gaussian Process Limit Reveals Structural
Benefits of Graph Transformers*, arXiv:2603.17569v1, March 2026. This is a
preprint. The full paper is archived as `2603_17569.pdf`. The author-linked
Figshare URL returned HTTP 403; the response and retrieval record are preserved.
The local code is an independent implementation of the equations, not an
execution or verified reproduction of the author code.

The mechanism is the linear-attention covariance from Corollary 8:

`Sigma = K * (A @ K @ A.T) + A @ (K * K) @ A.T`,

where `*` is elementwise multiplication. This retains correlations between
attention scores at different source/destination pairs. The primary control is
ordinary graph convolution covariance `Sigma = A @ K @ A.T`. Both receive the
same input covariance, support graph, two layers, output transformation,
conditional regression and residual calibration. No neural weights are trained.
The infinite-width/head derivation motivates the covariance, but the local
normalizations change the original architecture. No claim is made that this is
an exact softmax-attention GP, a biological causal model, or a reproduction of
the paper's node-classification benchmarks.

All choices below are frozen before candidate performance is computed:

1. Four type blocks use the original frozen entity attributes. Within a type,
   normalize feature rows and use `0.5 * (X @ X.T + 1)`; initial cross-type
   covariance is zero. Normalize its diagonal to one. This constant feature
   also handles empty feature rows. No context76 columns are added.
2. For each of the five existing training folds, remove all target facts in
   the fold before symmetrization or degree normalization. Retain the union of
   other physical edges, add self loops, and use symmetric degree normalization.
   Relation identity is represented by the initial type blocks; edge types are
   merged for this local homogeneous propagation. Query labels are never read.
3. Apply exactly two covariance layers. In both arms apply the analytic ReLU
   covariance (paper equation 2) and normalize marginal variances after each
   layer. These are explicit local architectural choices. The source's
   oversmoothing results under population block-model assumptions are not
   presumed to hold for this biological graph.
4. Form the typed pair kernel as the product of the final RNA–RNA and
   drug–drug covariances. Condition on support labels mapped to −1/+1 by the
   exact FP64 solve `(K_support + 10 I) alpha = y_signed`. The maximum absolute
   solve residual must be at most 1e−7. This is GP regression on binary targets,
   not a Bernoulli likelihood or calibrated probability posterior.
5. Calculate the full 605×216 posterior mean as `K_r @ alpha_matrix @ K_d.T`.
   This is algebraically equal to querying every pair individually, and makes
   query batching irrelevant. Each training row's OOF score comes only from its
   held-out support configuration; evaluation averages all five means. Kernel
   conditioning uses binary support labels, without potentially contaminated
   support OOF-reference residuals.
6. Share the previously tested two-parameter scalar residual calibration with
   the cut study: train on OOF scores and labels with the audited OOF reference
   as fixed logit offset; mean BCE plus 0.005 times coefficient squared norm;
   OOF score standardization; BFGS gradient tolerance 1e−7. Clip the calibrated
   residual to [−2,2] and add it with fixed weight 0.1 to the unchanged reference.
   Calibration and reference composition are not new mechanisms.
7. LncDrug is prospectively the exact strong-reference identity, with no new
   fit. MiDrug must improve both mean global AUC and AP over the reference and
   GCN-GP control, with each reference improvement positive in at least two of
   three reference contexts historically and two of three later partitions.
   Frozen outer checking remains mandatory before any success count. All
   views have prior development exposure; the original reserved test is unused.

The method is deterministic. Three historical reference initializations produce
three scoring contexts, not three independently fitted GP models. Count kernel
solves, scalar calibrations, identity scores and neural training separately.
Depth, regularization and residual weight are not selected on evaluation labels.
A failed joint gate stops this version; a better control is not renamed as a
successful attention method.

Four unit checks already passed: direct four-index covariance contraction and
PSD; exact tensor-kernel conditioning against an independent primal-feature
solution; nonlinear normalization and permutation; query deletion before reverse
edges and degrees. A real training-only resource probe for both arms used about
1.58 GB peak allocated GPU memory and obtained solve errors below 5e−14. It did
not read evaluation labels or calculate performance, and is not a formal fit.
