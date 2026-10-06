# Auxiliary-graph reference residual: revision within the analytic-attention family

The first analytic-attention version failed its historical mechanism comparison.
Its immutable result is in `results/attention_gp_rank_v1_20260916`. This revision
was motivated by that observed failure and is not an independent screening
exercise. It does not create a second innovation family.

Retain arXiv:2603.17569v1 Corollary 8, the type-block feature covariance, symmetric
graph normalization, two ReLU covariance layers and unit marginal variance from
version 1. The author Figshare URL returned 403; the implementation remains an
independent equation-based adaptation, not a verified author-code reproduction.

Make two explicit changes, frozen before new performance evaluation:

1. Remove **all** current-task MiDrug association edges before computing either
   full attention or GCN-GP covariance. Keep auxiliary physical relations and
   original entity attributes. Thus every training label, including labels of
   rows later used as kernel support, is absent from the covariance construction.
   Flipping all MiDrug labels leaves the covariances bitwise identical in the
   unit check. This also avoids using a training query label indirectly through
   another row's graph representation.
2. Fit the tensor-product kernel directly to the reference residual using
   convex kernel logistic regression. For `f = K alpha`, minimize
   `sum BCE(anchor_oof + f, y_train) + 0.5 * 10 * alpha.T @ K @ alpha`.
   The fixed OOF reference is the previously audited nested cache. Use a damped
   FP64 Newton solve of `I + sqrt(W) K sqrt(W) / 10`, without inverting K.
   Require maximum dual stationarity residual
   `abs(y - sigmoid(anchor_oof + K alpha) - 10 alpha)` at most 1e−7, with at most
   50 Newton iterations and 25 dyadic line-search trials per iteration.

This is a MAP function for a Bernoulli likelihood with a GP kernel prior.
There is no claim of exact posterior marginalization. No scalar calibration is
fitted. The precision is fixed at 10, depth at two, residual bound at ±2 and
reference-composition weight at 0.1. The full 605×216 residual field is computed
as `K_r @ alpha_matrix @ K_d.T`; prediction uses the same function for each
query regardless of order or batch. Reference evaluation logits stay unchanged.

The sole primary control retains the same auxiliary graph, features, likelihood,
precision, solver and output formula, replacing the linear-attention covariance
with ordinary graph-convolution covariance. LncDrug is an exact reference
identity without any new fitting. Historical and three-partition gates remain:
both mean global MiDrug AUC/AP must beat reference and primary control; each
reference difference must be positive in at least two of three contexts or
partitions; LncDrug must be identical. Frozen outer checking is mandatory before
success. Three historical reference initializations are scoring contexts for one
deterministic fit per arm, not three independent fits. All evaluation surfaces
have prior development exposure, and the original reserved test is untouched.

Three new checks passed: agreement with an independently optimized primal
logistic problem; full-universe prediction and observation-order invariance;
complete target-edge removal and label-independence. Both real training-only
resource probes converged in three Newton iterations, with dual residuals below
5e−10, matrix replay errors below 2e−14 and about 2.45 GB peak GPU allocation.
Probe outcomes were not scored and do not count as formal fits. The version-1
covariance tests and source remain unchanged.
