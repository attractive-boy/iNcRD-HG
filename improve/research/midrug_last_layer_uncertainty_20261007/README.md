# Fixed last-layer conditional uncertainty experiment

Hypothesis: pair-specific covariance from training-score variability may improve
MiDrug ranking beyond the matched converged point estimator and curvature-only
Laplace integration. Inspired by Ribbon arXiv:2606.27269v1 §2.3/3.1/3.3; this is a
Gaussian moment approximation restricted to a frozen last layer, not a full
Ribbon replication or exact posterior. Historical Copula Laplace was already tried.

## Frozen protocol

- Only existing miRNA and drug inputs; no external training information.
- Original training's three immutable folds, seed 42. Restore the verified
  selected `unmasked10` checkpoint per fold; freeze all parameters and BN buffers.
  Extract the 256 inputs to lin3 using a forward pre-hook in evaluation mode.
- Scale each feature by its current training RMS, floored at 1e-6. Do not center.
  Append an intercept of 1. Use the same transform for heldout pairs.
- Fit 257-dimensional logistic regression with objective
  `mean(BCE(X theta,y)) + ||theta||^2/(2n)`. This corresponds conditionally to an
  isotropic standard-Normal prior on *all* scaled coefficients including intercept.
  Precision 1 is fixed, not optimized. Start at zero, double precision Cholesky
  Newton, Armijo coefficient 1e-4, at most 100 updates, gradient infinity norm
  <=1e-9. No validation-based stopping or candidate checkpoint selection.
- At the solution, H is the exact Hessian of that mean objective;
  `g_i = (sigmoid(x_i theta)-y_i) x_i`, B is their centered second moment.
  `C_laplace=H^{-1}/n`, `C_sandwich=H^{-1} B H^{-1}/(n+1)` (alpha fixed at 1).
- The same theta provides four readouts: point sigmoid; Gaussian Laplace;
  Gaussian sandwich; and Gaussian smoothing with constant variance equal to
  the **training** mean of x C_sandwich x. Conditional marginal integration
  uses 64-point Gauss–Hermite quadrature. 128 points is a numerical check only.
- Only `sandwich` is eligible. Its mean inner AUC and AP must each exceed
  `control`, `unmasked10`, `point`, `laplace`, `constant` by at least .002.
  No averaging with earlier controls, scale sweep, picking controls or outer
  evaluation on failure. This gate is not the user's original-test +.02 target.
- Store coefficients, scaling, Hessian, both covariances, training convergence,
  predictions, immutable source hashes and replay checks.

## Interpretation boundaries

The conditional Laplace approximation applies to a converged last-layer objective.
It ignores uncertainty in the trained feature extractor and checkpoint selection.
The sandwich is the covariance of an edgewise influence approximation with fixed
features/graph/penalty; a Gaussian moment approximation adds another assumption.
Shared entities mean graph edges are not generally independent observations; no
calibration, independent-sample validity or full-network posterior is claimed.
Constant variance is a monotone-ranking control. BCE/Brier changes alone do not
satisfy the AUC/AP goal. PLMF remains protected by existing immutable artifacts.
The source checkpoint was selected on the same inner holdout, which has also
been reused across development rounds. These inner metrics are development
evidence, not an independent confirmation or a new unbiased generalization test.
