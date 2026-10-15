# Observed-entry logistic matrix completion

Goal remains the original dual +0.02 MiDrug test target, with unchanged PLMF protection.
Only independent frozen miRNA/drug inputs are permitted. No external observations are added.

Literature motivation: IRPCA, JCIM 2025, DOI 10.1021/acs.jcim.4c02385, discusses
nonconvex low-rank decomposition and noise; only its abstract and bibliographic metadata
have been verified. This experiment is an independent convex adaptation, not IRPCA reproduction.
No claim is made about using the authors' formulas, neural predictor, or GIP integration.

Fit an unrestricted 605 x 216 residual score matrix X, with observed-entry logistic loss,
nuclear norm shrinkage, and optional Laplacian smoothing on the two original similarity
topologies. The original target association edges do not define these Laplacians.
Unobserved entries have no likelihood target: missing pairs are not assigned negative labels.
Regularization does propagate information into unobserved cells.

Noise ablation: an L1-penalized per-observation logit correction is analytically eliminated
from the loss. Its derivative equals a clipped logistic residual. This is convex and is not
the nonconvex low-rank method in IRPCA. A high correction is not proof of a wrong label.
The correction is excluded from inference. The scalar prior is fitted using training labels;
for mixture comparability, inference scores use the nominal training log-odds plus X.
This prior alignment is not a claim of probability calibration.

This differs from prior neural latent8/latent32 fusion: there is no chosen latent dimension,
factorized nonconvex optimizer, or joint neural BCE; matrix rank is selected through
singular-value thresholding, and all fits stop using training convergence alone.
The matrix is a supervised predictor fitted to train labels, not an in-sample feature fed to
M123. Predictions are blended only for unseen internal pairs with fixed weights 0.1/0.25.
The inference blend retains the unchanged M1/M2/M3 model plus reconstruction training.

Predeclared 2 nuclear fractions x 2 graph weights x 2 error models x 3 folds = 24 fits.
Compare standalone scores descriptively and blends against unmasked10 and pure M123.
Both mean paired AUC/AP gains must reach 0.002 versus each control before outer validation.
No original validation/test set is loaded. The original final target is not relaxed.

Optimization: monotone restarted accelerated proximal gradient, float64 CPU SVD,
global safe step 1/(0.25+2*graph_weight), max 3000 steps, relative proximal-gradient
stationarity tolerance 1e-5. No held-out checkpoint selection or training-score stacking.
Final nonconverged fits cannot be promoted without further solver verification.

Run with ncRNAdrug Python, OMP/MKL/OPENBLAS_NUM_THREADS=2:
check.py -> launch.py -> audit.py -> summarize.py.
