# Observed-entry conjugate latent block experiment

Hypothesis: an explicitly regularized, categorical co-clustering likelihood can
extract useful training-association structure missed by fixed feature clusters.
This is a bounded classical statistical test, not a new block-model family or a
replication of TGSBM. No external data, lncRNA information, or new weights.

## Protocol fixed before inner evaluation

- Use the existing three folds of original MiDrug training, unchanged. Only
  observed current-training labels enter the likelihood. Unobserved entries are
  missing, never negatives. Features enter only through existing label-free
  KMeans16/8 groups (original miRNA expression and original drug ChemBERTa).
- Row and column categorical memberships have 16 and 8 categories. Mixture
  proportions independently have symmetric Dirichlet(1,...,1) priors. Each
  block probability has an independent Beta(0.5,0.5) prior. No degree correction.
- Mean-field q is categorical per entity, Dirichlet per side, Beta per block.
  Initialize memberships at 0.95 one-hot + 0.05 uniform. Two starts: archived
  feature groups and archived size-matched shuffled groups. No random restarts.
  Optimize globals first, then sequentially update all row memberships, all
  column memberships, then conjugate globals. Use float64 full-batch CAVI.
- Stop when maximum membership change <=1e-8 and relative ELBO change <=1e-10
  for three consecutive sweeps; maximum 2000 sweeps. Check full ELBO monotonicity
  within 1e-8 + 1e-12 abs(previous ELBO). Nonconvergence invalidates this experiment
  for promotion; do not increase budget or access heldout metrics on that failure.
- Per fold choose the converged start with larger TRAINING ELBO (feature wins
  an exact tie). This is approximate inference with local optima, not exact
  Bayesian model evidence selection. Freeze all six fitted states and all three
  start choices BEFORE producing any heldout prediction or metric.
- Fixed-feature and fixed-shuffled controls keep HARD memberships and fit only
  conjugate globals from the identical positive/negative observations and priors.
  Six iterative fits plus six closed-form fits; no neural network retraining.
- Prediction averages block posterior means under both entity membership
  probabilities. Fitted memberships and Beta parameters are trained parameters,
  not graph statistics recomputed per query. They are not leave-one-observation-
  out posteriors. All heldout queries are absent from their fitting observations.
- Fuse each block readout with the immutable unmasked10 probabilities using
  logit(p_fused)=logit(p_base)+0.1*(logit(p_block)-logit(p_global)). Here p_global
  is the training Beta(0.5,0.5) posterior mean. The 0.1 coefficient is fixed;
  no fitting or search. Exact 0/1 base probabilities remain endpoints. Block
  probabilities are clipped only at 1e-12 for numerical safety. The global
  readout gives exactly the base prediction (identity control).
- Only the ELBO-selected learned model is eligible. Mean inner AUC AND AP must
  each exceed control, unmasked10, fixed_feature fusion and fixed_shuffled fusion
  by >=0.002. Standalone block metrics and unselected-start diagnostics do not
  create additional eligible candidates. No original validation/test on failure.
- Archive independent scalar likelihood/count/ELBO checks, an exact tiny-model
  log-evidence bound, coordinate checks, convergence, parameter/prediction replay,
  query-order/batching/category-permutation invariance, and protected PLMF hashes.

These inner folds and neural checkpoint selections have been used repeatedly for
development. The inner gate is not independent confirmation or final success.
The actual target remains original-test mean seeds42/43/44 AUC>=0.9733929274189016
and AP>=0.933683373732677, with both PLMF datasets preserved.

The observation likelihood is conditional on the sampled benchmark entries;
block means are not estimates of population biological prevalence. Group labels
are latent statistical categories, not verified miRNA families or drug classes.

