# Review leading to the conjugate-block test

User scope: miRNA and drug only, no external datasets. Latest completed entity
Mixup experiment failed its fixed gate; do not tune its weight/layer/partner rule.

Reused primary-paper archive, not a new literature search:
Yang et al., **TGSBM: Transformer-Guided Stochastic Block Model for Link Prediction**,
WWW2026, DOI https://doi.org/10.1145/3774904.3792497,
author preprint https://arxiv.org/abs/2601.20646v1 (2026-01-28).
Existing `../new_mechanisms_20260915/tgsbm_export.pdf`, SHA256
`2c5274998c85c24a8040df813d22f5f2930e671e05ca0fa2b526ef2e53374e2a`.
Re-read extracted §3.2 and §4.2–4.4, particularly equations9–16. This is partial
primary reading, not a new full-paper or author-code audit. The prior review
`../new_mechanisms_20260915/READING_NOTES.md` records metadata/source validation.

The paper learns overlapping Bernoulli memberships and Gaussian strengths via
a sparse transformer, with a stick prior and nonlinear decoder. Its likelihood
is over observed positive and sampled negative pairs (eq15), not every unknown
matrix entry. It does not propose the conjugate categorical model below.

Historical `overlap_block_models.py` already adapted TGSBM, and new-partition
validation failed. Thus neither learned communities nor stochastic blocks are
untried families. The feature-group experiment `midrug_class_prior_20261006`
also failed: fixed KMeans16/8 positive group frequencies, Dirichlet10 smoothing,
query subtraction and a neural residual. Continuous molecular similarities,
weighted graphs, kernels and low-rank pair models have likewise been explored.
None should be advertised as new merely by renaming them.

The current narrower distinction is an observed-entry Bernoulli likelihood,
single categorical membership per endpoint, conjugate Beta/Dirichlet priors and
coordinate-ascent variational inference, without transformer, Monte Carlo,
latent strengths, sticks, neural residual training or hard pseudo-labels.
It is a classical model derived here, not an original research novelty claim.
It directly compares learned categories with fixed feature/shuffled categories.
There is no paper-backed assurance of AUC/AP improvement, much less +0.02.

For observed positive/negative sparse matrices P/N and memberships R,D:
block expected counts are R^T P D and R^T N D. Beta shapes are prior+counts;
Dirichlet concentrations are ones+membership column sums. Row logits are
Elog(pi_R) + (P D) Elog(theta)^T + (N D) Elog(1-theta)^T;
column logits use analogous transposed matrices and the updated rows.
The full ELBO includes expected observed-entry likelihood, Beta KL, both
Dirichlet KLs, expected categorical log priors and categorical entropies.
The experiment verifies this independently on tiny cases and checks the lower
bound against exact assignment enumeration with integrated Beta/Dirichlet terms.

Inference uses fitted parameters. Deleting a training observation and refitting
can change memberships, just as it can change neural weights; no exact deletion
invariance is claimed for these learned parameters. No heldout label enters CAVI
or training-ELBO start selection. Source base predictions are immutable reused
artifacts; this experiment does not claim new GPU neural or PLMF replays.

