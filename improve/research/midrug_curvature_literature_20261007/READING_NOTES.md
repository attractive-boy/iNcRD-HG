# Curvature and conditional uncertainty review — 2026-10-07

Workflow: nature-academic-search / multi-source-search. Academic MCP is unavailable;
public arXiv, Crossref and PubMed API preflight passed. Discovery returned 47 records,
47 after deduplication. Discovery metadata alone is not evidence of method efficacy.
No datasets, annotation files, sequences or pretrained weights were downloaded.

## Primary sources actually inspected

1. Gibson, G., Tipton, J., Rumsey, K. & Klein, N. **Ribbon: Scalable Approximation
   and Robust Uncertainty Quantification**. arXiv:2606.27269v1 (25 June 2026).
   https://arxiv.org/html/2606.27269v1
   Read introduction, methods §2.1–2.3, properties §3.1–3.3, baseline implementation
   §4.1, classification results §5.3/Table 4, discussion. Appendix proofs and author
   code were not audited. It is a preprint, not a verified peer-reviewed publication.
2. **From Deep Additive Kernel Learning to Last-Layer Bayesian Neural Networks via
   Induced Prior Approximation**. arXiv:2502.10540v1 (14 February 2025).
   https://arxiv.org/html/2502.10540v1
   Read the additive GP construction, induced prior approximation, inference and
   training (equations 5–11). Its Laplace *kernel* is not a Laplace posterior
   approximation: inference uses mean-field variational Gaussian weights and ELBO.
   It requires a different learned representation/kernel basis; it is not evidence
   for simply integrating uncertainty around our existing last linear layer.
3. **Do Existing Preconditioners Improve Biomedical Tabular Foundation Learning?
   An Empirical Study on TabPFN Optimization**. arXiv:2609.25013v1 (2026).
   https://arxiv.org/html/2609.25013v1
   Read abstract, results discussion §4.1 and conclusion §5. Authors report that
   original AdamW outperforms their generic preconditioners across 59 datasets.
   This is context against assuming extra curvature machinery guarantees gains,
   not a reproduced conclusion for MiDrug. Detailed tables/code not audited.

Raw public HTML, fetch status/time/hashes and readable extractions are retained.
The rest of the search results were screened as metadata/abstracts only.

## What Ribbon actually supports

For mean loss L, fixed penalty r, stationary penalized estimate theta, per-example
loss gradients G and H_r = Hessian(L+r), equation 2.3 gives the influence update
`delta = -H_r^{-1} sum_i (w_i - 1/n) g_i` for Dirichlet weights.
With alpha=1 and *centered* `B=G_centered.T G_centered/n`, covariance is
`C=H_r^{-1} B H_r^{-1}/(n+1)` (§3.1/3.3). The penalty stays fixed under reweighting.
This is a covariance of a first-order refitting approximation, not an exact
Bayesian posterior. A Gaussian having this covariance is an additional moment
approximation; it is not the finite-sample Dirichlet distribution.

Ordinary conditional Laplace uses `(n H_r)^{-1}` when the summed likelihood plus
Gaussian-prior penalty is the optimized objective. An early-stopped neural head
need not be stationary for that objective. We must refit a convex last layer to
stationarity and compare uncertainty methods to that *same* fitted point estimate.

Ribbon's MNIST Table 4 does not demonstrate an AUC/AP gain. Its variants closely
track Laplace (accuracy about 85.8%); the independently trained ensemble wins
accuracy/Brier/NLL in that experiment. The main hypothesis for us is narrower:
pair-specific conditional variance could reorder uncertain MiDrug predictions.
Constant-variance smoothing is monotone and cannot improve ranking in exact math.

## Distinction from existing project experiments

The historical CopulaLSP adaptation already used differentiable Newton/probit
latent inference and a Laplace integration. It failed fixed-partition confirmation;
we do not call Bayesian/Laplace ideas untried. Do not execute its historical
mixed-task loaders. The present design uses the strict miRNA/drug loader, fixed
M123 embeddings, convex Bernoulli last-layer fitting, and training-gradient
covariance instead of learned inter-edge latent dependence/Rasch support offsets.
Previous fusion-head fine-tuning selected stage zero; this experiment explicitly
includes a converged point-estimate control, so no gain is automatically credited
to uncertainty rather than to refitting.

## Proposed bounded test

New directory: `../midrug_last_layer_uncertainty_20261007/`. One fixed convex fit per
inner fold; same fit generates point, Laplace, Gaussian sandwich and constant
variance predictions. No prior/temperature sweep; no outer validation/test.
Only Gaussian sandwich is eligible for promotion, under the predeclared gate.
All conclusions are conditional on the frozen learned features and graph;
edgewise Dirichlet weighting does not model shared-entity dependence, graph
re-estimation, encoder uncertainty or checkpoint selection. No coverage claim.
