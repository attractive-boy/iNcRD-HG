# Representation interpolation review

Previous goal turn was progress: three converged last-layer fits, twelve readouts
replayed and a negative frozen selection. This review changes the next hypothesis,
not the required original-test dual +.02 outcome.

## Current-state and predecessor evidence

The miRNA name/scaffold groups were already evaluated in
`midrug_mirna_only_20261006/groups.py` and its REPORT. Names are not a new input.
Do not use the later external-sequence provenance stored with the old release.
The pure-miRNA pair-perturbation experiment already compared random/adversarial
local representation noise and failed. Mixup is distinct from additive noise:
it interpolates two supervised examples and their targets, with gradients to both.
It is not a new invention or a new family of molecular generative modeling.

`mean_fit_diagnostic.json` reuses immutable last-layer experiment feature/logit
files only. In every inner fold, converged conditional MAP fitting improves
training AUC (+.00639 to +.00779) but reduces heldout AUC (−.00411 to −.00578).
This supports testing a generalization regularizer, but does not uniquely diagnose
the encoder, prove causal overfitting, or exclude effects of the changed penalty.

## Search and primary reading

Applied nature-academic-search/multi-source-search using public APIs because the
academic MCP is absent. API preflight 3/3 passed. Two arXiv searches plus Crossref
returned 4+19+20=43 records, 43 deduplicated. Several hits are irrelevant; they
are not counted as useful evidence. Crossref permits 2024 references and the
arXiv queries target 2025 through 2026-10-06. Primary HTML and fetch hashes retained.

1. **Warmer for Less: A Cost-Efficient Strategy for Cold-Start Recommendations at
   Pinterest**, arXiv:2512.17277v1, 19 December 2025.
   https://arxiv.org/html/2512.17277v1
   Read §3.4–3.5, §4.1–4.2.1, §4.2.6, §4.3.4 and conclusion. Eq.7 mixes interaction
   embeddings and binary labels with the same Beta coefficient. Eq.9 combines
   original and mixed BCE. Their implementation uses random distinct examples;
   the text reports restricting warm/cold partners can hurt diversity. This is
   evidence to keep an unrestricted control, not support for assuming that our
   same-drug restriction is better. Their production data, multitask head,
   residual feature branch/MMD losses and online metrics are not used here.
   Class-specific flattening in foundational theory must not be confused with
   this paper's descriptive claim of higher overall feature rank.
2. **Enhancing Drug Discovery: Autoencoder-Based Latent Space Augmentation for
   Improved Molecular Solubility Prediction using LatMixSol**,
   arXiv:2506.00223v1 (metadata publication date 30 May 2025).
   https://arxiv.org/html/2506.00223v1
   Read abstract and §3.4–3.5 interpolation/settings, limitations. It uses spectral
   clusters, a learned decoder, Uniform[0,1] mixing and interpolated solubility.
   It does not enforce explicit chemical constraints. Lipschitz continuity alone
   does not imply the claimed near-affine reconstruction identity, so this is
   not a guarantee of biochemical validity. We borrow no data or molecular
   generator, and do not call interpolated MiDrug representations real samples.
3. Verma et al. **Manifold Mixup: Better Representations by Interpolating Hidden
   States**, arXiv:1806.05236v4, foundational ICML 2019 method.
   https://arxiv.org/html/1806.05236v4
   Read introduction, §2 and beginning of §3/3.1. Gradients flow through both
   source representations and the entire encoder. Alpha=1 is Uniform[0,1].
   A shared minibatch coefficient is explicitly supported. Theory is idealized;
   it is not a guarantee of AUC/AP gains on this graph task. Unlike its randomized
   layer selection, this experiment fixes the original M123 concatenation layer.

These are partial primary-paper readings. Author code, appendix proofs and full
benchmark replication were not audited. No new data or pretrained weights fetched.

## Concrete next test

New directory `../midrug_entity_mixup_20261007/`: drug-sharing manifold mixup versus
unrestricted mixup, each using existing reconstruction weight10 and the same M123
initialization, three fixed inner folds. Both preserve empirical target prevalence
using a permutation and one shared coefficient, with a common singleton policy.
The drug restriction is a testable task-specific hypothesis, not a paper result.
No synthetic graph edges, no new nodes, no external annotations, no inference-time
augmentation. Full fixed training/progression criteria are in the experiment README.
