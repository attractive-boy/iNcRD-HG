# Proper-loss shape review, 2026-10-07

The previous goal turn made concrete progress: six conjugate block CAVI fits and
six fixed-group fits, audited and archived. Its improvement over unmasked10 was
only +.000340593 AUC / +.000514640 AP on inner folds, failing the frozen gate.
The original-test dual +.02 requirement remains unchanged.

## Search scope

Continued nature-academic-search public-API workflow (academic MCP unavailable).
Two broad arXiv requests, a Crossref request, then two focused arXiv requests and
a focused Crossref request each returned 20 records: 120 raw records. Each request
is limited to 20 results, many irrelevant; this is not an exhaustive review.
Broad arXiv range 2025–2026-10-06, focused range 2024–2026-10-06. Crossref likewise
starts in 2024. Query strings, original API responses, authors and dates retained.
No training datasets, sequences, annotations or model weights were fetched.

## Primary readings

1. **Beyond One Epoch: Uncertainty-Weighted Sensitivity Regularization for
   Recommendation Models**, arXiv:2609.34083v1, 28 September 2026.
   https://arxiv.org/html/2609.34083v1
   Read abstract, §4, evaluation protocol and appendix discussion of expected
   Fisher/uniform controls. Eq7 penalizes downstream logit sensitivity to
   TRAINABLE sparse embedding rows using first-epoch frozen inverse information.
   The default rowwise squared-gradient accumulator is explicitly not exact
   Fisher information. This is a partial reading; neither local-influence proof
   nor author code has been fully audited. Current M123 uses shared feature/graph
   encoders, not the same sparse trainable lookup rows; applying this directly
   would need a separate justified uncertainty model. Not implemented here.
   Its reported .0058–.0231 AUC gains do not predict MiDrug gains.

2. **Enhancing Deep Neural Network Reliability with Refinement and Calibration**,
   arXiv:2605.23249v1, 26 May 2026.
   https://arxiv.org/html/2605.23249v1
   Read abstract and §4.1. RefCal pretrains an encoder using supervised
   contrastive learning, then freezes it and trains a classifier using
   classification/calibration losses. This is not an asymmetric proper-loss
   paper. Calibration improvements alone cannot establish our ranking target.
   The theorem connecting their refinement surrogate to contrastive loss was
   not audited, and this paper's workflow is not implemented in this experiment.

3. Mark D. Reid and Robert C. Williamson, **Composite Binary Losses**, JMLR11,
   2387–2422 (2010), official metadata and PDF:
   https://www.jmlr.org/papers/v11/reid10a.html
   https://www.jmlr.org/papers/volume11/reid10a/reid10a.pdf
   The attempted arXiv HTML returned404; retained failure and used official PDF.
   Read §3.2 Theorem1/Corollary2, symmetric-loss discussion and Appendix C's
   noise-robustness definitions/conclusion. Proper partial losses obey
   -l1'(p)/(1-p)=l0'(p)/p=w(p)>=0. Conditional risk derivative is
   (p-eta)w(p). Strictly positive interior weights uniquely elicit the observed
   conditional class probability in the unrestricted population problem.
   This is foundational work, not a new2026 method. Properness does NOT recover
   clean probabilities under label corruption or guarantee finite neural ranking.
   Nonconvexity by itself also does not establish label-noise robustness.

These are three partial primary-paper readings, not three implemented methods.

## Historical exclusions

Already covered: positive/zero-label graph messages, source-conditioned zero
messages, PULL-inspired expected-graph self-training, STORM meta reweighting,
old focal/label smoothing, and robust observed-entry nuclear-norm completion
(sparse-error derivative clipping). These are not new proposed experiments.
The latter clips gradients at a fixed cap; the present proper-loss family changes
the confidence-dependent gradient continuously while coupling both partial losses.
Do not load those historical mixed-task/external-feature experiment runners.

## Selected bounded hypothesis and derivation

Test likelihood gradient shape in the ORIGINAL miRNA-only M123 network using
the same reconstruction10 training. Let p=sigmoid(z), y in {0,1}.

- negative_tail: l0=2p; l1=2[-log(p)-(1-p)], weight w=2/p.
  Logit gradient is 2(1-p)(p-y); the negative-label tail at p→1 saturates.
- positive_tail: reflect labels/probabilities; w=2/(1-p).
  Logit gradient is 2p(p-y).
- brier: 2(p-y)^2; w=4; logit gradient 4p(1-p)(p-y).

All match BCE's logit gradient at p=.5. A BCE-to-loss convex transition preserves
strict propriety, although the neural parameter objective is not convex and loss
scales interact with reconstruction. negative_tail also changes gradients on
positive/easy examples; it is not solely a selective negative reweighting.
No sample is called mislabeled, no labels are changed and no noise rate inferred.
The candidate is negative_tail; positive_tail and brier are matched diagnostic
controls, not extra candidates to select after seeing performance. There is no
guarantee it exceeds BCE; fixed inner experimentation is required.

The next directory is `../midrug_proper_loss_20261007/`. Its README fixes training,
numerical implementation, controls and gate before metrics. No external data or
lncRNA information, and no original validation/test until a frozen gate is passed.

