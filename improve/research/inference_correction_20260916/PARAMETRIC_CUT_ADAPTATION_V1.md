# Two-stage parametric-cut ranking: prospective local adaptation

The source is Nitayanont and Hochbaum, *An Effective Flow-based Method for
Positive-Unlabeled Learning: 2-HNC*, arXiv:2505.08212v2 (2025 extended version),
with its KDIR 2024 predecessor and author repository fixed at
`b0a235e053303040253bca8cc879002374b8c6c1`. Source PDFs, code, and retrieval
records are archived alongside this note. This is an adaptation to association
ranking, not a full reproduction of the authors' prevalence-based classifier.

The proposed mechanism is the second parametric minimum-cut stage conditioned
on likely-negative seeds. Its sole primary control is the first-stage cut
ranking, sharing the graph and positive support exactly. Both arms use the
same training-only scalar calibration. Calibration and fixed composition with
the reference are infrastructure and do not count as innovations.

Before any candidate performance evaluation, freeze these choices:

- Use all 605 × 216 RNA–drug pairs, including unobserved pairs, in a fixed
  Cartesian product graph. The frozen, label-independent `kr` and `kd` kernels
  supply top-5 positive neighbor similarities on each biological axis.
  Symmetrize by maximum and normalize each axis to mean weighted degree 1.
  This is transductive inference over a fixed universe; it has no evaluation
  labels and does not depend on query batch composition.
- Use the existing five training folds. For each fold, only positive labels
  outside it become fixed positive seeds. All labels inside the fold are absent
  from seed construction. The stage-one grid is 501 values from −1 to 0; the
  stage-two grid is 501 values from 0 to 1.
- The likely-negative budget equals the number of sampled zero labels outside
  the query fold. Select lowest first-stage ranks; break ties by normalized
  positive affinity, then frozen pair identifier. These are inferred seeds,
  not verified biological negatives. The budget is an engineering choice,
  not an estimate of biological prevalence.
- Return `(502 - first_entry_index) / 501`. These quantities are ranks, not
  probabilities. Generate each training row's rank from its held-out fold.
  Average all five support-fold ranks for evaluation, matching the support
  fraction between calibration and inference. Do not add a full-support fit.
- In each arm fit exactly two coefficients on training OOF ranks, labels, and
  audited OOF reference logits. Standardize rank by OOF mean and population
  standard deviation (floor 1e−8). Minimize mean binary cross entropy of
  `anchor_oof + beta * standardized_rank + intercept`, plus
  `0.005 * (beta**2 + intercept**2)`. Use deterministic BFGS; maximum absolute
  final gradient must be ≤1e−7. No learning-rate or regularization search.
- Apply the calibration to mean evaluation rank, clip its residual to [−2,2],
  and add it to the unchanged strong reference with fixed weight 0.1.
  LncDrug is prospectively the exact reference identity, without any fit.

The author's Python stage-two wrapper overwrites a positive-affinity source
constant. The local implementation retains this term as required by the paper
objective. The C ABI computes `constant + max(0, lambda * slope)`; capacities
are constructed for this actual ABI. Exhaustive small-graph cut enumeration
and a counterexample for the lost constant already passed. The native code
uses global state; a process-local lock serializes calls.

Historical evaluation uses the three already-frozen reference initializations,
but the cut computation and calibration occur once. These are three scoring
contexts, not three independent method fits. Historical advancement requires
positive mean MiDrug AUC and AP differences from both reference and primary
control, with each reference difference positive in at least two contexts.
Then the same recipe must pass three fixed data partitions, with positive mean
differences from reference and control and each reference difference positive
in at least two partitions. Every LncDrug difference must be exactly zero.
Frozen historical outer rechecking is mandatory before any success claim.
All these data views have prior development exposure; none is an independent
test set. The original reserved test source is not accessed.

All parameters, code, inputs, ranks, breakpoints, seeds, calibrations, predictions,
and metrics will be archived. A failed gate stops this version. Only one
parametric-cut mechanism family can be counted if the complete chain passes;
native solver calls, calibrations, identity scoring, and neural fits are counted
separately. The current confirmed innovation count remains zero.
