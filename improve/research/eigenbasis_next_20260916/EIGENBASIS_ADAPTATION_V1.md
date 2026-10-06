# SOAP follow-up: coordinate-wise second moments in a learned eigenbasis

This study is motivated by **already observed** SOAP control results from the
failed KL-Shampoo study. It is not an independent screen, does not revise the
old primary, and cannot be counted as an extra optimizer-family innovation.
The old screen remains failed. New confirmation, fixed partitions and the
frozen outer recheck are still mandatory before any success claim.

Sources are the actually read SOAP arXiv:2409.11321v2 Algorithms 2--4,
implementation discussion, Section 7.2/Adafactor comparison, and the pinned
`nikhilvyas/SOAP` source at `a1e553530fde97d0e6b307d7c82ac6d38b072340`.
The full acquisition and numerical details remain in
`../preconditioning_next_20260916/PRECONDITIONING_ADAPTATION_V1.md` and its
immutable source manifests. Publisher verification was incomplete (HTTP 403).

The narrower hypothesis is that Adam's coordinate-wise second moments, kept in
the slowly updated eigenbasis of gradient covariance, improve graph residual
ranking over both a separable second moment and ordinary AdamW.

The three fixed arms are:

- `full`: exactly the already implemented SOAP arm, relabeled explicitly by a
  new protocol. No optimizer, graph, RNG, training or prediction operation changes.
- `separable_variance`: same covariance EMA, eigenbasis, original-space momentum,
  variance state, bias correction, variance permutation, numerical constants,
  initialization and update budget. Only replace V in the update denominator by
  `V_hat = V.sum(1)[:,None] * V.sum(0)[None,:] / V.sum()`.
  This is the nonnegative rank-one row/column factorization used in the paper's
  idealized Adafactor comparison, with the same adaptive eigenbasis. It preserves
  both marginals. We retain the full V state for matched bookkeeping and make
  no memory-saving claim. It is not a reproduction of DistributedShampoo.
- `adamw`: exactly the already implemented AdamW baseline with betas (0.9,0.98).

Full must beat both controls on mean MiDrug global ROC AUC and sklearn AP,
as well as the unchanged strong reference. LncDrug remains exact identity.
All numerical, fitting, selection and downstream gates match the parent:
one initialization gradient without updates, then 400 updates, five cyclic
query-deleted supports, lr=0.001/0.003, 20-update evaluation intervals, fixed
0.1 raw-residual composition, and cuda:1 only.

The two SOAP and two AdamW MiDrug seed-666 fitted models are reused byte for
byte, after verifying equal recipes, all input arrays, saved files and exact
prediction replay under the new wrapper. Four LncDrug identity records are
also reused; they are not trained models. Thus this screen performs **two new
MiDrug fits**, both for the new separable-variance control. No saved checkpoint
is reselected in the reused trajectories. Seeds 0/1, if eligible, require six
new MiDrug fits. The protocol does not silently count reused fits as new work.

The new training, prediction and run function bodies are text-identical to the
sealed parent. Numeric and CPU graph-trajectory checks establish the new label
mapping's equivalence for SOAP and AdamW. The only new 25-update GPU short fit
exercises separable variance across two QR updates. The parent graph support
audits are retained with verified hashes rather than re-counted as new audits.

Observed parent results are exploratory evidence only. Choosing this narrower
hypothesis after inspecting the old screen increases selection uncertainty;
the remaining partitions are also previously used development partitions,
not a new independent biological cohort. No global novelty or significance
claim follows from this adaptation or from a positive single screen.
