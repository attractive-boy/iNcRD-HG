# Encoder/consumer update-order intervention

The previous goal turn was progress: nine proper-loss fits and full selected/final
state audits, 161 archive hash checks. Its fixed candidate underperformed unmasked10
by -.000403437 AUC / -.000722359 AP on the three inner folds. Do not tune that loss.

This continuation reuses the previously downloaded primary paper, not a new search:
**Beyond One Epoch: Uncertainty-Weighted Sensitivity Regularization for Recommendation
Models**, arXiv:2609.34083v1 (28 September 2026),
https://arxiv.org/html/2609.34083v1
HTML SHA256 `efc7e8fdaca264ada4026894b06a6cba9abaad92dd4b258cb2bf06dcdf46b5aa`.
Source is `../midrug_loss_shape_literature_20261007/raw/uwsr.html`, with metadata,
authors and original public API response retained in that review archive.

New reading scope: §3.2's embedding-uncertainty assumptions, §3.3 update-order
experiment and Table1, and AppendixB.1 formulas8–9. Earlier reading covered §4.
No author-code or full proof audit, and no new data or pretrained weights.

The paper's §3.3 performs two forward/backward passes per minibatch. Embedding-first
updates sparse lookup rows, rescores the same examples, then updates the consumer;
consumer-first reverses the order. Both preserve the number of updates per parameter.
The authors report that embedding-first increases train advantage and sensitivity,
while consumer-first remains close to ordinary training. They explicitly leave the
exact contribution to overall multi-epoch degradation for further work (§6).
Their result does not imply that consumer-first will exceed ordinary M123 training.

Direct UWSR transplantation is not justified here: original M123 has shared feature
and graph projections, query-corrected endpoint/path states and M3 projections,
not the paper's independent trainable sparse embedding table. A damped inverse GGN
being positive definite alone does not establish a well-calibrated posterior.
Squared minibatch gradients can cancel occurrence gradients; AppendixB.1 discusses
the extra assumptions, shrinkage and normalization. We do not call original RMSprop
state a measured row posterior or implement a made-up uncertainty approximation.

The selected local experiment instead transplants only the explicitly stated
update-order intervention. Representation block: encoder, ncn_lin, pc, eb, em,
plus the original reconstruction decoder. Consumer block: lin1, lin2, lin3.
These blocks share parameters across many entities and are not sparse lookup rows.
The intervention thus tests sequential optimization on this architecture, not the
paper's row-uncertainty theorem or a unique causal diagnosis of self-influence.

Historical checks found head-only/frozen-encoder fine-tuning, staged reconstruction
fine-tuning, crossfit residuals, representation perturbation and SAM. Those did not
implement within-epoch sequential updates of these two blocks. STACO's primal/dual
block optimizer solves a different objective. This remains a classical block-update
experiment, not a claim that alternating optimization is newly invented.

The README fixes all nine fits and controls before any performance evaluation.
No lncRNA, new labels/data/annotations, posterior weights or inference changes.

