# Fixed proper-loss shape experiment

Hypothesis: coupled asymmetric partial losses can improve MiDrug training by
reducing the gradient of high-confidence negative-label errors while preserving
the unrestricted population optimum p=P(observed label1|x). This is a classical
proper-loss adaptation, not a label-error detector or a new invented loss family.

Only original miRNA/drug inputs and the three fixed inner folds; seed42. Preserve
M1/M2/M3 and original reconstruction10, all initialized exactly as unmasked10.
No new graph edges, node types, features, labels, weights or inference parameters.

With p=sigmoid(z), partial losses (l0,l1):

- negative_tail: (2p, 2[-log(p)-(1-p)]).
- positive_tail: (2[-log(1-p)-p], 2(1-p)).
- brier: (2p², 2(1-p)²).

Their logit gradients are BCE's (p-y) times 2(1-p), 2p and 4p(1-p), respectively.
Each factor equals1 at p=.5. Their proper weights w(p) are 2/p, 2/(1-p), and4;
the conditional probability-risk derivative (p-eta)w(p) has its unique zero at eta.
This does not imply noisy-label invariance, finite-model calibration, convex
neural optimization or improved AUC/AP. Easy-example gradients also change.

Training association objective: (1-alpha) BCE + alpha chosen_loss, alpha0 through
epoch30, linear to1 at60. From60 onward only the chosen proper loss. Reconstruction
weight10 ramps over50 epochs, unchanged. One original forward/BN update per epoch;
no augmentations or extra neural modules. RMSprop lr1e-4/alpha.99; original wd0,
reconstruction wd1e-4, zero_grad(set_to_none=True). Private reconstruction RNG
20261006+fold. max800/min300/patience200, earliest maximum mean(inner AUC,AP).
No weight/temperature/loss exponent sweep, second seed search or warm-start.

Stable arithmetic: phi(x)=x+expm1(-x), x=softplus(-z), gives l1/2 for negative_tail.
Compute softplus with logaddexp(0,-z) to avoid the default threshold20 approximation
switch in Torch softplus (caught by the pre-freeze finite-difference check).
For x<.01 use Taylor terms x²/2-x³/6+x⁴/24-x⁵/120+x⁶/720, with neglected x⁷/5040.
positive_tail uses reflection; Brier uses sigmoid(±z) without subtracting p from1.
Use ordinary autograd of these actual losses, not stopped-gradient sample weights.

Nine fits (three arms × three folds). Only negative_tail is eligible. Mean inner
AUC AND AP must each exceed control, unmasked10, positive_tail, brier by >=.002.
No selecting a control, loss-weight tuning or original validation/test on failure.
Original-test mean seeds42/43/44 must still reach AUC>=.9733929274189016 and
AP>=.933683373732677, and PLMF MDR/MDS must not decline. The inner gate is only
a development filter, not a replacement target or independent confirmation.

Prechecks: independent scalar loss and derivative, float64 gradcheck including
Taylor switch, extreme logits, conditional-risk minima, label reflection, BCE
gradient matching at zero, zero-transition whole-model gradient equivalence,
real full training step/BN count, heldout label isolation and physical query deletion.
Then freeze all implementation/plan dependencies, train, replay selected predictions,
audit selected/final RMSprop states and curves, protect PLMF hashes, archive.
