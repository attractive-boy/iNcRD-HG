# Fixed representation/consumer update-order experiment

Goal remains original MiDrug test mean seeds42/43/44 AUC>=.9733929274189016 and
AP>=.933683373732677, preserving PLMF MDR/MDS. Only original miRNA/drug inputs,
three immutable inner folds, seed42; no external information or lncRNA.

Hypothesis: updating the consumer before refreshing representations on the same
training examples can improve generalization compared with the reverse order.
Inspired specifically by UWSR (2026) §3.3, not its sparse-row covariance regularizer.

## Fixed blocks, objective and comparison

- Consumer C: original lin1/lin2/lin3 parameters only.
- Representation R: all other original parameters, including encoder, M2/M3
  projections and eb affine parameters; plus original reconstruction decoder.
- Same original model initialization, graph, losses and inference. Objective
  BCE + 10*min(1,epoch/50)*original unmasked reconstruction MSE.
- Both blocks use persistent RMSprop lr1e-4/alpha.99/eps1e-8/momentum0/centeredFalse.
  Main parameter wd0; reconstruction decoder wd1e-4. Separate optimizers for
  disjoint blocks; set_to_none=True. Every effective parameter receives exactly
  one optimizer update per epoch, never two. Mask RNG20261006+fold generates one
  set of reconstruction indices per epoch; reuse it in both passes.
- consumer_first: compute full objective/gradients, step only C; recompute full
  objective/gradients using updated C and unchanged R, then step only R.
- representation_first: step only R from first pass; recompute with updated R,
  then step only C. All blocks remain differentiable; unused gradients discarded.
- joint: compute both gradients at pre-update parameters in pass1; recompute the
  same full objective/backward in pass2 without a parameter update and discard
  these second gradients; restore pass1 gradients and step each optimizer once.
  This is an explicit two-pass compute control for ordinary joint updates, not
  two joint optimizer steps. The extra pass does not determine updates.
- First pass uses training BN normally. Second pass also uses the current batch
  statistics but temporarily disables tracking, preventing a second update of
  running_mean/running_var/num_batches_tracked. Restore tracking even on exceptions.
  eb is the only BN. Thus all arms advance BN exactly once per epoch, independent
  of order; changing train/eval normalization between passes is not permitted.

Nine fits: three arms × three folds. Same800 maximum epochs,300 minimum,200
patience; earliest maximum mean(inner AUC,AP). Two full association/reconstruction
forward/backward passes per epoch in every new arm; compared with old unmasked10
the new joint control has extra computation, not extra optimizer updates.

Only consumer_first is eligible. Its mean inner AUC AND AP must each exceed
control, unmasked10, representation_first and joint by >=.002. No order schedule,
learning-rate, block-definition or replay-budget sweep. Do not select a control
or access original validation/test on failure. Reused inner folds remain development
data, and this filter does not replace the user's original-test dual+.02 target.

## Verification and scope

Test a scalar three-parameter model against independent hand-derived RMSprop
updates (including weight decay/moment carryover), all three orders and step counts.
Verify disjoint/complete blocks, stationary joint equivalence to one-pass update,
BN batch output and buffer policy, finite gradients, mask reuse, label isolation
and physical query deletion on the actual original M123 model. Freeze source/plan
after prechecks. Replay nine selections and audit nine final states, each block's
selected/final optimizer history, curves and PLMF protection hashes.

The paper's sparse embedding assumptions do not hold exactly for shared graph
projections. A performance/order effect would not uniquely identify self-influence
as its cause. The sequential objective values refer to different parameter states;
report first/second pass separately, not their sum as one optimized scalar loss.

