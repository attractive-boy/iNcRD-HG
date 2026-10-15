# Same-drug representation Mixup

Goal: original MiDrug test seed42/43/44 mean AUC≥.9733929274189016 and
AP≥.933683373732677, protected PLMF MDR/MDS do not decline. Only miRNA and drug,
original features and training labels, fixed original splits. No external inputs.

## Hypothesis and controlled difference

The previous converged last-layer fit improved training metrics but reduced inner
holdout metrics. Test supervised interpolation during representation learning.
Sources and qualifications are in `../midrug_vicinal_literature_20261007/READING_NOTES.md`.
This is a Manifold Mixup adaptation, not an exact replication of the Pinterest or
LatMixSol methods and not a claim of a newly invented regularizer.

Capture the original M1/M2/M3 concatenation H just before lin1, from a single clean
training forward. The ordinary exact query-edge removal still constructs each H_i.
For a partner permutation pi and shared lambda~Uniform[0,1], form
`H_mix=lambda H+(1-lambda) H[pi]`, `y_mix=lambda y+(1-lambda) y[pi]`.
Pass H_mix through the original lin1/LeakyReLU/lin2/LeakyReLU/lin3 head. Gradients
flow through both original representations and all encoder/head parameters.
No detached mixed features, target-edge insertion, new node, or graph alteration.
These are supervised regularization targets, not newly observed biological links.

`drug_mix` chooses a uniform random ordering within each training drug group and
a uniform nonzero cyclic shift to obtain distinct partners within that drug.
`global_mix` uses a random ordering/nonzero shift over all eligible examples.
Eligibility means the example's drug has at least two training observations.
Examples from singleton drug groups map to themselves in both arms. Each mapping
is a permutation. A single common lambda per epoch exactly preserves mean target
prevalence even though the partner labels/mixed-target distribution can differ.
Same-drug mixing does not guarantee realistic miRNA interpolation; unrestricted
mixing is a meaningful standard method and tests whether the restriction helps.

Both arms use the same stateless numpy RNG seed `91007000 + 10000*fold + epoch`;
draw lambda first, then permutations. No labels or heldout examples are used to
choose partners. Numpy mixing RNG is separate from Torch initialization and the
reconstruction mask RNG, whose seed remains `20261006+fold`.

## Frozen training/selection

Two configurations × three fixed inner folds, seed42: six fits from the same
original M123 initialization, not from the failed last-layer MAP checkpoints.
Both retain original unmasked reconstruction weight10, ramped over50 epochs,
and unchanged masked-node sampling for computing that reconstruction loss.
Association loss `(1-alpha)*BCE_clean + alpha*BCE_mix`, with alpha=0 for epochs≤30,
linear ramp to .5 at epoch60, fixed thereafter. This preserves total association
loss weight1; it intentionally differs from Pinterest's additive weighting.
No mixture weight, Beta shape, layer, or partner-policy sweep.

RMSprop lr1e-4/alpha.99; original model wd0, reconstruction wd1e-4. Zero gradients
with set_to_none=True. Maximum800/minimum300 epochs/patience200. Earliest maximum
mean(inner AUC,AP) selects the checkpoint; save final state and both optimizer
states as well. One original BatchNorm update per epoch; decoding mixup adds no BN.

Only drug_mix is eligible. Both mean inner metrics must beat original `control`,
`unmasked10` and `global_mix` by ≥.002 to advance to original validation. Global
mix is not promoted retrospectively. Inner data are reused development folds,
not independent confirmation. Failure does not imply a family-level impossibility.

## Verification

Check distinct/permuted partners, same-drug constraints, identical eligible sets,
shared lambda/prevalence, RNG determinism, singleton behavior, independent scalar
loss and gradient references, gradients to both sources, no-mix limiting cases,
full-model initialization, decoding equivalence, one BN update, nonzero encoder
gradients, and heldout-label invariance. Replay selected predictions/curves,
selected/final BN and optimizer updates, and physical query deletion. Removing
the reconstruction module and training-only interpolation preserves inference.
No PLMF parameter/route changes; protect its existing artifacts and input hashes.
