# Degree-scaled iNcRD-HG encoder with M1/M2/M3

Original target remains both AUC/AP test mean +0.02, with unchanged PLMF performance.
Only the independent frozen miRNA/drug graph and original-training internal folds are loaded.
No external data, lncRNA nodes/relations/features, or heldout supervision is added.

Hypothesis: the original encoder sums neighbor messages; node degree therefore affects
message amplitude at both layers. Prior M2 path weighting did not change this encoder.
Test receiving-node degree scaling within target relations only or all four relations,
using exponent 0.5 or 1.0. Self projections and mean-over-relation aggregation remain intact.
M3 already retains query-deleted degree information. This is a standard normalization
adaptation, not a new-method originality claim or a claim of guaranteed improvement.

For edge type e, the message term is max(k_e(v),1)^(-alpha) times the sum of messages.
In M1, deleting query q removes its message AND changes the two target receiver degrees:

    delta p1(v) = [f(k-q)*(sum_messages-q*message(u)) - f(k)*sum_messages] / n_rel(v).

Only the two endpoint layer-1 activations change. At layer 2, the same target formula
uses the original h1 because the opposite endpoint's changed message travels exclusively
along the now-removed query edge. Same-type self-loop and explicit self projections
propagate the endpoint's own activation shift with their original multiplicities.

For M2, every intermediate is distinct from both endpoints, so its degree is unchanged.
Its contribution from an endpoint's h1 shift is multiplied by that intermediate's own
incoming relation scale. The weighted correction is summed inside each original path pool.
All M3 quantities and fusion-head dimensions remain the original definitions.
Positive target pairs in this clean graph are unique; the model asserts this property.

Six configurations x three folds = 18 fits. Four change the encoder alone. Two combine
alpha=0.5 with the previously best unmasked reconstruction weight 10. Pure M123 and
unmasked10 checkpoints serve as matched controls. Same initialization, optimizer,
800-epoch cap / minimum 300 / patience 200 / mean internal AUC+AP selection as controls.
Require both mean gains >=0.002 over each comparator before promoting to outer validation.
This development gate does not replace the final dual +0.02 target.

All trainable parameter shapes remain compatible with original M123. Encoder degree scales
are rebuilt from the current training graph and recorded config, not taken from heldout data.
The released implementation is untouched; PLMF remains on its protected original route.

Checks include alpha=0 identity, explicit positive/negative query deletion including a
degree-one-to-zero endpoint, M1 endpoints, all four M2 pools, M3 and complete logit,
finite nonzero encoder gradients, and heldout-label invariance. CUDA floating operations
are checked with numerical tolerances; underlying weights and normalization inputs must
be exactly equal in the label-flip check.

Reproduce with ncRNAdrug Python and OMP/MKL/OPENBLAS_NUM_THREADS=2:
check.py -> launch.py -> audit.py -> summarize.py.
