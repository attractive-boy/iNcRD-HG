# Conditional determinant selection of graph neighbors

This is a prospective, from-scratch graph adaptation of the conditional
diversity objective in *Beyond Attention or Similarity: Maximizing Conditional
Diversity for Token Pruning in MLLMs*, arXiv:2506.10967v2 (2025-07-01), sections
3.2–3.4 and appendix A. Appendix C.4 was also read; its additional balance-factor
sweep is not used here. The author repository identifies the work as NeurIPS
2025, but publisher venue attribution has not been independently verified.

The pinned author commit is `9541616c40fcd5625de1cdb8ea6c33c129eb7864`
in `Theia-4869/CDPruner`. Four source files have verified Git blob identifiers
and SHA256 digests in `CDPRUNER_AUTHOR_SOURCE_V1.json`. The original
`encode_images` method is extracted from the pinned AST and executed with
synthetic encoder outputs in eight nondegenerate, single-image cases. The
local greedy rule gives the same selected subsets in those cases. This verifies
the selector, not the paper's pretrained MLLM experiments; no author weights,
vision-language model, or graph pretraining corpus is used.

## Graph objective and implementation

For a queried RNA–drug pair, the entire query fold is removed before graph
construction, including reverse target facts. A two-layer, four-head, width-64
typed attention encoder produces hidden vectors. All one-hop neighbors of
either endpoint enter the candidate union, excluding the two endpoints. There
is no preliminary node-ID truncation or candidate cap. The fixed retained
budget is `min(16, number_of_candidates)`.

Let `z_i` be the unit-normalized hidden vector of candidate i and `u` the unit
normalization of the sum of the two endpoint vectors. Positive cosine
`r_i = z_i^T u` is min–max normalized **within each query**, then transformed to
`q_i = 1e-4 + (1 - 1e-4) * normalized_r_i`. If a query's relevance span is at most
`1e-12`, all its qualities are one. Exact zero vectors remain zero under
normalization. The subset objective is the principal determinant of

`L = diag(q) [Z Z^T + 1e-5 I] diag(q)`.

The mathematical diagonal of a nonzero unit vector is set to one explicitly;
unit-normalization roundoff must not create a hidden tie-breaking preference in
the no-condition arm. Zero feature rows retain a zero Gram diagonal, with the
same positive ridge. Qualities and the ridge are fixed numerical conventions,
not performance-selected hyperparameters.

Selection uses FP64 incremental Cholesky residuals. At each step it chooses the
largest conditional diagonal, updates each remaining candidate's coefficient
vector, and subtracts the squared new coefficient. All ragged query groups are
processed together; only kernel rows belonging to selected pivots are formed.
No per-query dense `n × n` Gram matrix is allocated. For dimension d and budget
k the selection arithmetic costs `O(n*k*d + n*k^2)` and stores feature rows and
`O(n*k)` Cholesky coefficients. PSD residuals, positive finite pivots, and valid
groups are checked. Empty candidate sets contribute a zero pooled vector.

Exact pivot ties use lexicographic hidden-feature order, then left/right
neighborhood membership. Only candidates with identical hidden vectors and
identical evidence membership use their array position to choose an
interchangeable representative. Sorting does not use labels or node IDs as a
substantive priority.

For each selected node, an MLP receives its hidden vector, its elementwise
products with the two endpoint vectors, and its two membership indicators.
The retained evidence vectors are averaged and supplied to a zero-initialized
pair residual head. Hard subset indices are detached. Gradients flow through
the gathered selected embeddings and the ordinary encoder and evidence MLP;
there is no invented derivative of sorting and no selection-only trainable
projection that would lack a learning signal. Any finite-difference claim is
restricted to regions in which the selected subset is unchanged.

## Registered mechanisms and controls

The three arms share identical module initialization, parameter count, graph
inputs, dropout draws, training loss, and update budget:

- `full`: conditional quality and the complete Gram kernel.
- `no_diversity`: the same qualities and kernel diagonal, with all off-diagonal
  entries removed. This retains individual relevance selection without the
  redundancy penalty.
- `no_condition`: the complete Gram kernel with all qualities fixed to one.
  This retains diversity selection without endpoint conditioning.

The full model must beat **both** controls on mean MiDrug AUC and AP at every
required validation stage. Better control results do not validate the full
mechanism. A positive increment over the strong reference is insufficient on
its own. The family can count at most once, even if a later, explicitly
registered revision is motivated by these controls.

## Differences from the pinned author implementation

The paper defines positive cosine relevance. The pinned `encode_images`
negates it before averaging. Its min/max reduction is global across the batch,
and a constant relevance tensor can divide by zero. The local implementation
uses the paper's positive cosine, per-query normalization, and the explicit
constant-quality fallback above. The source-comparison test supplies a negated
synthetic query to the original method to account for its sign, and restricts
the comparison to nondegenerate single-image cases with no ridge. It does not
assert arbitrary-batch equivalence.

The paper states a general `(1 - 1/e)` guarantee. An arbitrary fixed-cardinality
log-determinant objective need not be monotone, so that guarantee is not claimed
here. The rule is a **greedy MAP heuristic**, neither exact MAP nor DPP sampling.
Selected neighbors are computational evidence, not newly observed biological
interactions. The endpoint sum, local graph candidates, finite retention
budget, ridge, floor, training gradients, pooling, and residual task are local
adaptations.

## Evaluation boundary

Training uses BCE with the strictly nested OOF strong-reference logit as a
fixed offset, one omitted query fold per update. Prediction averages raw
residuals over five support graphs and adds `0.1 * residual` to the unchanged
evaluation reference. MiDrug alone receives a new model. LncDrug is declared
in advance as exact strong-reference identity, with no parameters or fitting;
those scoring records never count as neural fits or an additional innovation.

The historical screen has seed 666, two learning rates (0.001, 0.003), 400
AdamW steps and 20 common checkpoints per arm. Only an eligible screen advances
to seeds 0 and 1. The three fixed development partitions and a frozen outer
stage remain mandatory afterward. Earlier historical and fixed-partition
outcomes have been exposed during development; none are described as an
independent test. The original retained test source is not accessed.

Before formal performance, the seven V2 implementation checks cover an
independent NumPy dense-logdet oracle (maximum logdet error `8.89e-16`), pinned
author selection, a repeated-neighbor counterexample, batching and permutation,
piecewise gather gradients, omitted facts and labels, equal initialization and
RNG, and synthetic optimizer budgets. Failed V1 evidence and its exact source
are retained; the two failures were test calls to a nonexistent
`torch.flatnonzero` API, replaced with `torch.nonzero(..., as_tuple=True)[0]`.

The separate real-data short probe reads training arrays only, saves weights
and optimizer states, checks reordering and batching only on each training
pair's omitted graph, and audits forty task/split/fold boundaries. Its resource
and numerical evidence is recorded separately before the formal protocol is
frozen; it is not performance validation or a successful innovation.
