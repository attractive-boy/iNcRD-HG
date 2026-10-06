# Triangular pair-state attention: prospective adaptation

Source: Luis Müller, Daniel Kusuma, Blai Bonet and Christopher Morris,
*Towards Principled Graph Transformers*, NeurIPS 2024 Poster, OpenReview
`LJCQH6U0pl`, arXiv `2401.10119v4`. Author names are verified against the full
downloaded paper. The full
paper, expressivity conditions, experiments, limitations and Appendix A.1/B.1
were read. Author implementation: `luis-mueller/towards-principled-gts`.

The distinguishing operator maintains pair states and combines `(i,k)` with
`(k,j)` through the same intermediate entity:

```
a[i,k,j,h] = softmax_k(dot(Q(X[i,k]), K(X[k,j])) / sqrt(d_head))
out[i,j,h] = sum_k a[i,k,j,h] * (V1(X[i,k]) * V2(X[k,j]))
```

Equation 2 and the author's implementation normalize over the intermediate
index. The pseudocode's last-axis softmax should not be copied without checking
the stated tensor layout. The theoretical 3-WL equivalence has fixed-graph-size
and parameterization assumptions; neither the finite local network nor one
layer is claimed to satisfy the full theorem. The original molecular regression,
CLRS and BREC experiments do not demonstrate improvement on MiDrug/LncDrug AUC
or average precision.

## Local model and boundaries

- Pair dictionaries contain only the target RNA type and drug type, with all
  821 MiDrug entities or all 1,538 LncDrug entities retained.
- Input tokens combine separate type-specific raw-attribute projections, two
  ordered endpoint maps, induced physical-relation tokens, a recorded-zero token,
  and a diagonal identity token. Unknown relations have a fixed zero embedding.
- Every positive and zero outcome in the current neural query fold is omitted
  before creating its support token matrices. Evaluation labels and global OOF
  reference predictions are never used to create graph tokens.
- A support zero means a recorded zero label, not biological antagonism.
- The trained model uses the existing nested strong-reference residual BCE and
  fixed 0.1 residual scale. All five support folds are averaged at evaluation.
  The strong reference, scaling, execution infrastructure and version revisions
  are not independent innovations.
- The mandatory `additive_value` control preserves attention scores, both value
  projections and all parameters, and replaces the conjunction with their mean.
  The primary `full` must exceed this control on both mean MiDrug metrics.

## Resource selection before any performance result

The initial two-layer width-16 dense implementation passed six algebraic,
gradient, query-boundary and permutation checks. Four two-step, training-only
CUDA-1 smoke fits and 40 support-fold audits passed. LncDrug required 28.3 seconds
per full optimizer step and 7.85 GiB peak allocation. No AUC or AP was read.

A separate compiled contraction matched real-size values and gradients (maximum
value difference below 3e-8 and gradient difference below 9e-8), but gave only
modest warm speed improvement for the larger graph. Two failed compiler probes
are retained: the installed 64-bit CUDA library lacks its unversioned linker name,
and the bundled CUDA-12 assembler is incompatible with the old driver. Task-local
linking to the installed library and the already-installed CUDA-11.8 assembler
resolved the probe. System libraries were not changed. Compilation is not used
by the final selected implementation.

The prospective recipe is one layer, width 32, four heads and dropout 0.1. The
final layer computes only requested output pairs while retaining every
intermediate entity. This exactly matches a one-layer dense model at evaluation;
it does not approximate the attention. Training dropout has the same law for
distinct queries, with a different random-draw order. Two more checks compare
values and gradients against dense models at both one and two layers, including
duplicate queries and chunk/reverse replay. The architecture choice was written
to `RESOURCE_RECIPE_DECISION_V1.json` before any performance metric was computed.

The final resource smoke uses the one-layer query implementation, not a claim of
equivalent outputs to the original two-layer architecture. Formal learning rates,
400-step budget, checks every 20 steps, seed 666 screening and seed 0/1 confirmation
remain those of the existing executor. Historical confirmation, three fixed
partitions and outer recheck are required. Final verified innovations remain 0.

## Related next direction

Paolino, Maskey, Welke and Kutyniok, *Weisfeiler and Leman Go Loopy: A New Hierarchy
for Graph Representational Learning*, NeurIPS 2024 Oral, OpenReview `9O2sVnEHor`,
was also read. Its operator sends learned path-GIN messages from simple paths
between distinct neighbors that exclude the receiving center. This is distinct
from adding scalar cycle counts. Public author code at `RPaolino/loopy`, tree
`4151a6a7e67312cdc44bb379b979ff62e5a448e5`, was retrieved and inspected. The dense
graph path explosion discussed in the paper requires a separate support-graph
resource audit before any implementation. No Loopy performance claim is made.
