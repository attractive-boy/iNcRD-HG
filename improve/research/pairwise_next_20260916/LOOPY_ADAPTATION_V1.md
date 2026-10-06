# Learned closed-path messages: prospective adaptation

Raffaele Paolino, Sohir Maskey, Pascal Welke and Gitta Kutyniok,
*Weisfeiler and Leman Go Loopy: A New Hierarchy for Graph Representational
Learning*, NeurIPS 2024 Oral, OpenReview `9O2sVnEHor`. The official full PDF,
definitions of simple paths and r-neighborhoods, Section 6, experiments,
limitations and relevant appendix definitions were read. Author code was read
at tree `4151a6a7e67312cdc44bb379b979ff62e5a448e5` of `RPaolino/loopy`.

The paper augments each node's neighbors with simple paths joining two distinct
neighbors and excluding the center. A path-GIN processes each path before
aggregation to the center. It is not simply a cycle-count feature. The theory
depends on injective aggregation functions; the finite normalized network below
does not inherit the full r-loopy WL guarantee.

## Prospective local mechanism

1. Omit all current neural query-fold positive and zero facts. Reconstruct the
   physical typed support graph with the existing audited boundary function.
2. Use only positive target RNA–drug support associations to form the loop graph.
   Its first nonempty higher neighborhood consists of length-two simple paths
   around four-cycles. Similarity edges do not create these cycles.
3. Enumerate every four-cycle, every receiving center, and both path orientations.
   No path is sampled or truncated. The pre-performance support audit found at
   most 231,488 ordered rooted paths for MiDrug and 9,936 for LncDrug.
4. Apply an ordinary two-layer typed physical-attention backbone, shared by both
   study arms. After each physical update, gather each path's three current node
   states and its center-distance labels (1,2,1). A shared learned projection and
   path-GIN perform one message step and pool the path representation to its
   excluded center.
5. Aggregate paths with square-root-count scaling and LayerNorm for the large
   association graph. Project the aggregate into a residual node update, masked
   to nodes that actually have loop paths. Decode endpoint states and their
   interactions with a zero-initialized residual head.

The mandatory `no_path_message` control keeps the identical paths, center labels,
projections, MLPs, normalization and physical backbone. It removes only adjacency
messages inside each path. All projection parameters remain live in both arms.
A paired synthetic test preserves path-node marginals and shows that only the
full operator responds to reassignment of which middle node joins which pair of
endpoints. Both arms preserve path-reversal symmetry.

The raw node attributes and ordinary relation-degree inputs belong to the shared
backbone. There is no graph36/context76 feature expansion, pretrained frozen
kernel, scalar cycle-count column, or additional biological dataset. Association
four-cycles describe observed co-association, not chemical rings, feedback loops,
or confirmed molecular mechanisms.

## Explicit differences from the original implementation

- Link residual prediction on a heterogeneous graph replaces graph-level
  molecular prediction.
- Physical typed attention replaces the paper's direct-neighbor GIN branch.
- Only the positive target bipartite support defines the higher path branch;
  other physical relation types remain in the ordinary shared backbone.
- Only length-two paths are used, because the bipartite support has no triangles.
- The same feature/position projection feeds both the self and neighbor terms
  in path-GIN, keeping the control's parameters active.
- Square-root path-count scaling, LayerNorm and zero-initialized residual readout
  are local adaptations. They do not preserve an injective sum theorem.
- Author comments about scatter-sum gradients are not taken as mathematical
  evidence of incorrect additive derivatives. Local gradients and deterministic
  evaluation are independently checked.

## Verification and decision rules

Four core tests cover complete enumeration versus an independent neighbor-walk
definition and exact count formula, path-message dependence, query-fact deletion,
zero null, gradients, permutation and query replay. The training-only smoke uses
five steps per arm on CUDA-1 and covers all five support folds. Every historical
and fixed-partition support fold is audited before formal fitting.

The formal recipe uses width 64, depth 2, four heads, dropout 0.1, and exact path
chunks of 8,192. The existing nested strong-reference residual objective, 400
steps, checks every 20 steps, two learning rates, fixed shrinkage and gates are
unchanged. Full must improve both MiDrug global metrics and exceed its declared
control; every LncDrug metric/seed or partition must remain within an absolute
0.001 loss. Historical screening/confirmation, fixed partitions, and outer
recheck remain separate requirements. The family counts at most once.
