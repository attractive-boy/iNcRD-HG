# Differentiable conditional readout for graph residual learning

Source: *Mochi: Aligning Pre-training and Inference for Efficient Graph
Foundation Models via Meta-Learning*, arXiv:2604.22031v2 (2026 preprint).
The abstract, full PDF, Section 3 and the author's `mochi/model.py`, training
and sampler files were retrieved. Author commit:
`420966d106c28fde6583bc128b872b26263fb146`.
No independently verified conference publication is claimed. The ridge solver
itself predates this paper (R2-D2); it is not claimed as a new invention here.

The project has already improved through graph features, tabular foundation
models and fixed score combinations. This candidate leaves that reference
unchanged and learns a graph encoder by differentiating through a classifier
fitted to a separate support-label set in every training episode. It does not
add table columns, change TabICL attention, or count fusion as an innovation.
It is also distinct from the prior variational-last-layer and optimizer trials:
the head is an exact conditional ridge solution, not a gradient-trained weight
posterior or an alternative update of a fixed classifier.

The adaptation uses the existing raw typed graph encoder rather than SVD/GAMLP
pretraining; no author weights or universal graph-transfer claim is used. Each
of five training folds takes turns as optimization queries Q. One of the other
four folds supplies support pairs R and their binary labels. **Both Q and R
target edges are removed before reverse arcs, degree features and message
passing.** Only the other three folds provide graph facts. The support head
uses R labels only, never support anchor residuals whose OOF predictors could
have seen Q. The already audited nested-OOF reference is used only as the Q
loss offset. This prevents a new query-label path through a support offset.

For normalized learned pair embeddings Z, the head solves the same ridge
objective as Mochi, including the penalized bias column:

`W = solve(Zs_aug.T @ Zs_aug + 10 I, Zs_aug.T @ onehot(ys))`.

The primal 65-dimensional solve is used instead of the equivalent dual solve
because our support set is larger than the feature dimension. Solves use FP64;
no explicit matrix inverse is formed. A learned, clamped temperature rescales
both support and query embeddings. Binary score contrast is mapped through
`2 tanh(2 contrast)` to give a bounded residual. Inference averages the same
20 registered graph/support configurations. Query batching cannot change a
head or any graph. This gives the same three-fifths graph support and one-fifth
conditioning support used in training; the unused fold is not read as labels
by that episode.

Full differentiates through the support solve. The sole primary control
`stop_support_gradient` detaches only support embeddings inside that solve;
it has exactly the same initial forward function, parameters, support labels,
queries, graphs and query-embedding gradient. This isolates the contribution
of learning representations through the support-conditioned solution. It is a
gradient ablation, not a different mathematically justified objective.

Four numerical/isolation tests include primal-dual predictions and gradients,
finite-difference gradcheck, support/query deletion in all 20 episodes, forward
identity of the ablation, and query-order/batch invariance. Real-data smoke runs
do not read validation outcomes. Formal screening uses the inherited 400-step,
two-learning-rate protocol; LncDrug has an exact reference-identity branch
declared in advance. Three initialization confirmations, the three existing
overlapping development partitions, and the frozen historical outer recheck
are all required. These are development checks, not independent cohorts.

This is one candidate mechanism family, not 20 techniques. Training/inference
alignment motivates the experiment; it does not guarantee global AUC or AP gains.

Other search findings: arXiv:2604.25978 attributes a link-prediction shortcut to
BatchNorm and batch class composition; the current encoder uses LayerNorm,
so that specific cause does not match this implementation. F2LP-AP
(arXiv:2604.20736) uses hand-specified clustering-coefficient mappings and
geometric-median prototypes; it is not selected for this experiment.
