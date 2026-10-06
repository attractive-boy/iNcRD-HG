# Joint persistence event attributes: a narrower prospective revision

The frozen SpectRe screen failed its mechanism gate: the no-spectrum control
outperformed the full time-specific spectrum model. That observed result
motivates this narrower hypothesis. The primary is **exactly** that existing
no-spectrum implementation and its two seed-666 models. A new control tests
whether the joint assignment of persistence-event attributes adds information
beyond their separate empirical distributions. This is a development revision
within the same persistent-topology family, not an independent new search or an
additional successful innovation.

The source definition is section 2.2, Definition 2.3 of *Graph Persistence goes
Spectral*, arXiv:2506.06571v3 (2026-02-04), together with the already inspected
and executed author implementation pinned at commit
`90ad637b00481a33318306d776882dce8e2012e8`. That paper attributes RePHINE to
Immonen, Souza and Garg, *Going beyond persistent homology using persistent
homology*, NeurIPS 2023. The latter citation is secondary here; this revision
does not claim to have reproduced its full original experiment.

For each local node, the zero-dimensional event is `(0, death, alpha, gamma)`,
where alpha is its learned vertex filtration value and gamma is its first
incident edge-filtration value. The survivor convention maps infinite death
and missing incident edges to one. Cycle events `(1, birth, 0, 0)` are retained
as before. Query graphs include both endpoints and every common one-hop
neighbor; the entire queried fold and its reverse facts are absent. There is
no node cap. The typed encoder, learned filtration networks, zero-dimensional
mean, cycle sum, nonlinear tuple embedding and zero-initialized residual head
are unchanged.

The two registered arms are:

- `full`: call the frozen parent `no_spectrum` topology computation directly.
- `marginal_only`: within each query graph, sort **each coordinate separately**
  across the zero-dimensional tuples and align the resulting order statistics.
  This comonotone coupling preserves the count and the entire multiset of every
  coordinate, but removes the observed assignment of death, alpha and gamma to
  each other. Cycle tuples, cycle masks and all pooling and nonlinear layers
  remain unchanged.

The control deliberately creates synthetic tuple associations; they need not
be a valid persistence diagram of any graph. It is an information-removal
control, not an alternative topological theorem or a biological intervention.
It leaves information already present in each marginal intact, so a win would
support the narrower joint-correspondence claim rather than every possible
benefit of topology. The broader claim that spectra improve this task remains
failed regardless of this revision's outcome.

The ragged sorting implementation pads each query to its largest local node
count using positive infinity, sorts along the event dimension, then discards
all padding. It never mixes queries, reads labels or uses node identifiers as
sort keys. Gather derivatives are retained within unchanged sorting regions;
no differentiability at ties or discrete pairing changes is claimed. Both arms
still compute the parent's native events and spectra and execute the same
zero-masked spectrum encoders. This preserves the parent modules, initialization
and optimizer behavior, including zero spectrum gradients and ordinary AdamW
decay. The finite training workload and RNG stream remain equal. The native
code and binary are not modified.

Four CPU implementation checks verify the independent NumPy marginal oracle,
piecewise finite differences, a pair of measures with equal marginals and
different joint moments, query batching and permutation, exact parent training
state/optimizer/trace/RNG compatibility, and graph/label boundaries. The prior
eight native implementation checks and forty real fold audits are inherited by
hash; they are not counted again as new tests. Before performance, one new
five-step MiDrug control fit checks real-data replay, positive finite training,
and equal final RNG against the saved parent-primary short fit. The unchanged
primary's saved weights are strictly loaded and replayed. LncDrug stays exact
reference identity and is not trained.

The seed-666 primary's two already completed learning-rate runs are copied only
after all arrays, recipes, checkpoints and predictions have been verified.
Their hashes and provenance remain attached. No primary rerun or new checkpoint
selection is used to replace those observations. The new control uses the same
two learning rates, 400 AdamW updates, 400 gradients and 20 checkpoints. An
eligible screen alone advances both arms to seeds 0 and 1. The fixed development
partitions and a frozen outer stage are still mandatory, with both MiDrug AUC
and AP above the strong reference and the new control. All prior performance
exposure is disclosed, no original retained test source is read, and no default
model is replaced at registration.
