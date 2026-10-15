# Exact query deletion for an attentive two-layer M123 encoder

Status: implemented in model.py/events.py. The independent NumPy prototype passed 240
forward cases; check.py additionally passed 12 explicit-deletion score/gradient comparisons
in float64, real-data checks and memory measurement. Training is now a separate stage;
no original validation/test evaluation has been performed.
Only original miRNA/drug inputs are permitted; the final dual +0.02 target is unchanged.

## Why this differs from the existing attention work

Do not describe attention as a never-tried idea. `improve/propagation_attention.py` already
contains relation fusion, degree-preserving sigmoid gates, and GATv2, using an older base
predictor on a shared training graph. Its `PropagationModel.forward` delegates directly
to that base and does not contain M123 PairStruct or per-query message deletion.
The recent midrug_attention_20261006 experiment changes M2 path weights but retains
the original sum encoder. The current proposed adaptation is an attentive encoder
inside M123 with exact query deletion propagated through both nonlinear layers.
The older executors also read original test splits; do not import their loaders/runners.

The molecular branch experiment is complete and did not beat reconstruction-only;
do not repeat its parameter grid. Its weak branch-only diagnostic is not proof that
drug structures have no information, because that branch was jointly optimized.

## Relation aggregation

For a relation e=(s,t), let an existing GeneralConv produce value v(u)=W_msg h(u)+b.
Use a positive gate g(u,v)=sigmoid(scorer(h(u),h(v))) and cache per receiver:

The implemented variant bounds it to `0.1 + 0.8 * sigmoid(scorer)`, keeping the
initial value 0.5 and avoiding nearly zero mass in a nonempty receiver neighborhood.
The derivation applies to either strictly positive gate.

    D(v) = sum_u g(u,v)
    N(v) = sum_u g(u,v) * value(u)
    k(v) = incoming edge multiplicity
    M(v) = k(v)^(1-alpha) * N(v)/D(v), with M(v)=0 if k(v)=0.

Sum M(v)+original lin_self(h(v)) over incoming relation types, then divide by the
unchanged relation count. Scalar alpha=0 preserves neighbor-sum scale; alpha=.5 or 1
provides degree scaling. Initially zero the final gate scorer, making every g=.5.
At alpha=0 this must reproduce the original encoder and original M1/M2 results;
at alpha=.5 it must reproduce the already audited degree-normalized encoder.
Gate normalization and messages must both be recomputed under query deletion.
Bounded sigmoid scores avoid exponential overflow; protect actual zero-degree rows
using the integer degree mask, not merely a tiny floating denominator.

## First layer: only the two endpoints change

Raw node features are fixed, so the remaining first-layer gates do not change when the
query is removed. For each direction of the target relation, subtract the removed
edge gate mass and weighted message from D/N and subtract its multiplicity from k.
The receiver's shift is (M_after-M_before)/n_rel. Add that shift to the full-graph
preactivation and apply the original LeakyReLU to obtain dh1 for the two endpoints.
All other nodes have exactly unchanged first-layer states.

## Second layer: update only incident edge events

The two changed h1 vectors alter second-layer gates and values. For each relation,
the only potentially changed edges have source OR receiver equal to that relation's
query endpoint. Build their union by edge index (keep repeated edge copies distinct).
For each query and each event:

1. Gather its old source/receiver h1, and replace either endpoint with its query h1.
2. Recompute that event's gate and value; set its new gate to zero if it is the
   removed target edge. Same-type self-loops are retained and also recomputed.
3. Group events by (query index, receiver index, relation). Add the changes
   in gate mass and weighted values to the cached full-graph D/N.
4. Recompute the normalized message for each affected receiver, using k-q only
   for the receiver of the deleted target edge. Subtract its old message.

The relation message deltas, divided by n_rel, are sparse query-specific updates
to the second-layer states. The original explicit lin_self projection also changes
at the two endpoints; add W_self*dh1/n_rel there. Do not separately add the same-type
self-loop term: that edge was already handled by the incident-event update.

## M1 and M2

M1 endpoints are full-graph z(endpoint) plus their message/self updates.
For M2, keep original query-deleted path weights w, which already exclude endpoints.
Compute the original pool w @ z / (sum(w)+1), then add

    sum_affected_nodes w(query,node)*delta_z(query,node)/(sum(w)+1).

Use indexed sums per query; avoid constructing [queries, all_nodes, hidden] tensors.
Endpoint explicit-self updates have zero M2 weight. M3 and its train-only standardization
remain untouched. Nonedges have no query correction and can use the ordinary full-graph
forward. Original validation pairs must all be nonedges in the installed training graph.

## Efficient implementation and necessary checks

Cache structural event indices based on the actual query-pair values and immutable graph.
Never key only by tensor data_ptr: the allocator can reuse a pointer for another query.
Only queries with nonzero graph A[r,d] need events. The current internal graph has about
2,157 positive queries, 16,232 MiMi edges, 852 DrugDrug edges and about 2,157 edges in
each target direction. Event support is incident neighborhoods rather than full graph
copies per query. Measure actual event counts and memory before launching fits.

First validate zero-gate identity, finite/nonzero gate gradients, and explicit graph
deletion with deliberately nonuniform gates. Cover zero remaining degree, repeated
same-type self-loops, nonedges, both endpoints, every M2 pool, M3 and final logit.
Compare autograd gradients with explicit deletion on a small graph, not just scores.
NumPy prototype verification does not replace these model checks.

Then predeclare a small controlled experiment: target-only vs all-relation gates,
with appropriate uniform controls and reconstruction-only comparison. Do not import
old propagation-attention data or test results into selection. Exact configs are not
frozen yet; all new development remains within the original-training internal folds.
