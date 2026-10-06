# Polynormer: prospective feature-product adaptation

Chenhui Deng, Zichao Yue and Zhiru Zhang, *Polynormer: Polynomial-Expressive
Graph Transformer in Linear Time*, ICLR 2024 Poster, OpenReview hmv1LpNfXa,
arXiv 2403.01232v3. The paper, OpenReview metadata, and author repository
`cornell-zhang/Polynormer` agree on identity. Author tree:
`fc8c276c9c5dfbd616d83f65338a3392188a5e08`. The downloaded model, parser and
README are inspected as text; the author's training commands are not run.

The polynomial base model in section 3.1 has trainable dense W/B and a
specific expressivity definition. The practical equivariant model replaces
these with attention, projections and shared beta. The author code further
uses ReLU, LayerNorm, local skips and an optional global stage. Therefore the
finite typed adaptation below does **not** inherit the base model's theorem
verbatim, and is not an exact reproduction of the reported node benchmarks.

Three local layers follow the author's implemented update:
`h=relu(Wh*x); v=dropout(relu(GAT(x)+Ws*x));
x=(1-sigmoid(beta))*LN(h*v)+sigmoid(beta)*v`.
Their outputs are summed. Two subsequent global layers use shared Q/K,
sigmoid-kernel normalized all-node attention, then
`x=dropout(relu(Wout*(LN(attended)*(Wh*x+sigmoid(beta)))))`.
The global output projection is shared across layers, as in author code.
The attention is computed by associative contractions, never an N-by-N
attention matrix. Dense small-graph values and gradients are checked.

Explicit adaptations are: total width 64 with four heads; type-specific
input projections of the unchanged attributes and support-only relation
counts; input LayerNorm; learned per-relation local score biases; a bipartite
pair residual readout; dropout .1 throughout; beta logits initially zero;
joint local/global training for all 400 steps instead of the author's staged
training. Physical self-loops are removed, since the local self transform is
explicit. Global communication includes all nodes but no target labels or
query-dependent graph changes. Original support-count conventions are kept.

The declared primary is `full`. The sole mandatory control is
`additive_update`: both local `h*v` and global `LN(attended)*(h+beta)` become
addition. All projections, LayerNorms, depths, graphs, global attention,
parameters and their initial values are identical. This tests multiplicative
feature interactions, not whether global attention itself is novel. Pair
readout products are common to both. There is no architecture/weight sweep.

This mechanism differs from polynomial **spectral graph filtering**, which
forms powers of a propagation operator; here products are between feature
channels after attention. Polynormer was listed as a future candidate in the
20260910 backbone literature notes, not established as a successful method.
The family is counted at most once, only after all validation stages.

Before any new scores, LncDrug is assigned the exact frozen-reference
identity policy. MiDrug uses the existing nested OOF reference residual
likelihood and fixed .1 logit correction. Identity, reference infrastructure
and fusion are not innovations. Mandatory seeds/partitions/outer gates and
the .001 LncDrug bound are unchanged. Historical and repeated development
partitions are not an independent test; no unseen evaluation is claimed.
