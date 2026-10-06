This is a prospective, independent link-prediction adaptation of expected path
signatures. It is not a reproduction of Graph2Tens and is not a new 2026 paper.

Source evidence

- Toth, Lee, Hacker and Oberhauser, *Capturing Graphs with Hypo-Elliptic
  Diffusions*, NeurIPS 2022, DOI 10.52202/068431-2812, arXiv:2205.14092v1.
  Sections 2--4 and the Appendix D initial condition were read. The tensor
  product, exponential lifting, transition operator and rank-one functionals
  supply the mathematical basis. The main text's inconsistent initial-condition
  notation is resolved by explicitly starting our recurrence at tensor unit /
  endpoint indicator, as in the probabilistic definition and appendix.
- *Scalable Machine Learning Algorithms using Path Signatures*,
  arXiv:2506.17634v2 (2025), Chapter 4, especially equations 4.12--4.13, was read.
  It is a later exposition of the graph construction, not independent new
  graph-performance evidence.
- Exact Crossref metadata, both full texts and their retrieval hashes are
  retained. The public author repository's commit metadata was retrieved, but
  author code was neither copied nor executed. Recent arXiv and Crossref
  follow-up searches are retained in FOLLOWUP_RETRIEVAL_V1.json. Most newer
  "signature" hits concern topology, temporal data or unrelated applications;
  no newer verified rough-path graph result is claimed here.

Task-specific hypothesis, fixed before performance observation

For a support-graph Markov walk B_0,...,B_K, represent a MiDrug pair (i,j) by
E[S^1,S^2 | B_0=i, B_K=j], with K=4. Learned bounded node attributes provide
eight pairs of coordinates u,v. Only existing physical/support arcs, their
reverse arcs and self loops enter a fixed uniform outgoing-arc transition P.
No query edge, node-ID order, feature-column order or physical timestamp is
introduced. Backtracking can cancel in a spatial signature; this finite model
does not claim full path identifiability, a characteristic kernel, or a general
expressivity theorem.

The proposed addition is a *bridge-conditioned* ordered path representation
for association prediction. The older Graph2Tens source principally describes
node/graph random-walk summaries. The hypothesis here is that route-dependent
area between fixed RNA/drug endpoints contains information absent from endpoint
moments and arrival probability. It is distinct from the prior local four-cycle
message network and neighbour-sequence state-space model, although all are
related broadly to graph paths. This family can count at most once; shared GAT,
reference offsets, fusion and a second revision do not count as innovations.

Exact calculation

Let Q indicate all 216 drug endpoints, M_0=Q and M_k=P M_(k-1). For each
coordinate pair define B_0=0 and

    B_k = P B_(k-1)
          + 1/2 [diag(u) P diag(v) - diag(v) P diag(u)] M_(k-1).

This computes the unnormalized expectation of the sum of consecutive vertex
wedges over every length-k walk. For positive M_K(i,j), its conditional Levy
area is

    A_ij = B_K(i,j)/M_K(i,j) - (u_i v_j-v_i u_j)/2,
    S2_uv = (u_j-u_i)(v_j-v_i)/2 + A_ij.

The first two feature blocks are the deterministic endpoint increments du,dv;
the third is S2_uv. A zero arrival probability produces zero signature and a
zero reachability flag, using a safe denominator only for that exact-zero
branch; no small positive probability is clipped. All sparse products and
normalization use float64. Sparse matrices have fixed values and dense-coordinate
gradients use the exact diagonal-factorization identity. Endpoint h features
remain available when no bridge exists. Each forward constructs all-node /
all-drug signatures independently of the query list.

Registered mechanism arms

- full: du, dv and ordered S2_uv.
- symmetrized: du, dv and du*dv/2, preserving the complete degree-two
  symmetric contraction under the shuffle identity, deleting only ordered area.
- first_level: du, dv and a zero degree-two block.

All three receive identical learned endpoint states, arrival probability and
reachability; they share tensor shapes, initialization, dropout streams,
optimizer steps and checkpoint opportunities. The first-level ablation has
inactive decoder columns; equal tensor counts are not a claim of equal active
function class. Controls require fewer sparse products, so compute cost is not
claimed equal. Both controls are mandatory on global MiDrug AUC and AP.

Frozen recipe and boundaries

Two width-64, four-head typed GAT layers, dropout 0.1, eight coordinate pairs,
tanh-bounded coordinates, four walk steps, level two, five support folds;
AdamW weight decay 0.001, learning rates 0.001/0.003, 400 updates, checkpoints
every 20. No rank, walk length, signature degree, mass floor, auxiliary loss or
fusion-weight search. The loss is BCEWithLogits(nested OOF reference + residual),
and evaluation uses reference + 0.1 * mean raw residual across support folds.
LncDrug uses exactly its previous score and has no new neural fit.

Queries and reverse query arcs are omitted before transitions, counts and
moments are built. Training-only probes and fold audits do not load evaluation
labels. Historical seed 666 screening must pass before seeds 0/1 confirmation;
fixed development splits 61453/7919/104729 and a frozen old-outer recheck remain
mandatory. These partitions have prior research exposure and are not independent
untouched tests. The original retained test files are not used.

Independent checks include explicit enumeration and NumPy Chen products,
finite-difference autograd checks, translation and shuffle identities, opposite
ordered paths with equal endpoints, reversed closed-path area, node relabeling,
query batching, fold isolation, exact LncDrug identity, and failure snapshots.
Passing implementation checks does not establish a performance improvement.

Before formal freezing, the first real-graph setup exposed an overly strict
fixed row-sum assertion: a high-degree node accumulated 4.0634e-14 roundoff.
No training had started. The previous code/checks and failure description are
retained in pretraining_row_sum_roundoff_v1. The validation bound now scales
with the number of positive summands; transition entries and the model are
unchanged. The revised six checks, three five-update MiDrug GPU probes, three
LncDrug identity probes and 40 fold-boundary audits all passed. Full-model
probe peak allocation was about 1.87 GiB. No evaluation metrics were computed
by these probes.
