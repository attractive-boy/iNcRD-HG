# Cooperative node communication: prospective MiDrug experiment

Ben Finkelshtein, Xingyue Huang, Michael Bronstein and İsmail İlkan Ceylan,
*Cooperative Graph Neural Networks*, ICML 2024, PMLR 235,
<https://proceedings.mlr.press/v235/finkelshtein24a.html>, arXiv 2310.01267v2.
The publisher metadata, arXiv full paper and public author implementation at
`benfinkelshtein/CoGNN`, tree `79e930861043b39b31c49ac918ff545395811cc1`, were
read. This is an independent multi-relational link-prediction adaptation.

## Mechanism and source distinctions

Each node chooses whether to send and whether to receive messages. Directed
edge u -> v is active only when u sends and v receives. The action network reads
the original support graph and the current node state; the environment network
then updates states on the action-induced communication pattern. The action
network is shared across environment layers.

The paper describes a distribution over four actions (standard, listen,
broadcast, isolate). The inspected author code factorizes this into two binary
Gumbel-softmax networks and samples even in evaluation. We use independent
binary send/receive choices, with a shared typed policy trunk and separate
binary output heads. This retains the four possible joint actions but restricts
their joint distribution; it is not a general four-category policy.

The paper's inverse-temperature formula and the implementation put the constant
on different sides of softplus. We avoid that ambiguity by registering a fixed
temperature of 1.0. The author `WeightedGNNConv` with PyG mean aggregation
divides by the original number of edges even when some edge weights are zero.
We explicitly retain that fixed physical-degree normalization. It is not the
mean over only retained neighbors.

The local network uses original type-specific node features and current-support
relation counts, width 64 and four environment layers. The shared action trunk
is a one-hop typed mean operator. Each environment layer has its own typed
message projection and relation embeddings, a self-state update, LayerNorm,
GELU and dropout 0.1. Synthetic self loops are removed from message edges because
the self-state update is explicit; original count features follow the frozen
count convention. No absent edge is added. The raw pair residual uses the same
four endpoint combinations as the common graph baseline and a zero-initialized
last linear layer.

The primary is `full`. Mandatory controls are:

- `soft_actions`: identical binary Gumbel draws and parameters, but continuous
  Gumbel-softmax weights instead of straight-through discrete choices.
- `no_lookahead`: identical parameters and discrete choices, but the action
  network's neighbor input is replaced by the node's own state. Every policy
  projection remains in the computation. The environment still uses the same
  physical graph. This isolates the policy's additional neighborhood inspection.

All arms draw the same number of random values in training. Evaluation averages
eight raw-residual predictions using fixed, saved, label-independent Gumbel
draws per node and layer, common across arms and folds. These draws do not depend
on queried pairs. This gives exactly reproducible reordered and chunked queries.
It is a finite Monte Carlo predictor, not exact marginalization. Under a node
permutation the random arrays must also be permuted: equivariance is conditional
on that correspondence and holds in distribution without it. No universal graph
distinguishability or long-range approximation theorem is claimed for this
finite-width, four-layer model.

## Prospective scope and validation

Only MiDrug receives a new residual. LncDrug is the exact zero-residual identity
declared here before any candidate performance results. It has zero parameters
and zero training steps; its scoring records are not neural fits. The policy
does not revise any previous failed Loopy, Copula or other candidate.

The common nested OOF reference, BCE residual objective, final raw-residual
weight 0.1, AdamW weight decay 0.001, clip 5, 400 training steps, 20-step checks,
learning rates 0.001/0.003, screen seed 666 and confirm seeds 0/1 are fixed before
screening. Full must improve MiDrug AUC and sklearn AP and exceed both controls
on both metrics, then pass the existing fixed partitions and frozen outer
recheck. Those development surfaces have earlier exposure and overlap.

This family is an adaptive communication mechanism; it is related to the earlier
failed learned-adjacency family but changes directed information flow on present
edges instead of learning new continuous edges. Neither its controls, Monte Carlo
inference, identity branch, nor the evaluation infrastructure count as separate
innovations. The paper's node/graph classification results provide motivation,
not evidence of improved RNA-drug resistance ranking.
