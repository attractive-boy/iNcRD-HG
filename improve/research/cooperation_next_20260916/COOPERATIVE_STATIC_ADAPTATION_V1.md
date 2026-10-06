# Initial-state cooperative policies: a narrower same-family revision

Both earlier Co-GNN versions retain their frozen failures. The local evolving
policy passed historical gates but failed fixed-partition AP and the static
policy contrast. Its `static_policy` comparator has small positive mean AUC/AP
versus the reference across those partitions. This observation motivates a
new **within-family** hypothesis; it is not validation of the old full model,
an independent screening set, or a second cooperative innovation.

The new `full` is exactly `cooperative_graph_v2.static_policy`: its trained
node-specific send/receive probabilities read the initial projected attributes
and support relation counts, not the evolving environment state. Probabilities
are the same across layers; each layer still samples its own Gumbel actions.
Thus the graph realization is not fixed across layers. The four environment
layers, all parameters, initial values, noise, degree normalization, graph
boundaries and residual likelihood are unchanged. This is a restricted
adaptation of the original ICML 2024 Co-GNN mechanism and its author-code
factorization, documented in COOPERATIVE_ADAPTATION_V1.md.

Two new mandatory controls are declared before their fitting:

- `soft_actions`: the same initial-state policy with continuous Gumbel
  relaxations in place of straight-through discrete actions.
- `shared_policy`: the same trainable policy network, but its logits are
  averaged across nodes and broadcast back before discrete sampling. All
  parameters still participate, yet send/receive probabilities cannot depend
  on the individual sender/receiver. This tests the value of node-specific
  communication beyond a globally learned stochastic message-retention rate.

The claim is now discrete, node-specific cooperative communication. We no
longer claim benefits from neighborhood look-ahead or adaptation to evolving
layer states, both of which lacked support. No old control is silently renamed
as a success, and all previous judgments remain accessible.

Full's historical and fixed-partition predictions have already been observed.
Their use is explicit: if all inputs/recipes/hashes agree and saved predictions
replay exactly, existing artifacts are adopted byte-for-byte. Screen adopts
two neural weights plus two identities; confirmation, only if the new screen
passes, adopts two more of each. Two new controls require four new MiDrug
screen fits and four new MiDrug confirmation fits. After confirmation the
totals are eight new neural fits, four adopted neural fits and twelve identity
records (four adopted). The corresponding fixed-partition full weights may
also be adopted only after a separate exact recipe/input/replay audit.

LncDrug remains the prospectively declared zero-residual reference identity;
it is not neural training or a separate technical contribution. Learning
rates .001/.003, 400 steps, eight evaluation draws, five support folds, .1
logit correction, all historical/partition/outer gates and .001 tolerance
remain unchanged. A successful result could count at most once for the entire
cooperative family. Outer verification is still required and has not been
performed for this revision.
