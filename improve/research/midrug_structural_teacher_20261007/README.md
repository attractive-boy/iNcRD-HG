# Structured masking and visible-context latent prediction

Original MiDrug test target: seed42/43/44 means AUC >= .9733929274189016 and
AP >= .933683373732677. PLMF MDR/MDS protected results must not decline.
Only frozen miRNA/drug features, original same-type graphs and current training
positive associations. No lncRNA, external data, new sequence/annotation or
pretrained weight. All development stays in the three original-training inner
folds; no new original validation/test evaluation.

## Motivation, source and historical distinction

The normalized fusion experiment increased training fit but decreased held-out
metrics. Keep the original inference input and test an auxiliary context task.
NodeJEPA (arXiv2608.04381v1, Aug2026), read in the existing representation
literature archive, predicts masked node latents using structural masking and
visible-context predictors. Its published evaluation is mainly node classification.

Historical improve/masked_graph_teacher.py already implemented GraphMAE2-style
attribute reconstruction, two remasking views and an all-node squared-cosine EMA
target on a different four-type model. That family is not new. The old loader
and model are not imported. This adaptation retains the actual pure-miRNA M123,
uses clustered same-type masking, predicts only masked targets from visible
source states, and centers latent targets within each node type. It is not a
complete reproduction of NodeJEPA or GraphMAE2.

## Fixed auxiliary task

Before training, prepare 800 Boolean masks for each fold and arm. Each epoch
masks exactly round(.5*N) nodes of each type (302 miRNAs and108 drugs).
For each pair of epochs, use randomized breadth-first traversal of the original
same-type graph (undirected, unique, non-self neighbors) for one block mask,
restarting in another component when needed; its complement forms the other.
For605 miRNAs exclude one uniformly drawn complement node from both masks so
both contain302 nodes. Randomize the order of the two block masks. The random
control partitions the exact same available nodes uniformly into two sets.
Thus every two epochs match per-node exposure exactly between block/random,
and each node is masked once per pair except the one omitted miRNA. This is not
the paper's k-hop/radius curriculum. Masks can be disconnected. Record edge and
node exposure statistics. No association labels or model adjacency are changed.

An initial single-BFS schedule had zero/800 exposures for some nodes. That design
was rejected before any training or performance evaluation; source and arrays
are preserved in precheck_unbalanced_masks/. The paired design addresses this
measured sampling imbalance, with no change based on validation results.

Replace masked input rows by separate learned per-type mean-initialized tokens.
Encode with the original shared GeneralConv x2 encoder on the current training
graph. Row-normalize these context states with parameter-free LayerNorm eps1e-5.
For each target node, independently mean-pool incoming VISIBLE source states for
its same-type and cross-type relations, retaining released self-loop multiplicity.
An empty visible pool is zero. Concatenate the two 256-d pools and four descriptors:
log1p(total degree), log1p(visible degree) for each relation. A separate per-type
516->128->256 GELU MLP predicts the target latent. No target state is pooled as a
source; the context encoder can still propagate learned mask tokens.

The target is the clean-input teacher's layer2 embedding, centered over all nodes
of the same type and L2-normalized (eps1e-8), detached from gradient. Centering
prevents the large original message-sum mean from supplying an easy common-vector
target. It is a local adaptation, not a claim about the original paper's formula.
Only masked, nonzero targets enter mean(1-cosine); miRNA/drug losses are averaged.

- block_ema: clustered masking, teacher momentum.99. Sole candidate.
- random_ema: random masking, teacher momentum.99. Masking control.
- block_current: clustered masking, teacher copied from current student after
  each update (momentum0). Temporal-target control.

All teacher encoders start as exact copies of the initialized student; no teacher
projector. All arms have identical inference/auxiliary parameters and initialization.
Teacher parameters never receive gradients or enter the optimizer. After every
student update, teacher parameters use the declared EMA/copy and buffers copy;
an integer counter records updates. Auxiliary targets always use the pre-update
teacher. Context models never consume held-out labels.

## Training and selection

BCE + 10 original unmasked reconstruction + .1 latent loss. Reconstruction ramps
for50 updates; latent is off through30, ramps31..60. When latent weight is zero,
its parameters receive no gradient/weight-decay update. Teacher still tracks the
student every step. Reconstruction sampling remains its original private generator
seed20261006+fold; masks have independent preparation seed20261007+fold.
RMSprop lr1e-4, alpha.99; original main wd0, reconstruction/context heads wd1e-4.
set_to_none=True. Max800/min300/patience200; earliest maximum mean(inner AUC,AP).

Nine fits, seed42. Sole candidate block_ema must beat pure control, unmasked10,
random_ema and block_current by >=.002 in BOTH mean inner metrics before original
validation. Do not promote a control retrospectively or tune momentum/masking
fraction/loss weight based on these results. Final test dual+.02 remains required.

## Verification

Independent cosine/gradient and dense visible-pooling checks, target centering,
masked-attribute invariance and exactly zero masked-input gradient, EMA/copy
recurrence over real updates, teacher stop-gradient, no auxiliary updates before
warmup, BN once, fixed masks/counts, heldout-label invariance and full physical
query deletion. Replays check all source/input hashes, predictions, selection,
loss composition, teacher update counts, current-copy equality and inference
batch invariance. Removing auxiliary reconstruction/context modules must retain
the original M123 prediction. Geometry/context-loss diagnostics occur only after
selection is frozen and do not revise it.
