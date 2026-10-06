# Attribute-initialized Kuramoto phase propagation, v1

Frozen before any performance fitting of this candidate. This is a local sparse
heterogeneous graph adaptation of KoPE, not a reproduction of its image results.
Only an entirely validated mechanism family can count toward the requested three
innovations. KoPE and the failed Stuart–Landau trial share the oscillatory family
and could together contribute at most one innovation.

## Source and verified scope

- Xiao et al., *Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization
  for Improved Learning Efficiency*, arXiv:2604.07904v2, 24 June 2026.
  The paper and author repository report ICML 2026; the publisher record has not
  been independently verified. The full PDF and extracted text are retained.
- Author repository: `microsoft/Neuro-inspired_Phase_Encoding`, commit
  `0eade3cebc76978d2944f3ebb9f50f57f1b7ee53`. Its compact `vit_kope.py` has SHA256
  `1da2179783351c5e377f80fed0a47a783187894d1c8404d0d3027478272df5cc`.
  The README, MIT license, commit/tree and Git-blob verification are retained.
- CPU tests execute unchanged AST bodies for `rmsnorm`, `apply_rotary_pairs`,
  `KoPEAttention` and `KoPECoupling`. Copied weights on a complete support graph,
  with type embeddings zero, match attention/coupling forwards and input/parameter
  gradients in FP64. The full ViT, its training and its reported benchmarks were
  not executed. The larger `simdinov2` implementation was not downloaded.

## Adaptation and numerical boundaries

Initial node attributes use the existing row-RMS scaling and masked support-type
counts, followed by per-type projection, LayerNorm and GELU. Width is 64, heads 4,
depth 4, dropout 0.1. The local graph has no image coordinates. A bias-free linear
map of the initial node state produces eight phases, shared across heads. No node
ID, arbitrary feature-column order or synthetic image position is used as a
coordinate. This learned attribute initialization replaces the author's 2D RoPE.

Each layer linearly mixes cosine/sine channels using a shared matrix for the two
components, initialized to identity plus Xavier-normal noise with gain 0.1. The
paired squared norm is clamped below at 1e-5 before inverse square root. A clamp
can make a mixed vector nonunit; its occurrences are counted separately and are
not represented as successful exact unit normalization. Q, K and V rotate in
adjacent coordinate pairs. A source node's phase also rotates its type embedding
before key addition. The output rotates back using the destination phase. This
typed-edge adaptation passes a common-phase gauge invariance check.

After attention, residuals and FFN, one shared coupling module reads the updated
node features. Q and K use per-head RMS normalization; Q additionally has a
learnable channel scale. Typed sparse attention on the same support graph gives
the mean phase `m`. For every phase pair `u=(cos(theta),sin(theta))`,

```
t = m - dot(m,u) * u
u_next = normalize(u + 0.05*t, eps=1e-5)
```

With unit input this equals an angular increment
`atan(0.05*(m_s*cos(theta)-m_c*sin(theta)))`. An independent NumPy sparse typed
softmax plus this angular formula matches the implementation and directional
derivatives for features, angles and every coupling parameter group. Updated
unit-norm error must be at most 1e-5. Scores, states, loss, gradients, optimizer
state and predictions must remain finite. The post-layer update includes the
last layer, whose resulting phase is not consumed by the pair readout, following
the author's update ordering. The shared coupling is registered once.

The ordinary pair MLP has an exactly zero final layer. This project retains its
common PyTorch linear initialization, LayerNorm epsilon, FFN ratio 2 and dropout
placement, rather than the ViT-wide truncated-normal initialization, LayerNorm
epsilon 1e-6, FFN ratio 4 and image preprocessing. No full-model equivalence,
attention acceleration, equal effective capacity or first graph application is
claimed.

## Registered arms and budget

1. `full`: learned attribute phases, phase mixing and coupling updates.
2. `static_phase`: same stored initialization; freezes the coupling parameters and
   keeps the initial phases across layers. Attribute phases and mixing still learn.
3. `no_phase_rotation`: additionally freezes the initializer and mixing, using
   cosine 1 and sine 0 for feature rotations. With copied weights the entire
   feature block and its gradients equal ordinary typed sparse attention.

All arms compute the phase/mixing/coupling branches; ignored results retain zero
dependencies. This is an audited forward-work control, not a claim that active
capacity, backward work or runtime are equal. Initial stored tensors and random
streams match. Stored parameters total 228993. Active parameters are respectively
228993, 219457 and 217921, with frozen values excluded from AdamW and checked
unchanged in both selected and final models.

Screen seed 666, learning rates 0.001 and 0.003, 400 AdamW steps per arm/rate,
weight decay 0.001, gradient clip 5, checkpoint every 20 steps. Only the predeclared
two rates are selected; no phase-step, depth, width, normalization or fusion sweep.
Each training step omits the query fold, including reverse edges. Twenty
checkpoint evaluations and the selected-weight prediction give 505 forwards per
fit, each with four feature layers and four phase updates. The frozen scripts
derive exact counts from the five support graphs; data-dependent normalization
clamps and score/state maxima are audited separately. Formal GPU work is serial
on `cuda:1` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Evaluation and stop rule

MiDrug fits BCE on fixed nested OOF strong-reference logits plus the raw residual.
Evaluation always adds 0.1 times the mean raw residual over the five support
graphs to the frozen strong reference. LncDrug is prospectively an exact identity:
no graph fit, no new parameters, no optimizer and no new neural-fit count.

The seven CPU tests, three five-step MiDrug GPU checks, three Lnc identity checks
and 40 fold audits read no evaluation labels and compute no performance metrics.
Their hashes are frozen with the protocol. Exact final weights, AdamW and RNG are
saved separately from selected checkpoint weights; numerical failures preserve
the exact completed-step partial state.

Only a passed historical screen permits seeds 0 and 1. Three initializations must
give positive mean MiDrug AUC/AP increments, each positive in at least two of
three, and the primary must beat both controls on both means. Lnc metric losses
must be at most 0.001. Three fixed development partitions and a separately frozen
outer stage remain mandatory after historical success. History, fixed partitions
and earlier outer results have development exposure; they are not independent
tests. No original held-out test source has been accessed. Stop this version on
the first failed numerical or joint gate, preserve its evidence, and do not count
a single-arm gain as a validated innovation.
