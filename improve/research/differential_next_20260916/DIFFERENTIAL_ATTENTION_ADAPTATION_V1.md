# Differential attention: prospective typed graph adaptation

Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang and Furu Wei,
*Differential Transformer*, arXiv 2410.05258v2, is marked as an ICLR 2025
conference paper in its verified full text. The arXiv exact-ID metadata and
author Microsoft repository match. The OpenReview title query returned HTTP
403; it was not bypassed or retried. CrossRef's title results concerned
electrical transformers and were excluded. The initial PDF title check
failed on spaced small caps; a new audit verifies the existing PDF after
case/whitespace normalization, preserving the failed audit and original file.

Section 2 of the paper subtracts two separately normalized attention maps,
using a layer-shared exponentially parameterized lambda and per-head RMSNorm.
The author repository additionally publishes **Diff Transformer V2**, whose
README and code replace this with two query heads sharing the same key/value
head, a token/head sigmoid lambda projection, and no post-attention RMSNorm.
V2 is treated here as an author implementation release, not as a second
peer-reviewed ICLR paper. Sources are pinned to Diff-Transformer subtree
`9fb60a455ed2865348e75d0d07d3a6907842d220`; downloaded blobs are SHA-verified.
Only source text is read; author training/example programs are not executed.

The local adaptation uses that V2 pairing on a sparse typed physical graph:
`A1=softmax_dst(Q1*(K+edge_key)/sqrt(d));
A2=softmax_dst(Q2*(K+edge_key)/sqrt(d));
lambda=sigmoid(Wlambda*LN(h));
out=((A1-lambda*A2)/(1+lambda))*V`.
Q layout is `(node, output_head, 2, head_dim)`: adjacent Q heads share
**exactly the same K/V**, as required by the author's V2 note. Each softmax
is over incoming support arcs only, including the ordinary self loops.

Dividing by `1+lambda` is our explicit graph adaptation, not in V2 code.
It bounds the incoming signed coefficient L1 norm by one and is applied to
both arms. The mandatory `additive_attention` control changes only the minus
sign to plus; its normalized incoming coefficients sum to one. The candidate
can attenuate common signals and assign negative message weights. These
weights are not negative biological associations and do not change graph facts.

Total width 64, four output heads (eight Q heads), two layers, dropout .1,
type-specific attribute/count projections, LayerNorm/GELU feedforward blocks,
and the existing bipartite residual readout are fixed. Unlike V1, output head
count is not halved. There is no RoPE, causal sequence mask, FlashAttention,
compiler, or pretrained language model. All candidate/control parameters,
initial values, RNG streams and support arrays are identical. We claim only
this graph-level test of differential cancellation, not language-model scaling,
hallucination improvements, a biological noise model or a universal guarantee.

The new family is distinct from KAA's spline score mapping and Polynormer's
feature products. The 20260913 recommendation notes mentioned differential
interactions as possible future work; no successful implementation was found.
Before results, LncDrug is an exact reference identity. MiDrug uses unchanged
nested OOF reference residual training, 400 steps, lr .001/.003 and .1 logit
correction. Historical, fixed-partition and outer gates remain mandatory.
Only one family could count; its control, reference and normalization do not
count as separate innovations. Existing failures and defaults are preserved.
