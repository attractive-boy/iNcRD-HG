# KAA: prospective spline attention scoring

Taoran Fang, Tianhong Gao, Chunping Wang, Yihao Shang, Wei Chow, Lei Chen and
Yang Yang, *KAA: Kolmogorov-Arnold Attention for Enhancing Attentive Graph
Neural Networks*, ICLR 2025 Poster, OpenReview atXCzVSXTJ, arXiv 2501.13456v4.
The paper, OpenReview metadata and author repository `zjunet/KAA` agree.
Author tree: `00899bbb4b3d247d79f2d631c65b77a4148531fb`. The downloaded
KAA_GAT, KAA_GLCN, GT layer, KAN modules and README are read as source; no
author training scripts are executed.

Section 4.1 separates a fixed alignment function AF from a learned scalar
score mapping. KAA replaces that mapping with KAN and uses independent maps
per attention head. The GLCN variant uses absolute node-feature differences.
Its author code applies KAN to those differences, then neighbor softmax;
this is the closest concrete ancestor of this adaptation. GT implementations
instead replace a key projection; we do not conflate these variants.

Our fixed AF is `abs(tanh(LN(h_dst))-tanh(LN(h_src)))/2`, bounded in [0,1].
For each head the scalar map is `sum_d w_d*SiLU(AF_d) +
sum_d,b c_db*B_b(AF_d) + bias`. Bases are cubic B-splines on four uniform
intervals with extended knots, evaluated by Cox-de Boor. There are seven
basis coefficients per input dimension. Knots never update from any data.
Spline coefficients start at Gaussian std .02/sqrt(width), and base weights
at uniform +/-1/sqrt(width); this differs from the author's interpolated
noise initialization and optional independent spline scaling.

All score heads consume the same width-64 AF. A learned relation/head bias
and stable receiving-node softmax complete attention. Values and feedforward
blocks use ordinary linear layers, not KAN. Two layers, four heads, width
64, dropout .1 and the existing bipartite residual readout are fixed.

The primary is `full`; the required `mlp_score` control replaces only that
scalar map with independent 64->8->1 SiLU MLPs. This control has 2,116 score
parameters per layer versus KAN's 2,052 (128 more overall), so the candidate
has no parameter-count advantage. All other initial parameters and RNG state
are exactly paired using a CPU RNG fork during score-module creation. The
comparison is to this specific capacity-matched MLP, not all possible MLPs.
Scipy B-spline values/derivatives, finite-difference gradients, symmetry and
nonzero attention gradients are independently checked.

The paper's MRD proof relies on special nondegenerate alignment matrices,
dimension/sample relations and modified **zero-order** splines. It does not
prove unlimited ranking for this finite cubic network, and we make no such
claim. Our fixed AF, grid, initialization, typed graph and link-prediction
likelihood are disclosed adaptations. Spline scoring is a separate mechanism
from feature-product updates and from prior successful tabular replacements,
feature extensions, anchor shrinkage or frozen representation kernels.

LncDrug is prospectively the exact reference identity. MiDrug uses the
existing masked physical support graphs, nested OOF reference likelihood,
400 steps, two registered learning rates and .1 logit correction. No grid,
spline-order or architecture search is performed. The family can count only
after the same mechanism, seed, partition and outer gates; identity and
fusion count as neither training nor separate innovations.
