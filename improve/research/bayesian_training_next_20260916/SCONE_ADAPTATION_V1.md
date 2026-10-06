# Score-based stochastic augmentation

The source is Lee, Choi, Wi, Cho and Park, *SCONE: A Novel Stochastic Sampling
to Generate Contrastive Views and Hard Negative Samples for Recommendation*,
WSDM 2025, DOI 10.1145/3701551.3703522. Crossref returned the exact title,
authors, ACM proceedings and 10 March 2025 publication date. The OA version read
is arXiv:2405.00287v2. Sections 2.2 and 4, Algorithms 1–3, and Equations
(7)–(18) were examined alongside the author's fixed repository
`jeongwhanchoi/SCONE`, commit `37292a56d58d580cbcf73de4aab5b9b5572544fd`.
The primary model, score loss, SDE, sampler, utilities, defaults and score-layer
files were read; author code was not executed. The seven supporting source
files were checked against the pinned Git tree's blob hashes. Raw GitHub
timeouts and successful public Git-blob recovery remain in separate manifests.

The proposed mechanism is learning a score model of node embeddings and using
its reverse stochastic process to train the graph encoder with contrastive
views and progressively harder negative representations. The generator is
used during training, not as a new predictor of graph adjacency. No synthetic
embedding is recorded as an observed edge or new biological association. It is
related to the broader diffusion/generative family already explored in this
project; only one validated generative family can count, and no previous failed
variant is relabelled a success.

This local adaptation retains the typed two-layer graph encoder (width 64,
four heads, dropout 0.1), query-deleted support folds, BCE likelihood offset by
the nested OOF reference, and fixed evaluation composition. The auxiliary
losses are 0.1 times a pairwise logistic ranking loss and 0.05 times symmetric
InfoNCE at temperature 0.2. Triplets use only actual positive and zero-labelled
training pairs in the currently omitted fold, with the same RNA on both sides.
The negative embedding is replaced by a synthetic representation, retaining
that original pair's reference logit. This auxiliary objective is a local
regularizer; it is not a likelihood for a newly observed negative drug.

The score network follows the paper's encoder/bottom/decoder structure with a
skip connection and sinusoidal noise-level embedding. Widths are 64 and 128.
It predicts Gaussian noise, with score defined as `-predicted_noise / sigma`;
this is an explicit output reparameterization of the weighted denoising score
loss. The author VE implementation instead returns its score directly. The
noise schedule remains sigma_min=0.01, sigma_max=50, intermediate time=0.1,
ten reverse steps and injection weight=0.9. Noise prediction and the use of a
typed attribute encoder instead of LightGCN are adaptations, not reproductions.

The reverse process follows the paper's variance-increment Algorithms 1/2:
initialize with variance `sigma(start)^2 - sigma(0)^2`; at each step inject
the evolving positive embedding into the negative, add the variance increment
times the learned score, and add independent Gaussian noise scaled by the
square root of that increment. Every step includes noise and injection, as
written in the paper. The released implementation uses Euler–Maruyama with
fixed dt, has one fewer injection in its negative sampler, and returns the
last predictor mean. These differences are not hidden behind a reproduction
claim. Finite-step samples are not claimed to follow the exact data distribution.
Score training uses the author's marginal standard deviation sigma(t), whereas
initial sampling uses the paper's variance difference. With nonzero sigma_min,
these are different finite-noise conventions; this adaptation does not claim
an exactly matched continuous-time marginal or an exact likelihood bound.

Three arms share all parameters, initialization, score training, random tensors
and the two optimizer updates per step. `no_injection` sets the injection
weight to one. `no_reverse_drift` sets only the learned reverse drift to zero,
keeping every noise draw and injection. Full must beat both on mean global
MiDrug ROC AUC and sklearn AP. Both counterfactuals continue training the
same score model, preventing a difference in optimization budget.

At each of 400 updates, first train the score network with detached graph
features using AdamW at 0.001 and zero decay. Then freeze score parameters and
differentiate the augmented loss through reverse sampling into the encoder.
The encoder uses learning-rate candidates 0.001 and 0.003, decay 0.001 and
clip norm 5; the score gradient is also clipped at 5. Dropout and sampling RNG
are separate and paired across arms. There are 400 encoder updates and 400
score updates, i.e. 800 backward evaluations, not 800 independent model fits.

Prediction uses only the trained graph residual, with deterministic five-support
averaging and `reference + 0.1 * residual`. LncDrug is a parameter-free identity
with no graph or generator training. Screening, three initializations, three
fixed development partitions and the frozen outer recheck remain mandatory;
every LncDrug metric/seed/split may decline by at most 0.001. Source/inputs are
sealed only after numerical tests and training-only GPU/isolation checks pass.
This document does not claim an improvement before those formal results exist.
