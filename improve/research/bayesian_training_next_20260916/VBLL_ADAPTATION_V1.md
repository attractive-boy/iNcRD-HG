# Correlated variational last-layer experiment

This is an independent binary graph adaptation of Harrison, Willes and Snoek,
*Variational Bayesian Last Layers*, arXiv:2404.11599v1 (17 April 2024). The PDF
marks ICLR 2024; the publisher record was not independently verified in this
session. The full text, especially Sections 2.2 and 3.2, Equation (14), training,
prediction and classification initialization details, and the authors'
`DiscClassification` / `DenseNormal` implementations were read. The pinned
repository is `VectorInstitute/vbll`, commit
`142f21edd606c1584a15b003bbfb732b4ba2affc`. Author code was read, not executed.
The paper and code manifests in this directory record exact bytes and sources.

The hypothesis is that learning correlated uncertainty in the classifier weights
improves ranking through joint representation learning and predictive
marginalization. It does not change context76 columns, introduce a tabular
foundation model, or freeze REIFM features and attach an attribute kernel.
This is a Bayesian *weight-learning* experiment; no claim of globally new
invention or an exact reproduction is made. At most one such family can count
towards the requested innovations after all validation gates pass.

All arms share the existing two-layer typed graph encoder (width 64, four heads,
dropout 0.1), five query-deleted support graphs and nested OOF reference logits.
The classifier has two independently distributed Gaussian class weight vectors.
An explicit bias feature gives dimension D=65. The full arm learns a dense
Cholesky factor for each class; the diagonal arm removes within-class covariance;
MAP removes posterior variance but retains the same Gaussian mean prior. MAP is
not described as a delta posterior with zero Gaussian KL.

Class means start at zero, Cholesky diagonals at 1/D and off-diagonals at zero.
These deterministic initializations differ from the authors' random means and
small random covariance perturbations, and pair the encoder parameters and
random-number streams across all arms. Prior variance is 2/D. Independent logit
noise is zero, a case explicitly allowed by the paper; no noise/Wishart terms
are learned. The reference enters class logits as [0, anchor_oof].

For full and diagonal, the minimized objective is the mean over the omitted
training fold of `logsumexp(mean + offset + variance/2) - target_mean`, plus the
analytic Gaussian KL divided by the *total* number of training rows. This is
the negative Jensen lower bound, not the exact expected cross entropy. MAP uses
ordinary cross entropy plus the Gaussian mean penalty with identical scaling.
The encoder and head are jointly trained for 400 AdamW updates, with learning
rate candidates 0.001 and 0.003, betas (0.9, 0.999), clip norm 5. Encoder decay
is 0.001; classifier decay is zero because the explicit prior already supplies
its regularization. No stochastic weight samples are used during training.

For each support graph, binary predictive probability integrates
`sigmoid(anchor_eval + mean_1 - mean_0 + Gaussian(0, var_0 + var_1))` using
64-point Gauss-Hermite in log space. Its logit minus the reference is the
residual. The five support residuals are averaged and enter the fixed
`reference + 0.1 * residual` scoring formula. This is a numerical approximation
and a fixed support aggregation, not exact Bayesian marginalization of graphs.
After each selected formal fit, all evaluation queries are checked against a
128-point reference without labels; maximum residual error must not exceed
3e-5. Integration order cannot be selected using performance.

The selected weights also produce posterior-mean-only scores for description;
these get no checkpoint or hyperparameter search and are not an independent
training run or a primary control. All query reorder and batch checks reorder
the corresponding evaluation reference logits as well. The common metric
reader iterates recorded score names and supports this extra diagnostic field.

LncDrug uses the prospectively registered parameter-free identity: no graph,
classifier, posterior or training is created. The candidate must beat the
strong reference and both diagonal and MAP on mean MiDrug global ROC AUC and
sklearn AP, first in screening, then over three paired initializations. Each
metric must improve in at least two initializations; all LncDrug changes must
be at least -0.001. Three fixed development partitions and a frozen outer
recheck remain mandatory. Previously exposed development/outer outcomes are
acknowledged; none can be represented as an untouched independent cohort.

Five core numerical tests and three executor checks passed before formal
registration. A separate training-only GPU smoke and support-fold audit must
also pass before the protocol is sealed. Test fits, integration checks,
identity scoring and fixed composition do not count as validated innovations.
