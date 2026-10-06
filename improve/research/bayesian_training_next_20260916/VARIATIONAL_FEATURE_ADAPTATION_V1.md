# Narrower VBLL feature-learning hypothesis

The complete-covariance VBLL v1 failed its declared screen: its AP was lower
than the diagonal posterior's AP. That failure and every original record remain
unchanged. This follow-up uses the already observed diagonal posterior as full
and the already observed MAP baseline. It is an adaptive revision of the same
Bayesian-weight family, not an independent screen or an additional innovation.
It makes no claim that dense within-class covariance helps.

The new scientific question is whether marginal-variance gradients shaping the
graph representation explain the advantage of diagonal variational training.
The new `no_feature_variance` control detaches the encoder features *only* in
`variance = feature_squared @ exp(2 * log_scale).T`. The mean path, classifier
variance gradients, Gaussian KL/N, numerical forward loss, parameter counts,
initial values, query-deleted support graphs, RNG, training budget and predictive
quadrature match full. This control is a partial-gradient ablation of Equation
(14); it is not claimed to optimize a proper alternative ELBO. MAP remains a
necessary independent baseline with the same Gaussian mean prior.

Full and MAP call their previous diagonal/MAP implementations exactly. Unit
checks compare saved CPU training states, traces and RNG; formal adoption must
compare all input arrays and reproduce every saved residual bit for bit before
copying four MiDrug model records and four LncDrug identity records. Only the
two new feature-gradient control runs are newly fitted in screening. If all
screen gates pass, all three arms receive seeds 0 and 1, adding six new fits.
Original fits and identity records never count as newly trained models.

The source is the same Harrison, Willes and Snoek VBLL paper and pinned author
code documented in `VBLL_ADAPTATION_V1.md`. The training, prior, initialization,
integration, composition, diagnostic, literature-verification and data-exposure
boundaries from that note remain applicable. The common `fit`, `predict`,
`predict_components` and `run` bodies are unchanged. The setup adapter changes
only the control's feature-variance gradient. No covariance or integration-order
search is introduced. Existing 40-fold isolation evidence is inherited with
hash checks, not recounted as 40 new audits.

The new full must exceed both the stop-feature-variance control and MAP in mean
global MiDrug ROC AUC and sklearn AP, improve against the strong reference in
at least two of three initializations, and keep every LncDrug change above
-0.001. Fixed-partition and frozen outer validation are still mandatory. LncDrug
continues to be exactly unchanged through a parameter-free identity branch.
