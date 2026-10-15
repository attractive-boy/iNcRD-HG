# Query-corrected endpoint variance/covariance regularization

Only the original miRNA/drug inputs, original training inner three folds, seed42.
Final requirement remains original MiDrug test mean seeds42/43/44 AUC >=
0.9733929274189016 and AP >= 0.933683373732677; protected PLMF cannot decline.
No lncRNA nodes/features/relations/supervision, external datasets, annotations,
sequences or pretrained weights. Original validation/test are not opened here.

## Hypothesis and evidence

Training-only checkpoint spectra in ../midrug_representation_geometry_20261007/
show concentrated M1 endpoint representations. Reconstruction increases their
participation ranks but does not make them isotropic. Low rank is not itself an
error: binary supervision, repeated entities and correlated original attributes
can all explain it. This controlled experiment asks whether maintaining variance
and reducing dimension redundancy improves held-out association ranking.

NodeJEPA (arXiv2608.04381v1, 2026) uses VICReg-style variance and covariance
regularization alongside masked latent prediction. See the verified primary-text
review in ../midrug_representation_literature_20261007/READING_NOTES.md.
This is a single-view supervised adaptation, not a reproduction of NodeJEPA or
VICReg: no EMA teacher, latent prediction, invariant view loss, SIGReg, learned
projector, graph augmentation or new inference parameter is introduced.

## Frozen design

During the one ordinary training forward, capture the two corrected M1 endpoint
blocks entering lin1, before M2/M3 concatenation is fused. Both have width256.
For each endpoint separately, use parameter-free per-row LayerNorm (eps1e-5)
only inside this training objective; let Z be the normalized full training-query
matrix and C=(Z-mean(Z)).T@(Z-mean(Z))/(N-1).

V = mean(relu(1-sqrt(diag(C)+1e-4))).
K = sum_{i!=j}(C_ij^2)/256.
Average V and K equally across the two endpoint types.
Per-row LayerNorm imposes a zero row sum, so perfectly identity covariance is
not attainable; this is a soft redundancy penalty, not exact whitening. At a
fully constant representation the finite stabilized objective can have zero
gradient; the initial nonconstant model's encoder gradients are checked.

Three configurations, each on three inner folds:
- varcov: BCE + .2 V + .2 K + 10 reconstruction. Sole candidate.
- var: BCE + .2 V + 10 reconstruction. Variance-only ablation.
- cov: BCE + .2 K + 10 reconstruction. Covariance-only ablation.

The .2 coefficients are fixed before fits, inspired by NodeJEPA's stated values;
they are not assumed scale-equivalent to its GCN embeddings or transferable as
optimal weights. Both ramp from zero at epoch30 to full at epoch60.
Reconstruction ramps during first50 updates and uses original private sampler
seed20261006+fold. Main RMSprop lr1e-4, alpha.99, wd0; reconstruction wd1e-4;
zero_grad(set_to_none=True). Max800 updates, min300, patience200, earliest best
mean(inner AUC,AP). No weight/temperature/margin sweep after seeing results.

The sole candidate must improve BOTH mean metrics by at least .002 over pure
control, original unmasked10, var and cov to enter original validation.
No retrospective promotion of var/cov. Six controls reuse existing full replays
with their metadata/checkpoint/prediction/input hashes verified by path.

All M1/M2/M3 prediction definitions remain original; normalization and losses
are absent at inference. Training predictions use the actual query-edge removal.
Each full training set contributes once; validation labels enter neither term.
Variance/covariance couple training representations as an auxiliary loss, not
inference queries. BN must update exactly once per training step. Immutable
checkpoint replay, full physical query deletion, loss composition, and batch
invariance are checked before selection. Post-selection spectra are diagnostic.
