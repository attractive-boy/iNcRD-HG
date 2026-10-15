# M1/M2 fusion input direction and scale

The goal remains original MiDrug test mean seeds42/43/44 AUC >= .9733929274189016
and AP >= .933683373732677, with protected PLMF MDR/MDS not declining.
Only frozen miRNA/drug data and original training labels; no lncRNA, external
datasets, new annotations/sequences or pretrained weights. Inner folds remain
the three fixed partitions of original training. No original validation/test.

## Evidence and hypothesis

The prior variance/covariance regularizer did not pass the gate. It made its
normalized endpoint copies more diverse, while the raw M1 vectors supplied to
the fusion head remained dominated by few directions. See the frozen geometry
POST_EXPERIMENT.md and variance/covariance report. This motivates testing the
representation actually consumed by the original fusion head.

Earlier receiver-degree normalization, normalized auxiliary branches and TabM
heads address different operations; this is a basic fusion preprocessing
experiment, not a novel normalization algorithm or a reproduction of NodeJEPA.

## Fixed intervention and matched arms

Compute the original two M1 endpoint blocks and four projected M2 blocks. For
each query and each block independently (dimension D), calculate
  mu = mean(h), s = sqrt(mean((h-mu)^2) + 1e-5), u = (h-mu)/s.
M3 statistics and EASE blocks remain unchanged. Append twelve slots to the
fusion input (two per graph block), extending lin1 with twelve zero columns.
All original weights and reconstruction initialization are shared across arms;
no extra random draws or learned normalization parameters.

- raw: original blocks h, twelve zeros. Matched original-input control.
- direction: replace six graph blocks by u, twelve zeros.
- direction_scale: replace them by u, append asinh(mu), log1p(s) for each block.

The direction/scale mapping is invertible: mu=sinh(t0), s=expm1(t1), h=u*s+mu.
This preserves input information mathematically; it does not guarantee the same
function is equally easy for the finite MLP to learn. The direction-only arm
discards mean and radial information. M3 retains its original degree/path facts.
All arms store the same parameters, but dummy columns in raw/direction receive
zero gradient; equal storage is not equal effective capacity. Initial raw must
match original unmasked10, and direction_scale must match direction because
their added weights start at zero. Neither normalized arm has to match raw.

Every arm uses original BCE+10 reconstruction (first50-update reconstruction
ramp). No variance/covariance loss remains. Original RMSprop lr1e-4, alpha.99,
main wd0, decoder wd1e-4, set_to_none=True; reconstruction private generator
seed20261006+fold. Max800/min300/patience200, earliest maximum mean(inner AUC,AP).
Nine fits: three arms x three inner folds, seed42. The sole candidate is
direction_scale; mean AUC and AP must both exceed pure control, unmasked10,
matched raw and direction by at least .002 before original validation.
No retrospective promotion of controls or post-result normalization sweep.

## Checks and interpretation

Independent NumPy transformation, differentiable reconstruction identity,
float64 gradient checks, row/within-block-column permutation and constant
blocks. The new transformation is per query, so inference batching cannot
change it. Original M1/M2/M3 physical deletion checks cover the entire output.
Training must update BN once; heldout labels cannot determine inputs. Replays
check saved metrics, selection, loss composition, finite weights and initial
zero columns remaining exactly zero in raw/direction. Decoder removal preserves
the trained inference computation; normalization itself is part of inference.

After freezing selection, report train metrics, normalized geometry, scale
input gradients/weights and a zero-scale-input intervention on the trained
direction_scale head. This intervention does not undo joint training and cannot
be used to revise selection. Preserve all prior evidence and PLMF artifacts.
