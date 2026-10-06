# Typed sum-of-squares relation circuits: prospective study

2026-09-16, before performance screening for this candidate.

The hypothesis is that a **jointly learned, tractable relational distribution
with multiple compatible squared components** supplies useful MiDrug residual
ranking information. Exact normalization, signed within-component interference,
separation of squared components, and auxiliary relation likelihood are tested
explicitly. Neither a scalar calibration nor the fixed integration weight is
counted as a technique. All circuit variants belong to one family.

## Primary literature and retrieval limits

1. Lorenzo Loconte, Stefan Mengel and Antonio Vergari, *Sum of Squares Circuits*,
   AAAI 2025, DOI [10.1609/aaai.v39i18.34100](https://doi.org/10.1609/aaai.v39i18.34100),
   arXiv:2408.11778v3, updated 24 May 2025. Full PDF and text retained; Introduction,
   circuit definitions and §4 Definition 4 read. Exact title and conference DOI
   matched Crossref. The linked AAAI page timed out and that failure is retained.
   The paper's exponential expressivity separations do not guarantee a benefit
   for this finite, shallow, three-variable adaptation.
2. Loconte, Di Mauro, Peharz and Vergari, *How to Turn Your Knowledge Graph
   Embeddings into Generative Models*, NeurIPS 2023, DOI
   [10.52202/075280-3395](https://doi.org/10.52202/075280-3395), arXiv:2305.15944v3,
   updated 16 January 2024. Full text obtained; §2, §4.1–4.2 and typed-constraint
   discussion read. This is the older relational-circuit foundation, not a 2026
   result. Exact MLE and efficient CP-squared normalization motivate the local
   construction.

Author commits were retrieved for `april-tools/sos-npcs` and `april-tools/gekcs`
(`8b3c8592ab1db852166c9d1cdf202c1cccc7e6d4` and
`ad53540323d7519c570bfd85cda237233685e646`). No author training code has been run or
copied. The implementation uses explicit CP formulas checked against independent
enumeration. It is not a reproduction of the papers' deep-circuit or benchmark
experiments. Related 2025 rule-circuit and Inception-PC papers were retrieved for
triage; their specific algorithms are not implemented or counted separately.

## Local probabilistic model

There are four compatible components of rank 16. For an allowed typed relation r,
component m has the real amplitude

`a_m(s,r,o) = sum_k E_subject[s,m,k] W[r,m,k] E_object[o,m,k]`.

The full nonnegative mass is `f(s,r,o)=sum_m a_m(s,r,o)^2`. Subject, object and
predicate tables are learned from scratch, with common N(0, 0.1²) initialization.
This is a transductive entity-ID model: no claim of new-entity or cold-start
generalization is made. It uses existing relation observations rather than a
new tabular feature bank or pretrained model; raw sequence and molecular feature
arrays are not used by this particular candidate.

For each relation, the legal subject/object types are fixed by the supplied
schema. Its exact normalizer is the sum of per-component Hadamard products of
the two Gram matrices and the predicate outer product. No sampled approximation
to the denominator is used. After normalization, a fixed 1% uniform component
on that typed Cartesian product prevents zero likelihoods. There is no silent
normalizer clipping. The mixture is still exactly normalized.

The original physical relations are augmented with one *sampled unobserved
MiDrug label* relation, populated only from the supporting folds. It does not
mean biological inhibition, validated absence, or a known biological prevalence.
The two MiDrug relation log densities provide a density ratio. A common learned
slope and intercept, both initially zero, convert it to an initially zero
residual on the unchanged nested-OOF strong reference.

In addition to `BCE(anchor_oof + residual, y)`, the full model minimizes 0.1 times
the mean of the conditional negative log likelihoods over the supported relation
types. Each relation gets equal weight; its uniform-model NLL is subtracted as
a parameter-independent diagnostic constant. This is a composite discriminative
and generative objective, not pure MLE. The role-type constraints are syntactic
schema constraints and do not establish any new biological law.

## Frozen controls

- `full`: sum of four squared CP amplitudes with the joint support likelihood.
- `single_square`: combine all 64 CP terms coherently before squaring. The same
  parameter tensors are used; this removes the separation between components.
- `monotone`: sum the squares of the individual rank-one products, eliminating
  signed interference. Its mass is a nonnegative CP mixture with 64 terms.
- `no_joint`: identical sum-of-squares predictor, but the auxiliary likelihood
  has coefficient zero. It is still computed for numerical and execution audit.

All three are mandatory controls. Parameters and random streams match across
arms, but circuit operation counts and effective relation supervision differ;
equal FLOPs or equal effective function classes are not claimed. In particular,
unused auxiliary predicates in `no_joint` cannot gain from relation labels.

## Evaluation and boundaries

The current optimization fold is removed from both target-positive facts and
sampled-zero support facts. Fixed-step fitting only accepts training arrays.
Evaluation labels do not enter the circuit or support objective. The baseline
offset is the existing independently checked nested OOF reference. The learned
parameters can use earlier optimization labels in the usual supervised sense;
the fold mask does not claim that trained entity embeddings are cross-fitted.

The deployment rule is MiDrug `anchor + 0.1 * residual`, with exact reference
identity for LncDrug. LncDrug scoring records are not new neural fits. The common
score-integration and scalar density-ratio calibration are not innovation claims.

Before performance, require six CPU checks, four five-step real MiDrug probes,
four LncDrug identities, 40 fold-isolation checks, and probability-sum checks over
the entire 605×216 Cartesian product. Then freeze learning rates **0.003/0.01**,
400 AdamW updates, weight decay 0.001, and 20-step checkpoints. Rank, component
count, smoothing, generative coefficient and integration weight are not tuned.

Historical seed 666 selects each arm by mean global ROC AUC and sklearn AP;
the full primary must improve both metrics over the strong reference and every
mandatory control. Only eligible models proceed to seeds 0 and 1, three fixed
development partitions and a separately frozen old-outer recheck. Earlier
development and old-outer exposure is acknowledged; these are not independent
cohorts. The original retained test source is not opened. Until all gates pass,
the candidate counts as zero validated innovations.
