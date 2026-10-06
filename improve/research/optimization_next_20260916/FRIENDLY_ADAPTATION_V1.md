# Friendly-SAM: prospective MiDrug-priority study

Tao Li, Pan Zhou, Zhengbao He, Xinwen Cheng and Xiaolin Huang,
*Friendly Sharpness-Aware Minimization*, CVPR 2024,
DOI `10.1109/CVPR52733.2024.00538`, arXiv `2403.12350`. The CVF open-access full
paper was downloaded and its empirical motivation, Sections 3–6, assumptions,
algorithm, experiments and limitations were read. Author code at
`nblt/F-SAM`, tree `2e2f9899ab473880430640fa5def4ad8f5aa2401`, was inspected.

## Mechanism and exact implementation choice

SAM evaluates the training gradient at a bounded adversarial parameter
perturbation. F-SAM removes an estimate of the common gradient component from
the perturbation direction. Following the public author implementation:

```
g_t = gradient of the current fold's clean residual BCE
first step: m_t = g_t; d_t = g_t
later: d_t = g_t - sigma * m_(t-1)
       m_t = decay * m_(t-1) + (1-decay) * g_t
epsilon_t = rho * d_t / ||d_t||_2
g'_t = gradient of the same stochastic loss at theta_t + epsilon_t
theta_(t+1) = AdamW(theta_t, g'_t)
```

The paper pseudocode updates EMA before subtracting it and initializes it at
zero. The author code subtracts the previous EMA and initializes it with the
first clean gradient. With sigma=1 and an identical EMA history, the old/new EMA
directions differ only by a positive scale, which normalization removes; the
initial EMA history still differs. This study explicitly chooses the author-code
initialization/order. It does not call the finite EMA residual an exact
orthogonal projection or an unbiased full-gradient estimate.

All arms use the same ordinary typed graph-attention network, original node
attributes, relation-degree inputs, nested reference offsets and final 0.1 raw
residual scale. The primary is `full` (F-SAM); mandatory controls are plain `sam`
and `erm`. Each performs two gradient evaluations per optimizer step, using the
same support fold and exactly the same dropout realization in the two passes.
ERM has zero perturbation, so its second pass reproduces the first. It is checked
against ordinary one-pass AdamW at zero radius. Parameters are restored by exact
copies before the base optimizer update and after a failed second pass.

The fixed optimizer settings are rho=0.05, sigma=1, decay=0.9, and gradient clip 5.
Width 64, depth 2, four heads, dropout 0.1, 400 optimizer steps, check interval 20,
two learning rates (0.001/0.003), and seed 666 screening with 0/1 confirmation are
declared before evaluating this candidate. No radius/momentum search is planned.

## LncDrug policy and counting

This fresh study changes only MiDrug. LncDrug always returns a zero new residual,
so its score is exactly its existing frozen strong reference. This policy is
declared before any F-SAM performance results and directly preserves the user's
LncDrug requirement. It is not retroactively applied to failed prior candidates.

The identity branch constructs no new graph, has no parameters, performs zero
gradient/optimizer steps, and saves an empty state with an explicit identity
record. Its query scores, including reordered/chunked queries, are verified
exactly. Identity records are **not new neural fits or new techniques**. The
common legacy assessor's `completed_fit_records` field counts serialized scoring
records; companion classification files give the actual neural fitting count.
The study therefore has six new MiDrug fits at screen and six more at confirm if
eligible, plus six identity scoring records at each stage. A claim of improved
training on LncDrug would be unsupported.

## Evidence limits and validation

The paper's convergence result assumes SGD, smoothness/noise conditions and
specified learning/radius schedules. Our AdamW, constant-radius, cross-fitted
graph setting does not establish these conditions. The paper's vision accuracy
and label-noise experiments are not evidence of MiDrug AUC or AP improvement.

Seven core/executor checks passed: closed-form two-step updates, clean-gradient
EMA tracking, stochastic zero-radius equivalence, exception restoration, equal
architectures and held-out support, exact LncDrug identity, and actual gradient
counts/identity accounting. Training-only CUDA-1 smoke contains three five-step
MiDrug fits and three identity replays. All 20 MiDrug support folds and 20 LncDrug
identity cases were checked, without evaluation labels or performance metrics.

The primary must increase both MiDrug global AUC and sklearn AP, improve each in
at least two of three historical initializations, and exceed both controls on
both mean metrics. Three fixed development partitions and a frozen outer recheck
remain mandatory. LncDrug's unchanged reference is verified at every stage.
Only one optimizer-mechanism family can be counted; identity deployment, SAM,
ERM, shared graph architecture and fixed reference blending are not innovations.

## Other sources found during this search

- *Efficient Sharpness-Aware Minimization for Molecular Graph Transformer Models*,
  ICLR 2024 Poster (`Od39h4XQ3Y`, GraphSAM), was verified from OpenReview metadata.
  It targets gradient approximation/efficiency. Its abstract is not a full-paper
  review or justification of the selected F-SAM implementation.
- *Rethinking Graph Generalization through the Lens of Sharpness-Aware
  Minimization*, WWW 2026, DOI `10.1145/3774904.3792707`, arXiv `2602.08855v1`,
  was verified by CrossRef, OpenAlex and the arXiv API. It proposes E2A latent
  augmentation. ACM returned 403 and the two arXiv PDF attempts returned a
  connection reset and 406. No access control was bypassed. The public author
  repository tree currently contains only a README. Full method reading and
  implementation have not been completed; no performance claim is made.
