# Two-sided Kronecker covariance learning for graph residual training

This is a prospective independent adaptation. No performance metrics for this
candidate have been consulted when writing this recipe. Prior failures,
including F-SAM and Muon, remain visible and preserved. At most one successful
optimizer-family direction may be counted; SOAP and KL variants do not count
as separate innovations.

## Sources actually read

- Wu Lin, Scott C. Lowe, Felix Dangel, Runa Eschenhagen, Zikun Xu and Roger B.
  Grosse, *Understanding and Improving Shampoo and SOAP via Kullback-Leibler
  Minimization*, arXiv:2509.03378v10, revised 2026-06-22. Read Sections 3--5,
  equations (5)--(7), Figures 3 and 12, experimental scope and limitations.
  [Open PDF](https://arxiv.org/pdf/2509.03378v10), SHA256
  `41e7edd5b683d4a370bd17bb84c2ea276097fe28181b52f768264e672a89dd04`.
  Authors label an earlier version ICLR 2026. Publisher API returned HTTP 403;
  independent verification of that publication status remains incomplete.
- Nikhil Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam*,
  arXiv:2409.11321v2, revised 2025-01-31. Read Algorithm 3, QR Algorithm 4,
  initialization/1D-layer discussion, and experimental/generalization scope.
  [Open PDF](https://arxiv.org/pdf/2409.11321v2), SHA256
  `d6f33bb71eab0f046d911b60800846fd3332f1dcf5f2f1434ab444ab89d0b22a`.
  A later citation labels an ICLR 2025 paper with “for Language Modeling”
  appended to its title; this is not an independently verified publisher record.
- Read `optim/kl_opt.py` in
  [yorkerlin/KL-Methods](https://github.com/yorkerlin/KL-Methods) at commit
  `a02e6221556301cf84f9607c73f327910afb2b95`, SHA256
  `442d743e516f7ed453e60d0502460ffdc744b8e1708dbdac56d4aef0929a9c1a`.
  Read `soap.py` in [nikhilvyas/SOAP](https://github.com/nikhilvyas/SOAP) at
  commit `a1e553530fde97d0e6b307d7c82ac6d38b072340`, SHA256
  `9e9021335ad02371584622fa256e73fe75f3eb74446a8d8a216e956534148f6b`.
  Author code is retained as evidence, not imported/executed by our experiment.
  Fetch manifests record URLs, access status and content hashes.
- The abstract of arXiv:2509.22938v1 reports settings where SOAP has no final
  loss advantage over Adam/Shampoo. That paper has not been fully read here;
  its abstract is a caution against assuming transfer of the original LLM gains.

## Hypothesis and independent controls

The typed graph encoder's matrix gradients can have anisotropic row and column
correlations. KL-Shampoo estimates both covariance factors jointly instead of
treating each opposite factor as identity. For a gradient G of size m by n:

    P_L = Q_L diag(r_L^2) Q_L^T; P_R = Q_R diag(r_R^2) Q_R^T
    Delta_L = G P_R G^T / n; Delta_R = G^T P_L G / m

The moving averages of covariance factors and their diagonals in the current
basis use beta2=0.98. Both targets use the same OLD factors, before either is
updated. Inverse square roots r are updated every step; each eigenbasis is
updated every 10 steps by `qr(S @ Q)`. They are approximate eigenbases, not an
exact eigendecomposition every step. Original-space momentum uses beta1=0.9.
The parameter direction uses the pre-update basis and inverse square roots,
as in the pinned author code. There is no Adam step-size grafting for these
matrix directions.

Four arms are fixed before screening:

1. `full`: the two-sided KL targets above and Kronecker-preconditioned momentum.
2. `scalar_coupling`: replace each P by `mean(r^2) * I` in the covariance and
   eigenvalue targets. Keep average inverse scale, matrix state, QR frequency,
   momentum, clipping and initialization; remove only directional coupling
   through the opposite factor. This is a purpose-built ablation, not an
   official Shampoo or VN-Shampoo implementation. Both arms coincide when
   the opposite factors are isotropic.
3. `soap`: Adam in the EMA gradient-covariance eigenbasis. Momentum is stored in
   original coordinates and projected each step, equivalent in exact arithmetic
   to the author's basis-transported momentum. At QR steps, second moments are
   permuted using the author's eigenvalue ordering; no exact full covariance
   transport is claimed. SOAP retains bias correction and its epsilon convention.
4. `adamw`: ordinary AdamW with the same betas, epsilon, decay and budget; CPU
   trajectory is checked against PyTorch AdamW after the common initialization
   gradient. No old weights are adopted, because the betas/warm start differ.

Every declared control is mandatory: full must exceed all three on mean global
MiDrug ROC AUC and sklearn AP. The strong reference and zero-residual null are
separate necessary checks. No control will be relabeled as a successful full
model after seeing outcomes.

## Local adaptation and fixed budget

- Use the unchanged two-layer, width-64, four-head typed sparse-attention graph
  backbone from `sharpness_graph_v1`, including the original attributes and
  train-support degree inputs. No new static feature columns, learned adjacency
  or graph-representation mechanism is introduced.
- Precondition only nondegenerate `nn.Linear.weight` matrices. Scalar-output
  weights, bias, normalization and relation embeddings use the same AdamW path
  in every arm. This is a matrix-only local adaptation, not the author's full
  tensor/LLM training reproduction.
- Fix lr candidates 0.001/0.003, betas (0.9,0.98), epsilon 1e-8, decay 0.001,
  frequency 10, initial eigenvalues 0.1, no curvature damping, and no gradient
  normalization. Clamp each inverse square root to
  `max(10, min(axis_dimension, 4000))`, as in the author's implementation.
  Use fp32 states and QR on cuda:1; mathematical checks use fp64 on CPU.
- All arms consume one initialization gradient on support-deleted fold 0,
  with no weight/momentum/variance update. Then take 400 parameter updates,
  cycling through the five omitted folds, with one complete fold BCE and
  gradient clipping at 5 per update: 401 gradient evaluations per fitted model.
  Matrix curvature is initialized from the first gradient as in the author's
  code; AdamW consumes and discards the same initialization gradient.
- Apply decoupled decay BEFORE the parameter update in all arms, matching
  Figure 12 and standard PyTorch AdamW. The pinned author's source applies
  decay after the update; this small order difference is explicit.
- Train `BCE(anchor_oof + residual, y)`. The existing nested OOF reference is
  immutable and excludes each query fold. Five-support mean residual is
  composed as `evaluation_anchor + 0.1 * residual`. Neither the residual
  infrastructure nor composition is an innovation claim.
- Evaluate every 20 updates; no initialization-only checkpoint. Select by mean
  global AUC/AP and the existing MiDrug-priority learning-rate rule. Screen
  seed 666, then seeds 0/1 only after all screening gates pass. Freeze median
  training steps before the established fixed-partition stage; the frozen
  outer recheck remains mandatory. The later partitions cannot enter fitting.
- LncDrug is prospectively an exact reference identity: no graph, optimizer,
  gradient or trained parameters. Scoring records do not count as neural fits.
- CPU initialization and cuda:1 dropout are seeded separately. Never call
  `torch.manual_seed` or initialize another CUDA device.

## Limits

LLM convergence speed and final training loss are not evidence of link-ranking
improvement. Correlated finite-sample gradients need not follow the idealized
matrix-Gaussian model, and the practical EMA/QR solver does not solve the KL
projection exactly. The scalar ablation and all baselines are necessary to
distinguish useful covariance learning from optimizer-scale differences.
Only the complete validation chain can establish a successful new technique.
