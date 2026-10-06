# Spectral matrix updates for the existing graph residual

Jingyuan Liu et al., *Muon is Scalable for LLM Training*, arXiv 2502.16982v1,
2025. Metadata, the open full paper (methods, experiments, optimizer mismatch
and discussion) and `MoonshotAI/Moonlight` author code were inspected at tree
`c2ad5b20c605086526a179d36901bfc41b52b44b`. The matching OpenReview record
`0TJvq9UQ9z` is an archive direct upload, not evidence of conference acceptance.
The PDF SHA-256 is
`6b8e3b47da4db09c952a35486c905f416c57081a32b445a36a3744c441b1c7f3`.

## Registered mechanism

All arms use exactly the existing ordinary typed graph-attention architecture:
width 64, depth 2, four heads, dropout 0.1. The only change is the optimizer.
For each hidden linear weight matrix, Muon first forms an accumulated momentum
buffer and its Nesterov direction, then applies five quintic Newton–Schulz steps:

```
m <- 0.95 * m + gradient
d <- gradient + 0.95 * m
X <- d / (FrobeniusNorm(d) + 1e-7)
repeat five times: X <- 3.4445 X - 4.7750 X X^T X + 2.0315 (X X^T)^2 X
W <- (1 - learning_rate * weight_decay) W
     - learning_rate * 0.2 * sqrt(max(rows, columns)) * X
```

For tall matrices the iteration runs on the transpose, as in the author code.
The finite iteration approximates a polar factor and changes singular values;
it does not produce exact orthogonality, constrain the learned weights to be
orthogonal, or prove an optimal spectral update. Small graph matrices use fp32,
with fp64 in the independent SVD-polynomial check. There is no compiler or bf16
approximation in this implementation.

Only weights owned by hidden `nn.Linear` modules with both dimensions larger
than one receive the matrix update. Relation embeddings, biases, normalization
parameters and the scalar output head use AdamW. All backup AdamW arms use
betas (0.9, 0.999) and epsilon 1e-8 to preserve this project's ordinary baseline;
the author's LLM example uses a different beta2. Model weights are initialized
from scratch, so this is not the pretraining/fine-tuning optimizer switch that
the paper reports can perform poorly.

The primary `full` must beat both mandatory controls:

- `frobenius` uses exactly the same momentum, Nesterov direction and five-step
  computation, but replaces the matrix direction by the original momentum
  direction rescaled to exactly the finite-NS Frobenius norm. This matches the
  per-matrix update magnitude and isolates singular-value reweighting.
- `adamw` is the project's unmodified AdamW optimizer on all parameters.

Learning rates 0.001/0.003, 400 updates, checks every 20 updates, decay 0.001,
gradient clip 5, screen seed 666 and confirm seeds 0/1 are fixed. The nested
OOF residual BCE and final weight 0.1 are unchanged infrastructure. Muon and
its matrix control each evaluate one training gradient per update. No claim
of lower total compute follows from the LLM paper's scaling-law results.

## Scope and predeclared baseline reuse

LncDrug is prospectively the zero-residual identity, with no new training.
Its scoring records are classified separately from all neural fits.

The already completed F-SAM study's ERM arm has the same ordinary architecture,
training inputs, initialization, updates and prediction rule as this study's
AdamW arm. It evaluated two gradients at the same dropout draw per update. A
new CPU test verifies exact weights and final random state versus the one-pass
AdamW trajectory. The CUDA smoke additionally compares its weights with the
archived ERM smoke; floating point atomic training reductions may differ.

Before screening the new directions, baseline adoption must check the frozen
recipes, every training and evaluation input array, all original artifact hashes,
and exact replay of the stored residual scores with the current model class.
Copied model, trace and score files retain their bytes; original results remain
untouched and the new result explicitly records its origin. Two MiDrug screen
baselines and two LncDrug identity records can then be adopted. Adopted neural
weights are not new fits; their original 800 gradient evaluations remain in the
provenance. If any compatibility check fails, stop and version the decision.

Expected screen accounting is four new MiDrug fits, two adopted MiDrug fits,
and six LncDrug identity records (two adopted). If eligible, confirmation adds
six new MiDrug fits and six identity records. Already observed baseline values
and all earlier failures are acknowledged; no Muon result has informed this
recipe. The same frozen partition and outer gates remain mandatory, and this
optimization family can count at most once after every gate passes.
