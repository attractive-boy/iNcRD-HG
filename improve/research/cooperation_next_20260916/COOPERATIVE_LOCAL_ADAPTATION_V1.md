# Cooperative local-state policy: same-family revision

The original cooperative study's three-arm screen has finished and failed its
registered mechanism gate. Its full model had lower MiDrug AUC/AP than the
`no_lookahead` arm, and lower AUC than its soft-action control. Those files,
source versions, primary choice and conclusions remain unchanged.

This is a prospective revision **within the same cooperative communication
family**, motivated by that negative component result. It can count at most as
one family together with all earlier cooperative variants. Its main model is
exactly the earlier `no_lookahead` predictor: the send/receive policy uses each
node's evolving state and incident relation mix without a second observation of
neighbor states. Environment messages still depend on physical neighbors.
It is an explicitly restricted adaptation of ICML 2024 Co-GNN, using the same
author-code factorization, fixed temperature and finite Monte Carlo inference
specified in `COOPERATIVE_ADAPTATION_V1.md`.

Two **newly paired mandatory controls** address the remaining claim:

- `soft_actions` retains the local policy and its Gumbel draws, replacing only
  straight-through discrete actions with their continuous relaxations. The old
  soft-action arm also had neighborhood look-ahead, so it is not reused here.
- `static_policy` uses the initial projected node state for every layer's policy
  logits. It preserves every policy parameter and discrete Gumbel draw, but the
  policy cannot adapt to the environment's evolving representations.

Architecture, depth, width, temperature, dropout, saved evaluation draws,
support masking, OOF reference and residual weight are identical to v1. The two
learning rates and 400-step budget remain fixed. LncDrug retains its original
prospective identity policy. No result from either new control has been observed
when this revision is declared.

The existing `no_lookahead` screen model/trace/score artifacts can be adopted as
`full` only after checking recipe and every input array, verifying the old file
hashes, and reproducing stored predictions exactly. A CPU check additionally
compares forward values, every parameter gradient and random state against the
old implementation in training and evaluation. Source files are copied byte
for byte to the new study; old files are never modified. Adopted weights are
not counted as new fits. The screen therefore has four new MiDrug fits, two
adopted MiDrug fits and six identity records, two of which are adopted.

The main candidate's seed-666 performance was already seen and helped motivate
this revision; its screen is not independent evidence. Seeds 0/1, fixed
partitions and outer recheck are still required under the original gates.
No prior failure is erased, no prior gate is relaxed, and the higher old
control score is not by itself a success.
