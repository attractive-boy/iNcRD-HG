Stuart–Landau amplitude–phase propagation for RNA–drug residual prediction

The prospective hypothesis is that nonlinear amplitude stabilization coupled to amplitude-dependent phase evolution improves graph-conditioned residual ranking beyond both the same stabilized oscillator without phase feedback and a linear oscillator. These are mechanism controls, not separate innovations. Oscillatory propagation is at most one family, including possible later Kuramoto or HoloGraph adaptations; source recency, a numerical solver, the typed graph wrapper and reference infrastructure do not independently count as innovations. Earlier diffusion, geometric and graph-dynamics failures remain visible and preserved.

Source and verification

- Zhang, Reynolds, Deidda and Tudisco, *Stuart-Landau Oscillatory Graph Neural Network*, arXiv:2511.08094v2, updated 20 February 2026; first posted 11 November 2025. Full PDF and extracted text are retained. The authors report WWW 2026 and DOI 10.1145/3774904.3792675. The exact Crossref request returned HTTP 404, so publisher confirmation has not been established.
- Repository `kevvzhang/StuartLandauGNN`, fixed commit `40944b1e0d2b954de6b46f31b6a9a6f3dc04f095`. Downloaded README, `src/slmodel.py`, `src/layer.py` and `src/best_params.py` have verified Git blobs and SHA256 hashes in `SLGNN_AUTHOR_SOURCE_V1.json`.
- Read the main theory and method, Algorithm 1, Appendix D and SLGCN implementation. The pinned best-parameter dictionaries were inspected as source examples without this candidate's evaluation outcomes. The unit test extracts and executes the unchanged AST bodies of `SLGCN.solve_implicit_r` and `SLGCN.ode_step` with synthetic coupling. It does not execute the complete author graph model or reproduce their benchmark scores.
- The paper's Algorithm 1 differs from its accompanying code: it puts coupling before the local update, omits an R factor in the displayed radial alpha term, and writes the phase gamma term with a plus sign. The body ODE, standard cubic in the appendix and source code use alpha times R and omega minus gamma times R squared. The local implementation follows the source order, `local(old_state) + dt*k*coupling(old_state)`, and the body/code signs. `slgnn_algorithm_page6_v1.png` records the inspected algorithm page; the earlier page-4 image is retained but is not evidence for Algorithm 1.

Mathematical and numerical mechanism

Each node has width-32 Cartesian complex state z. With fixed dt=0.1, alpha=1, beta=1, omega=1, gamma=0.5 and coupling strength k=1, define a=1-dt*alpha=0.9, s=dt*beta=0.1 and v=|z| squared. Solve the monotone cubic in squared radius q:

    q*(a+s*q)^2 = v,  q >= 0.
    local(z) = exp(i*dt*(omega-gamma*q))*z/(a+s*q).

The root is computed in float64 using the hyperbolic inverse of the depressed cubic, avoiding Cardano subtraction. Its custom backward uses the exact implicit derivatives

    dq/dv = 1/((a+s*q)*(a+3*s*q)),
    dq/da = -2*q/(a+3*s*q),
    dq/ds = -2*q^2/(a+3*s*q).

The Cartesian expression has a smooth extension at zero, avoiding division by old amplitude and `atan2`. Independent SciPy roots, gradcheck for all three root inputs, and Cartesian gradcheck including zero pass. The author's nonzero forward and gradient agree to approximately 7e-16. At exactly zero, the author's `abs/angle/polar` autograd does not match the smooth Jacobian: its maximum error in the unit fixture is about 1.10556. The local Jacobian matches the analytic limit and a central difference of the author's forward. Consequently, full author-autograd equivalence at zero is explicitly not claimed. Relative residual `abs(q*(a+s*q)^2-v)/(1+v)` must remain <=2e-12, and states, losses, gradients and optimizer states must be finite.

The coupling is a complex linear GCN with symmetric degree normalization, positive type-channel gates `2*sigmoid(gate)`, and separate real/imaginary LeakyReLU(0.2). Each of 16 layers owns its coupling weights. Real weights use Xavier initialization; imaginary weights and gate logits start at zero. The graph is the common support-only typed physical graph. Its self-loop convention is preserved. This finite-depth, trainable typed coupling is an adaptation; no continuous-system stability, convergence or synchronization theorem is asserted for it. Componentwise nonlinear coupling also prevents a claim of global complex-phase equivariance for the full graph network.

Architecture and controls

Original per-type node attributes are divided by their row RMS and concatenated with common support-only relation counts. Dropout 0.1 precedes a linear projection into the real/imaginary state. After 16 local-plus-coupling updates, take the real component, apply dropout, and form `[left, right, left*right, abs(left-right)]` for the queried pair. A 128→64→1 GELU/dropout head has a zero-initialized last layer. No LayerNorm, state clipping or new context76 features are introduced.

- `full`: stabilized amplitude and amplitude-dependent phase as above.
- `no_phase_feedback`: the same nonlinear amplitude update with fixed phase increment dt*omega, removing only the gamma*q feedback.
- `linear_oscillator`: local map `exp(i*dt*omega)*z/(1-dt*alpha)`, removing nonlinear stabilization and phase feedback.

All three arms actually compute the root and every local branch; unused branch outputs are multiplied by zero. This matches operation counts without claiming equal function classes. Total and trainable parameter counts are both 114,369 in each real-data smoke model. All initializer and dropout draws match across arms. Coupling graphs, query folds, 400 AdamW updates, 400 gradient evaluations, weight decay 0.001, clipping norm 5, two learning rates {0.001,0.003}, and checks every 20 steps are common. No width, depth, local coefficient, normalization, fusion or initialization sweep is planned.

For 4,901 nodes, width 32 and depth 16, each complete historical model has 505 forward calls, 8,080 local updates/root solves and 1,267,202,560 scalar root entries. Coupling messages are counted independently from each omitted-fold physical graph, including 400 training forwards and 21 five-fold evaluation passes. The three-arm five-step smoke recorded 5,889,392 messages each; the formal expected total is 101 times that count, 594,828,592. These counts concern registered model fitting and checkpoint/final prediction; additional saved-weight audit forwards are separately read-only work.

Evaluation, evidence and stopping

Only cuda:1 is used for real neural fitting, serially, with CUBLAS_WORKSPACE_CONFIG=:4096:8. CPU work is confined to implementation/synthetic checks and read-only audits. MiDrug learns BCEWithLogits(nested OOF strong-reference logit + raw residual); evaluation adds 0.1 times the mean raw residual over the five omitted-fold graphs to the fixed strong reference. LncDrug is prospectively the exact reference identity, with zero parameters and no neural training. Identity scoring records do not count as neural fits.

Six CPU mathematical/execution tests and three five-step MiDrug GPU smoke fits passed before formal screening. Forty historical/fixed-partition fold checks exclude queried target edges and their reverses; omitted-label changes cannot affect graph support. Smoke prediction uses only the graph omitting the queried fold. Evaluation-label arrays were not read and performance was not computed during these checks. Formal source and input hashes are frozen after these checks.

Seed 666 selects one learning rate per arm by the existing shared-task mean AUC/AP rule. Only a joint passing screen enables seeds 0 and 1. MiDrug mean AUC and AP must both improve over the strong reference, each in at least two of three repeats, and both means must beat every registered mechanism control. Either LncDrug metric may lose at most 0.001. A passing historical stage is followed by the three fixed development partitions and a separately frozen outer recheck. Historical, fixed-partition and previously seen outer evaluations are development-exposed; none is described as an independent test. The original reserved test source is not accessed.

Each completed fit saves selected weights, raw predictions, trace, final weights, named AdamW state and CPU/current-device RNG. An in-training exception saves the exact completed optimizer-step count, failing step/operation and partial state. All scores, selections, budgets, final state, random streams and source hashes are independently rechecked. Saved weights must replay in reversed query order and new batches with absolute error <=3e-5. A failed numerical, performance or mechanism gate stops this frozen version before subsequent stages. No implementation check, isolated metric gain or better control score counts as a completed innovation.
