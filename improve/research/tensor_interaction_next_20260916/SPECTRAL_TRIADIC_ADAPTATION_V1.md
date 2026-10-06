SHONN spectral sharing in the RNA–drug pair head

The prospective hypothesis is that a fixed random quadratic basis, modulated by learned input-pair and output spectral coefficients, generalizes better than either a nonquadratic head or independently trained quadratic coefficients. This is one candidate in the broader feature-interaction family. Polynomial interactions are longstanding; a 2026 preprint date, the two constituent layers, a new executor, and the residual reference do not count as separate innovations. The earlier failed Polynormer model used featurewise products inside local/global message aggregation. Here the mechanism is a constrained complete quadratic coefficient tensor in the query decoder, while graph propagation is held fixed. Only passing the declared controls and every subsequent validation stage could support the narrower claim.

Source and verification

- Peri, Carletti, Fanelli and Febbe, *Spectral Higher-Order Neural Networks*, arXiv:2603.28420v1, 30 March 2026. The publicly available PDF, its extracted text and retrieval record are retained. Crossref's exact-title search did not establish a matching journal DOI; this work is treated as a preprint.
- Author repository `gianluca-peri/hyperspectral`, fixed commit `bf38adfcfc441d4bf71e9eff43d87e41a0f1e7a7`. The downloaded Git blobs and local SHA256 checks are recorded in `SHONN_AUTHOR_SOURCE_V1.json`.
- Read Sections 2.1–2.3, 4.1–4.3 and Appendix B, the full author `lib/layers.py`, and the MNIST and regression construction/training recipes. Unit checks execute the unchanged layer module, with both bias settings; they do not run the author's benchmark scripts or reproduce reported benchmark scores.
- The local bias constant uses the input's dtype and device. The source creates a default-dtype ones tensor. Forward and active-parameter gradients are checked at float64; equivalence for every possible mixed-precision setting is not claimed.

Mathematical mechanism

For input width N and output width K, let P = N(N+1)/2. All i <= j products are retained, including squares and counting cross terms once. With the bias present only in the linear term,

    y_k = sum_i (lambda_in_i - lambda_out_k) phi_ki x_i
        + sum_(i<=j) (lambda2_in_ij - lambda2_out_k) phi2_kij x_i x_j.

The linear basis and both linear spectral factors are trainable. The quadratic basis phi2 is Xavier-uniform and fixed; both quadratic spectral factors are trainable. All spectral factors use the author's uniform[-1,1] initialization and draw order. The basis-normalized quadratic tensor obeys the additive row/column constraint; shifting both quadratic factor vectors by the same constant changes no coefficient. No graph eigendecomposition or Laplacian positional encoding is used.

The quadratic part has P+K trainable factors but stores K*P fixed entries and needs K*P multiply-adds per query, in addition to the explicit products. These costs remain cubic when K scales with N. The implementation does not repeat the paper's occasional identification of parameter complexity with actual computation. Including the linear part and bias, the two chosen layers have 23,202 active spectral parameters, versus 674,242 in the direct-quadratic control; fixed and trainable totals for the complete graph model are recorded separately. The quadratic tensor holds 661,504 entries across both layers in either arm. Spectral factors left inactive in the direct arm remain stored and are explicitly counted as fixed parameters.

Local architecture

Use the unchanged typed sparse-attention backbone with two layers, width 64, four heads, dropout 0.1, per-type original attributes divided by per-row RMS, and support-only relation counts. Every neural query fold and its reverse target edges are removed before graph construction. Concatenate the RNA and drug endpoint states, without adding context76 columns, to form 128 query features. Apply spectral-triadic 128→64 and 64→64 layers, each followed by LayerNorm, GELU and dropout 0.1. A final 64→1 linear layer is initialized to zero. LayerNorm/GELU/dropout, the graph backbone, and the zero residual readout are local adaptations. The author's regression example stacks quadratic layers without these operations. No universal approximation theorem or biological higher-order interaction discovery is asserted for this finite decoder.

Controls and initialization

- `full`: the spectral quadratic mechanism above.
- `linear_only`: build the same input products and perform the same quadratic matrix multiplication, then multiply its contribution by zero. Its quadratic spectral factors therefore have zero gradients. Common nonlinearity and graph attention remain; it is a nonquadratic query head, not a globally linear graph predictor. Equal nominal parameter counts do not imply equal effective capacity.
- `direct_quadratic`: keep the common linear spectral block; replace each fixed quadratic basis deterministically with the initial effective quadratic tensor and train every coefficient independently. Freeze the now-unused quadratic input/output spectral factors. This starts at the same function and input Jacobian as `full` in float32; float64 checks construct both directly at float64 to avoid casting an already-rounded product. No additional random draw is consumed. Subsequent optimization and trainable dimensions intentionally differ.

All three arms use the same initializer draw stream, graph, query folds, dropout schedule, training labels and nested strong-reference offsets. Formal runs use 400 AdamW updates, weight decay 0.001, gradient clipping at 5, two learning rates {0.001, 0.003}, and checkpoints every 20 steps. Source-theorem conditions are not substituted for the required comparisons. No layer-width, degree, dropout, fusion-weight or spectral-initialization search is planned.

Evaluation and stopping

Only cuda:1 is used for real neural fitting, serially. CPU work is restricted to implementation tests, synthetic short fits and read-only checks. The existing strong anchor and the fixed 0.1 residual composition are preserved. MiDrug training is BCE(nested OOF anchor + residual). Evaluation uses the existing anchor plus 0.1 times the mean raw residual across support folds. LncDrug is the prospectively registered exact identity with no parameters or neural training.

First complete source/formula checks and five training-only GPU steps per MiDrug arm, verify zero initialization, finite gradients, fixed parameters, optimizer states, query reorder/batch replay, and all historical/fixed-partition fold boundaries without loading evaluation-label arrays. Then freeze source/input hashes and the formal screen protocol. Seed666 chooses one learning rate per arm by the existing shared-task mean AUC/AP rule. Only a passing joint screen enables seeds0/1. MiDrug mean AUC and AP must both improve, each in at least 2/3 initializations, and both means must exceed every registered mechanism control. LncDrug loss in either metric may not exceed 0.001. Three pre-existing fixed development partitions and a separately frozen outer recheck remain mandatory after historical confirmation. All these data are development-exposed, not independent tests. The original reserved test source is not accessed.

Every successful run retains selected weights, raw predictions, checkpoint traces, final weights, AdamW states, named optimizer parameters and CPU/current-device RNG. A numerical stop retains the exact completed step count and partial state when available; it does not authorize silent clipping or a changed run. Stop this frozen version on any failed numerical, performance or mechanism gate. Keep earlier failed sources, evidence, protocols and results. No implementation check or single-arm improvement counts as a final successful technique.
