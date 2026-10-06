Contextual tensor-product attention on the RNA–drug support graph

The prospective hypothesis is that input-dependent low-rank factorizations of the query, key and value activations improve residual ranking beyond input-independent head factors and unconstrained linear Q/K/V maps. This is one candidate in the broader feature-interaction family, not an additional innovation for each projection or control. Earlier failed Polynormer and SHONN outcomes are known: the former used feature products in propagation; the latter constrained a quadratic pair decoder. Here contextual tensor factors change the graph attention scores and transmitted values. The better classical quadratic control in the earlier study does not count as a successful innovation and is not renamed here.

Sources and reading scope

- Zhang, Liu, Yuan, Qin, Yuan, Gu and Yao, *Tensor Product Attention Is All You Need*, Advances in Neural Information Processing Systems 38 (NeurIPS 2025), DOI 10.52202/085713-3748. Exact Crossref metadata verifies title, author list and proceedings venue. The full source read here is arXiv:2501.06425v7, updated 11 January 2026; the first version was posted 11 January 2025. The v7 PDF has SHA256 `f411822f67707b62a14694477f4da5210b982da33810d1f3aea9537741b354f2`.
- Author repository `tensorgi/TPA`, fixed commit `c276c80d5ad807881dedb4707d8d3c20b4e97ec6`. README, `model/T6.py`, KV-only and MHA implementations, small-model configuration and MIT license were downloaded as verified Git blobs. The implemented projection and initializer were checked against `model/T6.py`.
- Read the main factorization and attention definitions, non-contextual special cases, Appendix A's contractions and Appendix G/H's variants, initialization and architecture settings. The unit check executes unchanged `CPLinear` and `apply_rotary_emb` AST bodies, with an explicitly substituted all-zero-position rotary provider. Full T6, original language-model training and FlashTPA kernels are not executed or benchmarked.
- Repository discovery also returned `Kkyn-ltcode/GraphTPA-Parameter-Efficient-Tensor-Product-Attention-for-Network-Intrusion-Detection`. Its publication status and full implementation have not been verified. Consequently this work makes no first-use-of-TPA-on-graphs claim.

Mechanism and exact source choices

For each node feature x, and separately for Q, K and V, form two linear factor maps A(x) of shape heads×rank and B(x) of shape rank×head_dim. The activation matrix is

    T(x) = A(x) B(x) / rank.

Use width 64, four heads, head dimension 16, query rank 6 and key/value rank 2. Rank choices follow the source's standard TPA recipe. Factors have no bias. These are quadratic functions of the node state, with mixed input Hessians; at fixed input the head-by-channel activation has rank at most the corresponding factor rank. This is a factorization of activations, not a low-rank update to pretrained weights.

The pinned `CPLinear.reset_parameters` initializes three-dimensional views of stored weights with Xavier uniform. The local implementation preserves that exact view shape, draw order and resulting weights. Appendix G's generic two-dimensional Xavier description does not fully specify this source convention, so the local recipe is tied to the actual code. Head-major storage for A is used consistently. No node-index RoPE or causal token mask is introduced: arbitrary node IDs are not sequence positions.

At each support edge u→v, compute a per-head score `q_v dot (k_u + relation_embedding) / sqrt(head_dim)`, softmax over all incoming edges at v, and sum attention-weighted values from u. This uses the common typed sparse-attention relation convention. Two such layers retain the common LayerNorm, residual, output projection, width-128 GELU feed-forward block and dropout 0.1. Per-type original attributes are row-RMS scaled and concatenated with support-only relation counts, then projected to width 64. The ordinary pair head receives `[left,right,left*right,abs(left-right)]`; its last layer is zero initialized. The common graph wrapper, GELU blocks and nonzero attention-output initialization differ from T6's RMSNorm/SwiGLU/zero-output language-model blocks and are held identical across arms. No context76 features are added.

Controls and accounting

- `full`: both A and B depend on the input, following the contextual TPA equation.
- `noncontextual_a`: B depends on the input, while A is a directly learned heads×rank parameter. Its initial value is the initial contextual A evaluated at the fixed all-ones vector. The unused contextual A weights are frozen.
- `dense_qkv`: each Q/K/V is a freely trained bias-free linear map. Its initial matrix is the exact effective linear map of `noncontextual_a`. All factor and static-A parameters are frozen in this arm.

Every arm stores all three representations and executes all three projection branches, multiplying unused outputs by zero. Initial stored tensors and random streams match exactly across arms. The two controls start at the same function and input Jacobian; the full projection is a different function. The zero output head gives all three models the same initial residual score of zero. Optimizer parameters include only active tensors. Parameter counts and frozen tensors are checked separately; equal effective capacity, equal backward FLOPs or equal optimization geometry are not claimed.

The complete real-data smoke models each store 174,481 parameters. Active counts for full, noncontextual-A and dense-QKV are 149,825, 144,785 and 148,801; the remaining 24,656, 29,696 and 25,680 parameters are frozen. All three five-step smoke fits passed, as did three identity replays and 40 fold-isolation checks. Maximum replay error was 3.73e-9 and no evaluation labels were read. These checks are implementation evidence, not performance validation.

The static-A initialization and dense-equivalence construction are local control designs, not an assertion that every author ablation uses this initialization. The source's generic non-contextual A family motivates the first control; the ordinary dense projection tests whether contextual factor products are needed beyond a full linear map.

All branches materialize full Q/K/V. There is no autoregressive KV cache, no FlashTPA execution and no claim of the source's inference speed or memory savings on this graph. Registered counts distinguish contextual product terms, static product terms, dense projection outputs, edge-head scores and value-message entries. They audit actual forward work and do not substitute for measured runtime or full FLOP accounting.

Validation and stopping

MiDrug uses the unchanged nested OOF strong-reference offset with BCEWithLogits(offset+raw_residual); evaluation adds exactly 0.1 times the mean raw residual over five omitted-fold graphs to the existing strong reference. LncDrug is prospectively the exact reference identity with no parameters or neural training. Query target edges and their reverses are removed before graph construction and support-derived counts. No original reserved test source is accessed.

Seven CPU mathematical/synthetic tests check pinned-source initialization, projections and active gradients, an independent quadratic tensor expansion and Hessian, rank bounds, control-function equivalence, incidence-based sparse attention, node relabeling, query-label isolation, RNG/frozen parameters, exact LncDrug identity and exact partial failure state. Then run three five-step MiDrug GPU smoke fits, three identity replays and 40 historical/fixed-partition fold-isolation checks using training arrays only. No evaluation labels or performance are used for these checks.

Freeze sources and inputs before seed666 screening. Use 400 AdamW steps with weight decay 0.001, gradient clipping 5, learning rates {0.001,0.003}, and a checkpoint every 20 steps, serially on cuda:1 with CUBLAS_WORKSPACE_CONFIG=:4096:8. No rank, width, depth, initializer, normalization or fusion search is planned. Only a joint passing screen permits initialization seeds0/1. Both mean MiDrug AUC and AP must increase, each in at least two of three repeats, and both means must exceed both declared controls. Either LncDrug metric may lose at most 0.001. A passing historical result still requires three fixed development partitions and a separately frozen outer recheck. Historical, fixed-partition and prior outer evaluations are development-exposed, not independent tests.

Stop this version on any failed numerical, performance or mechanism gate. Preserve selected/final weights, raw predictions, traces, named AdamW state, current-device and CPU RNG, all source/input hashes and failures. Strictly reload and replay every saved model in reversed order and new query batches with absolute error <=3e-5. Identity records and read-only replay do not count as new neural fits. No formula check, isolated gain, control score or source-paper result counts as a completed innovation.
