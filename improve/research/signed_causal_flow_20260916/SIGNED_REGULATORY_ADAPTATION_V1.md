This prospective experiment tests whether curated regulatory signs improve
MiDrug ranking through a bounded gene-mediation network. It does not claim
identified causal effects, measured gene activity, or reproduction of GSNN.

Sources and availability

- *SIGNOR 4.0: the 2025 update with focus on phosphorylation data*, DOI
  10.1093/nar/gkaf1237, verified from Crossref. The downloaded interaction table
  is the current OmniPath SIGNOR-containing subset, not a verified exact dump of
  SIGNOR 4.0. Full UniProt identifiers, sources, references, direction and
  stimulation/inhibition fields are retained.
- Evans et al., *Graph Structured Neural Networks for Perturbation Biology*,
  bioRxiv 2024, DOI 10.1101/2024.02.28.582164. The bioRxiv API still reports
  `published: NA`; no journal publication is asserted. Metadata and abstract
  were read. Both public full-text XML and PDF endpoints returned HTTP 403;
  the full paper has not been read.
- Author GSNN documentation and model/sparse-layer definitions were read from
  fixed commit aacb74bc65722cad5123b778c2ab38b2d2fde936 via verified Git blobs.
  The README distinguishes the current implementation from the paper's v1.0.0
  release. No author benchmark was run and no author layer code is imported.
  GSNN motivates imposing known latent-variable interactions on architecture;
  the present node/channel-specific, sign-constrained recurrence is different
  from its edge-state sparse function networks.
- OpenAlex HTTP 429s, Europe PMC timeouts, the bioRxiv 403s and the raw GitHub
  README timeout are retained in the retrieval records. Fixed author blobs and
  the public OmniPath table were retrieved successfully.

Evidence construction, before performance observation

Only rows with unambiguous stimulation versus inhibition, matching consensus
direction/sign, and an explicit SIGNOR PMID are eligible. Unique HGNC-approved
protein-coding mappings use the full UniProt accession; no isoform suffix is
stripped, no symbol fallback is used, and conflicting collapsed gene pairs are
discarded. Thus 65,532 downloaded rows produce 12,684 edges (8,316 positive,
4,368 negative) after filtering and mapping. No reverse edges are invented.
RNA/gene annotation weights (1, 0.25, 0.1) and expanded high-confidence ChEMBL
binding evidence reuse frozen sources with the same gene dictionary. The
expanded drug matrix previously failed as a standalone feature expansion; its
use here is common to all controls and is not counted as a new technique.

The resulting 17,317 gene nodes include all regulatory and binding endpoints.
594/605 miRNAs and 156/216 drugs have binding annotations; 594 miRNAs and 144
drugs touch at least one regulatory-network gene. The original shared graph's
fourth node type is circRNA, not gene: this is a separate gene evidence channel.
RNA and drug name order is checked against the original identity-only CSVs.
No association labels or retained test files enter this input build.

Positive binding weights indicate annotation strength, not activation or
inhibition. Neither drug action nor RNA suppression is inferred from a binding
assay. The new network transforms latent feature probes; its coordinates are
not calibrated biological concentrations or intervention responses.

Mathematics and registered model

Let R and D be the fixed positive RNA/gene and drug/gene annotation matrices.
Column-mean normalized transposes inject bounded projections of the common
query-masked graph embeddings into genes:

    U = 0.5 * mean_RNA_binding(tanh(W_R h_R))
        + 0.5 * mean_drug_binding(tanh(W_D h_D)).

Missing roles contribute zero. Let P[j,i] = sign(i->j)/indegree(j) on the
curated directed graph, with zero rows for nodes with no incoming regulation.
Each gene/channel has a trainable positive gain a=2*sigmoid(g) and bias b.
Use z_0=tanh(U) and exactly eight updates

    z_(k+1) = tanh(a * (U + 0.4 P z_k) + b).

The absolute row norm of P is at most one and tanh is 1-Lipschitz, giving an
upper contraction bound of 0.8 for the recurrent state map. This does not claim
eight updates satisfy a fixed-point residual tolerance. Local edge derivatives
respect the curated sign, whereas combined effects along multiple conflicting
paths need not have a single sign. Gains/biases are gene/channel-specific and
shared across the eight updates; hidden width is 16.

Row-mean normalized R,D pool final gene states back to all RNA/drug entities.
Their normalized vectors, interactions and coverage counts enter the same
pair readout as the existing two-layer width-64 typed GAT embeddings. Gene
states are computed once per full support-graph forward, independent of query
order or batch composition. Query folds and reverse arcs are removed before
the common GAT, and hence before feature injection into the gene network.

Full, unsigned and receiver-stratified shuffled-sign arms share learned
parameter tensors, initialization, dropout streams, annotation inputs,
normalization and all update counts. Only the immutable regulatory signs vary.
The shuffled control changes 2,994/12,684 edges (23.60%) while preserving each
receiver's positive and negative incoming-edge counts. It is one fixed finite
permutation, not a uniform random-network ensemble or a significance test.
Both unsigned and shuffled-sign comparisons are mandatory for both global
MiDrug metrics. The family is specifically signed regulatory evidence with
constrained nonlinear mediation; it is related broadly to biological evidence
learning, but unordered pathway overlap, appended feature count, the common
GAT, reference offset and fusion are not claimed as new contributions.

Training/validation contract

Five support folds; gene width 16; eight regulatory updates; coupling 0.4;
gain limit 2; two width-64/four-head GAT layers; dropout 0.1. AdamW decay 0.001,
learning rates 0.001/0.003, 400 updates, 20 common checkpoints. No gene-width,
coupling, iteration count, evidence-tier, shuffle-seed or fusion sweep. The
nested OOF reference remains a fixed training offset; prediction is the
unchanged reference plus 0.1 times the mean raw residual over support folds.
LncDrug has exactly zero residual and no new fit.

Seed-666 historical screening precedes seeds 0/1, the fixed development
partitions 61453/7919/104729, and frozen old-outer verification. These existing
partitions have prior research exposure. Any failed performance, mechanism or
numerical gate stops this version. No successful technique is counted from
implementation checks, new annotation coverage, or historical gains alone.

Before formal freezing, the first five-update GPU probe completed but a later
paired-initialization check found nondeterministic float32 atomic-sum roundoff
in fixed annotation normalizers (maximum absolute difference 3.7253e-9).
This was a preparation issue, not a performance result. The probe and previous
sources/checks are archived under pretraining_fixed_normalization_v1. Fixed
annotation sums now use a deterministic CPU float64 order and are cast once to
float32 before device transfer, ensuring equal constant buffers across arms.
The biological inputs, architecture and selection rule are unchanged. The
valid unit/probe evidence refers to the corrected implementation; the archived
extra five-update neural probe is reported separately from the three final
probes and from formal fits.
