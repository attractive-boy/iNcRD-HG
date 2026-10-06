# DuplexFM

DuplexFM is a five-expert RNA foundation-model framework for miRNA--mRNA
interaction prediction and frozen-representation transfer to siRNA efficacy
prediction. This repository contains the complete project code, the three
released miRNA source checkpoints, the 15-head siRNA P13 transfer ensemble,
compact paper results, and the matched OligoFormer comparison package.

The repository supports both reproduction of the reported benchmarks and
inference on completely new guide--target sequences. No benchmark identifier,
label, or precomputed DuplexFM feature cache is required for new inputs.

## What is included

- Final five-expert source architecture and training/evaluation scripts.
- Three distinct miRNA source checkpoints: seeds 42, 43, and 44.
- Frozen mRNA-accessibility LoRA adapter and nucleotide-level prediction head.
- Exact IntaRNA21 scaler used by the source models.
- Fifteen Huesken P13 transfer heads: 3 initialization seeds x 5 folds.
- Compact strict-FP32 miRNA and six-setting siRNA results under `results/`.
- OligoFormer code, matched splits, compact feature caches, OOF predictions,
  metrics, and audit files under `baselines/oligoformer_paper/`.

Project-trained checkpoints total about 61.7 MB; the largest individual file
is about 16.4 MB, so they are versioned directly in this repository. The two
third-party foundation-model bases are not included because each exceeds
GitHub's 100-MB per-file limit.

## Final model contract

The fixed source expert order is:

1. mRNA accessibility: 92-D profile input mapped to a 32-D slot;
2. 11-lite target-to-guide cross-attention;
3. FiLM- and position-conditioned score-matrix attention;
4. Bio23 biological priors; and
5. IntaRNA21 interaction energy, accessibility, and geometry features.

All five 32-D slots have independently learned sigmoid gates. Source training
uses fused BCE plus matched expert auxiliary BCE with weight 0.3. The released
runs use BF16 training, batch size 256, at most 25 epochs, patience 5, and
strict FP32 benchmark inference.

The released siRNA transfer route is P13:

```text
z160  = concat(frozen five ungated 32-D source slots)
score = sigmoid(TD24_anchor(TD24) + rho * residual(z160))
loss  = MSE(score, efficacy) + 0.03 * mean((rho * residual(z160))^2)
```

The source seed-44 representation is frozen. Fifteen fold/head predictions are
averaged for the final siRNA efficacy estimate. P13 has no binary auxiliary
loss.

## 1. Installation

Linux and an NVIDIA GPU are recommended. The audited environment used Python
3.11, CUDA 12.8, PyTorch 2.7.0, and IntaRNA 3.4.1.

```bash
git clone https://github.com/cbaiming/DuplexFM.git
cd DuplexFM

conda env create -f environment.yml
conda activate duplexfm
# If necessary, install the PyTorch wheel matching the local CUDA driver.
```

Check Python packages, CUDA visibility, and required runtime files:

```bash
python duplexfm.py check-environment
```

Download and checksum-verify the two foundation-model backbones:

```bash
python duplexfm.py download-backbones --models-root models
```

This creates:

```text
models/
  mrnabert/    # YYLY66/mRNABERT, pinned snapshot
  rnafm/       # exact RNA-FM snapshot used to train the released LoRA weights
  manifest.json
```

Both backbones are hosted on Hugging Face. The expected repositories,
revisions, file sizes, and SHA-256 hashes are locked in
`configs/backbones.json`: mRNABERT is obtained from `YYLY66/mRNABERT`, while
RNA-FM is obtained from the checksum-matched training snapshot configured
there. If the RNA-FM snapshot repository is private before publication, run
`huggingface-cli login` first. The currently public `multimolecule/rnafm`
repository remains the upstream model source, but the locked snapshot avoids
silently changing the LoRA base when upstream files are updated.

Verify the bundled DuplexFM checkpoint files and the complete repository:

```bash
python duplexfm.py download-checkpoints --verify-only
python duplexfm.py check
```

`download-checkpoints` normally only verifies local files. If a partial source
archive omitted them, running it without `--verify-only` restores their exact
package-relative paths from this GitHub repository.

## 2. Inference on new guide--target pairs

### Input

Input may be CSV or TSV. Required columns are:

- guide: one of `guide`, `siRNA`, `sirna`, `miRNA`, or `mirna`;
- target: one of `target`, `mRNA`, or `mrna`.

An optional `td24`, `td`, or `TD24` column may contain 24 comma-separated
values. If it is absent during siRNA inference, TD24 is computed automatically
from the first 19 guide nucleotides.

Sequence requirements:

- guide length: 18--30 nt;
- candidate mRNA target-region length: 50--60 nt;
- DNA `T` is automatically normalized to RNA `U`.

Example (`examples/pairs_sequences_only.csv`):

```csv
pair_id,guide,mRNA
example_1,CUAAUAUGUUAAUUGAUUU,AAAUUGAAAGGAAUUGUAUAAAUCAAUUAACAUAUUAGCUGAGUUGAACCGGAUUAA
```

### Run both released ensembles

```bash
python duplexfm.py predict \
  --input examples/pairs_sequences_only.csv \
  --output outputs/example_predictions.csv \
  --task both \
  --intarna-bin "$(command -v IntaRNA)" \
  --models-root models
```

Use `--task interaction` for only the miRNA interaction ensemble or
`--task sirna` for only the siRNA P13 efficacy ensemble. Add
`--td24-mode required` to forbid automatic TD24 generation.

### Prediction output

The output CSV retains all original input columns and appends:

| Output column | Meaning |
|---|---|
| `interaction_probability_seed42/43/44` | Binary interaction probability from each independent source seed |
| `interaction_probability_soft_vote` | Arithmetic mean of the three interaction probabilities |
| `interaction_probability_seed_sd` | Population standard deviation across the three source seeds |
| `gate_seed{seed}_{expert}` | Five learned expert gates for each source seed (15 columns total) |
| `sirna_efficiency_p13_mean` | Mean efficacy prediction across 15 P13 heads |
| `sirna_efficiency_p13_sd` | Population standard deviation across the 15 P13 heads |
| `sirna_efficiency_p13_min/max` | Minimum and maximum of the 15 P13 predictions |

Intermediate IntaRNA and source-slot files are written below
`outputs/custom_inference/target_width_<L>_<sequence-hash>/`. The cache key
contains every normalized guide--target pair, preventing feature reuse across
different input batches.

### Compute TD24 only

```bash
python duplexfm.py compute-td24 \
  --input examples/pairs_sequences_only.csv \
  --output outputs/pairs_with_td24.csv
```

The output retains the input table and adds one comma-separated `td24` column.

## 3. Reproduce miRNA source training

The convenience command downloads miRBench v7, obtains the two backbones,
builds restartable Bio23/IntaRNA21/accessibility caches, and trains seed 42:

```bash
python duplexfm.py train-mirna \
  --intarna-bin "$(command -v IntaRNA)" \
  --models-root models \
  --data-root data/mirbench_v7 \
  --feature-root outputs/source_features \
  --output-dir outputs/duplexfm_seed42
```

Static feature construction is the expensive stage because IntaRNA is run for
approximately 2.84 million source pairs. It is restartable. To build only the
shared inputs, add `--prepare-only`.

After the shared inputs exist, train the other two seeds independently:

```bash
python duplexfm.py train-mirna-core --seed 43 \
  --models-root models --data-root data/mirbench_v7 \
  --feature-root outputs/source_features \
  --output-dir outputs/duplexfm_seed43

python duplexfm.py train-mirna-core --seed 44 \
  --models-root models --data-root data/mirbench_v7 \
  --feature-root outputs/source_features \
  --output-dir outputs/duplexfm_seed44
```

Each source run writes its best trainable-delta checkpoint and accompanying
configuration, architecture audit, gate summary, validation history, and test
metrics below its `--output-dir`. The protocol uses all annotated transcript
regions, retains the 57 official miRBench v7 cross-dataset overlap rows, and
uses the fixed 15% validation split grouped by miRNA family.

The lower-level accessibility-pretraining and frozen-cache stages are exposed
through `run_pipeline.py`:

```bash
python run_pipeline.py --help
python run_pipeline.py accessibility-prepare-two-view -- --help
python run_pipeline.py accessibility-train -- --help
python run_pipeline.py accessibility-cache92 -- --help
```

GSE145805 raw/all-human training material and generated token embeddings are
not committed; obtain the raw data from GEO and rebuild them with these stages.

## 4. Strict-FP32 miRNA evaluation and soft voting

Evaluate the three bundled source checkpoints and create a probability-level
soft vote:

```bash
python duplexfm.py evaluate-mirna-ensemble \
  --seeds 42,43,44 \
  --models-root models \
  --data-root data/mirbench_v7 \
  --feature-root outputs/source_features \
  --output-root outputs/source_ensemble_fp32
```

Important outputs are:

```text
outputs/source_ensemble_fp32/
  seed42|43|44/fp32/test/<dataset>/predictions.tsv
  seed42|43|44/fp32/test/<dataset>/metrics.json
  seed42|43|44/fp32/aggregate_test_metrics.json
  soft_voting/<dataset>/predictions.tsv
  soft_voting/<dataset>/metrics.json
  soft_voting/summary.json
```

`soft_voting/summary.json` reports the four-dataset macro APS. Compact copies
of the released metrics are already provided under `results/miRNA/`.

## 5. Reproduce the six siRNA transfer settings

```bash
python duplexfm.py benchmark-sirna \
  --intarna-bin "$(command -v IntaRNA)" \
  --models-root models \
  --source-seed 44 \
  --output-root outputs/sirna_p13_six_settings
```

The six settings are Huesken, Mixset, Takayuki, Huesken-to-Mixset,
DuplexFM-Wang, and ENsiRNA-ALAS1. In-domain cohorts use grouped five-fold OOF
evaluation; transfer-head initialization seeds are 42, 43, and 44.

Main outputs:

```text
outputs/sirna_p13_six_settings/
  feature_export/                 # frozen five-slot source representations
  oligo_p13_heads/                # Huesken/Mixset/Takayuki/inter-dataset heads and metrics
  external_p13_heads/             # Wang and ALAS1 fold heads, OOF predictions, metrics
  caches/                         # restartable IntaRNA artifacts
  PIPELINE_COMPLETE               # created only after the full workflow finishes
```

The reported compact ensemble metrics and OOF predictions are under
`results/siRNA/`.

## 6. Run the matched OligoFormer comparison

The complete compact baseline package is in
`baselines/oligoformer_paper/`. It contains the exact processed cohorts,
matched split identities, small feature caches, code snapshot, seed-level OOF
predictions, and reference metrics.

```bash
cd baselines/oligoformer_paper
# Read the environment and per-setting commands first:
cat RUN_COMMANDS.md
```

The audit in `AUDIT_REPORT.md` reports:

- 33/33 metric checks passed;
- 78/78 paper split/pairing checks passed; and
- 6/6 baseline Python syntax checks passed.

The comparison uses a **matched split and evaluation protocol**. Optimization
hyperparameters remain architecture-specific rather than being forced to be
numerically identical.

## 7. Included result summary

The latest complete rerun produced:

- miRNA four-test-set soft-voting macro APS: **0.876184**;
- siRNA six-setting macro Pearson: **0.575200**;
- siRNA six-setting macro Spearman: **0.594761**;
- siRNA six-setting macro AUPRC: **0.801804**; and
- siRNA six-setting macro F1: **0.736628**.

The siRNA aggregate values differ from the rounded manuscript table by only
about 0.001--0.003 and are treated as ordinary rerun/seed variation. The
released P13 deployment uses source seed 44; the compact latest-complete-rerun
artifacts under `results/ensemble_reproduction/` used source seed 43, as
recorded in `REPRODUCIBILITY.md`.

Result locations:

| Directory | Contents |
|---|---|
| `results/ensemble_reproduction/` | Complete-run summary, binary-label contract, and completeness audit |
| `results/miRNA/` | Per-seed strict-FP32 metrics and soft-voting summaries |
| `results/siRNA/` | Six-setting contracts, fold metrics, ensemble predictions, and summaries |
| `baselines/oligoformer_paper/reference_results/` | OligoFormer seed metrics and OOF predictions |

The optional 354,980,080-byte full prediction archive is not committed. Its
SHA-256 and recommended external-hosting options are recorded in
`LARGE_ARTIFACTS.json`.

## 8. Repository map

```text
DuplexFM/
  duplexfm.py                     # public command dispatcher
  run_pipeline.py                 # lower-level audited stage dispatcher
  configs/                        # locked source/transfer/backbone contracts
  checkpoints/                    # all project-trained models (bundled)
  scripts/                        # public preparation, training, benchmark, inference tools
  runtime/                        # implementation modules used by the public wrappers
  data/sirna/                     # compact released siRNA tables
  baselines/oligoformer_paper/    # matched OligoFormer reproduction package
  results/                        # compact released DuplexFM results
  examples/                       # new-input examples
  tests/                          # source, TD24, checkpoint, and inference tests
```

## 9. Reproducibility and release checks

```bash
python duplexfm.py build-manifests
python duplexfm.py check
python -m unittest discover -s tests -p 'test_*.py'
```

The manifests record package-relative paths, byte sizes, and SHA-256 hashes.
The released seed checkpoints are verified to be distinct; the accessibility
predictor stays frozen during source and transfer training; cross-attention is
asymmetric with mRNA target tokens querying guide tokens; and encoder special
tokens are retained for contextualization but excluded from valid score-matrix
candidate cells.

## Data and licensing notes

Full miRBench v7 and GSE145805 corpora, generated million-row caches, and the
two foundation-model bases are not committed. Third-party datasets and models
remain subject to their original licenses and terms. Before public upload,
confirm redistribution rights for the processed OligoFormer tables,
DuplexFM-Wang reconstruction, and ENsiRNA-ALAS1 patent-derived table. See
`LICENSE-NOTICE.md`, `HF_ASSETS.md`, and `REVIEW_BEFORE_UPLOAD.md`.
