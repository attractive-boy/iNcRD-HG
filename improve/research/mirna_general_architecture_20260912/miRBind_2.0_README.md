# miRBind 2.0

Deep-learning models for predicting miRNA–mRNA interactions.

This repository ships two models:

- **Pairwise binding-site model** — a CNN that predicts whether a given miRNA binds a given target site (≈50 nt window). Use this to score candidate binding sites.
- **Gene-level repression model** — predicts the gene-level fold change a miRNA induces from a full 3'UTR sequence. Built on top of the binding-site model via transfer learning.

## Installation

Clone the repo and install the dependencies (Python ≥ 3.9, PyTorch ≥ 1.9):

```bash
git clone https://github.com/BioGeMT/miRBind_2.0.git
cd miRBind_2.0
pip install -r code/pairwise_binding_site_model/requirements.txt
```

A GPU is recommended but not required — the models will fall back to CPU automatically.

## Quick start: predicting miRNA binding sites

The trained binding-site model is included in [models/pairwise_onehot_model_20260105_200141.pt](models/pairwise_onehot_model_20260105_200141.pt).

### 1. Prepare your input

A TSV file with at minimum these columns:

| column 0 (target/mRNA) | column 1 (miRNA) | label |
|---|---|---|
| `TTTTTTTT...GACAGTGG` | `TGTGCAAATCTATGCAAAACTGA` | 0 |

The `label` column is required by the data loader but is ignored at inference. Set it to `0` if you don't have ground truth. A small example is provided in [data/chimeric_datasets/sample_dataset/](data/chimeric_datasets/sample_dataset/).

### 2. Run inference

```bash
cd code/pairwise_binding_site_model

python -m inference.predict \
    --model_path ../../models/pairwise_onehot_model_20260105_200141.pt \
    --input_file path/to/your_sites.tsv \
    --output_file predictions.tsv \
    --model_type pairwise_onehot \
    --batch_size 32
```

The output TSV is your input plus two columns:

- `prediction_score` — binding probability in [0, 1]
- `predicted_class` — 1 if `prediction_score > 0.5`, else 0

There is also a ready-to-edit wrapper script at [analysis/pairwise_binding_site_model/inference.sh](analysis/pairwise_binding_site_model/inference.sh).

## Quick start: predicting gene-level repression

See [analysis/gene_level_model/README.md](analysis/gene_level_model/README.md) for the full walkthrough. Briefly:

```bash
# install gene-level model dependencies
pip install -r analysis/gene_level_model/requirements.txt

# download the training/eval data
bash analysis/gene_level_model/download_data.sh

# evaluate on a test set (or train your own — see analysis/gene_level_model/train.sh)
bash analysis/gene_level_model/evaluate.sh
```

The gene-level model takes a full 3'UTR sequence (up to several thousand nt) and a miRNA sequence and predicts a scalar fold change.

## Explainability

The binding-site model supports SHAP-based attribution (via Captum's GradientShap). See [code/pairwise_binding_site_model/README.md](code/pairwise_binding_site_model/README.md) for the SHAP, clustering, and aggregation pipelines.

## Downloading the public datasets

To reproduce the published results or train from scratch:

```bash
bash data/scripts/run_zenodo_downloader.sh
```

This pulls the AGO2 eCLIP Manakov 2022 train / test / leftout splits from Zenodo into [data/chimeric_datasets/](data/chimeric_datasets/).

## Repository layout

- [code/](code/) — model definitions, encoders, training and inference scripts.
- [analysis/](analysis/) — runnable wrapper scripts (`train.sh`, `inference.sh`, etc.) for each model.
- [data/](data/) — placeholder; populated by the download scripts above.
- [models/](models/) — trained model checkpoints.

## Models leaderboard

We track model performance on the Manakov22 test and leftout datasets, ranked by Average Precision score (AP) `AP(test) + AP(leftout)`.

| Rank | Model | AP(test) | AP(leftout) | Model | Code | Date | Authors |
|------|-------|-------------|----------------|-------|------|------|---------|
| 1 | Pairwise encoding with conservation (+2 channels) | 85.93 | 82.26 | [model](https://drive.google.com/drive/folders/17pGBXqX7aoH_KyyoulRa3zFegzRB2P2q?usp=drive_link) | [code](https://github.com/BioGeMT/miRBind_2.0/tree/dimos/conservation_channels/code/pairwise_cnn_conservation) | 2025-03-27 | Dimos, David, Panos |
| 2 | Pairwise encoding CNN | 84.97 | 83.08 | [model](https://drive.google.com/drive/folders/1dFsm0CcC7WL2mP4h5a6UZtVt57ICH3vB?usp=drive_link) | [code](https://github.com/BioGeMT/miRBind_2.0/tree/main/code/pairwise_binding_site_model) | 2025-03-19 | David, Panos |
| 3 | Retrained miRBind CNN (miRBench) | 84.00 | 81.00 | — | — | 2025-03-19 | Eva |
| 4 | TargetScanCNN | 77.00 | 76.00 | — | — | 2025-03-19 | TargetScan |

## Transcript repression predictions
[Transcript repression predictions stored on Google Drive](https://drive.google.com/drive/folders/1Wuw9aFpO5oNpHrW3Tna9DNC7H5LLqYc-?usp=sharing)  

## Citation

If you use miRBind 2.0 in your work, please cite the corresponding manuscript: [miRBind2 enables sequence-only prediction of miRNA binding and transcript repression](https://www.biorxiv.org/content/10.64898/2026.03.19.712027v1).
