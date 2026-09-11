## iNcRD-HG

This repository contains the source code for our paper "Semantic-Enhanced Heterogeneous Graph Learning for Identifying ncRNAs Associated with Drug Resistance". It includes the implementation of the proposed predictor, along with datasets and tutorials to help users reproduce and extend our work.

---

## Introduction

Identifying non-coding RNAs (ncRNAs) associated with drug resistance is crucial for understanding drug response mechanisms and discovering new therapeutic targets.  
We propose **iNcRD-HG**, a framework for predicting ncRNA–drug resistance associations. iNcRD-HG integrates molecular interactions, expression profiles, and drug structures to construct a biologically informative network. Through relation-type-aware message passing and an interpretability module, it captures complex contextual dependencies and uncovers cooperative ncRNA roles in drug resistance.

---

## Datasets

All datasets are available at: [Zenodo Link] (The data link will be made public upon the completion of the review process)  

- **Heterogeneous graph/**: Files for constructing the heterogeneous graph.  
- **Train-val-test-data/**: Training, validation, and test sub-datasets of ncRNA–drug resistance associations.  
- **Extract latent feature/**: Embeddings of ncRNA–drug resistance pairs extracted from iNcRD-HG.  
- **Case studies/**: Datasets used for case study experiments.  

---

## Environment Setup

- **OS**: Ubuntu 20.04.5  
- **GPU**: NVIDIA GeForce RTX 3060  
- **CUDA**: 11.3  
- **Python**: 3.8.8  
- **PyTorch**: 1.12.1  

Install the environment via:
```bash
conda env create -f iNcRD-HG-env.yml
```

## Main Scripts

- `drug_language_feature.py` + `ChemBERTa-zinc-base-v1.zip`: Extraction of drug features using ChemBERT.  
- `drug_similarity.py`: Computation of drug similarities.  
- `heter_data_prepare.py`: Construction of the heterogeneous graph and preparation of train/val/test datasets.  
- `find_best_params.py`: Parameter tuning on the validation dataset.  
- `predict_association.py` + `compare_base_GNN.py`: Training and testing of iNcRD-HG and baseline GNN models.  
- `graph_explain.py`: Computation of node and edge contributions in the heterogeneous graph.  
- `ablation_predict_association.py`: Training and testing for the ablation study.  
- `feature.py`: Visualization of ncRNA–drug resistance pairs under different strategies.  
- `predict_simulated_drug.py` + `compare_base_simulated_drug.py`: Training and testing under simulated novel drug scenarios.  
- `plot_case_drug.py`: Case study visualization.  

## Running Examples

Create the environment:
```bash
conda env create -f iNcRD-HG-env.yml
```

Train and test iNcRD-HG:
```bash
python predict_association.py
```

Compute node and edge contributions:
```bash
python graph_explain.py -e 'MiDrug' -f 'test' -i '0' -k 100 –p
```

Case study visualization:
```bash
python plot_case_drug.py -i '0' -k 100
```

## Replacement backbone experiments (2026-09-10)

The current experimental replacement is **graph-derived pair features + TabM
(ICLR 2025)**. It has its own training and prediction entry points, frozen
configuration files and verified checkpoints. The original scripts remain
available for reproduction and comparison.

On the existing outer validation partition, the three-seed mean AP changes from
0.943892 to 0.966260 for LncDrug and from 0.919169 to 0.921461 for MiDrug.
Global AUC also improves on both tasks. LncDrug's per-drug macro AP decreases;
these results concern a reused validation partition, not a new independent test.

See [usage, architecture and fixed configurations](improve/GRAPH_TABM.md),
[the four-architecture comparison](improve/results/backbone_comparison_20260910/REPORT.md),
and [the literature check](improve/research/backbone_replacement_20260910/RESEARCH.md).

```bash
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.predict_backbone \
    --task MiDrug --drug-index 0 --out mi_drug0_scores.csv --device cuda:0
```
