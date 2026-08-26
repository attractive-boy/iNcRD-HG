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

## Popularity-Controlled Benchmark

Generate the debiased benchmark after preparing `./3.heter_data`:

```bash
python analysis/ncrd_bench.py --out ./9.bench
```

The benchmark writes one directory per split: `random`, `matched_drug`,
`matched_rna`, `matched_both`, `cold_rna`, `cold_drug`, and `cold_pair`.
The files use the same `HeteroData` schema as the released training scripts.
Audit every generated split before reporting results:

```bash
python analysis/audit_bench.py \
  --bench ./9.bench \
  --splits random matched_drug matched_rna matched_both cold_rna cold_drug cold_pair \
  --json-out ./9.bench/audit.json
```

The audit fails if held-out positives appear in message-passing edges, sampled
negatives collide with known positives, requested matching marginals are broken,
or cold-drug graphs contain cross-task association edges to invisible drugs.
Run the model and popularity controls with:

```bash
python analysis/run_bench.py --bench ./9.bench --out ./9.bench/results.csv
python analysis/summarize_bench.py --results ./9.bench/results.csv
```

## Multi-task Extension (improvement direction #2)

The released `predict_association.py` trains lncRNA-drug and miRNA-drug as two
completely separate single-task models. `multitask_train.py` introduces a
genuine multi-task design that the paper review identified as a key improvement
direction:

  1. **Shared heterogeneous GNN encoder** with **task-specific decoders**
  2. **Cross-task consistency** via lncRNA-miRNA interaction edges (LncMi)
  3. **Task-aware contrastive** objective (InfoNCE within ncRNA type)
  4. **Uncertainty-weighted** adaptive loss reweighting (Kendall et al., 2018)
  5. **Validation-based** model selection (fixes test-set selection bias §2.3)

Results and comparison with the single-task baseline are in
`analysis/MULTITASK_RESULTS.md`. The multi-task model matches the paper's
LncDrug AUPR (0.9585) with a shared encoder, and improves LncDrug perDrugAUC by
+0.187 on the cold-drug debiased split.

