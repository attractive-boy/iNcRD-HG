# Stuart-Landau Oscillatory Graph Neural Network

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.0](https://img.shields.io/badge/pytorch-2.5.0-orange.svg)](https://pytorch.org/)

Official implementation of **"Stuart-Landau Oscillatory Graph Neural Network"**, accepted at **TheWebConf (WWW) 2026**.

## Overview

**Four Oscillatory / Non-oscillatory Frameworks:**
- **Stuart-Landau** (Proposed): `SLGCN`, `SLGAT`, `SLTrans`
- **GraphCON** (Harmonic): `GraphCONGCN`, `GraphCONGAT`, `GraphCONTrans`
- **Kuramoto**: `KuramotoGCN`, `KuramotoGAT`, `KuramotoTrans`
- **Baseline**: `BaseGCN`, `BaseGAT`, `BaseTrans`

**Total: 12 model variants** (4 frameworks × 3 coupling types)

---

## Installation

```bash
# Clone repository
git clone https://github.com/kevvzhang/StuartLandauGNN.git
cd StuartLandauGNN

# Create environment
conda create -n slgnn python=3.12
conda activate slgnn

# Install dependencies
pip install -r requirements.txt

# Or use Docker
docker build -t slgnn .
docker run --gpus all -it slgnn
```

---

## Supported Datasets

**Node Classification (9):** Cora, Citeseer, Pubmed, Texas, Wisconsin, Cornell, Amazon-ratings, Chameleon, Squirrel

**Graph Classification (3):** MUTAG, ENZYMES, PROTEINS

**Graph Regression (3):** ESOL, FreeSolv, Lipo

---

## Usage

### 1. Direct Execution

```bash
# Single run (uses pre-tuned hyperparameters from best_params.py automatically)
python run.py --dataset Cora --model SLGCN

# Hyperparameter tuning
python tune.py --dataset Cora --model SLGCN 

# Extract best parameters from logs
python parse_logs.py --log_dir ./logs

# Multiple repeats
python run.py --dataset Cora --model SLGCN --test_repeats 100
```

**Note:** `run.py` automatically loads tuned hyperparameters from `best_params.py` when available.

### 2. Kubernetes

```bash
# Batch tuning jobs
python launch_kubernetes.py
```

See `tune_kubernetes.sh` and `run_kubernetes.sh` for single job examples.

### 3. SLURM

```bash
# Batch tuning jobs
python launch_slurm.py

# Batch verification jobs 
python launch_batch_verify.py
```

---

## Project Structure

```
SLGNN/
├── slmodel.py                 # All 12 model architectures
├── layer.py                   # Custom GNN layers
├── data.py                    # Dataset loaders
├── experiment.py              # Training loops
├── run.py                     # Single run script
├── tune.py                    # Hyperparameter tuning
├── parse_logs.py              # Extract best parameters
├── best_params.py             # Pre-tuned hyperparameters
├── launch_kubernetes.py       # Kubernetes launcher
├── launch_slurm.py            # SLURM launcher
├── launch_batch_verify.py     # Verification launcher
├── requirements.txt           # Dependencies
├── logs/                      # Tuning logs (auto-generated)
├── data/                      # Downloaded datasets (auto-generated)
└── yml/                       # Kubernetes configs
```

---

## Citation

```bibtex
@inproceedings{zhang2026stuart,
  title={Stuart-Landau Oscillatory Graph Neural Network},
  author={Zhang, Kaicheng and Reynolds, David N. and Deidda, Piero and Tudisco, Francesco},
  booktitle={Proceedings of the Web Conference},
  year={2026}
}
```
