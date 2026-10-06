# scRADAR Training Code

![scRADAR Overview](docs/flowchart.png)

This repository contains the training pipeline for scRADAR (Response Analysis via Drug-Aware Routing), a single-cell drug response prediction framework.

## Reproducible Setup (Step-by-step)

### Step 1: Install Miniconda/Conda
Make sure `conda` is available on your system.

### Step 2: Create the environment
The full environment is pinned in `scRADAR.yml`.

```bash
conda env create -f scRADAR.yml
conda activate psych310
```

If you prefer a custom environment name:
```bash
conda env create -f scRADAR.yml -n scRADAR
conda activate scRADAR
```

### Step 3: Prepare data and resources
Place the required files in these paths:

```
data/processed/                  # input .h5ad datasets
data/resource/                   # pathway resources (GMT/TSV)
data/resources/                  # drug fingerprint cache
```

Required resource files:
- `data/resource/c2.cp.reactome.v2025.1.Hs.symbols.gmt`
- `data/resource/progeny_human_top500.tsv`
- `data/resources/drug_fingerprint_cache.npz`

Dataset list and paths are defined in `config.py`. Ensure your `.h5ad` files match the dataset names configured there.

### Step 4: (Optional) Rebuild the drug fingerprint cache
If you want to regenerate the cache:
```bash
python scripts/build_drug_fingerprint_cache.py
```

### Step 5: Run training
Example run using the provided overrides:
```bash
PIPELINE_OVERRIDES="$(cat overrides/run_global.json)" \
python code/run.py --config config.py --run-name run_global_base
```

### Step 6: Check outputs
Outputs are saved under:
```
${RESULTS_ROOT}/run_global_base/
```

Key files:
- `results.csv`
- `summary.json`
- `prototype_diagnostics.json`
- `model_checkpoint.npz`

## Environment Specification (from scRADAR.yml)

### Conda packages
- python=3.10
- pip
- numpy=1.26.4
- pandas=1.5.3
- scipy=1.15.2
- scikit-learn=1.7.2
- scanpy=1.9.8
- anndata=0.11.4
- h5py=3.13.0
- igraph=0.10.17
- python-igraph=0.11.9
- leidenalg=0.10.2
- louvain=0.8.2
- umap-learn=0.5.9.post2
- pynndescent=0.5.13
- matplotlib=3.7.3
- statsmodels=0.14.5
- networkx=3.4
- numba=0.59.1
- llvmlite=0.42.0
- pytables=3.10.1
- openpyxl=3.1.5
- tqdm=4.65.2
- pillow=11.3.0
- requests=2.32.5
- lxml=6.0.2

### Pip packages
- adjusttext==1.3.0
- blinker==1.9.0
- click==8.3.1
- cloudpickle==3.1.1
- decoupler==2.1.1
- docrep==0.3.2
- filelock==3.20.3
- flask==3.1.2
- fsspec==2026.1.0
- imbalanced-learn==0.14.0
- itsdangerous==2.2.0
- jinja2==3.1.6
- legendkit==0.3.6
- loguru==0.7.2
- markupsafe==3.0.3
- marsilea==0.5.6
- mpmath==1.3.0
- pandas-stubs==2.3.3.260113
- platformdirs==4.5.0
- psignifit==4.3
- python-pptx==1.0.2
- scikit-misc==0.5.2
- seaborn==0.13.2
- session-info2==0.2.3
- shap==0.45.1
- slicer==0.0.8
- sympy==1.14.0
- torch==2.9.1
- urllib3==2.5.0
- werkzeug==3.1.5
- xlsxwriter==3.2.9

## Repository Layout
```
config.py                 # Paths/dataset metadata & PIPELINE_FLAGS
overrides/run_global.json # Override items for this run
code/
  run.py                  # Training entry
  data_loader.py          # Data loading and preprocessing
  features/               # Pathway and fingerprint feature builders
  models/                 # ProtoMechanism
  postprocess/            # Threshold selection
scripts/                  # Fingerprint cache tools
data/                     # processed inputs, results outputs, resource files
docs/                     # README figures
```
