# SPD-Sheaf

Code for the paper associated with the ICML 2026 poster:

https://icml.cc/virtual/2026/poster/64132

This repository contains the implementation of SPD-Sheaf Graph Neural Networks
for molecular property prediction.

## Data

The processed `.pth` files for seven molecular datasets are hosted on
Hugging Face:

https://huggingface.co/datasets/YuhanPeng0524/sheaf

Datasets:

- BACE
- BBBP
- ClinTox
- HIV
- MUV
- SIDER
- Tox21

Place the downloaded files under:

```bash
data/processed/
```

## Baselines

Baseline implementations and settings refer to:

https://github.com/LongLee220/KA-GNN

## Usage

```bash
python main.py --config config/bbbp.yaml
python main.py --config config/bace.yaml
python main.py --config config/clintox.yaml
python main.py --config config/hiv.yaml
python main.py --config config/muv.yaml
python main.py --config config/sider.yaml
python main.py --config config/tox21.yaml
```
