# IGNN-Solver

This repository is the official PyTorch implementation of "IGNN-Solver: A Graph Neural Solver for Implicit Graph Neural Networks".


# Environment Settings 
* python == 3.10   
* Pytorch == 1.13.0  
* Numpy == 1.26.4 
* pandas == 2.2.1
* SciPy == 1.11.3
* scikit-learn == 1.3.0
* torch_scatter == 2.1.1
* torch_sparse == 0.6.17
* torch_geometric == 1.6.1

# Usage 

## Quick Start

Use the unified training script `train.py` for all datasets:

```bash
python train.py --dataset DATASET_NAME --model MODEL_NAME
```

### Supported Datasets
- **citeseer** - CiteSeer dataset
- **acm** - ACM dataset  
- **blogcatalog** - BlogCatalog dataset
- **flickr** - Flickr dataset
- **coraml** - Cora-ML dataset
- **reddit** - Reddit dataset
- **pubmed** - PubMed dataset
- **ogbn-arxiv** - OGB arXiv dataset
- **ogbn-products** - OGB Products dataset
- **amazon-all** - Amazon All dataset

### Supported Models
- **IGNN_solver** - IGNN with solver
- **IGNN_ori** - Original IGNN
- **IGNN_solver_withNN** - IGNN solver with neural network (default)

### Examples

Basic training:
```bash
python train.py --dataset citeseer --model IGNN_solver
```

With custom parameters:
```bash
python train.py --dataset acm --model IGNN_solver_withNN --epochs 100 --lr 0.002 --hidden 128
```

Specify GPU and other settings:
```bash
python train.py --dataset flickr --model IGNN_solver --gpu 0 --threshold 20 --dropout 0.5
```

View all options:
```bash
python train.py --help
```

For more details on dataset configuration, see [DATASET_CONFIG.md](DATASET_CONFIG.md).

# Data Link
* **Citeseer**: [Semi-Supervised Classifcation with Graph Convolutional Networks.](https://github.com/tkipf/gcn)  
* **ACM**: [Heterogeneous Graph Attention Network.](https://github.com/Jhy1993/HAN)  
* **BlogCatalog, Flickr**: [Co-Embedding Attributed Networks.](https://github.com/mengzaiqiao/CAN)  
* **CoraFull**: [Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking.](https://github.com/abojchevski/graph2gauss/)  
* **Amazon-all**: [Implicit Graph Neural Networks](https://github.com/SwiftieH/IGNN)
* **Reddit** [GRAND+: Scalable Graph Random Neural Networks](https://github.com/THUDM/GRAND-plus)
* **ogbn-arxiv, ogbn-products**: [Node Property Prediction|Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/)

* **MUTAG, PTC\_MR, COX2, PROTEINS, NCI1**: [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/)

# Baselines Link

## Implicit GNNs
* **MIGNN** https://github.com/Utah-Math-Data-Science/MIGNN
* **EIGNN** https://github.com/liu-jc/EIGNN
* **IGNN** https://github.com/sczhou/IGNN

## Explicit/Traditional GNNs
* **AM-GCN** https://github.com/zhumeiqiBUPT/AM-GCN
* **GCN** https://github.com/tkipf/pygcn
* **GAT** https://github.com/gordicaleksa/pytorch-GAT
* **SGC** https://github.com/Tiiiger/SGC
* **APPNP** https://github.com/benedekrozemberczki/APPNP
* **JKNet** https://github.com/mori97/JKNet-dgl
* **DEMO-Net** https://github.com/junwu6/DEMO-Net
* **GCNII**  https://github.com/chennnM/GCNII
* **ACM-GCN** https://github.com/SitaoLuan/ACM-GNN

# Graph classification

## Datasets

For all datasets used for graph classification, simply have `torch_geometric` installed and run the code.

## Run
- The experiments are self-contained. Specify the dataset name and run the script, for example:
```bash
python train_graphclassification.py --dataset MUTAG
```

You should obtain the paper-reported results.





# License

This project is open sourced under MIT license.
