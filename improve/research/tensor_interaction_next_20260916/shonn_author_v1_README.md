# Spectral Hypergraph Neural Networks

Repository for spectral hypergraph neural network experiments.

## Installation

To install the required dependencies it is advised to use miniconda:

```bash
conda create -n "hyperspectral" python=3.14
conda activate hyperspectral
pip install -r requirements.txt
```

## Important Notes for Training

**GPU Configuration**: All training scripts currently hardcode the GPU selection to device `1` (`os.environ["CUDA_VISIBLE_DEVICES"] = "1"`). If your machine does not have a second GPU, or you wish to use a different one, you will need to modify this line at the top of the training scripts (`mnist.py`, `cifar10.py`, etc.) to match your system.