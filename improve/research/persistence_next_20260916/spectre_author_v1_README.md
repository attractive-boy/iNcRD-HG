# Spectre

[Mattie Ji](https://github.com/maroon-scorch) | [Amauri H. Souza](https://www.amauriholanda.org) |  [Vikas Garg](https://www.aalto.fi/en/people/vikas-kumar-garg)

This is the official repo for the paper [Graph persistence goes spectral](https://openreview.net/pdf?id=wU8IKGLpbi) (NeurIPS 2025).


## Installing routines to compute SpectRe (FastSpectre) diagrams
This repo is highly based on https://github.com/Aalto-QuML/RePHINE. Please, see the base repo for installing basic dependencies. 

```
cd torch_ph
python setup_spectre.py install 
python setup_spectre_scheduling.py install
```

## Experiments on expressivity

To reproduce the experiments on Cayley graphs, see the python notebook ```Experiments - Cayley.ipynb```.

For BREC datasets, run the python notebook ```Experiments - BREC.ipynb```. 


## Experiments using real data

For the main experiments, we run the ```main.py``` with the arguments in ```cli_main.py```. For instance, to run FastSpectre on NCI109 combined with a GCN model, we run:
```
python main.py --dataset NCI109  --diagram_type fast_spectre --gnn gcn --max_epochs 200 --out_dim_eigen_deep_set 32 --num_filtrations 2  
```

## Citation

```bibtex
@inproceedings{spectre,
title={Graph Persistence goes Spectral},
author={Mattie Ji and Amauri H. Souza and Vikas K. Garg},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
year={2025},
url={https://openreview.net/forum?id=wU8IKGLpbi}
}
```