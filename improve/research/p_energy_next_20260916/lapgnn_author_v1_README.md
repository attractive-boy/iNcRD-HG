
<h2 align="center">Enhancing Robustness of Graph Neural Networks through $p$-Laplacian </h1>


## Requirements
See that in https://github.com/DSE-MSU/DeepRobust/blob/master/requirements.txt

## Installation
To run the code, first you need to install DeepRobust:
```
pip install deeprobust
```
Or you can clone it and install from source code:
```
git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install
```

## Run the code
```
python train.py --two_stage y --seed 10  --dataset cora --attack no --ptb_rate 0 --epochs 200  --epochs_pre 400 --alpha 1.0  --gamma 1.0 --beta 0.10 --lr_optim 1e-2 --lr 1e-3 
```

## Acknowledgements
The code is based on :
- DeepRobust [(https://github.com/DSE-MSU/DeepRobust)](https://github.com/DSE-MSU/DeepRobust)
- [Pro-GNN](https://github.com/ChandlerBang/Pro-GNN)

