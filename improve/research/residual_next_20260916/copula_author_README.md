# Official Implementation of CopulaLSP

This is an official implementation repository for **CopulaLSP**.
Please refer to our accepted paper for details:  
_A Scalable Inter-edge Correlation Modeling in CopulaGNN for Link Sign Prediction, ICLR 2026._ [[paper](https://openreview.net/forum?id=U7tR3lCRr5)]

## Setup

### Install

```bash
pip install -r requirements.txt
```

### Details

```txt
# python>=3.10
torch==2.7.0
torch-geometric==2.6.1
scikit-learn==1.6.1
hydra-core==1.3.2
hydra-optuna-sweeper==1.2.0
pytorch-memlab
tabulate
```

## Run

### BitcoinAlpha  

```bash
python main.py -m --config-name CopulaLSP \
    seed='range(2017, 2027)' \
    dataset=BitcoinAlpha \
    model.eps=0.04 \
    model.eta=0.0008
```

### BitcoinOTC  

```bash
python main.py -m --config-name CopulaLSP \
    seed='range(2017, 2027)' \
    dataset=BitcoinOTC \
    model.eps=0.05 \
    model.eta=0.0001
```

### WikiElec  

```bash
python main.py -m --config-name CopulaLSP \
    seed='range(2017, 2027)' \
    dataset=WikiElec \
    model.eps=0.04 \
    model.eta=0.005
```

### WikiRfa  

```bash
python main.py -m --config-name CopulaLSP \
    seed='range(2017, 2027)' \
    dataset=WikiRfa \
    model.eps=0.04 \
    model.eta=0.01
```

### SlashDot  

```bash
python main.py -m --config-name CopulaLSP \
    seed='range(2017, 2027)' \
    dataset=SlashDot \
    model.eps=0.04 \
    model.eta=0.01
```

### Epinions  

```bash
python main.py -m --config-name CopulaLSP \
    seed='range(2017, 2027)' \
    dataset=Epinions \
    model.eps=0.04 \
    model.eta=0.001
```

## BibTex

```bibtex
@inproceedings{
    CopulaLSP,
    title={A Scalable Inter-edge Correlation Modeling in {C}opula{GNN} for Link Sign Prediction},
    author={Jinkyu Sung and Myunggeum Jee, and Joonseok Lee},
    booktitle={In Proceeding of The International Conference on Learning Representations (ICLR)},
    year={2026}
}
```
