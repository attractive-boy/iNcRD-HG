## Introduction

STORM (Self-Taught On-the-fly Rescaling via Meta loss) is a flexible loss rescaling method for learning from noisy labels.

STORM
- uses a novel meta learning scheme called meta loss rescaling that eliminates the need for clean validation data by rescaling both the loss in the inner loop and the meta loss in the outer loop, using noisy validation data.
- is flexible as it dynamically decides how much importance to assign to a sample at each training stage and keeps learning from the model's own signals.
- is efficient as it uses features based on sample losses and prediction probabilities instead of sample gradients, reducing computational complexity.
- is robust as it handles class imbalance, different types of noise, and prevents overfitting.

## Recent updates

- 2025.02.24: Initial commit

## How to run

### Preparations

- Clone the TripPy source code:

```
git clone https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public.git trippy
```

- Download and prepare the datasets. See data/README.md for instructions.

### Run

Scripts are provided for demonstrating how to use STORM.

`DO.example.*` will train and evaluate a model with settings that were used for experiments in our paper "Learning from Noisy Labels via Self-Taught On-the-Fly Meta Loss Rescaling".
- `DO.example.spam` applies STORM to TF-IDF encoded Youtube and SMS datasets.
- `DO.example.glue` applies STORM to CoLA, MRPC and RTE datasets from the GLUE benchmark.
- `DO.example.dst` applies STORM to dialogue state tracking on the MultiWOZ 2.4 dataset.

Using the command line parameter `--simulate_only` to storm.py or storm_dst.py will recreate the baselines without applying STORM.
Note that this will deactivate the loss rescaling, but not the training of the loss rescaler. The loss rescaler merely remains unused.

## Datasets

STORM is not limited to a particular set of datasets. In the paper, we evaluated STORM on the following datasets:
- Youtube (https://doi.org/10.24432/C58885)
- SMS (https://doi.org/10.24432/C5CC84)
- CoLA (https://gluebenchmark.com/tasks)
- MRPC (https://gluebenchmark.com/tasks)
- RTE (https://gluebenchmark.com/tasks)
- MultiWOZ 2.1 (https://github.com/budzianowski/multiwoz.git)
- MultiWOZ 2.4 (https://github.com/smartyfh/MultiWOZ2.4.git)

The ```--task_name``` is
- 'youtube', for Youtube
- 'sms', for SMS
- 'cola', for CoLA
- 'mrpc', for MRPC
- 'rte', for RTE
- 'multiwoz21', for MultiWOZ 2.1
- 'multiwoz21_legacy', for MultiWOZ 2.4

## Requirements

- torch (tested: 2.0.0)
- transformers (tested: 4.18.0)
- tensorboardX (tested: 2.1)

## Citation

This work is published as [Learning from Noisy Labels via Self-Taught On-the-Fly Meta Loss Rescaling](https://arxiv.org/abs/2412.12955)

If you use STORM in your own work, please cite our work as follows:

```
@inproceedings{heck2024storm,
    title = "Learning from Noisy Labels via Self-Taught On-the-Fly Meta Loss Rescaling",
    author = "Heck, Michael and Geishauser, Christian and Lubis, Nurul and van Niekerk, Carel and 
    	      Feng, Shutong and Lin, Hsien-Chin and Ruppik, Benjamin Matthias and Vukovic, Renato and
              Ga{\v{s}}i{\'c}, Milica",
    booktitle = "Proceedings of the AAAI Conference on Artificial Intelligence",
    month = "Mar.",
    year = "2025",
    volume = "39",
    address = "Philadelphia, Pennsylvania, USA",
    publisher = "AAAI Press, Washington, DC, USA",
    organization = "Association for the Advancement of Artificial Intelligence"
}
```
