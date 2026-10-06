---
license: apache-2.0
pipeline_tag: other
tags:
- Tabular
- In-Context-Learning
- Transformer
arxiv: 2410.18164
---

# TabDPT: Scaling Tabular Foundation Models on Real Data

**TabDPT** is an open-source foundation model for tabular data based on in-context learning (ICL). It is trained on real-world data and can generalize to new tasks across classification and regression without additional training or hyperparameter tuning.

- **Paper:** [TabDPT: Scaling Tabular Foundation Models on Real Data](https://huggingface.co/papers/2410.18164)
- **GitHub:** [layer6ai-labs/TabDPT](https://github.com/layer6ai-labs/TabDPT)

## Installation

To set up the environment, ensure you have Python 3.10 or 3.11, then run:

```bash
git clone https://github.com/layer6ai-labs/TabDPT.git
cd TabDPT
pip install -e .
```

## Example Usage 

TabDPT performs zero-shot prediction on new tabular datasets. For detailed working examples, please refer to the following files in the GitHub repository:
- `tests/cls_example.py` for classification tasks.
- `tests/reg_example.py` for regression tasks.

**Performance Tips:**
For better performance, you can increase `context_size` or increase `n_ensembles` to trade off speed and accuracy.

## Updates

### Update April 2025: New Model
**Version 1.1 is now available.** We have improved the prediction performance of TabDPT through increased training stability.

### Update December 2024: Faster Inference
Added support for flash attention (with bf16 precision) and compile flag. Both are enabled to True by default and should lead to a significant speed-up.

## Citation

```bibtex
@inproceedings{
    ma2025tabdpt,
    title={Tab{DPT}: Scaling Tabular Foundation Models on Real Data},
    author={Junwei Ma and Valentin Thomas and Rasa Hosseinzadeh and Alex Labach and Hamidreza Kamkari and Jesse C. Cresswell and Keyvan Golestan and Guangwei Yu and Anthony L. Caterini and Maksims Volkovs},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=pIZxEOZCId}
}
```

© Copyright 2024-2025 The Toronto-Dominion Bank and/or its affiliates