# ResistanceCurvature

A powerful Python package for calculating resistance curvature

## Features

- **High performance**: GPU acceleration supported
- **Easy to Use**: Fully based on PyTorch, with a clear interface design

## Installation

You can install the package using pip:

```bash
pip install ResistanceCurvature
```


## Simple Example

```python
import torch
from ResistanceCurvature.resistance_curvature import ResistanceCurvature

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rc = ResistanceCurvature(device=device)
adj_tensor = torch.tensor([[0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0],], dtype=torch.float32, device=device)
node_curvature, edge_curvature = rc.cal_curvature(adj_tensor)
print("Node Curvature:", node_curvature)
print("Edge Curvature:", edge_curvature)

```
