# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from abc import abstractmethod
from torch import nn
try:
    # Optional dependency chain (torch_sparse / torch_geometric). This module's
    # learner classes below don't require it, so keep import optional.
    from .lib import laplace as lap
except Exception:
    lap = None

torch.set_default_dtype(torch.float64)


class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""
    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index):
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features (src + dst) and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], hidden_channels: int = 64, sheaf_act="tanh"):
        super(LocalConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        # Calculate output dimension (e.g., if out_shape is (2,3), dim is 6)
        out_dim = int(np.prod(out_shape))

        # Layer 1: Projection / Feature Mixing (128 -> 64)
        self.linear1 = nn.Linear(in_channels * 2, hidden_channels)

        # Layer 2: Compression to Restriction Map (64 -> 6)
        self.linear2 = nn.Linear(hidden_channels, out_dim, bias=False)

        # Activation for the hidden layer
        self.hidden_act = nn.ELU() # or ReLU, GELU

        # Activation for the final restriction map (controls the range of the map values)
        if sheaf_act == 'id':
            self.final_act = lambda x: x
        elif sheaf_act == 'tanh':
            self.final_act = torch.tanh
        elif sheaf_act == 'elu':
            self.final_act = F.elu
        else:
            raise ValueError(f"Unsupported activation {sheaf_act}")

    def forward(self, x, src, dst):
        # x shape: [N, 128] if using the concatenated features you mentioned
        x_src = x[src]
        x_dst = x[dst]

        # 1. Concatenate
        cat_feat = torch.cat([x_src, x_dst], dim=1) # [E, 256] (if input is 128 each) -> or [E, 128] if sum is 128

        # 2. MLP First Layer + Non-linearity
        h = self.linear1(cat_feat)
        h = self.hidden_act(h)

        # 3. MLP Second Layer -> Prediction
        maps = self.linear2(h)

        # 4. Final Activation (Constrain the map properties)
        maps = self.final_act(maps)

        # reshape to desired output
        if len(self.out_shape) == 2:
            maps = maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            maps = maps.view(-1, self.out_shape[0])

        return maps


class QuadraticFormSheafLearner(SheafLearner):
    """
    Re-implemented as an MLP-based learner (similar to LocalConcatSheafLearner)
    for consistency in the linear part.
    (The original quadratic form logic is removed for structural uniformity.)
    """
    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], hidden_channels: int = 64, sheaf_act="tanh"):
        super(QuadraticFormSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        out_dim = int(np.prod(out_shape))

        # Layer 1: Projection / Feature Mixing

        self.linear1 = nn.Linear(in_channels * 2, in_channels*4)

        self.linear2 = nn.Linear(in_channels * 4, in_channels*2)

        self.linear3 = nn.Linear(in_channels * 2, in_channels)

        # Layer 2: Compression to Restriction Map
        self.linear4 = nn.Linear(in_channels, out_dim, bias=False)

        # Hidden layer activation
        self.hidden_act = nn.ELU()

        # Final Activation for the restriction map
        if sheaf_act == 'id':
            self.final_act = lambda x: x
        elif sheaf_act == 'tanh':
            self.final_act = torch.tanh
        elif sheaf_act == 'elu':
            self.final_act = F.elu
        else:
            raise ValueError(f"Unsupported activation {sheaf_act}")

    def forward(self, x, src, dst):
        x_src = x[src]
        x_dst = x[dst]

        # 1. Concatenate
        cat_feat = torch.cat([x_src, x_dst], dim=1) # [E, 256] (if input is 128 each) -> or [E, 128] if sum is 128

        # 2. MLP First Layer + Non-linearity
        h = self.linear1(cat_feat)
        h = self.hidden_act(h)
        # 3. MLP Second Layer -> Prediction
        h = self.linear2(h)
        h = self.hidden_act(h)

        h = self.linear3(h)
        h = self.hidden_act(h)

        h = self.linear4(h)
        h = self.hidden_act(h)
        # 4. Final Activation
        maps = self.final_act(h)

        # reshape to desired output
        if len(self.out_shape) == 2:
            maps = maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            maps = maps.view(-1, self.out_shape[0])

        return maps


class GatedLocalConcatSheafLearner(SheafLearner):
    """
    Gated Sheaf Learner.
    Learns a restriction map P and a gate value alpha in [0, 1].
    The final output parameters are scaled by alpha: params_final = alpha * params.

    Interpretation:
    - If params map to a rotation (Lie Algebra), scaling by alpha < 1 reduces the rotation angle.
    - alpha = 0 => Identity transformation (no transport).
    - alpha = 1 => Full learned transformation.
    """

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], hidden_channels: int = 64, sheaf_act="tanh"):
        super(GatedLocalConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        # Calculate output dimension
        out_dim = int(np.prod(out_shape))

        # Layer 1: Projection / Feature Mixing
        self.linear1 = nn.Linear(in_channels * 2, hidden_channels)

        # Layer 2: Prediction
        # We output out_dim (for params) + 1 (for gate)
        self.linear2 = nn.Linear(hidden_channels, out_dim + 1, bias=False)

        # Activation for the hidden layer
        self.hidden_act = nn.ELU()

        # Activation for the final restriction map parameters
        if sheaf_act == 'id':
            self.final_act = lambda x: x
        elif sheaf_act == 'tanh':
            self.final_act = torch.tanh
        elif sheaf_act == 'elu':
            self.final_act = F.elu
        else:
            raise ValueError(f"Unsupported activation {sheaf_act}")

    def forward(self, x, src, dst):
        # x shape: [N, in_channels]
        x_src = x[src]
        x_dst = x[dst]

        # 1. Concatenate
        cat_feat = torch.cat([x_src, x_dst], dim=1)

        # 2. MLP First Layer
        h = self.linear1(cat_feat)
        h = self.hidden_act(h)

        # 3. MLP Second Layer -> Prediction (Params + Gate)
        out = self.linear2(h)

        # Split into params and gate
        # params: [E, out_dim], gate_logit: [E, 1]
        params_raw = out[:, :-1]
        gate_logit = out[:, -1:]

        # 4. Activate
        params = self.final_act(params_raw)
        gate = torch.sigmoid(gate_logit)

        # 5. Apply Gate
        # Scaling the parameters effectively interpolates towards Identity
        # (assuming 0 params -> Identity, which is true for Lie Algebra / exp map)
        gated_params = params * gate

        # Reshape
        if len(self.out_shape) == 2:
            maps = gated_params.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            maps = gated_params.view(-1, self.out_shape[0])

        return maps
