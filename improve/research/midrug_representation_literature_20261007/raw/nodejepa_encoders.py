"""Graph neural network encoders for Graph-JEPA.

Implements context encoder and target encoder (EMA-updated) based on
standard GNN architectures (GCN, GAT, GIN, GraphSAGE).
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool,
)
from typing import Optional


class GNNEncoder(nn.Module):
    """General GNN encoder supporting GCN, GAT, GIN, and GraphSAGE backbones.

    Used as both the context encoder (updated by gradient) and the
    target encoder (updated by EMA) in Graph-JEPA.

    Architecture:
        [Input features] -> Linear projection -> [L x GNN layers + LayerNorm]
        -> Output embeddings

    For graph-level tasks, a global pooling is applied.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        dropout: float = 0.0,
        norm: str = "layer",
        graph_level: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.graph_level = graph_level

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(self._make_conv(hidden_dim, hidden_dim))
            if norm == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norms.append(nn.Identity())

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # Learnable [MASK] token used to replace masked-node features instead of
        # zeroing them out. A shared learnable vector avoids the "zero pollution"
        # problem where zero-vectors dilute neighbours' representations during
        # normalized GCN aggregation (see get_context_embeddings docstring).
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def _make_conv(self, in_dim: int, out_dim: int) -> nn.Module:
        if self.gnn_type == "gcn":
            return GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
        elif self.gnn_type == "gat":
            return GATConv(in_dim, out_dim, heads=4, concat=False, dropout=0.0)
        elif self.gnn_type == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
            return GINConv(mlp)
        elif self.gnn_type == "sage":
            return SAGEConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        mask_mode: str = "token",
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [N, in_dim] node features.
            edge_index: [2, E] edge indices. For proper context/target
                isolation this should already be the *restricted* edge set
                (e.g. the subgraph induced by visible nodes); this method does
                NOT delete edges itself.
            batch: [N] batch assignment for graph-level pooling.
            node_mask: [N] boolean mask, True for VISIBLE (context) nodes,
                False for masked nodes. If provided, masked nodes have their
                projected features replaced according to ``mask_mode``.
            mask_mode: How to treat masked-node features:
                - "token": replace with the learnable [MASK] token (default,
                  GraphMAE/BERT style; avoids zero-pollution of neighbours).
                - "zero": legacy behaviour (multiply by 0) — kept only for
                  ablation / backward compatibility.

        Returns:
            [N, hidden_dim] node embeddings, or [B, hidden_dim] if graph_level.
        """
        h = self.input_proj(x)

        if node_mask is not None:
            visible = node_mask.bool()
            if mask_mode == "zero":
                h = h * visible.unsqueeze(1).float()
            else:  # "token"
                masked = ~visible
                if masked.any():
                    h = h.clone()
                    h[masked] = self.mask_token.to(h.dtype)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h, edge_index)
            h = norm(h)
            if i < self.num_layers - 1:
                h = self.act(h)
                h = self.dropout(h)

        if self.graph_level and batch is not None:
            h = global_mean_pool(h, batch)
        return h

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get node-level embeddings without masking."""
        return self.forward(x, edge_index, batch=batch, node_mask=None)


class GraphEncoder(nn.Module):
    """Wrapper that manages context encoder + EMA target encoder.

    The context encoder is updated by backpropagation.
    The target encoder is updated by exponential moving average (EMA)
    of the context encoder weights, following I-JEPA.

    No gradients flow through the target encoder (stop-gradient).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        dropout: float = 0.0,
        norm: str = "layer",
        graph_level: bool = False,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.context_encoder = GNNEncoder(
            in_dim, hidden_dim, num_layers, gnn_type, dropout, norm, graph_level
        )
        self.target_encoder = GNNEncoder(
            in_dim, hidden_dim, num_layers, gnn_type, dropout, norm, graph_level
        )
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.ema_momentum = ema_momentum
        self.ema_start_momentum = ema_momentum

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None):
        """Update target encoder via EMA of context encoder weights."""
        m = momentum if momentum is not None else self.ema_momentum
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

    def schedule_ema(self, current_step: int, total_steps: int):
        """Schedule EMA momentum from initial value toward 1.0 (cosine schedule).

        Following I-JEPA, the EMA momentum increases over training so that
        the target encoder becomes more stable as training progresses.
        """
        progress = min(current_step / max(total_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        self.ema_momentum = 1.0 - (1.0 - self.ema_start_momentum) * cosine

    def forward_context(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        mask_mode: str = "token",
    ) -> torch.Tensor:
        """Forward through context encoder (with gradients).

        ``edge_index`` should be the visible-subgraph edge set when structural
        isolation is enabled (the caller is responsible for restricting it).
        ``node_mask`` marks visible nodes; masked nodes are handled per
        ``mask_mode`` ("token" or "zero").
        """
        return self.context_encoder(
            x, edge_index, batch=batch, node_mask=node_mask, mask_mode=mask_mode
        )

    @torch.no_grad()
    def forward_target(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through target encoder (no gradients).

        The caller decides what ``edge_index`` / ``x`` cover. For proper JEPA
        isolation the target encoder should be run on the *full* graph to
        produce stable targets, but only the embeddings at target positions are
        used downstream (context nodes' target embeddings are discarded)."""
        return self.target_encoder(x, edge_index, batch=batch, node_mask=None)
