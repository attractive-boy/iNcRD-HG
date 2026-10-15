"""Variance-Covariance regularization (from I-JEPA / VICReg).

Prevents representation collapse by enforcing:
1. Variance: Each embedding dimension should have non-trivial variance
   (above a threshold), preventing all embeddings from collapsing to a point.
2. Covariance: Off-diagonal entries of the covariance matrix should be
   small, preventing dimensional collapse (where only a few dimensions
   are used).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VarianceCovarianceLoss(nn.Module):
    """Variance-Covariance regularization from VICReg/I-JEPA.

    Args:
        variance_target: Minimum standard deviation per dimension.
        variance_epsilon: Numerical stability epsilon.
        covariance_weight: Weight for covariance penalty.
    """

    def __init__(
        self,
        variance_target: float = 1.0,
        variance_epsilon: float = 1e-4,
    ):
        super().__init__()
        self.variance_target = variance_target
        self.variance_epsilon = variance_epsilon

    def forward(self, embeddings: torch.Tensor) -> tuple:
        """Compute variance and covariance regularization.

        Args:
            embeddings: [N, D] batch of embeddings.

        Returns:
            (variance_loss, covariance_loss) tuple of scalar tensors.
        """
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device), torch.tensor(
                0.0, device=embeddings.device
            )

        # Variance: maintain std >= target per dimension
        std = embeddings.std(dim=0)
        std = torch.clamp(std, min=self.variance_epsilon)
        variance_loss = torch.mean(torch.relu(self.variance_target - std))

        # Covariance: decorrelate embedding dimensions
        N, D = embeddings.shape
        emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        cov = (emb_centered.T @ emb_centered) / (N - 1)
        # Off-diagonal penalty
        cov_loss = (cov.fill_diagonal_(0.0) ** 2).mean()

        return variance_loss, cov_loss
