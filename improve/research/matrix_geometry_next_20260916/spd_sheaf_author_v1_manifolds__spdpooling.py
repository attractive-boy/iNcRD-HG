import torch
import torch.nn as nn
from loguru import logger
from .spd import SPDManifold

class SPDPooling(nn.Module):
    """
    SPD-specific pooling respecting Riemannian geometry
    Based on ICLR 2025: "Understanding Matrix Function Normalizations in Covariance Pooling"

    Three methods:
    - log_euclidean: Map to tangent space, average, map back
    - power_euclidean: Use matrix power (RECOMMENDED, θ=0.5)
    - affine_invariant: Karcher mean (most accurate but slow)
    """
    def __init__(self, pooling_type='power_euclidean', power=0.5, max_iter=10, eps=1e-6):
        super().__init__()
        self.pooling_type = pooling_type
        self.power = power
        self.max_iter = max_iter
        self.eps = eps
        # Non-essential detail; keep at debug to reduce noise
        logger.debug(f"SPDPooling init: type={pooling_type}, power={power}")

    def matrix_power(self, X, power):
        """Matrix power via eigendecomposition: X^θ"""
        eigenvalues, eigenvectors = torch.linalg.eigh(X)
        eigenvalues = torch.clamp(eigenvalues, min=self.eps)
        powered_eigenvalues = torch.pow(eigenvalues, power)
        return eigenvectors @ torch.diag_embed(powered_eigenvalues) @ eigenvectors.transpose(-2, -1)

    def log_euclidean_mean(self, matrices):
        """
        LEM: P_mean = exp(mean(log(P_i)))
        Uses SPDManifold's logmap/expmap
        """
        log_matrices = SPDManifold.logmap_id(matrices)  # [N, d, d]
        mean_log = log_matrices.mean(dim=0)  # [d, d]
        return SPDManifold.expmap_id(mean_log.unsqueeze(0)).squeeze(0)

    def power_euclidean_mean(self, matrices):
        """
        PEM: P_mean = (mean(P_i^θ))^(1/θ)
        RECOMMENDED: θ=0.5 (matrix square root)
        From paper Table 5: Power-EMLR outperforms Log-TMLR
        """
        powered = self.matrix_power(matrices, self.power)  # [N, d, d]
        mean_powered = powered.mean(dim=0)  # [d, d]
        return self.matrix_power(mean_powered.unsqueeze(0), 1.0 / self.power).squeeze(0)

    def affine_invariant_mean(self, matrices):
        """
        AIM: Karcher mean via iterative algorithm
        Most geometrically faithful but computationally expensive
        """
        mean = matrices.mean(dim=0)  # [d, d]
        for _ in range(self.max_iter):
            sqrt_mean = self.matrix_power(mean.unsqueeze(0), 0.5).squeeze(0)
            inv_sqrt_mean = self.matrix_power(mean.unsqueeze(0), -0.5).squeeze(0)

            tangent_sum = torch.zeros_like(mean)
            for P in matrices:
                temp = inv_sqrt_mean @ P @ inv_sqrt_mean

                eigvals, eigvecs = torch.linalg.eigh(temp)
                eigvals = torch.clamp(eigvals, min=self.eps)
                log_eigvals = torch.log(eigvals)
                log_temp = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.T

                tangent_vec = sqrt_mean @ log_temp @ sqrt_mean
                tangent_sum += tangent_vec

            grad = tangent_sum / len(matrices)

            if torch.norm(grad) < self.eps:
                break

            temp = inv_sqrt_mean @ grad @ inv_sqrt_mean
            eigvals, eigvecs = torch.linalg.eigh(temp)
            exp_eigvals = torch.exp(eigvals)
            exp_temp = eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.T
            mean = sqrt_mean @ exp_temp @ sqrt_mean

        return mean

    def forward(self, mat_feats, batch_indices):
        """
        Pool SPD matrices by graph

        Args:
            mat_feats: [N, d, d] SPD matrices for all nodes
            batch_indices: [N] batch assignment for each node

        Returns:
            [batch_size, d, d] pooled SPD matrices
        """
        device = mat_feats.device
        batch_size = batch_indices.max().item() + 1
        d = mat_feats.shape[-1]

        pooled_mats = []

        for i in range(batch_size):
            # Get matrices for graph i
            mask = (batch_indices == i)
            graph_mats = mat_feats[mask]  # [n_i, d, d]

            if len(graph_mats) == 0:
                # Empty graph: return identity
                pooled_mats.append(torch.eye(d, device=device))
                continue

            # Apply SPD pooling
            if self.pooling_type == 'log_euclidean':
                pooled = self.log_euclidean_mean(graph_mats)
            elif self.pooling_type == 'power_euclidean':
                pooled = self.power_euclidean_mean(graph_mats)
            elif self.pooling_type == 'affine_invariant':
                pooled = self.affine_invariant_mean(graph_mats)
            elif self.pooling_type == 'euclidean':
                # Baseline: simple average (not respecting geometry)
                pooled = graph_mats.mean(dim=0)
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling_type}")

            pooled_mats.append(pooled)

        return torch.stack(pooled_mats)  # [batch_size, d, d]