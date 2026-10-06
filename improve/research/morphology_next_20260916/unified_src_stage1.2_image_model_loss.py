import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Bidirectional InfoNCE (NT-Xent) loss for contrastive learning."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, drug_proj, target_proj):
        """
        Args:
            drug_proj:   [B, D] normalized drug projections
            target_proj: [B, D] normalized target projections
        Returns:
            Scalar loss (symmetric drug↔target InfoNCE)
        """
        batch_size = drug_proj.shape[0]
        logits = torch.matmul(drug_proj, target_proj.T) / self.temperature
        labels = torch.arange(batch_size, device=drug_proj.device)

        loss_d2t = F.cross_entropy(logits, labels)
        loss_t2d = F.cross_entropy(logits.T, labels)
        return (loss_d2t + loss_t2d) / 2


def compute_alignment_metrics(drug_proj, target_proj):
    """Compute positive/negative similarity statistics for a batch.

    Returns a dict with pos_sim_mean, neg_sim_mean, gap, alignment.
    """
    batch_size = drug_proj.shape[0]
    sim_matrix = torch.matmul(drug_proj, target_proj.T)

    pos_sim = torch.diagonal(sim_matrix)
    mask = torch.eye(batch_size, device=drug_proj.device).bool()
    neg_sim = sim_matrix[~mask]

    return {
        'pos_sim_mean': pos_sim.mean().item(),
        'neg_sim_mean': neg_sim.mean().item(),
        'gap': (pos_sim.mean() - neg_sim.mean()).item(),
        'alignment': pos_sim.mean().item(),
    }
