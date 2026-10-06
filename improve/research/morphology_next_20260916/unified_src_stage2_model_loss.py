import torch
import torch.nn as nn


class MultiPositiveInfoNCELoss(nn.Module):
    """InfoNCE loss that supports multiple positive pairs per sample.

    Pairs sharing the same 'id' label are treated as positives.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, struct_proj, img_proj, labels):
        """
        Args:
            struct_proj: [B, D] normalized structure projections
            img_proj:    [B, D] normalized image projections
            labels:      [B] integer pair IDs (same ID = positive pair)
        Returns:
            Scalar symmetric loss
        """
        logits = torch.matmul(struct_proj, img_proj.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(struct_proj.device)

        loss = (self._compute_loss(logits, mask) + self._compute_loss(logits.T, mask.T)) / 2
        return loss

    def _compute_loss(self, logits, mask):
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()


def compute_alignment_metrics(struct_proj, img_proj, labels):
    """Compute positive/negative similarity statistics for multi-positive pairs."""
    device = struct_proj.device
    sim_matrix = torch.matmul(struct_proj, img_proj.T)

    labels = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float().to(device)
    neg_mask = 1 - pos_mask

    pos_sim = sim_matrix[pos_mask.bool()]
    neg_sim = sim_matrix[neg_mask.bool()]

    return {
        'pos_sim_mean': pos_sim.mean().item() if len(pos_sim) > 0 else 0.0,
        'neg_sim_mean': neg_sim.mean().item() if len(neg_sim) > 0 else 0.0,
        'gap': (pos_sim.mean() - neg_sim.mean()).item() if len(pos_sim) > 0 and len(neg_sim) > 0 else 0.0,
        'alignment': pos_sim.mean().item() if len(pos_sim) > 0 else 0.0,
    }
