import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Three-layer MLP projection head with LayerNorm and Dropout."""

    def __init__(self, input_dim, hidden_dim=1024, output_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ContrastiveModel(nn.Module):
    """Two-tower contrastive model for image-based drug-target alignment.

    Args:
        drug_dim:   input dimension of compound image embeddings (e.g. 737)
        target_dim: input dimension of CRISPR image embeddings (e.g. 599)
        hidden_dim: hidden dimension of projection heads
        output_dim: output dimension of projection heads
        dropout:    dropout rate in projection heads
    """

    def __init__(self, drug_dim, target_dim, hidden_dim=1024, output_dim=512, dropout=0.1):
        super().__init__()
        self.drug_projector = ProjectionHead(drug_dim, hidden_dim, output_dim, dropout)
        self.target_projector = ProjectionHead(target_dim, hidden_dim, output_dim, dropout)

    def forward(self, drug_embeds, target_embeds):
        drug_proj = self.drug_projector(drug_embeds)
        target_proj = self.target_projector(target_embeds)
        return drug_proj, target_proj
