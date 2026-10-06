"""iNcRD-HG + M1 + M2 + M3: released final model of the miRNA-drug association study.

M1 QMP       exact query-edge correction of the two message-passing layers
M2 SP-RNCN   pooled intermediate states of simple 2-hop paths (co/knn/dknn)
M3 MC-EASE   masked-cell EASE collaborative scores (lambda = 10, 50, both sides)
"""
from .config import ModelConfig

__all__ = ["ModelConfig"]
