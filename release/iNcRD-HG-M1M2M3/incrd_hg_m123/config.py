"""Fixed configuration of the reported iNcRD-HG + M1 + M2 + M3 model."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 256
    layers: int = 2
    ncn_dim: int = 64
    ncn_rels: tuple = ("co", "knn", "dknn")
    ease_lambdas: tuple = (10.0, 50.0)
