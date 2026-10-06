from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .model import NanoTabPFNPUModel

ArrayLike = Union[np.ndarray, torch.Tensor]


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _to_feature_tensor(x: ArrayLike, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        tensor = x.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {tuple(tensor.shape)}.")
    return tensor


def _to_label_tensor(y: ArrayLike, device: torch.device) -> torch.Tensor:
    if isinstance(y, torch.Tensor):
        tensor = y.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
    if tensor.ndim != 1:
        raise ValueError(f"Expected a 1D label vector, got shape {tuple(tensor.shape)}.")
    return tensor


def _load_checkpoint(checkpoint: str | Path | None) -> tuple[dict, Path]:
    if checkpoint is not None:
        resolved = Path(checkpoint).expanduser().resolve()
        payload = torch.load(resolved, map_location="cpu")
        return payload, resolved

    resource = files("puicl").joinpath("checkpoints/latest.pt")
    with as_file(resource) as checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu")
        return payload, Path("puicl/checkpoints/latest.pt")


@dataclass
class PUICLModel:
    """Convenience wrapper around the packaged pretrained model."""

    model: NanoTabPFNPUModel
    device: torch.device
    checkpoint_path: Path

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path | None = None,
        device: str | torch.device = "auto",
    ) -> "PUICLModel":
        resolved_device = _resolve_device(device)
        payload, checkpoint_path = _load_checkpoint(checkpoint)
        model_cfg = payload.get("config", {}).get("model", {})
        model = NanoTabPFNPUModel(
            embedding_size=int(model_cfg.get("embedding_size", 128)),
            num_attention_heads=int(model_cfg.get("num_attention_heads", 8)),
            mlp_hidden_size=int(model_cfg.get("mlp_hidden_size", 256)),
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_outputs=int(model_cfg.get("num_outputs", 2)),
        ).to(resolved_device)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return cls(model=model, device=resolved_device, checkpoint_path=Path(checkpoint_path))

    def predict_logits(
        self,
        x_task: ArrayLike,
        *,
        train_test_split_index: int,
        y_train: ArrayLike | None = None,
    ) -> torch.Tensor:
        x = _to_feature_tensor(x_task, device=self.device)
        split = int(train_test_split_index)
        if split <= 0:
            raise ValueError("train_test_split_index must be >= 1.")
        if split >= x.shape[0]:
            raise ValueError("train_test_split_index must be smaller than the number of rows.")

        if y_train is None:
            y = torch.zeros(split, dtype=torch.float32, device=self.device)
        else:
            y = _to_label_tensor(y_train, device=self.device)
            if y.shape[0] != split:
                raise ValueError(
                    "y_train length must match train_test_split_index. "
                    f"Got len(y_train)={y.shape[0]} and split={split}."
                )

        with torch.no_grad():
            logits = self.model(
                (x.unsqueeze(0), y.unsqueeze(0)),
                train_test_split_index=split,
            ).squeeze(0)
        return logits.detach().cpu()

    def predict_proba(
        self,
        x_task: ArrayLike,
        *,
        train_test_split_index: int,
        y_train: ArrayLike | None = None,
    ) -> torch.Tensor:
        logits = self.predict_logits(
            x_task,
            train_test_split_index=train_test_split_index,
            y_train=y_train,
        )
        return torch.softmax(logits, dim=-1)

    def predict_labels(
        self,
        x_task: ArrayLike,
        *,
        train_test_split_index: int,
        y_train: ArrayLike | None = None,
    ) -> torch.Tensor:
        probs = self.predict_proba(
            x_task,
            train_test_split_index=train_test_split_index,
            y_train=y_train,
        )
        return torch.argmax(probs, dim=-1)

    def score_unlabeled(
        self,
        labeled_positive_features: ArrayLike,
        unlabeled_features: ArrayLike,
    ) -> torch.Tensor:
        labeled = _to_feature_tensor(labeled_positive_features, device=self.device)
        unlabeled = _to_feature_tensor(unlabeled_features, device=self.device)
        if labeled.shape[1] != unlabeled.shape[1]:
            raise ValueError(
                "labeled_positive_features and unlabeled_features must have the same feature dimension. "
                f"Got {labeled.shape[1]} and {unlabeled.shape[1]}."
            )
        x_task = torch.cat([labeled, unlabeled], dim=0)
        probs = self.predict_proba(
            x_task,
            train_test_split_index=labeled.shape[0],
            y_train=torch.zeros(labeled.shape[0], dtype=torch.float32),
        )
        return probs[:, 1]


def load_pretrained_model(
    checkpoint: str | Path | None = None,
    *,
    device: str | torch.device = "auto",
) -> PUICLModel:
    """Load the packaged pretrained checkpoint and return an inference wrapper."""

    return PUICLModel.from_pretrained(checkpoint=checkpoint, device=device)
