from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MultilabelMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_labels: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        dropout: float = 0.0,
        use_missingness_mask: bool = True,
    ) -> None:
        super().__init__()
        self.use_missingness_mask = use_missingness_mask
        input_dim = num_features * (2 if use_missingness_mask else 1)
        dims = [input_dim, *hidden_dims, num_labels]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), _activation(activation)])
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, missingness_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_missingness_mask:
            if missingness_mask is None:
                raise ValueError("missingness_mask is required when enabled")
            features = torch.cat([features, missingness_mask], dim=-1)
        return self.network(features)


def build_model(
    framework_config: dict[str, Any], project_config: dict[str, Any]
) -> MultilabelMLP:
    return MultilabelMLP(
        num_features=len(project_config["features"]),
        num_labels=len(project_config["labels"]),
        **framework_config["model"],
    )
