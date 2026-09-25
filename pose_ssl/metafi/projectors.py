"""Projection heads shared by MetaFi SSL methods."""

from __future__ import annotations

from torch import Tensor, nn


class Projector(nn.Module):
    """Two-layer Linear-BatchNorm-ReLU projection head for global MetaFi features."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        for name, value in (("in_dim", in_dim), ("hidden_dim", hidden_dim), ("out_dim", out_dim)):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Project a batch of global encoder vectors."""

        return self.net(features)
